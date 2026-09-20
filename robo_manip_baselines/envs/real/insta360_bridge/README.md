# insta360_bridge

A small C++ process that connects to an Insta360 X3/X4 over USB via the
[Insta360 CameraSDK](https://github.com/Insta360Develop/CameraSDK-Cpp), feeds
its video + gyro into [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
(Monocular-Inertial), and publishes both the decoded camera frames and the
estimated 6-DoF pose to Python over a local Unix domain socket. See
`Insta360BridgeProtocol.h` (and its Python counterpart,
`common/utils/Insta360Protocol.py`) for the wire format, and `main.cc`'s
top-of-file comment for the overall data flow.

This bridge replaces the earlier Orbbec Gemini 305 camera setup on the UMI
rig, and the Insta360-estimated pose replaces the HTC Vive Tracker as this
rig's teleoperation input source -- see `teleop/Insta360InputDevice.py` and
`envs/configs/RealUMIDemo.yaml`.

## Status

**Built and verified against real hardware**: an Insta360 X3 (firmware
v1.0.83) over USB, with the real CameraSDK 2.1.8 (`third_party/insta360_sdk`)
and a prebuilt `third_party/ORB_SLAM3`. All of the items below were unknowns
before this and are now confirmed:

- `ins_camera::GyroData` (`include/stream/stream_types.h`) carries both
  accelerometer (`ax/ay/az`) and gyro (`gx/gy/gz`) fields (not the originally
  guessed `gyro_x/y/z`/`accel_x/y/z` names) -- `INSTA360_GYRO_HAS_ACCEL`
  defaults **on**, running ORB-SLAM3 in `IMU_MONOCULAR` mode.
- **`OnGyroData` only fires when the camera is physically set to dual-lens
  ("360") capture mode**, not its single-lens mode -- this is a camera-side
  setting (its own screen: Settings -> General -> Camera Mode on this X3), not
  something forced from software. An earlier version of this file tried to
  force it via `Camera::SetVideoCaptureParams` before `StartLiveStreaming`,
  which turned out to be unnecessary (confirmed by comparing against
  [ai4ce/insta360_ros_driver](https://github.com/ai4ce/insta360_ros_driver), a
  working ROS2 driver using the same CameraSDK struct/callback, which never
  calls it) and caused its own problems (forced resolution to 3840x1920
  regardless of `LiveStreamParam.video_resolution`); removed. With the camera
  confirmed in 360 mode via its own menu, `LiveStreamParam.video_resolution`
  is still not always honored exactly (observed both `3840x1920` and
  `2880x2880` across runs with identical code before the camera's mode was
  confirmed via its screen; consistently `3840x1920` after) -- if you see a
  different resolution, verify the camera's on-screen mode/resolution
  setting first.
- **`ins_camera::GyroData::ax/ay/az` are in units of g (9.80665 m/s^2), not
  m/s^2** -- confirmed via `ai4ce/insta360_ros_driver`'s `OnGyroData`, which
  multiplies by standard gravity before publishing as `sensor_msgs/Imu`
  (which, like `ORB_SLAM3::IMU::Point`, expects m/s^2). `main.cc` now applies
  this conversion; feeding raw g-units in was a real bug that would have
  broken IMU preintegration/gravity alignment.
- **Sphere/360 mode delivers raw dual-fisheye, not a stitched equirectangular
  panorama** -- verified by saving and visually inspecting a live frame: two
  side-by-side circular fisheye images (one per lens, black corners), not one
  panoramic image. `main.cc` crops to the front lens's circle only
  (`frame(Rect(0,0,frame.rows,frame.rows))`) before `TrackMonocular`, since
  `KannalaBrandt8` (`Insta360_X4.yaml`) models exactly one fisheye lens --
  feeding it the full double-wide frame put both lenses under one distortion
  model centered on the gap between them, which can't produce a coherent map.
  The full dual-fisheye frame is still broadcast as-is to Python.
- `GetVideoEncodeType()`/`VideoEncodeType::H265` matched the originally
  guessed API exactly -- no change needed.
- The CameraSDK's actual shared library is `lib/libCameraSDK.so`, matching
  the `CameraSDK` name already in `CMakeLists.txt`.
- Fixed along the way: `CMakeLists.txt`'s `ORB_SLAM3_DIR` path (was one `..`
  short), missing `find_package(Eigen3)`/`find_package(Pangolin)` (needed
  transitively by ORB-SLAM3 headers) and the missing Pangolin link, a
  `Camera::SetStreamDelegate(shared_ptr<StreamDelegate>&)` binding issue
  (needs a named lvalue, not an implicit temporary), and a real bug in
  `StreamDecoder::Decode` that only allocated the output `cv::Mat` on the
  very first decoded frame ever (every frame after that wrote through a
  null buffer, corrupting/crashing on `cv::cvtColor`).

Verified end-to-end with real camera motion: camera connects, live video
decodes (3840x1920 dual-fisheye, cropped to the 1920x1920 single-lens
circle and resized to 800x800 with a 370px circular mask for SLAM),
IMU samples flow (partial per-frame coverage, see below), and **tracking
reaches the `OK` state with plausible pose estimates** (tens of centimeters
of translation matching real hand motion) -- confirmed across 3 separate
init cycles in one 30s motion test. Frame/pose messages publish correctly
over the Unix socket (checked against `Insta360Protocol.py`'s wire format
with a throwaway Python client).

**RESOLVED -- sustained tracking now works.** Measured on an X3 (firmware
v1.1.6) in the Docker image, 70s of continuous handheld motion:

| metric | value |
|---|---|
| tracking `OK` | **79.0%** (979 poses) |
| `INIT` | 19.0% (235) |
| `LOST` | 2.1% (26) |
| `Empty IMU measurements vector!!!` | **0** |
| crashes | none (previously 4-27s to a hard crash, every run) |

Three separate bugs had to be fixed to get here; all are documented at their
fix sites:

1. **`-march=native`** (stripped in `Dockerfile.insta360`). It enables AVX,
   raising Eigen's alignment requirement for fixed-size vectorizable types
   from 16 to 32 bytes, which the vendored 2011-era g2o does not honour when
   allocating graph vertices/edges. Symptom: `double free or corruption (out)`
   inside `g2o::HyperGraph::clear()` the instant the first global bundle
   adjustment finished -- i.e. immediately after `New Map created with N
   points`.
2. **The IMU queue race** (`main.cc`, `TakeImuSamplesForRecording`). The main
   broadcast loop and the SLAM thread both drained the same destructive IMU
   queue, and the main loop ran on every decoded frame (~30fps) even with no
   `--record`, throwing the samples away. The SLAM thread was left feeding
   `TrackMonocular` empty windows; keyframes born from those frames carry no
   preintegration, and `LocalMapping::InitializeIMU` ->
   `Optimizer::InertialOptimization` then dereferences a null
   `mpImuPreintegrated`. Symptom: SIGSEGV in
   `IMU::Preintegrated::SetNewBias`, seconds after tracking first reached
   `OK`. Recording now has its own queue.
3. **`IMU.Frequency`** was 318.7 (firmware v1.0.83) against a real ~500Hz on
   v1.1.6 -- see `Insta360_X4.yaml`; ORB-SLAM3 scales all four noise terms by
   `sqrt(frequency)`.

Remaining known imperfection: the largest single OK-to-OK position step in
that run was 1.512 m, which is a relocalisation jump rather than real motion
(`Insta360InputDevice.py`'s `POSE_JUMP_POS_THRESHOLD` exists for this). The
19% `INIT` share is re-initialisation after those events.

Earlier history, for context -- two distinct failure modes seen before the
fixes above:
- `TRACK_REF_KF: Less than 15 matches!! / Fail to track local map!` -- vision
  tracking loses too many feature matches against the reference keyframe.
  This was the dominant reset cause (7 of 10 in one session), and was
  originally attributed to the then-placeholder zero distortion coefficients.
- `IMU is not or recently initialized. Reseting active map...` -- ORB-SLAM3's
  IMU initializer (scale/gravity/bias estimation) doesn't converge before the
  map resets. This was attributed to `OnGyroData`'s batched/intermittent
  delivery (roughly 20-40% of frames seeing zero new IMU samples, i.e.
  `PreintegrateIMU` logging "Empty IMU measurements vector!!!").

  **That intermittency is gone on firmware v1.1.6.** Measured from a 20.5s
  `--record` capture on an X3 running v1.1.6 (the earlier testing was on
  v1.0.83): 10450 gyro + 10450 accel samples, a steady ~500 Hz (mean
  inter-sample dt exactly 2.000 ms), 616 frames at 29.95 fps, and **0 of 615
  frame intervals with zero new IMU samples** -- a consistent 17 IMU samples
  per frame interval. Sanity checks on the same capture: accelerometer
  magnitude 9.708 m/s^2 at rest (sd 0.016), gyro magnitude 0.0037 rad/s,
  confirming the g -> m/s^2 conversion in `main.cc` is correct.

  Note this also means `IMU.Frequency` changed from 318.7 (v1.0.83) to 500
  -- see `Insta360_X4.yaml`, where it is load-bearing for noise scaling.

**Both observations predate the real calibration.** `Insta360_X4.yaml` now
carries measured Basalt intrinsics (0.50px reprojection), a real `IMU.T_b_c1`
and measured Allan-deviation noise, and `ORBextractor.nFeatures` was raised
1250->2000. The reset rate has not yet been re-measured against real hardware
with those values in place -- doing so is the first thing to check.

**Still unverified / not yet done:**
- IMU timestamp unit (assumed microseconds) is still unverified.
- ORB vocabulary load was previously guessed at "up to 30s". Measured on the
  Docker image (i9-13900HX): **3.4s** to construct a full
  `ORB_SLAM3::System` in `IMU_MONOCULAR` with this YAML and the viewer on.
  Not worth optimizing.
- `Insta360_X4.yaml`'s intrinsics, `IMU.T_b_c1` and IMU noise are now real
  measured values (see Calibration below). The one open item is that
  `IMU.T_b_c1` comes from `basalt_calibrate_imu`'s phase-1 result, extracted
  by hand from log output -- see that file's header comment and the GUI
  procedure below, which supersedes the `--no-gui` run that made this
  necessary.
- Only tested on an X3; X4/X5 behavior is unverified.
- Whether `OnGyroData`'s batched delivery rate is a hard SDK/firmware limit
  or something tunable is unknown.

## Build

### Docker (recommended)

`Dockerfile.insta360` / `docker-compose.insta360.yaml` at the repo root build
ORB-SLAM3, Pangolin, Basalt and the python calibration tooling into one image,
leaving the host clean. This is the supported path.

```
# host prerequisites, once per session
newgrp docker                 # if your login session predates `usermod -aG docker`
xhost +local:docker           # revoke later with `xhost -local:docker`
mkdir -p /tmp/insta360

docker compose -f docker-compose.insta360.yaml build
docker compose -f docker-compose.insta360.yaml run --rm insta360_bridge bash
```

Inside the container, ORB-SLAM3 is prebuilt at `/opt/ORB_SLAM3`
(`$ORB_SLAM3_DIR`, vocabulary at `$ORB_VOCAB`) and the bridge is built from the
bind-mounted source:

```
cmake -B /tmp/bridge_build -S . \
    -DINSTA360_SDK_DIR=$INSTA360_SDK_DIR \
    -DORB_SLAM3_DIR=/opt/ORB_SLAM3 \
    -DCMAKE_PREFIX_PATH=/opt/pangolin-0.6
cmake --build /tmp/bridge_build -j16
```

The CameraSDK is **not** baked into the image (proprietary, not
redistributable) -- unpack it to `third_party/insta360_sdk/` on the host, where
the compose file bind-mounts it from. The image needs no GPU and no NVIDIA
Container Toolkit; `/dev/dri` gives the Pangolin viewers hardware GL.

### Native

1. **Apply for the Insta360 CameraSDK** at
   <https://www.insta360.com/sdk/apply> (requires approval; the public
   `Desktop-CameraSDK-Cpp` repo is a demo/docs wrapper only, not the SDK
   itself). Unpack it somewhere and note the path (must contain
   `include/camera/` and `lib/`).
2. **Build ORB-SLAM3** (vendored as `third_party/ORB_SLAM3`, a git
   submodule -- run `git submodule update --init third_party/ORB_SLAM3` if
   not already checked out) per that project's own `README.md`/`build.sh`.
   This step is independent of the Insta360 SDK and can be done first.
   Note: upstream appends `-std=c++11` directly to `CMAKE_CXX_FLAGS`, which
   overrides `-DCMAKE_CXX_STANDARD=14`; Pangolin v0.6 needs C++14, so that
   has to be patched out (see `Dockerfile.insta360` for the exact sed).
3. Install system dependencies: OpenCV, ffmpeg dev packages
   (`libavcodec-dev`, `libavutil-dev`, `libswscale-dev`), `libusb-dev`,
   `libudev-dev` (the last two per the CameraSDK README's Linux driver
   section).
4. Build this bridge:
   ```
   cmake -B build -DINSTA360_SDK_DIR=/path/to/unpacked/sdk \
       [-DINSTA360_GYRO_HAS_ACCEL=OFF]   # to force the IMU-less MONOCULAR fallback
   cmake --build build
   ```

## Camera setup (one-time, per camera)

- **Set the camera to dual-lens ("360") capture mode** via its own screen
  (Settings -> General -> Camera Mode on the X3 tested here) -- confirmed
  required for `OnGyroData`/IMU delivery to work at all; single-lens mode
  never delivers gyro/accel data regardless of any SDK call. If the live
  stream's resolution looks wrong (this code expects `3840x1920`
  dual-fisheye), check this setting first.
- **Set the video framerate to 30fps specifically (e.g. "4K/30fps"), not
  60fps or 100fps.** Verified against real hardware, extensively: at
  60fps or 100fps, `StartLiveStreaming` still reports success and the
  camera confirms the requested `VIDEO RES` internally, but
  `StreamDelegate::OnVideoData` is never invoked at all -- zero video
  packets reach the SDK client, regardless of `LiveStreamParam.
  video_resolution`/`using_lrv`/bitrate. Tried and ruled out: matching the
  live-stream resolution request to the camera's current fps, the LRV
  proxy stream, is_h265 detection (consistently correct). No alternative
  high-framerate live-streaming API exists in the SDK headers. Conclusion:
  this SDK's live-streaming path appears to only support 30fps on this
  camera/firmware; higher framerates are presumably recording-only. Only
  30fps has been confirmed to actually deliver video.
- **Switch the camera to Android USB mode**: by default an Insta360 camera
  enumerates as a USB mass-storage ("U disk") device when plugged in, which
  the CameraSDK cannot talk to. On X4/X5, a mode-selection popup appears on
  the camera's own screen when connected -- choose Android mode there. See
  the CameraSDK README's "Switching the Camera to Android Mode" section for
  older models.
- **The camera drops out of Android mode when it sleeps.** Observed
  repeatedly in real use: after a few minutes idle (or between back-to-back
  bridge runs) the camera stops being visible to the SDK while still being
  electrically enumerated. The symptom is unambiguous and worth learning:

  | check | still works | means |
  |---|---|---|
  | `lsusb` shows `2e1a:0002` | yes | USB/power/cable/cgroup rules are all fine |
  | `DeviceDiscovery` finds 0 devices | no | camera left Android/accessory mode |

  So `[insta360_bridge] No Insta360 camera found` while `lsusb` *does* show
  the device is **not** a permissions, container or udev problem -- do not go
  chasing `--privileged`. Wake the camera and re-select Android mode on its
  screen (or replug it). Budget for this between calibration runs.

- **USB permissions**: the CameraSDK README notes Linux may need `sudo` to
  access the camera, or a udev rule granting the USB device (vendor ID
  `0x2e1a` per that README) access without it -- prefer a udev rule
  (`/etc/udev/rules.d/`) over running this bridge as root.

## Calibration

`Insta360_X4.yaml` now carries **real measured values throughout**: Basalt
KannalaBrandt8 (kb4) intrinsics at 0.50px reprojection, an `IMU.T_b_c1`
extrinsic from `basalt_calibrate_imu`, and IMU noise parameters from an Allan
deviation measurement. See that file's own header comment for the provenance
and caveats of each, and `calibration/README.md` for the script inventory and
the KB4 monotonic-range check.

The raw recordings and Basalt `calibration.json` behind those numbers did not
survive a machine move -- only the distilled values in the YAML. Re-deriving
them needs a fresh recording (below).

### Intrinsics

The current values come from `basalt_calibrate` on an AprilGrid recording (see
the next section -- the same recording serves both stages). An older
`cv2.fisheye.calibrate` path also exists in `calibration/`
(`gen_checkerboard.py` -> `capture_calib_frames.py` ->
`run_fisheye_calibration.py`) and produced the superseded 26.6px fit; it is
kept as a fallback and is documented in `calibration/README.md`. Two hard-won
notes from it, if you ever use it again:

- `cv2.findChessboardCornersSB` is required -- classic `findChessboardCorners`
  fails on this lens's distortion even when the board is clearly visible.
- Coverage across center/mid/outer radial zones is essential; with centrally
  clustered images `cv2.fisheye.calibrate` diverges to 100s-1000s of px RMS.
  Pass `None` (not zeroed arrays) for `K`/`D`, and use a tight optimizer
  (200 iterations, 1e-8 eps) -- looser settings settle into bad local minima
  on the same data.

### Camera-IMU extrinsic (`IMU.T_b_c1`) calibration

Uses [Basalt](https://gitlab.com/VladyslavUsenko/basalt) rather than the more
commonly-referenced [Kalibr](https://github.com/ethz-asl/kalibr): same
underlying approach (B-spline trajectory + IMU noise model + reprojection,
same noise-parameter convention), but Basalt is plain CMake+vcpkg with a
prebuilt binary installer. It is preinstalled in the Docker image
(`--target calib`); natively, `curl -LsSf https://gitlab.com/VladyslavUsenko/\
basalt/-/raw/master/scripts/install.sh | sh` installs to `~/.local`.

#### 1. Target

`python3 calibration/gen_aprilgrid.py` generates a printable 6x6 AprilGrid
(`aprilgrid_a4.png`/`.pdf`, A4 landscape, nominal 25mm tags). **Print at 100%
scale, no "fit to page", then measure the printed tag and update `tagSize` in
`calibration/aprilgrid.json`** if it differs.

Tags are rendered locally from `calibration/t36h11_codes.json` using Kalibr's
bit-layout/rotation/**2-cell black border** convention -- NOT the standard
1-cell-border PNGs from `AprilRobotics/apriltag-imgs`. Basalt vendors its own
AprilTag implementation configured with `blackTagBorder=2`, and a 1-cell target
detects as **zero** corners on every frame. See `calibration/README.md`.

#### 2. Recording

**Fix the target to a wall; move the camera.** (Earlier revisions of this file
said "moving the printed target (or the camera)" -- for the camera-IMU stage a
moving target breaks the static-world assumption the spline fit rests on.)

```
insta360_bridge --socket /tmp/insta360/calib.sock --no-slam \
    --lens back --preview --record <prefix>
```

Motion profile -- this is what the two failed earlier attempts got wrong:

- **Distance 0.4-0.8 m, roughly constant, all 36 tags in view throughout.**
  The most important constraint. An earlier 5423-frame sweep with "close-up to
  full-room pull-back" range stalled at ~9-26px reprojection because distant
  tags subtend too few pixels in an 800x800 fisheye frame, poisoning the
  initial trajectory estimate that `initCamImuTransform` and `initOptimization`
  both depend on. Camera-only intrinsics survived it (they average over many
  frames), which is why that sweep "converged fine" for the camera and not the
  IMU.
- **60-90 s total** (1800-2700 frames at 30fps): 3 s still -> ~10 s per axis
  translation-only at 1-2 Hz and 15-30 cm amplitude -> ~20 s combined 6-DoF
  (simultaneous translation and +-45 deg rotation on all three axes) -> 3 s
  still. Rotation projects gravity differently onto each accel axis, which is
  what makes the extrinsic observable at all.
- Watch for motion blur. `main.cc` already forces a 1/120 s shutter; light the
  target well and slow down rather than dropping shutter further.

Then `python3 calibration/convert_record_to_euroc.py <prefix> <euroc_dir>`.
This writes `cam0` and symlinks `cam1` to it -- Basalt's `EurocIO` loader
hardcodes `num_cams=2`, so `--cam-types` must be given **twice**. Frames are
converted to real grayscale rather than left as BGR.

#### 3. Camera stage

```
basalt_calibrate --dataset-path <euroc_dir> --dataset-type euroc \
    --aprilgrid calibration/aprilgrid.json --result-path <result_dir>/ \
    --cache-name <name> --cam-types kb4 kb4
```

Run this even if you only want the extrinsic: `basalt_calibrate_imu` loads
`<result_dir>/calibration.json`, and the corner detections are cached under
`--cache-name` and reused by the IMU stage. It is also a free cross-check --
if the fresh intrinsics land within ~1% of the YAML's current values, the
recording, target and conversion pipeline are all independently validated.

#### 4. IMU stage -- use the GUI

```
basalt_calibrate_imu --dataset-path <euroc_dir> --dataset-type euroc \
    --aprilgrid calibration/aprilgrid.json --result-path <result_dir>/ \
    --cache-name <name> \
    --gyro-noise-std 1.6858e-04 --accel-noise-std 4.0236e-03 \
    --gyro-bias-std 7.9248e-05 --accel-bias-std 9.5337e-04
```

**Pass the raw, uninflated Allan measurements** (the values above), not the
x5/x8-inflated numbers in `Insta360_X4.yaml`. That inflation exists to make
ORB-SLAM3's *online* optimizer appropriately distrustful under handheld
vibration; feeding inflated values to an offline batch fit just downweights the
IMU and makes the extrinsic *less* observable.

**Do not use `--no-gui`.** `calibrate_imu.cpp`'s headless path is a hardcoded
sequence that unconditionally runs a second phase with `setOptImuScale(true)`
after the good one, and only calls `saveCalib()` after all phases complete. On
an under-excited recording that scale phase converges to a degenerate solution
(accel_scale diag ~0.1, gravity norm 0.93 m/s^2 instead of 9.8) -- a real
minimum of an ill-posed problem, not a numerical bug. That is why the current
`IMU.T_b_c1` had to be scraped by hand from phase-1 log output.

The GUI constructor defaults `opt_cam_time_offset`, `opt_imu_scale` and
`opt_mocap` all to **off**, so clicking straight through saves exactly the
phase-1 solution. Button order:

`load_dataset` -> `detect_corners` -> `init_cam_poses` -> `init_cam_imu` ->
`init_opt` -> `optimize` (or tick `opt_until_converge`) -> **inspect stdout** ->
`save_calib`

Accept the result only if:

- gravity norm is in [9.6, 9.9] m/s^2 (the current value came from a run at
  9.58 -- acceptable but at the edge),
- mean reprojection error < 1.0 px (current: 0.66 px),
- `T_i_c` translation is physically plausible for the rig (current:
  16.5/12.8/-2.2 cm).

Then click `save_calib` and stop. Optionally afterwards tick
`opt_cam_time_offset` (usually well-observable) and re-inspect. Only then
consider `opt_imu_scale` -- and if gravity drifts off 9.81 or the accel-scale
diagonal departs from ~1.0, **do not save again**. The GUI's value here is that
saving is a decision rather than a consequence.

#### 5. Into the YAML

Basalt's `calibration.json` stores `T_imu_cam` as `{px,py,pz,qx,qy,qz,qw}`
(**qw last**). ORB-SLAM3's `IMU.T_b_c1` is camera-to-body(IMU), the same
direction -- a direct 4x4 composition, **no inversion**. Sanity-check the
translation magnitude and signs against the physical rig before pasting; a sign
flip here produces plausible-looking-but-wrong tracking rather than an obvious
error.

### IMU noise parameters

The values in `Insta360_X4.yaml` come from `allantools` `oadev` on a 213 s
stationary recording, then inflated (Noise x5, Walk x8) -- see that file's
comment for why the inflation is a real modelling fix rather than a crash
guard.

The white-noise terms are solid (log-log slope within ~2% of -0.5). The two
random-walk terms are not: 213 s caps the longest reliable cluster time at
tau ~= 21 s, but bias instability and random walk appear at tau ~= 10-100 s.
A **60 minute** stationary run (tau_max ~= 360 s) after a 10+ minute thermal
soak with the stream already running would settle them. Note `--record` also
writes ~6-7 GB/hour of video; the IMU CSV alone is ~30 MB/hour.

## Run

In the Docker image (bridge built to `/tmp/bridge_build`, see Build above):

```
/tmp/bridge_build/insta360_bridge \
    --socket /tmp/insta360/hand.sock \
    --vocab  $ORB_VOCAB \
    --settings ./Insta360_X4.yaml \
    --lens back \
    [--viewer]   # opens ORB-SLAM3's Pangolin viewer (map/keypoints/camera
                 # pose); needs a display (DISPLAY set, X server reachable).
                 # Off by default -- production teleop runs headless.
```

`--lens back` matters: the current `Camera1.*` intrinsics were fit against the
back lens. Drop it only after recalibrating for the front lens.

The socket lives under `/tmp/insta360/`, which `docker-compose.insta360.yaml`
bind-mounts from the host so the teleop container and host-side Python clients
can reach it too. Its path must match `insta360_ids` in
`envs/configs/RealUMIDemo.yaml` and `device_params.socket_path` in
`teleop/configs/Insta360UMI.yaml` (both currently say
`/tmp/insta360_hand.sock`, and `insta360_ids` is still commented out).

### Without hardware

`misc/MockInsta360Bridge.py` serves the same wire protocol with synthetic
frames and poses, for exercising the socket path, the protocol and any client:

```
python3 /workspace/RoboManipBaselines/robo_manip_baselines/misc/MockInsta360Bridge.py \
    --socket /tmp/insta360/hand.sock
```
