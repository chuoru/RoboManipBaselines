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
- **A second physical Insta360 X3 unit delivered a genuinely square
  `2880x2880` frame** (confirmed via `main.cc`'s one-time "Decoded frame
  size" startup log line) even with the on-camera mode set to dual-lens/
  "360" and after a power cycle -- unlike the first unit, this one
  actually honored `main.cc`'s `LiveStreamParam.video_resolution` request
  at face value instead of ignoring it in favor of its native 3840x1920.
  `main.cc` requests `RES_3840_1920P30` directly for exactly this reason
  (previously requested the lower `RES_1920_960P30` "preview" resolution,
  relying on the first unit's override behavior) -- confirmed fixed on
  the second unit and harmless on the first (which already always
  delivered 3840x1920 regardless of the request). If a frame still comes
  back square, re-check the camera's on-screen Camera Mode setting and
  try a power cycle before suspecting the code.
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
decodes (3840x1920 dual-fisheye, cropped to 1920x1920 single-lens for SLAM),
IMU samples flow (partial per-frame coverage, see below), and **tracking
reaches the `OK` state with plausible pose estimates** (tens of centimeters
of translation matching real hand motion) -- confirmed across 3 separate
init cycles in one 30s motion test. Frame/pose messages publish correctly
over the Unix socket (checked against `Insta360Protocol.py`'s wire format
with a throwaway Python client).

**Still fragile -- tracking loses lock after ~0.1-0.3s and re-initializes,
repeatedly.** Two distinct failure modes observed in the same test:
- `TRACK_REF_KF: Less than 15 matches!! / Fail to track local map!` -- vision
  tracking loses too many feature matches against the reference keyframe.
  Consistent with `Insta360_X4.yaml`'s placeholder zero distortion
  coefficients (`k1-k4: 0.0`) not matching the real ~190 degree fisheye lens.
- `IMU is not or recently initialized. Reseting active map...` -- ORB-SLAM3's
  IMU initializer (scale/gravity/bias estimation) doesn't converge before the
  map resets. Plausibly related to `OnGyroData`'s batched/intermittent
  delivery relative to video frame rate (roughly 20-40% of frames see zero
  new IMU samples in the interval since the last frame, i.e. `PreintegrateIMU`
  logs "Empty IMU measurements vector!!!" for that step) combined with
  placeholder `IMU.T_b_c1`/noise parameters.

Real calibration (see Calibration below) is the natural next step for both --
this was never expected to produce sustained accurate tracking without it.

**Still unverified / not yet done:**
- IMU timestamp unit (assumed microseconds) and the up-to-30s ORB
  vocabulary-load startup latency are unverified/unoptimized.
- `Insta360_X4.yaml`'s camera intrinsics, `IMU.T_b_c1` extrinsic, and IMU
  noise parameters are still placeholders -- see Calibration below.
- Only tested on an X3; X4/X5 behavior is unverified.
- Whether `OnGyroData`'s batched delivery rate is a hard SDK/firmware limit
  or something tunable is unknown.

## Build

1. **Apply for the Insta360 CameraSDK** at
   <https://www.insta360.com/sdk/apply> (requires approval; the public
   `Desktop-CameraSDK-Cpp` repo is a demo/docs wrapper only, not the SDK
   itself). Unpack it somewhere and note the path (must contain
   `include/camera/` and `lib/`).
2. **Build ORB-SLAM3** (vendored as `third_party/ORB_SLAM3`, a git
   submodule -- run `git submodule update --init third_party/ORB_SLAM3` if
   not already checked out) per that project's own `README.md`/`build.sh`.
   This step is independent of the Insta360 SDK and can be done first.
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
- **USB permissions**: the CameraSDK README notes Linux may need `sudo` to
  access the camera, or a udev rule granting the USB device (vendor ID
  `0x2e1a` per that README) access without it -- prefer a udev rule
  (`/etc/udev/rules.d/`) over running this bridge as root.

## Calibration

`Insta360_X4.yaml`'s `IMU.T_b_c1` extrinsic and IMU noise parameters are
still placeholders (see that file's own comments); `Camera1.fx/fy/cx/cy/
k1-k4` now come from a real OpenCV fisheye calibration (26.6px RMS -- a
first-pass fit, not a final precise one) using the tooling in
`calibration/`:

1. `python3 calibration/gen_checkerboard.py` generates a printable 9x6-
   internal-corner checkerboard target (`calibration/checkerboard_a3.png`/
   `.pdf`, A3, nominal 35mm squares -- print at 100% scale, no "fit to
   page", and measure the actual printed square size with a ruler
   afterward, since printer scaling drifts).
2. Run the bridge in lightweight capture-only mode (skips ORB-SLAM3/vocab
   load entirely -- much faster and lower CPU than tracking mode, and the
   only mode that gave a smooth enough live preview for aiming the
   camera): `./build/insta360_bridge --socket /tmp/insta360_calib.sock
   --no-slam`
3. `python3 calibration/capture_calib_frames.py /tmp/insta360_calib.sock
   <output_dir> <duration_sec> <target_count> [front|back]` opens a live
   `cv2.imshow` preview (checkerboard corners overlaid in green when
   detected via `cv2.findChessboardCornersSB` -- the classic
   `findChessboardCorners` fails on this lens's fisheye distortion even
   when the board is clearly visible) with three guide rings marking
   center/mid/outer radial zones. **Actively move the checkerboard (or the
   camera) so it visits all three zones, especially outer/near the lens
   edge** -- verified essential: with only centrally-clustered images,
   `cv2.fisheye.calibrate` diverges badly (100s-1000s of px RMS, nonsensical
   distortion coefficients); mixing zones is what let it converge. The
   script caps per-zone saves (biased toward outer) so it won't finish
   until real coverage is achieved.
4. `python3 calibration/run_fisheye_calibration.py <output_dir>
   <square_size_mm>` re-detects corners at full resolution and runs
   `cv2.fisheye.calibrate`. Print the resulting `fx/fy/cx/cy/k1-k4` into
   `Camera1.*` in `Insta360_X4.yaml`. Note: passing zero-initialized
   `K`/`D` with `CALIB_RECOMPUTE_EXTRINSIC` crashes cv2's `InitExtrinsics`
   (`fabs(norm_u1) > 0` assertion) on this data -- pass `None` for `K`/`D`
   instead, and use a reasonably tight optimizer (`200` iterations, `1e-8`
   eps): looser settings were observed to settle into bad local minima
   (900-2000+ px RMS) on the *same* data that converges to 26.6px with
   tighter ones.

The steps above are intrinsic-only and don't touch `IMU.T_b_c1`/noise.
Getting those needs a proper camera-IMU joint calibration -- solving for
the spatial (and implicitly temporal) alignment between the camera and IMU
requires a continuous-time joint optimization over trajectory + IMU bias +
reprojection error, not something with a simple library-call equivalent to
`cv2.fisheye.calibrate` above, so this reuses an established, validated
tool rather than hand-rolling it.

### Camera-IMU extrinsic (`IMU.T_b_c1`) calibration

Uses [Basalt](https://gitlab.com/VladyslavUsenko/basalt) rather than the
more commonly-referenced [Kalibr](https://github.com/ethz-asl/kalibr):
same underlying approach (B-spline trajectory + IMU noise model +
reprojection, same noise-parameter convention), but Basalt is plain
CMake+vcpkg with a prebuilt binary installer for Ubuntu 22.04+ amd64 (no
ROS, no Docker) -- confirmed working: `curl -LsSf
https://gitlab.com/VladyslavUsenko/basalt/-/raw/master/scripts/install.sh
| sh`, installs to `~/.local` (source the printed env file, or restart
your shell, before running any `basalt_*` command below).

1. `python3 calibration/gen_aprilgrid.py` generates a printable 6x6
   AprilGrid target (`calibration/aprilgrid_a4.png`/`.pdf`, A4 landscape,
   nominal 25mm tags -- print at 100% scale, no "fit to page", and measure
   the actual printed tag size afterward; update `tagSize` in the
   generated `calibration/aprilgrid.json` if it differs). Tags are
   rendered from `calibration/t36h11_codes.json` (the tag36h11 family's
   36-bit codes, copied from `ethz-asl/kalibr`'s
   `kalibr_create_target_pdf`) using Kalibr's own bit-layout/rotation/
   **2-cell black border** convention -- NOT the standard pre-rendered
   tag36h11 PNGs from
   [AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs)
   (1-cell border) this originally used. That distinction matters a lot:
   Basalt vendors its own independent AprilTag implementation
   (`thirdparty/apriltag/ethz_apriltag2`, same lineage as Kalibr's codes)
   configured with `blackTagBorder=2` -- confirmed against real hardware
   that a target built from the standard 1-cell-border images (readable
   by `cv2.aruco` and the modern `apriltag` reference library, both found
   all 36 tags on a real captured frame) was detected as exactly **zero**
   corners by `basalt_calibrate`'s `detect_corners` on every frame of two
   full recording sweeps -- tracing that border-width mismatch is what
   led here. Tag placement (grid position, not per-tag rendering) follows
   Kalibr's `id = n_cols*row + col` convention, which Basalt's AprilGrid
   expects too since it's the same target format.
   `calibration/live_aprilgrid_preview.py`'s live detection overlay still
   uses `cv2.aruco` (1-cell border) for lack of a simple alternative, so
   it will not actually detect this target -- it still works as a plain
   live view for aiming, just without the detection count/overlay.
2. Record a calibration clip moving the printed target (or the camera)
   through the field of view with **varied rotation on all three axes**,
   not just translation -- camera-IMU extrinsic calibration specifically
   needs rotational excitation to be observable (unlike the intrinsics-only
   capture above, which just needs radial coverage):
   `./build/insta360_bridge --socket /tmp/insta360_calib.sock --no-slam
   --lens <front|back> --preview --record <prefix>`. This now also writes
   `<prefix>_frame_timestamps.csv` (real per-frame camera-SDK timestamps,
   same clock domain as `<prefix>.csv`'s IMU samples) alongside the usual
   `<prefix>.avi`/`.csv` -- added specifically so the conversion below can
   use real capture times instead of assuming a constant frame rate.
3. `python3 calibration/convert_record_to_euroc.py <prefix> <euroc_dir>`
   converts that recording into the EuRoC MAV dataset folder layout
   `--dataset-type euroc` expects (`<euroc_dir>/mav0/{cam0,imu0}/...`).
   Only `cam0` is written -- Basalt's `EurocIO` loader hardcodes
   `num_cams=2`, but confirmed empirically that a `cam0`-only dataset with
   a single `--cam-types` entry loads without error (nothing ever looks
   for `cam1`'s files). Frames are converted to real grayscale
   (`cv2.cvtColor`) rather than left as BGR -- Basalt's own loader would
   otherwise take a crude single-channel slice as "intensity", worse for
   corner detection.
4. Run the camera calibration first (both are interactive Pangolin GUI
   tools -- `load_dataset`, `detect_corners`, etc. are buttons you click
   through, not flags):
   `basalt_calibrate --dataset-path <euroc_dir> --dataset-type euroc
   --aprilgrid calibration/aprilgrid.json --result-path <result_dir>
   --cam-types kb4` (`kb4` = Kannala-Brandt 4-coefficient, matching
   `Insta360_X4.yaml`'s `KannalaBrandt8` model).
5. Then the camera-IMU calibration, same `--result-path`:
   `basalt_calibrate_imu --dataset-path <euroc_dir> --dataset-type euroc
   --aprilgrid calibration/aprilgrid.json --result-path <result_dir>
   --cam-types kb4 --gyro-noise-std <..> --accel-noise-std <..>
   --gyro-bias-std <..> --accel-bias-std <..>` (noise params: real Insta360
   X3 IMU datasheet values aren't known -- start from the placeholders
   already in `Insta360_X4.yaml`, which are generic VINS-Mono defaults, not
   measured against this hardware either).
6. `save_calib` in the GUI writes `calibration.json` to `<result_dir>` --
   its `T_imu_cam` (or `T_cam_imu`, inverted as needed -- check Basalt's
   output convention against ORB-SLAM3's `IMU.T_b_c1` = camera-to-body
   before copying) is the extrinsic to paste into `Insta360_X4.yaml`.

Not yet done (as of when this section was written): an actual calibration
recording+run via this Basalt path (this only sets up and smoke-tests the
tooling/pipeline -- confirmed `basalt_calibrate` loads a real converted
recording without error, but no target had actually been photographed for
a real calibration yet). `Insta360_X4.yaml`'s own header comments document
real Basalt runs completed later, with results noticeably rougher on the
extrinsic (15.0px reprojection) than this path's own 0.66px first-unit
number -- see the next section for the current recommended alternative.

### Camera-IMU joint calibration via OpenImuCameraCalibrator (current default)

[`OpenImuCameraCalibrator`](https://github.com/urbste/OpenImuCameraCalibrator)
(OpenICC) -- the tool UMI's own SLAM recipe uses for GoPro -- is now
preferred over the Basalt path above: it gave UMI's team good results on a
similarly wide-FOV action camera, and its own per-run reprojection-error
output makes it easy to directly compare against the numbers already
recorded in `Insta360_X4.yaml`'s header. It has no GoPro-specific
dependency in its *calibration* step (only its `run_gopro_calibration.py`
convenience script assumes GPMF telemetry muxed into the video -- irrelevant
here either way, since Insta360 has no such muxing); the smartphone/Pilotguru
path (originally written for a phone with a separately-recorded IMU stream,
documented in its `docs/samsung_s20_calibration.md`) is the structurally
closest match to this bridge's own `--record` output (separate video +
IMU csv), so that's what `calibration/convert_record_to_openicc.py` targets.

**Build** (not yet packaged as a simple installer, unlike Basalt above):
1. `git clone https://github.com/urbste/OpenImuCameraCalibrator.git` --
   NOT vendored as a submodule of this repo (see the commit that added it
   for why: it only needs a one-time system-wide `find_package(Theia)`,
   not an in-tree link, so pinning it in-tree just adds friction for no
   benefit).
2. Its `find_package(Theia REQUIRED)` needs
   [`urbste/pyTheiaSfM`](https://github.com/urbste/pyTheiaSfM) built+installed
   first, at the exact commit OpenICC's own `Dockerfile` pins (`69c3d37`) --
   clone that as a **plain** clone too (its own CMake git-hook-install logic
   breaks if cloned as a submodule, since `.git` is then a gitlink file, not
   a directory), `git checkout 69c3d37`, then `cmake -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_INSTALL_PREFIX=<prefix> && make install`.
3. pyTheiaSfM needs Ceres **>= 2.1** (`ceres::Manifold`/`SetManifold` --
   this repo's system Ceres may be older, e.g. 2.0.0 on Ubuntu 22.04's own
   package; confirmed building Ceres 2.1.0 from source to the same
   `<prefix>` and pointing pyTheiaSfM's `-DCMAKE_PREFIX_PATH=<prefix>` at
   it fixes a `'Manifold' is not a member of 'ceres'` compile error).
4. Build OpenICC itself: `cmake -DCMAKE_PREFIX_PATH=<prefix>
   -DCMAKE_BUILD_TYPE=Release && make`, then `pip install -r requirements.txt`.

**KNOWN BUILD BLOCKER on OpenCV >= 4.7** (confirmed against OpenCV 4.11):
`src/core/board_extractor.cc` (ChArUco board detection) and
`camera_calibrator.cc` are written against OpenCV's pre-4.7 **legacy**
`aruco` contrib module (`cv::aruco::DetectorParameters::create()`,
`cv::aruco::CharucoBoard::create()`, a `.chessboardCorners` member,
`#include <opencv2/aruco/dictionary.hpp>`), which OpenCV 4.7 restructured
into the `objdetect` module with a constructor/class-based API instead (no
`::create()`, no that header path, `CharucoBoard`'s corners now behind
`getChessboardCorners()`) -- and, unlike the handful of call sites this
same break caused in `third_party/ORB_SLAM3` (fixed by wrapping return
values in `cv::makePtr<...>`, see
`orb_slam3_hardware_fixes/README.md`), OpenICC's break is NOT a mechanical
fix: `cv::aruco::interpolateCornersCharuco` (the function
`board_extractor.cc`'s detection loop is actually built around) was
**removed entirely**, replaced by a different, class-based workflow
(`cv::aruco::CharucoDetector::detectBoard()`) with no 1:1 call-for-call
equivalent. Porting this properly means rewriting
`BoardExtractor::InitializeCharucoBoard`/its detection loop against the new
`CharucoDetector` API, which needs real calibration-board imagery to verify
corner-detection correctness against -- not something to do blind. Lowest-risk
options, in order: (a) build against an OpenCV < 4.7 (matching OpenICC's own
`Dockerfile`, which apt-installs Ubuntu 22.04's `libopencv-contrib-dev` --
check what version that resolves to on your system before assuming it's new
enough to hit this), e.g. via a pinned conda env; (b) do the `CharucoDetector`
port yourself with board images in hand to validate against; (c) fall back to
`--board_type=apriltag` (`InitializeAprilBoard`, a separate code path in
`board_extractor.cc` -- not checked for the same break, but not confirmed
clean either) instead of the default ChArUco board.

**Run** (per calibration session -- recapture if you get a new camera unit,
same as the Basalt path above):
1. Capture 3 recordings via the bridge's existing `--record` mode (same
   command as step 2 of the Basalt section above, run 3 times):
   - `cam`: slowly circle the printed checkerboard/ChArUco board for
     ~20-30s (intrinsics).
   - `imu_bias`: camera stationary for ~10-20s (IMU bias).
   - `cam_imu`: fast motion exciting all 6 DOF, avoiding motion blur
     (extrinsics + time offset).
2. Convert each into OpenICC's dataset layout: `python3
   calibration/convert_record_to_openicc.py <prefix> <OpenICCDataset_dir>
   --role {cam,imu_bias,cam_imu}` (once per recording/role).
3. Calibration board: start with OpenICC's own ready-made ChArUco board
   (`resource/board.png` + `resource/charuco_detector_params.yml`, its
   default) -- lower risk than regenerating this bridge's existing
   AprilGrid tooling (`gen_aprilgrid.py` etc., currently sized for Basalt's
   2-cell-border minimal grid, not verified at OpenICC's expected scale/
   border convention).
4. Run: `python3 python/run_smartphone_calibration.py
   --path_calib_dataset=<OpenICCDataset_dir> --checker_size_m=<measured>
   --camera_model=FISHEYE --known_gravity_axis=UNKOWN` (note: `UNKOWN`,
   not a typo in this doc -- that's the actual argparse choice spelling
   upstream). `FISHEYE` is OpenICC's closest match to
   `Insta360_X4.yaml`'s `KannalaBrandt8` model (both Kannala-Brandt-family
   equidistant models).
5. Compare the printed reprojection-error stats against
   `Insta360_X4.yaml`'s documented 0.48-15.0px range (particularly the
   extrinsic, the current weak point at 15.0px) before trusting the new
   numbers over the existing ones; paste `Camera1.*`/`IMU.T_b_c1` in if
   they're a clear improvement, following the same inline-documentation
   style the file's header already uses for every prior calibration run.

## Run

```
./build/insta360_bridge --socket /tmp/insta360_hand.sock \
    --vocab /path/to/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --settings ./Insta360_X4.yaml \
    [--viewer]   # opens ORB-SLAM3's Pangolin viewer (map/keypoints/camera
                 # pose); needs a display (DISPLAY set, X server reachable).
                 # Off by default -- production teleop runs headless.
```

The socket path must match `insta360_ids` in
`envs/configs/RealUMIDemo.yaml` and `device_params.socket_path` in
`teleop/configs/Insta360UMI.yaml`.
