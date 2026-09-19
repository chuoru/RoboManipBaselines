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

## Known unresolved risk

**This has not been built or run against real hardware or the real SDK.**
The Insta360 CameraSDK's public repo (linked above) ships only a demo
`main.cc` and documentation, not the actual headers/libraries -- those come
from the SDK package obtained via the application process below, which was
not available while writing this code. `main.cc` and `StreamDecoder.cc`
were written against the public README/demo's documented API surface (class
names, method signatures), which is solid, but the following details are
genuine unknowns until the real `camera/camera.h` is in hand:

1. **Does `ins_camera::GyroData` include accelerometer samples?** The README
   only documents "gyro data" for `OnGyroData`, separately from
   `OnExposureData`. If it does not, ORB-SLAM3's `IMU_MONOCULAR` mode cannot
   run (it needs both gyro and accel). `main.cc` has both code paths behind
   the `INSTA360_GYRO_HAS_ACCEL` compile definition (default off = plain
   `MONOCULAR`, no IMU, scale ambiguous -- tune `Insta360UMI.yaml`'s
   `pos_scale` by hand in that case).
2. The exact field names of `GyroData` (assumed `gyro_x/gyro_y/gyro_z` and,
   if (1) holds, `accel_x/accel_y/accel_z`) and its timestamp's unit
   (assumed microseconds).
3. `GetVideoEncodeType()`'s return type/enum values (assumed
   `ins_camera::VideoEncodeType::H265`/implicitly H264 otherwise).
4. The CameraSDK's actual shared library name(s) in `lib/` (assumed
   `CameraSDK` in `CMakeLists.txt` -- adjust to match).

Search for `TODO(insta360-sdk)` in `main.cc` for the exact spots to fix once
the real SDK is available. None of this affects the Python side, MuJoCo
teleop wiring, or `misc/MockInsta360Bridge.py` -- those were built and
tested against the wire protocol alone, independent of this file.

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
       [-DINSTA360_GYRO_HAS_ACCEL=ON]   # only once confirmed, see above
   cmake --build build
   ```

## Camera setup (one-time, per camera)

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

`Insta360_X4.yaml`'s camera intrinsics, `IMU.T_b_c1` extrinsic, and IMU noise
parameters are all placeholders (see that file's own comments) and must be
calibrated against the real unit before trusting any tracking result --
e.g. with [Kalibr](https://github.com/ethz-asl/kalibr) for the camera-IMU
extrinsic/intrinsics. No calibration tooling is provided here; adapt
`teleop/calibrate_vive_rotation.py`'s general approach, or use an existing
camera-IMU calibration tool directly.

## Run

```
./build/insta360_bridge --socket /tmp/insta360_hand.sock \
    --vocab /path/to/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --settings ./Insta360_X4.yaml
```

The socket path must match `insta360_ids` in
`envs/configs/RealUMIDemo.yaml` and `device_params.socket_path` in
`teleop/configs/Insta360UMI.yaml`.
