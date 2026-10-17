# Offline SLAM post-processing for Insta360-mounted robot-teleop episodes

Extracts an independent, SLAM-derived EEF trajectory from the Insta360
camera mounted on the robot's own end-effector during teleoperation,
alongside (not replacing) the robot's own FK-derived
`DataKey.MEASURED_EEF_POSE` already recorded in every episode. This is
this repo's adaptation of UMI's own offline SLAM recipe
(`run_slam_pipeline.py` + `cheng-chi/ORB_SLAM3`'s `gopro_slam.cc`, run on
recorded GoPro footage) -- substituting Insta360 for GoPro, and robot-teleop
episodes for UMI's own handheld-rig demonstrations (this repo has no
handheld UMI gripper rig; the camera here always rides on the robot's own
EEF -- see `envs/real/insta360_bridge/README.md`'s Status section).

## One-time setup

1. Swap+rebuild `third_party/ORB_SLAM3` to `cheng-chi/ORB_SLAM3` and
   re-harden it against this hardware's known crash classes -- see
   `envs/real/insta360_bridge/orb_slam3_hardware_fixes/README.md`. This also
   builds the `mono_inertial_video` executable this pipeline's step 2 below
   runs directly (ported from the previously-vendored UZ-SLAMLab build,
   already designed for exactly this raw-video+IMU-csv input shape -- no
   GoPro-GPMF-JSON adapter needed).
2. Recalibrate `envs/real/insta360_bridge/Insta360_X4.yaml` (used by the
   *live* bridge) via OpenImuCameraCalibrator -- see that README's
   Calibration section. `Insta360X3_2880.yaml` (in
   `orb_slam3_hardware_fixes/`, copied into this directory) is the
   settings file for THIS offline pipeline instead -- it targets the
   on-device recording's native, un-cropped resolution, not the live
   bridge's already-cropped/masked 800x800 SLAM-input frames.
3. Calibrate the camera-to-EEF extrinsic:
   `teleop/calibrate_insta360_eef_extrinsic.py` (see its own docstring).
   Paste the result into `teleop/configs/Insta360UMI.yaml`'s
   `insta360_to_eef_frame_rotation`/`insta360_to_eef_translation`
   (currently a PLACEHOLDER identity there).
4. `pip install robo_manip_baselines[insta360]` for `telemetry-parser`
   (extracts the Insta360's embedded telemetry from on-device `.insv`
   recordings -- see `misc/ConvertInsvToMp4AndImu.py`).

## Per-episode workflow

1. Record teleop episodes as usual, adding `--record_insta360_slam` (see
   `teleop/TeleopBase.py`'s `--record_insta360_slam` help) to
   `--input_device insta360` teleop runs. This starts/stops the Insta360's
   **on-device** recording (via the CameraSDK's own `StartRecording()`/
   `StopRecording()`, wired through `envs/real/insta360_bridge/main.cc`'s
   recording-control callback and `Insta360InputDevice.start_recording()`/
   `stop_recording()`) for each episode, downloading the result and saving
   it as `<episode>.rmb/insta360_raw.insv` once the episode is kept.
   UNVERIFIED ON REAL HARDWARE: whether this SDK/camera supports
   concurrent live-streaming (for the live bridge's own frame/pose
   broadcast, used for teleop input) and on-device recording -- confirm
   this on your rig before relying on it.
2. Post-process each episode:
   `python ./misc/insta360_slam_pipeline/AddSlamPoseToRmb.py
   <episode>.rmb --vocab third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt
   --settings envs/real/insta360_bridge/Insta360X3_2880.yaml
   --mono_inertial_video_bin third_party/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_video
   --insta360_config teleop/configs/Insta360UMI.yaml`
   This converts the `.insv` (via `ConvertInsvToMp4AndImu.py`'s converter
   class), runs `mono_inertial_video` on it, transforms the resulting
   camera trajectory into the EEF frame using the calibrated extrinsic,
   realigns it onto the episode's own `DataKey.TIME` axis (see its
   docstring for why a realignment step is needed at all -- the
   trajectory's clock and the episode's clock have different origins), and
   writes it into `<episode>.rmb/main.rmb.hdf5` under
   `DataKey.MEASURED_CAMERA_EEF_POSE`.
3. Sanity-check before trusting a session's worth of episodes: the
   `mono_inertial_video` run prints its own trajectory file path;
   `python ./misc/insta360_slam_pipeline/PlotSlamTrajectory.py
   <path>_CameraTrajectory.txt --show` plots position/orientation and
   flags frame-to-frame speed spikes / sample-interval gaps (SLAM silently
   drops untracked frames rather than flagging them in-line, so a gap in
   sample spacing is the signal to look for).
4. Use the SLAM-derived pose instead of FK when converting to a UMI-style
   training dataset: `python ./misc/ConvertRmbDataToUmiZarr.py <dir>
   --output <out>.umi.zarr --pose_source slam` (default is `fk`, the
   robot's own forward kinematics, unchanged from before this pipeline
   existed).

## Files

- `AddSlamPoseToRmb.py` -- the per-episode post-processing script (step 2).
- `PlotSlamTrajectory.py` -- trajectory sanity-check plot (step 3), modeled
  on `misc/PlotInsta360Imu.py`.
- `../../envs/real/insta360_bridge/orb_slam3_hardware_fixes/mono_inertial_video.cc` --
  the offline SLAM executable's source (built as part of
  `third_party/ORB_SLAM3`, not from this directory).
