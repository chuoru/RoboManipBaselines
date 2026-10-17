# UZ-SLAMLab/ORB_SLAM3 real-hardware fixes (reference, pre-fork-swap)

Before `third_party/ORB_SLAM3` was repointed from `UZ-SLAMLab/ORB_SLAM3`
(`v1.0-release-2`) to `cheng-chi/ORB_SLAM3`, the vendored UZ-SLAMLab checkout
had accumulated ~420 lines of uncommitted, gdb-verified patches fixing real
crashes hit during live Insta360 X3 IMU_MONOCULAR tracking on this rig. These
were never committed anywhere (submodule working-tree changes aren't tracked
by the superproject beyond "dirty" status), so they're preserved here before
being overwritten by the fork swap.

- `UZ_SLAMLab_v1.0-release-2.patch` — the full `git diff` from that checkout
  (`Frame.cc`, `ImuTypes.cc`, `LocalMapping.cc`, `Optimizer.cc`,
  `Tracking.cc`, `System.cc`, `CMakeLists.txt`). Every hunk has an inline
  comment explaining the bug, how it was confirmed (gdb backtrace on real
  hardware), and the fix. Recurring bug classes worth specifically checking
  for in `cheng-chi/ORB_SLAM3`'s own code, since this academic codebase
  lineage tends to repeat them:
  - `Frame`'s default/early-return constructor paths leaving `mpMutexImu`
    uninitialized.
  - Unguarded derefs of `mpImuPreintegrated` / `mpImuPreintegratedFrame` /
    `mpLastKeyFrame` / `mpPrevFrame`, which can legitimately be null when a
    frame/keyframe lands in a gap with no queued IMU samples (routine on
    this gyro-batching hardware, not just a rare edge case).
  - Division by a degenerate (zero/negative) `dt`/`tab` in IMU
    preintegration/interpolation.
  - `Sophus::SO3f::exp()` fed a non-finite or wildly-out-of-linearization-
    range rotation correction (bias-Jacobian linearization, or a
    near-zero-norm gravity-direction vector before normalizing).
  - Non-finite/non-orthogonal rotation matrices coming out of an
    ill-conditioned inertial pose optimization, fed straight into
    `Frame::SetImuPoseVelocity`/`Sophus::SE3f` with no validation.
  - Under-constrained `InertialOptimization`/`FullInertialBA` calls (too
    few valid inertial edges) producing NaN-inducing g2o updates.
- `mono_inertial_video.cc` — the offline Monocular-Inertial driver already
  written against this fixed UZ-SLAMLab build, reading a raw video file +
  this repo's native long-format IMU CSV directly (no GPMF/JSON conversion
  needed). Includes its own hardware-informed fixes: a circular safety mask
  for the fisheye calibration's valid angular range, correct
  Shutdown()-vs-trajectory-save ordering (avoids a Pangolin-viewer-thread
  segfault on exit), and `SaveTrajectoryEuRoC`'s null-map guard (from
  `System.cc`'s patch above). Use this as the reference/template for the
  offline SLAM executable on the new fork, rather than starting from
  `cheng-chi/ORB_SLAM3`'s own `gopro_slam.cc` from scratch.
- `Insta360X3_2880.yaml` — the matching ORB_SLAM3 settings file this driver
  was run against (native 2880x2880 resolution case).

None of this was validated against `cheng-chi/ORB_SLAM3` — it's a reference
for re-deriving equivalent fixes there, not a drop-in patch (the fork's file
contents will differ). Re-verification against real hardware is required
after porting; this sandbox has no camera attached to reproduce any of these
crashes directly.

## Porting status (as of the fork swap)

All of the above was re-applied directly to `third_party/ORB_SLAM3` (now
`cheng-chi/ORB_SLAM3`, `master`). Two things worth knowing before trusting
that build:

- **Several bug classes were already independently fixed upstream** in
  `cheng-chi/ORB_SLAM3` (presumably from UMI's own real GoPro hardware
  debugging) and did NOT need porting: the `LocalMapping::Run()`
  no-motion-reset check is fully disabled rather than just rescaled; its
  `KeyFrameCulling`'s `MergePrevious` call already null-checks
  `mpImuPreintegrated` on both sides; and every `SetNewBias`/edge-building
  call site in `Optimizer.cc` (`FullInertialBA`, both `InertialOptimization`
  overloads with `Rwg`/`scale`) already null-checks `mpImuPreintegrated`.
  `SaveKeyFrameTrajectoryEuRoC` in `System.cc` already null-checks its
  `pBiggerMap`.
- **Everything else was ported as real, new patches** against this fork's
  (structurally near-identical) code: the `Frame::mpMutexImu`
  initialization-order fix across all 4 non-trivial constructors + the
  default constructor, `SetImuPoseVelocity`'s non-finite/non-orthogonal
  `Rwb` guard, both `ImuTypes.cc` fixes (degenerate `dt`, non-finite bias
  correction), `LocalMapping.cc`'s `dirG` zero-norm guard, a new
  `vpei.size()<3` under-constrained-optimization guard in the
  `InertialOptimization(Rwg,scale,bg,ba,...)` overload (cheng-chi's other
  overloads didn't have an equivalent either, but this one specifically
  lacked any edge-count check), the `PoseInertialOptimizationLastKeyFrame`/
  `PoseInertialOptimizationLastFrame` null guards (`pKF`, `pFp`,
  `mpImuPreintegrated(Frame)`, the `ep`/`mpcpi` handling), all three
  `Tracking.cc` fixes (`PreintegrateIMU`'s `tab` guards,
  `PredictStateIMU`'s and `UpdateFrameIMU`'s null-preintegration/
  null-keyframe guards), and `System.cc`'s `pBiggerMap` guard in
  `SaveTrajectoryCSV`/`SaveTrajectoryTUM`/`SaveTrajectoryEuRoC` (the first
  two reject `MONOCULAR` before reaching that code anyway, but were
  hardened for consistency).
- `cheng-chi/ORB_SLAM3` also needed NEW fixes unrelated to this hardware's
  IMU fragility, to build at all against a modern OpenCV (4.11): its
  `aruco_dict` support was written against OpenCV < 4.7's `cv::aruco` API
  (`getPredefinedDictionary` returning `Ptr<Dictionary>` instead of
  `Dictionary` by value, and `DetectorParameters::create()`, both removed
  in 4.7+). Fixed at all 4 call sites (`System.h`, `Frame.h`, `Frame.cc`,
  `Examples/Monocular-Inertial/gopro_slam.cc`) by wrapping in
  `cv::makePtr<...>(...)`.
- `mono_inertial_video.cc` was copied into this fork's
  `Examples/Monocular-Inertial/` and registered in `CMakeLists.txt`
  unchanged (no API differences hit) and builds clean.
- Both `libORB_SLAM3.so` and the live `insta360_bridge` executable rebuild
  cleanly against this fork in this environment. **None of this has been
  tested against real hardware** (this sandbox has no Insta360 camera
  attached) — do that before trusting it over the previous build.
