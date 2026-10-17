# insta360_bridge calibration tooling

Scripts for calibrating the Insta360 single-fisheye crop that `main.cc` feeds to
ORB-SLAM3, plus the camera-IMU extrinsic. Results land in `../Insta360_X4.yaml`.

Everything here runs inside the `insta360_bridge` container
(`docker compose -f docker-compose.insta360.yaml run --rm insta360_bridge bash`),
which already has OpenCV, Basalt and `allantools`.

## Script inventory

| Script | Purpose |
|---|---|
| `gen_aprilgrid.py` | Generates `aprilgrid_a4.png/.pdf` + `aprilgrid.json`. **Primary** target for Basalt. |
| `gen_checkerboard.py` | Generates `checkerboard_a3.png/.pdf`. Only for the legacy OpenCV intrinsics path. |
| `t36h11_codes.json` | tag36h11 36-bit codes, copied from Kalibr's `kalibr_create_target_pdf`. |
| `capture_calib_frames.py` | Live checkerboard capture with radial-zone coverage quotas. Legacy OpenCV path. |
| `run_fisheye_calibration.py` | `cv2.fisheye.calibrate` over captured frames. Legacy OpenCV path. |
| `live_aprilgrid_preview.py` | Live view for aiming the camera. See the detection caveat below. |
| `aprilgrid_2cell_detector.py` | Minimal tag36h11 2-cell-border detector used by the preview. |
| `convert_record_to_euroc.py` | `--record` output → EuRoC `mav0/` layout that Basalt loads. |

## Run order (current, Basalt-based)

1. `python3 gen_aprilgrid.py` — print at **100% scale, no "fit to page"**, then
   measure the printed tag with a ruler and update `tagSize` in
   `aprilgrid.json` if it differs from the 25 mm nominal.

   > **Careful:** `gen_aprilgrid.py` rewrites `aprilgrid.json` with the
   > *nominal* 25 mm (`tagSize: 0.025`), silently discarding whatever measured
   > value is in there. The committed value is `0.022` — a real 22 mm
   > measurement from the previous print, and the one the current
   > `../Insta360_X4.yaml` calibration was computed against. After
   > regenerating, either re-measure your own print and set that, or
   > `git checkout -- aprilgrid.json` if you are reusing the old target.
   > A wrong `tagSize` scales the whole calibration: it makes `IMU.T_b_c1`'s
   > translation and the map scale wrong by exactly that ratio, while
   > reprojection error still looks fine — so it fails silently.
2. Record a sweep: `insta360_bridge --socket /tmp/insta360/calib.sock --no-slam
   --lens back --preview --record <prefix>`.
3. `python3 convert_record_to_euroc.py <prefix> <euroc_dir>`.
4. `basalt_calibrate ... --cam-types kb4 kb4` → camera intrinsics.
5. `basalt_calibrate_imu ...` (**GUI mode**) → `IMU.T_b_c1`.

Steps 4-5 and the recording motion profile are documented in `../README.md`.

## Why the AprilGrid is generated here instead of downloaded

Basalt vendors its own AprilTag implementation
(`thirdparty/apriltag/ethz_apriltag2`) configured with **`blackTagBorder=2`**.
The standard pre-rendered tag36h11 PNGs from `AprilRobotics/apriltag-imgs` have
a **1-cell** border. A target built from those was detected as **zero** corners
by `basalt_calibrate`'s `detect_corners` on every frame of two full recording
sweeps — while `cv2.aruco` and the reference `apriltag` library both found all
36 tags on the same frames. `gen_aprilgrid.py` therefore renders tags itself
from `t36h11_codes.json` using Kalibr's bit-layout/rotation/2-cell-border
convention. Tag placement follows Kalibr's `id = n_cols*row + col`.

**Caveat:** `live_aprilgrid_preview.py`'s overlay uses `cv2.aruco` (1-cell
border), so it will *not* detect this target. It still works as a plain live
view for aiming — just without the detection count.

## The KB4 monotonic range and `kMaskRadiusPx`

A Kannala-Brandt equidistant model is only physically valid while its
projection is monotonic in `theta`:

```
r(theta)     = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
dr/dtheta    = 1 + 3*k1*t^2 + 5*k2*t^4 + 7*k3*t^6 + 9*k4*t^8
```

Monotonic up to the first `theta` where `dr/dtheta <= 0`. Re-check this whenever
`Camera1.k1-k4` change:

```python
import math
k1, k2, k3, k4 = 0.0925616, -0.0371436, 0.0156295, -0.0037079   # from ../Insta360_X4.yaml
fx = 214.0900
drdt  = lambda t: 1 + 3*k1*t**2 + 5*k2*t**4 + 7*k3*t**6 + 9*k4*t**8
r_px  = lambda t: t*(1 + k1*t**2 + k2*t**4 + k3*t**6 + k4*t**8)*fx
t = 0.0
while t < math.pi and drdt(t) > 0:
    t += 1e-5
print("monotonic to %.2f deg = %.1f px from centre" % (math.degrees(t), r_px(t)))
```

For the current coefficients this gives **103.08 deg**, matching the value
recorded in `../Insta360_X4.yaml`.

**However** — and this corrects a claim in that YAML — 103.08 deg corresponds to
**398.1 px** from the centre at `fx = 214.09`, essentially the full 400 px
inscribed circle of the 800x800 frame. So the monotonic range is *not* what
constrains `main.cc`'s `kMaskRadiusPx = 370`. Inverting the same model, 370 px
corresponds to **theta = 90.06 deg** — i.e. the mask is currently the 90 deg
hemisphere, a choice ~28 px tighter than the model's validity requires.

That is worth knowing for tuning: widening the mask toward ~395 px would recover
a 28 px annulus of still-valid field of view and therefore more ORB features,
which bears directly on the `Fail to track local map!` resets that
`../Insta360_X4.yaml` records as the dominant live-tracking failure mode. It has
**not** been changed here — it needs validation against real hardware, since the
outer annulus is also where lens vignetting and resolution loss are worst.

## Where artifacts live

All gitignored:

- `aprilgrid_a4.png/.pdf`, `checkerboard_a3.png/.pdf` — generated targets
- `recordings/` — `--record` output (`.avi`, `.csv`, `_frame_timestamps.csv`)
- `euroc_*/` — `convert_record_to_euroc.py` output
- `results/` — Basalt `--result-path` (`calibration.json`)

Basalt's corner-detection cache lives in the container's `/root/.cache`, kept in
the `insta360_cache` named volume so the IMU stage reuses the camera stage's
`detect_corners` pass.
