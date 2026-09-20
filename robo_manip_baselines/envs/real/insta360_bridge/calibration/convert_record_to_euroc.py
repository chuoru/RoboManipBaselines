"""Convert an insta360_bridge --record output (<prefix>.avi,
<prefix>_frame_timestamps.csv, <prefix>.csv) into the EuRoC MAV dataset
folder layout Basalt's basalt_calibrate/basalt_calibrate_imu expect with
--dataset-type euroc, for camera-IMU extrinsic (T_b_c1) calibration.

Usage: python3 convert_record_to_euroc.py <record_prefix> <out_dir>
  <record_prefix>.avi                    -- 640x640 BGR frames (MJPG/AVI)
  <record_prefix>_frame_timestamps.csv   -- frame_index,timestamp_ms
  <record_prefix>.csv                    -- long-format IMU (see below)
Writes <out_dir>/mav0/{cam0,imu0}/...

cam1 is a symlinked DUPLICATE of cam0 (same images/timestamps), not left
absent -- num_cams=2 is hardcoded in Basalt's EurocIO loader, and a
cam0-only dataset loads fine for basalt_calibrate's --no-gui path *until*
its post-intrinsics-init "find the frame with all valid images" loop
(cam_calib.cpp): since cam1 never has any image, that loop's validity
check never succeeds and img_idx walks off the end of
get_image_timestamps(), reading out of bounds -- confirmed via gdb
backtrace (SIGSEGV at the same address whether or not the KB4 intrinsics
initializer itself succeeded, i.e. downstream of it, not caused by it).
Duplicating cam0 into cam1 sidesteps this entirely; --cam-types must then
be passed twice (e.g. "kb4 kb4") to match num_cams=2. cam1's own
"calibration" result is meaningless (it's identical data to cam0) and
should be ignored -- only cam0's.

Frames are converted to real 8-bit grayscale (cv2.cvtColor, proper
luminance weighting) before writing -- Basalt's own EurocIO loader would
otherwise take a crude single-channel (blue) slice of a BGR PNG as
"intensity" (see dataset_io_euroc.h's CV_8UC3 branch), which is worse for
AprilGrid corner detection than converting properly ourselves.

IMU format: insta360_bridge's <prefix>.csv is long-format
("timestamp_ms,sensor_type,x,y,z", one gyro row then one accel row per
sample sharing the same timestamp -- see main.cc's OnGyroData/record loop).
EuRoC's imu0/data.csv is wide-format, one row per sample:
"#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z"
(gyro then accel, same units we already use: rad/s and m/s^2 -- confirmed
against Basalt's dataset_io_euroc.h read_imu_data, no unit conversion
needed).
"""
import csv
import os
import sys

import cv2

record_prefix = sys.argv[1]
out_dir = sys.argv[2]

cam0_data_dir = os.path.join(out_dir, "mav0", "cam0", "data")
cam1_data_dir = os.path.join(out_dir, "mav0", "cam1", "data")
imu0_dir = os.path.join(out_dir, "mav0", "imu0")
os.makedirs(cam0_data_dir, exist_ok=True)
os.makedirs(cam1_data_dir, exist_ok=True)
os.makedirs(imu0_dir, exist_ok=True)


def ms_to_ns(t_ms):
    # Round to the nearest ns rather than truncating -- t_ms carries
    # microsecond-level precision (see main.cc's setprecision(6) fix for
    # the gyro-batch interpolation), and int(t_ms * 1e6) alone would
    # silently truncate that.
    return round(float(t_ms) * 1e6)


# --- cam0 ---
frame_timestamps_ns = []
with open(f"{record_prefix}_frame_timestamps.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        frame_timestamps_ns.append(ms_to_ns(row["timestamp_ms"]))

cap = cv2.VideoCapture(f"{record_prefix}.avi")
frame_count = 0
cam0_rows = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count >= len(frame_timestamps_ns):
        print(f"WARNING: more video frames ({frame_count + 1}+) than "
              f"timestamps ({len(frame_timestamps_ns)}) -- stopping early. "
              f"This shouldn't happen (both are written in the same loop "
              f"iteration in main.cc); check for a truncated recording.")
        break
    t_ns = frame_timestamps_ns[frame_count]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = f"{t_ns}.png"
    cv2.imwrite(os.path.join(cam0_data_dir, filename), gray)
    cam0_rows.append((t_ns, filename))
    frame_count += 1
cap.release()

cam0_rows.sort(key=lambda r: r[0])
for cam_name in ("cam0", "cam1"):
    with open(os.path.join(out_dir, "mav0", cam_name, "data.csv"), "w") as f:
        f.write("#timestamp [ns],filename\n")
        for t_ns, filename in cam0_rows:
            f.write(f"{t_ns},{filename}\n")

for t_ns, filename in cam0_rows:
    link_path = os.path.join(cam1_data_dir, filename)
    if not os.path.exists(link_path):
        # Absolute target: a symlink target is resolved relative to the
        # symlink's OWN directory (cam1_data_dir), not the CWD, so a
        # relative os.path.join(cam0_data_dir, filename) would only work
        # by accident depending on out_dir's shape.
        os.symlink(os.path.abspath(os.path.join(cam0_data_dir, filename)),
                   link_path)

print(f"cam0: {frame_count} frames written to {cam0_data_dir}")
print(f"cam1: {frame_count} symlinked duplicates written to {cam1_data_dir}")

# --- imu0 ---
gyro_by_ts = {}
accel_by_ts = {}
with open(f"{record_prefix}.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        t_ns = ms_to_ns(row["timestamp_ms"])
        xyz = (float(row["x"]), float(row["y"]), float(row["z"]))
        if row["sensor_type"] == "gyro":
            gyro_by_ts[t_ns] = xyz
        elif row["sensor_type"] == "accel":
            accel_by_ts[t_ns] = xyz

if gyro_by_ts.keys() != accel_by_ts.keys():
    only_gyro = gyro_by_ts.keys() - accel_by_ts.keys()
    only_accel = accel_by_ts.keys() - gyro_by_ts.keys()
    print(f"WARNING: {len(only_gyro)} gyro-only and {len(only_accel)} "
          f"accel-only timestamps (expected every sample to have both -- "
          f"see main.cc's OnGyroData, which always sets both fields from "
          f"the same underlying sample). Dropping the unpaired ones.")

imu_rows = sorted(t_ns for t_ns in gyro_by_ts if t_ns in accel_by_ts)
with open(os.path.join(imu0_dir, "data.csv"), "w") as f:
    f.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,"
            "a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
    for t_ns in imu_rows:
        wx, wy, wz = gyro_by_ts[t_ns]
        ax, ay, az = accel_by_ts[t_ns]
        f.write(f"{t_ns},{wx},{wy},{wz},{ax},{ay},{az}\n")

print(f"imu0: {len(imu_rows)} samples written to {imu0_dir}/data.csv")
print(f"Done. Run e.g.: basalt_calibrate_imu --dataset-path {out_dir} "
      f"--dataset-type euroc --aprilgrid aprilgrid.json "
      f"--result-path <result_dir> --cam-types kb4 kb4 "
      f"--gyro-noise-std ... --accel-noise-std ... "
      f"--gyro-bias-std ... --accel-bias-std ... (cam-types passed twice "
      f"to match num_cams=2 -- see cam1's docstring note above)")
