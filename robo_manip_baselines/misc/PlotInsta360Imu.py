"""Plots Insta360 IMU CSV data (see ConvertInsvToMp4AndImu.py or the
insta360_bridge's --record mode) so gyro/accelerometer readings, sampling
consistency, and gravity magnitude can be visually sanity-checked.

Usage:
    python ./PlotInsta360Imu.py <csv_path> [<csv_path> ...] \
        [--plot_output <plot_output.png>] [--show] [--gravity 9.81]
"""

import argparse
import csv
import os

import numpy as np


def parse_argument():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="+",
        help="one or more Insta360 IMU CSV file(s) "
        "(timestamp_ms,sensor_type,x,y,z; see ConvertInsvToMp4AndImu.py)",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="output plot PNG path. If omitted, or if multiple csv_path "
        "are given, derived as <csv_path>_imu_plot.png next to each input",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display the plot interactively",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="[m/s^2] reference line for the accel-magnitude sanity check",
    )
    return parser.parse_args()


def load_imu_csv(csv_path):
    """Parse the long-format IMU CSV into
    {"gyro": {"t": ..., "x": ..., "y": ..., "z": ...},
     "accel": {"t": ..., "x": ..., "y": ..., "z": ...}}
    Time is timestamp_ms converted to seconds and zero-based against the
    minimum timestamp across both sensor streams (timestamp_ms can be
    negative, e.g. relative to a video-start epoch)."""
    raw = {"gyro": [], "accel": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_type = row["sensor_type"]
            if sensor_type not in raw:
                continue
            raw[sensor_type].append(
                (
                    float(row["timestamp_ms"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
            )

    all_ts_ms = [r[0] for rows in raw.values() for r in rows]
    t0_ms = min(all_ts_ms) if all_ts_ms else 0.0

    data = {}
    for sensor_type, rows in raw.items():
        arr = np.array(rows, dtype=float) if rows else np.zeros((0, 4))
        data[sensor_type] = {
            "t": (arr[:, 0] - t0_ms) / 1000.0 if len(arr) else np.array([]),
            "x": arr[:, 1] if len(arr) else np.array([]),
            "y": arr[:, 2] if len(arr) else np.array([]),
            "z": arr[:, 3] if len(arr) else np.array([]),
        }
    return data


def plot_imu(data, csv_path, gravity, plot_output, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gyro, accel = data["gyro"], data["accel"]
    accel_mag = np.sqrt(accel["x"] ** 2 + accel["y"] ** 2 + accel["z"] ** 2)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    for axis_name in "xyz":
        axes[0].plot(gyro["t"], gyro[axis_name], label=axis_name)
    axes[0].set_ylabel("gyro [rad/s]")
    axes[0].legend()
    axes[0].set_title("Gyroscope")
    axes[0].grid(True, alpha=0.3)

    for axis_name in "xyz":
        axes[1].plot(accel["t"], accel[axis_name], label=axis_name)
    axes[1].set_ylabel("accel [m/s^2]")
    axes[1].legend()
    axes[1].set_title("Accelerometer")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(accel["t"], accel_mag, label="|accel|")
    axes[2].axhline(gravity, color="gray", linestyle="--", label=f"g={gravity:.2f}")
    axes[2].set_ylabel("|accel| [m/s^2]")
    axes[2].legend()
    axes[2].set_title("Accelerometer magnitude (gravity sanity check)")
    axes[2].grid(True, alpha=0.3)

    for sensor_type in ("gyro", "accel"):
        t = data[sensor_type]["t"]
        if len(t) > 1:
            dt_ms = np.diff(t) * 1000.0
            axes[3].plot(t[1:], dt_ms, label=sensor_type, alpha=0.7)
    axes[3].set_ylabel("sample interval [ms]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend()
    axes[3].set_title("Sampling interval (consistency / dropped-sample check)")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Insta360 IMU check: {os.path.basename(csv_path)}")
    fig.tight_layout()

    fig.savefig(plot_output)
    print(f"[PlotInsta360Imu] Saved plot to {plot_output}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_argument()
    multiple = len(args.csv_path) > 1

    for csv_path in args.csv_path:
        data = load_imu_csv(csv_path)

        if args.plot_output is not None and not multiple:
            plot_output = args.plot_output
        else:
            plot_output = os.path.splitext(csv_path)[0] + "_imu_plot.png"

        plot_imu(data, csv_path, args.gravity, plot_output, args.show)


if __name__ == "__main__":
    main()
