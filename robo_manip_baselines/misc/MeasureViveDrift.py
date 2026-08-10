"""Standalone diagnostic: measure how much a STATIONARY HTC Vive Tracker's
reported pose drifts over time (libsurvive lighthouse-tracking/IMU-fusion
drift), with no robot involved at all.

Motivated by misc/ReplayUmiOnFairino5.py's filter_drift(): a UMI demo
recorded with no intentional tracker motion still showed the retargeted FR5
pose climbing ~5cm / ~5.7deg over 10.5s when replayed. That was inferred
indirectly from a recorded demo file; this script instead measures the drift
directly at the source (the raw tracker reading, no UMI/FR5 retargeting
involved), so the number is unambiguous -- if it drifts here with the
tracker truly held still, that confirms the drift is in the tracker/
lighthouse system itself, not something introduced downstream.

Usage: put the tracker down somewhere stationary (same physical setup you'd
use for a real UMI recording -- same lighthouse geometry, same mount) and do
not touch it once recording starts:
    python ./misc/MeasureViveDrift.py --serial_number LHR-904A2704 --duration 60

If --serial_number is omitted, this auto-picks the tracker if exactly one
"LHR-..." device (not a lighthouse, not a controller) is detected -- see
teleop/check_vive_devices.py to look up the serial number otherwise.
"""

import argparse
import csv
import sys
import threading
import time

import numpy as np
import pinocchio as pin
import pysurvive


class PosePoller:
    """Polls target_obj.Pose() continuously on a background daemon thread,
    publishing the latest (wall_time, pos, quat, timecode) under a lock.

    Why: a real run of this script (both in this sandbox AND independently
    in the user's own terminal, i.e. not an environment-specific fluke)
    reproducibly HUNG inside a plain `target_obj.Pose()` call partway
    through a 60s recording (consistently around the 55-60s mark), with no
    exception and no further output -- Ctrl+C was needed to stop it. Root
    cause unconfirmed (a pysurvive/libsurvive-internal stall of some kind),
    but polling on a separate daemon thread means the MAIN thread never
    calls the possibly-blocking Pose() itself: it only reads the
    lock-protected "latest" value with a staleness check, so a stalled
    poller thread is detected (no update for STALL_TIMEOUT) and the main
    loop can exit gracefully with whatever data was collected, instead of
    hanging indefinitely. Being a daemon thread, if the poller genuinely is
    stuck forever inside Pose(), the process can still exit normally at the
    end of main() without waiting for it."""

    STALL_TIMEOUT = 3.0  # [s] no update within this long => treat as stalled

    def __init__(self, target_obj):
        self._target_obj = target_obj
        self._lock = threading.Lock()
        self._latest = None  # (wall_time, pos, quat, timecode)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            pose, timecode = self._target_obj.Pose()
            now = time.time()
            if timecode > 0:
                pos = np.array(pose.Pos[:3], dtype=np.float64)
                quat = np.array(pose.Rot[:4], dtype=np.float64)
            else:
                pos, quat = None, None
            with self._lock:
                self._latest = (now, pos, quat, timecode)
            # Without this, if Pose() returns near-instantly (e.g. it just
            # reads a cached value rather than blocking for a fresh one),
            # this loop spins as fast as possible with no yield point,
            # fighting the main thread for the GIL. A first version without
            # this sleep reproducibly got stuck around the 55-60s mark of a
            # 60s recording with NO stall ever detected by is_stalled() --
            # i.e. this thread was still updating _latest, but the main
            # thread's own print()/loop bookkeeping was being starved of
            # scheduling time, not blocked. 200Hz is far above --hz's
            # typical sampling need, so this doesn't reduce data quality.
            time.sleep(0.005)

    def get_latest(self):
        """Returns (wall_time, pos, quat, timecode) or None if no sample has
        arrived yet."""
        with self._lock:
            return self._latest

    def seconds_since_update(self):
        """inf if no sample has ever arrived."""
        latest = self.get_latest()
        if latest is None:
            return float("inf")
        return time.time() - latest[0]

    def is_stalled(self):
        return self.seconds_since_update() > self.STALL_TIMEOUT


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--serial_number",
        type=str,
        default=None,
        help="tracker serial number (e.g. LHR-904A2704, see "
        "teleop/check_vive_devices.py). If omitted, auto-picks the single "
        "detected 'LHR-...' tracker device -- errors if there's more than "
        "one connected.",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="[s] how long to record"
    )
    parser.add_argument(
        "--hz", type=float, default=20.0, help="sampling rate [Hz]"
    )
    parser.add_argument(
        "--settle_time",
        type=float,
        default=2.0,
        help="[s] wait this long after the tracker's pose first becomes "
        "valid before starting to record, to let libsurvive's pose solver "
        "converge (mirrors ViveInputDevice.POSE_SETTLE_TIME) -- otherwise "
        "the solver's own initial convergence would be misread as drift.",
    )
    parser.add_argument(
        "--discovery_timeout",
        type=float,
        default=15.0,
        help="[s] give up if the tracker isn't discovered within this long",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="vive_drift.png",
        help="output plot path",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="vive_drift.csv",
        help="output CSV path (raw pose + drift-from-start per sample)",
    )
    return parser.parse_args()


def main():
    args = parse_argument()

    # stdout defaults to full (block) buffering when not a TTY (e.g. when
    # piped to a file/background task), so without this, print() progress
    # messages sit invisible in the buffer for a long time -- confirmed:
    # a first run showed literally zero Python-level output (not even the
    # very first print(), before any blocking call) for minutes, making a
    # slow-but-working run indistinguishable from a hung one.
    sys.stdout.reconfigure(line_buffering=True)

    print("[MeasureViveDrift] Connecting to libsurvive...")
    ctx = pysurvive.SimpleContext([])

    known_objects = {}
    target_obj = None
    target_serial = None
    discovery_start = time.time()
    while target_obj is None:
        if time.time() - discovery_start > args.discovery_timeout:
            raise RuntimeError(
                f"[MeasureViveDrift] No matching tracker found within "
                f"{args.discovery_timeout}s. Detected so far: "
                f"{list(known_objects.keys())}. Run "
                "teleop/check_vive_devices.py to debug lighthouse/tracker "
                "visibility."
            )
        while True:
            obj = ctx.NextUpdated()
            if obj is None:
                break
            serial = pysurvive.simple_serial_number(obj.ptr)
            if isinstance(serial, bytes):
                serial = serial.decode()
            known_objects[serial] = obj

        if args.serial_number is not None:
            if args.serial_number in known_objects:
                target_obj = known_objects[args.serial_number]
                target_serial = args.serial_number
        else:
            trackers = {
                s: o for s, o in known_objects.items() if s.startswith("LHR")
            }
            if len(trackers) == 1:
                target_serial, target_obj = next(iter(trackers.items()))
            elif len(trackers) > 1:
                raise RuntimeError(
                    f"[MeasureViveDrift] Multiple trackers detected "
                    f"({list(trackers.keys())}) -- pass --serial_number to "
                    "pick one."
                )
        time.sleep(0.1)
    print(f"[MeasureViveDrift] Using tracker serial={target_serial}")

    poller = PosePoller(target_obj)

    print(
        f"[MeasureViveDrift] Waiting for valid tracking, then "
        f"{args.settle_time}s settle..."
    )
    settle_start = None
    wait_start = time.time()
    while True:
        if time.time() - wait_start > args.discovery_timeout:
            raise RuntimeError(
                f"[MeasureViveDrift] Tracker serial={target_serial} was "
                f"discovered but never reported a valid Pose() (timecode "
                f"stayed 0) within {args.discovery_timeout}s -- likely out "
                "of lighthouse view, or this is actually a lighthouse/"
                "controller object misidentified as the tracker. Check with "
                "teleop/check_vive_devices.py."
            )
        if poller.is_stalled() and poller.get_latest() is not None:
            raise RuntimeError(
                "[MeasureViveDrift] Pose() polling thread stalled (no "
                f"update for >{PosePoller.STALL_TIMEOUT}s) before tracking "
                "ever settled -- see PosePoller's docstring."
            )
        latest = poller.get_latest()
        timecode = latest[3] if latest is not None else 0
        if timecode > 0:
            if settle_start is None:
                settle_start = time.time()
                print("[MeasureViveDrift] Tracking acquired, settling...")
            elif time.time() - settle_start >= args.settle_time:
                break
        else:
            settle_start = None
        time.sleep(0.05)

    print(
        f"[MeasureViveDrift] Recording for {args.duration:.0f}s -- "
        "DO NOT MOVE THE TRACKER..."
    )
    dt = 1.0 / args.hz
    t_list = []
    pos_list = []
    quat_list = []
    start_time = time.time()
    next_sample = start_time
    last_print = start_time
    stalled = False
    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= args.duration:
            break
        if poller.is_stalled():
            print(
                f"[MeasureViveDrift] WARNING: Pose() polling thread stalled "
                f"(no update for >{PosePoller.STALL_TIMEOUT}s) at "
                f"{elapsed:.1f}/{args.duration:.0f}s -- stopping early and "
                "using what was collected so far instead of hanging "
                "indefinitely."
            )
            stalled = True
            break
        latest = poller.get_latest()
        if latest is not None:
            _wall_time, pos, quat, timecode = latest
            if timecode > 0:
                t_list.append(elapsed)
                pos_list.append(pos)
                quat_list.append(quat)
        if now - last_print >= 5.0:
            print(f"[MeasureViveDrift] ...{elapsed:.0f}/{args.duration:.0f}s")
            last_print = now
        next_sample += dt
        sleep_time = next_sample - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # pysurvive.simple_close(ctx.ptr) is NOT called synchronously here: every
    # run so far (across a sandbox environment and independently in the
    # user's own terminal) got stuck at exactly this point -- right as the
    # recording loop ends and this call would happen -- with no exception,
    # requiring Ctrl+C. The sampling loop's own watchdog (PosePoller) never
    # caught a stall, meaning the freeze is specifically inside
    # simple_close(), not the Pose() polling. Since the process is about to
    # finish anyway (write results and exit), and both PosePoller's thread
    # and this one are daemon threads, closing the survive context isn't
    # required for correctness here -- run it in a background daemon thread
    # and don't wait for it, so a hang in simple_close() can no longer block
    # this script from finishing and reporting results.
    threading.Thread(
        target=lambda: pysurvive.simple_close(ctx.ptr), daemon=True
    ).start()

    t = np.array(t_list)
    pos = np.array(pos_list)
    n = len(t)
    print(
        f"[MeasureViveDrift] Collected {n} samples over {t[-1] if n else 0:.1f}s"
        + (" (STOPPED EARLY due to stall)" if stalled else "")
    )
    if n < 2:
        raise RuntimeError(
            "[MeasureViveDrift] Not enough samples collected (tracking lost "
            "during recording?)."
        )

    r0 = pin.Quaternion(*quat_list[0]).toRotationMatrix()
    ang_drift_deg = np.array(
        [
            np.rad2deg(
                np.linalg.norm(
                    pin.log3(r0.T @ pin.Quaternion(*q).toRotationMatrix())
                )
            )
            for q in quat_list
        ]
    )
    pos_drift_from_start = np.linalg.norm(pos - pos[0], axis=1)
    pos_range = np.ptp(pos, axis=0)

    print(
        f"[MeasureViveDrift] Position drift from start: "
        f"max={pos_drift_from_start.max():.4f} m, "
        f"final={pos_drift_from_start[-1]:.4f} m"
    )
    print(
        f"[MeasureViveDrift] Orientation drift from start: "
        f"max={ang_drift_deg.max():.2f} deg, final={ang_drift_deg[-1]:.2f} deg"
    )
    print(f"[MeasureViveDrift] Per-axis (x,y,z) position range: {np.round(pos_range, 5)} m")
    print(
        f"[MeasureViveDrift] Average drift rate: "
        f"{1000.0 * pos_drift_from_start[-1] / t[-1]:.2f} mm/s (position), "
        f"{ang_drift_deg[-1] / t[-1]:.3f} deg/s (orientation)"
    )

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "tx",
                "ty",
                "tz",
                "qw",
                "qx",
                "qy",
                "qz",
                "pos_drift_from_start_m",
                "rot_drift_from_start_deg",
            ]
        )
        for i in range(n):
            writer.writerow(
                [t[i]]
                + list(pos[i])
                + list(quat_list[i])
                + [pos_drift_from_start[i], ang_drift_deg[i]]
            )
    print(f"[MeasureViveDrift] Saved CSV to {args.csv}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis_idx, axis_name in enumerate("xyz"):
        axes[0].plot(t, pos[:, axis_idx] - pos[0, axis_idx], label=axis_name)
    axes[0].set_ylabel("position drift [m]")
    axes[0].legend()
    axes[0].set_title("Per-axis position drift from start")

    axes[1].plot(t, ang_drift_deg)
    axes[1].set_ylabel("orientation drift [deg]")
    axes[1].set_title("Cumulative orientation drift from start")

    axes[2].plot(t, pos_drift_from_start)
    axes[2].set_ylabel("total position drift [m]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_title("Cumulative position drift magnitude from start")

    fig.suptitle(
        f"Stationary Vive tracker drift (serial={target_serial}, "
        f"{args.duration:.0f}s)"
    )
    fig.tight_layout()
    fig.savefig(args.plot)
    print(f"[MeasureViveDrift] Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
