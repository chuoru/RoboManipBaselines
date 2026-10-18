"""Connects to a running insta360_bridge and tallies ORB-SLAM3 tracking_state
("pose" messages, see common/utils/Insta360Protocol.py) over a live session,
so tracking quality can be measured the same way as this repo's own
documented baseline (feat/insta360-docker-slam-fixes, measured on an X3,
firmware v1.1.6, 70s of continuous handheld motion: 79.0% OK, 19.0% INIT,
2.1% LOST, 0 crashes) instead of eyeballing scrolling stdout.

This only sees the socket side (tracking_state per pose message) -- it
canNOT see things only visible in the bridge process's own stdout/exit code,
notably crash backtraces and "Empty IMU measurements vector!!!" counts. A
dropped connection is reported here as a likely crash, but check the bridge
process's own terminal/exit code to confirm and get a backtrace (gdb --args
./build/insta360_bridge ...) if so.

Usage:
    python ./misc/MeasureInsta360TrackingPerformance.py \\
        --socket_path /tmp/insta360_hand.sock [--duration 70]

Move the camera with real 6-DoF motion while this runs (just like the
documented baseline run) -- holding it still mostly measures how long
initialization takes, not steady-state tracking quality. Omit --duration to
run until Ctrl+C.
"""

import argparse
import socket
import time

from robo_manip_baselines.common import read_insta360_message


def parse_argument():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--socket_path",
        type=str,
        required=True,
        help="insta360_bridge's Unix domain socket path (e.g. /tmp/insta360_hand.sock)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to measure; omit to run until Ctrl+C",
    )
    parser.add_argument(
        "--live_update_sec",
        type=float,
        default=1.0,
        help="how often to refresh the live status line (0 to disable)",
    )
    return parser.parse_args()


def main():
    args = parse_argument()

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(args.socket_path)
    print(f"[MeasureInsta360TrackingPerformance] Connected to {args.socket_path}")
    print(
        "[MeasureInsta360TrackingPerformance] Move the camera with real 6-DoF "
        "motion now -- holding it still mostly measures init time, not "
        "steady-state tracking."
    )

    # Counts every pose message by its tracking_state ("OK"/"INIT"/"LOST").
    # reset_count: OK -> non-OK transitions, i.e. how many times tracking
    # that was previously locked on was lost/reinitialized -- the metric
    # this repo's own hardware-debugging notes call "resets" (e.g. "26 in a
    # 70s run" before the fixes that motivated this script).
    counts = {}
    reset_count = 0
    prev_state = None
    start_time = time.time()
    last_print_time = start_time
    disconnect_reason = None

    try:
        while True:
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break
            try:
                message = read_insta360_message(connection)
            except ConnectionError as e:
                disconnect_reason = str(e)
                break

            if message["type"] != "pose":
                continue

            state = message["tracking_state"]
            counts[state] = counts.get(state, 0) + 1
            if prev_state == "OK" and state != "OK":
                reset_count += 1
            prev_state = state

            now = time.time()
            if args.live_update_sec > 0 and (now - last_print_time) >= args.live_update_sec:
                elapsed = now - start_time
                total = sum(counts.values())
                ok_pct = 100.0 * counts.get("OK", 0) / total if total else 0.0
                print(
                    f"\r[{elapsed:6.1f}s] OK {ok_pct:5.1f}%  resets {reset_count}  "
                    f"msgs {total}  ",
                    end="",
                    flush=True,
                )
                last_print_time = now
    except KeyboardInterrupt:
        print("\n[MeasureInsta360TrackingPerformance] Interrupted by Ctrl+C.")
    finally:
        connection.close()

    elapsed = time.time() - start_time
    total = sum(counts.values())

    print("\n")
    if disconnect_reason is not None:
        print(
            f"[MeasureInsta360TrackingPerformance] Bridge connection closed "
            f"after {elapsed:.1f}s ({disconnect_reason}) -- likely a crash; "
            "check the bridge process's own stdout/exit code (and gdb for a "
            "backtrace if so)."
        )
    if total == 0:
        print(
            "[MeasureInsta360TrackingPerformance] WARNING: no pose messages "
            "received at all -- check the bridge is running and tracking, "
            "and that --socket_path is correct."
        )
        return

    print(
        f"[MeasureInsta360TrackingPerformance] Summary over {elapsed:.1f}s "
        f"({total} pose messages, {total / elapsed if elapsed > 0 else 0:.1f} msg/s):"
    )
    print(f"{'  state':<12}{'count':>8}{'pct':>8}")
    for state in ("OK", "INIT", "LOST"):
        n = counts.get(state, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  {state:<10}{n:>8}{pct:>7.1f}%")
    other_states = set(counts) - {"OK", "INIT", "LOST"}
    for state in sorted(other_states):
        n = counts[state]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {state:<10}{n:>8}{pct:>7.1f}%  (unexpected tracking_state)")
    print(f"  resets (OK -> non-OK transitions): {reset_count}")
    print(
        "\n[MeasureInsta360TrackingPerformance] Compare against this repo's "
        "documented baseline (79.0% OK / 19.0% INIT / 2.1% LOST, 0 resets to "
        "a crash, over 70s -- see "
        "envs/real/insta360_bridge/README.md's Status section)."
    )


if __name__ == "__main__":
    main()
