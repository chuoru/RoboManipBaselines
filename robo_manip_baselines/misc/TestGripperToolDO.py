"""Directly exercise the Fairino's tool DO to find the correct (id,
close_status) for the IAI gripper, without going through the full env/replay
pipeline -- see RealFairino5EnvBase's gripper_type="tool_do" support.

Usage:
    python ./misc/TestGripperToolDO.py --robot_ip 192.168.57.2
"""

import argparse
import socket
import time

from fairino import Robot


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--robot_ip", type=str, default="192.168.57.2")
    return parser.parse_args()


def connect_robot(robot_ip):
    """Same CNDE/XML-RPC fallback as RealFairino5EnvBase._connect_robot(): the
    vendored SDK only proceeds with XML-RPC calls once both the CNDE state-feed
    channel (port 20005) and the XML-RPC command channel (port 20003) have
    connected, but some controllers only expose XML-RPC -- without this, calls
    like SetToolDO/GetToolDO fail with fr_code=-4 (communication error) or
    return malformed values, exactly as seen when calling Robot.RPC() directly."""
    print(f"[TestGripperToolDO] Connecting to {robot_ip}...")
    robot = Robot.RPC(robot_ip)

    if not Robot.RPC.is_connect:
        print(
            "[TestGripperToolDO] CNDE state channel (port 20005) is unreachable. "
            "Verifying the XML-RPC command channel (port 20003) independently."
        )
        xmlrpc_ok = False
        try:
            socket.setdefaulttimeout(1)
            robot.robot.GetControllerIP()
            xmlrpc_ok = True
        except Exception as e:
            print(f"[TestGripperToolDO] XML-RPC verification failed: {e}")
        finally:
            socket.setdefaulttimeout(None)

        if not xmlrpc_ok:
            raise RuntimeError(
                f"[TestGripperToolDO] Failed to connect to the Fairino arm at "
                f"{robot_ip} (neither the CNDE nor the XML-RPC channel is reachable)."
            )
        print(
            "[TestGripperToolDO] XML-RPC is reachable; proceeding without the "
            "CNDE state feed."
        )
        Robot.RPC.is_connect = True

    return robot


def get_tool_do(robot):
    """Robustly unpack GetToolDO()'s return, which has been observed to be a
    bare int (just the fr_code, no bitmask) rather than the documented
    (fr_code, bits) tuple -- possibly connection-state dependent."""
    result = robot.GetToolDO()
    if isinstance(result, tuple):
        return result
    return result, None


def main():
    args = parse_argument()

    robot = connect_robot(args.robot_ip)

    for do_id in (0, 1):
        for status in (0, 1):
            input(
                f"\n[TestGripperToolDO] About to set tool DO{do_id} = {status}. "
                "Press Enter to send, then watch the gripper..."
            )
            fr_code = robot.SetToolDO(do_id, status, 0, 0)
            print(f"[TestGripperToolDO] SetToolDO({do_id}, {status}) -> fr_code={fr_code}")
            readback_fr_code, tl_dgt_output_l = get_tool_do(robot)
            if tl_dgt_output_l is None:
                print(
                    f"[TestGripperToolDO] Readback GetToolDO() -> fr_code={readback_fr_code} "
                    "(no bitmask returned -- ignore, judge by what you observed)"
                )
            else:
                print(
                    f"[TestGripperToolDO] Readback GetToolDO() -> fr_code={readback_fr_code}, "
                    f"bits={tl_dgt_output_l:#04b}, DO{do_id}={(int(tl_dgt_output_l) >> do_id) & 1}"
                )
            observed = input(
                "[TestGripperToolDO] What did the gripper do? "
                "(o=opened, c=closed, n=nothing, Enter to skip note): "
            ).strip()
            print(f"[TestGripperToolDO]   -> noted: {observed or '(none)'}")
            time.sleep(0.3)

    print(
        "\n[TestGripperToolDO] Done. Use whichever (do_id, status->observed) pair "
        "opened/closed the gripper to set --gripper_do_id and "
        "--gripper_do_close_status (the status value that CLOSES it) correctly."
    )


if __name__ == "__main__":
    main()
