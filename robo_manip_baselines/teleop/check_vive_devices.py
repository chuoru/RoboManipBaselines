"""Standalone diagnostic to check which libsurvive devices (lighthouses,
trackers, controllers) are currently detected, and print each one's live pose.

Uses pysurvive (libsurvive) directly over USB/wireless -- no SteamVR/Steam
installation required. Run this before teleop to confirm the lighthouses and
trackers are recognized, and to look up each tracker's stable hardware serial
number (e.g. "LHR-904A2704") for device_params.serial_number in
teleop/configs/Vive*.yaml. Note: lighthouse serials also start with "LH" but
read "LHB-..." -- don't confuse them with the tracker's "LHR-..." serial.

Devices only appear here once libsurvive finishes negotiating with them, which
can take a few seconds after startup (trackers typically take longer than
lighthouses), so give it 5-10s before concluding one is missing.

Usage:
    python ./teleop/check_vive_devices.py
"""

import time

import pysurvive


def main():
    ctx = pysurvive.SimpleContext([])
    print("[check_vive_devices] Connected to libsurvive. Press Ctrl+C to stop.\n")

    known_objects = {}  # serial_number -> SimpleObject

    try:
        while ctx.Running():
            # SimpleContext.Objects() is a static snapshot taken at startup, so
            # devices that finish negotiating afterwards (usually the trackers)
            # never appear in it -- NextUpdated() is the only way to catch them.
            while True:
                obj = ctx.NextUpdated()
                if obj is None:
                    break
                serial_number = pysurvive.simple_serial_number(obj.ptr)
                if isinstance(serial_number, bytes):
                    serial_number = serial_number.decode()
                known_objects[serial_number] = obj

            rows = []
            for serial_number, obj in known_objects.items():
                name = obj.Name()
                if isinstance(name, bytes):
                    name = name.decode()

                pose, timecode = obj.Pose()
                if timecode > 0:
                    pos = pose.Pos
                    status = f"xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                else:
                    status = "NOT tracked yet"

                button_mask = pysurvive.simple_object_get_button_mask(obj.ptr)
                status += f" button_mask={button_mask:#06x}"

                rows.append(f"  name={name:<6s} serial={serial_number:<16s} {status}")

            print("\033c", end="")  # clear terminal each frame
            print(f"[check_vive_devices] Detected {len(rows)} device(s):")
            print("\n".join(sorted(rows)) if rows else "  (none yet)")

            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        pysurvive.simple_close(ctx.ptr)


if __name__ == "__main__":
    main()
