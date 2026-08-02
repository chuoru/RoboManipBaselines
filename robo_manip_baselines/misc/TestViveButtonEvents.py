"""Check whether pysurvive's event-queue API (simple_next_event ->
SimpleEventType_ButtonEvent) picks up button presses on a tracker that
simple_object_get_button_mask() (the polling API used by ViveInputDevice)
fails to report -- observed on this rig for a tracker libsurvive enumerates as
"WM0" (Wireless Watchman/controller) rather than a plain tracker.

Usage:
    python ./misc/TestViveButtonEvents.py
Then press the tracker's button a few times and watch for output.
"""

import time

import pysurvive


def main():
    ctx = pysurvive.SimpleContext([])
    print("[TestViveButtonEvents] Connected. Press the tracker's button a few "
          "times. Press Ctrl+C to stop.\n")

    event = pysurvive.SurviveSimpleEvent()
    try:
        while True:
            event_type = pysurvive.simple_next_event(ctx.ptr, event)
            if event_type == pysurvive.SimpleEventType_None:
                time.sleep(0.001)
                continue
            if event_type == pysurvive.SimpleEventType_ButtonEvent:
                button_event_ptr = pysurvive.simple_get_button_event(event)
                be = button_event_ptr.contents
                name = pysurvive.simple_object_name(be.object)
                if isinstance(name, bytes):
                    name = name.decode()
                print(
                    f"[TestViveButtonEvents] BUTTON EVENT: object={name} "
                    f"event_type={be.event_type} button_id={be.button_id} "
                    f"time={be.time:.3f}"
                )
            else:
                print(f"[TestViveButtonEvents] (other event_type={event_type})")
    except KeyboardInterrupt:
        pass
    finally:
        pysurvive.simple_close(ctx.ptr)


if __name__ == "__main__":
    main()
