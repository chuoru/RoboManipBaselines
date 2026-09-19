"""Pure-Python stand-in for envs/real/insta360_bridge (the real C++ process
that talks to the Insta360 CameraSDK and ORB-SLAM3 -- see that directory's
README for the real thing). Speaks the exact same wire protocol (see
common/utils/Insta360Protocol.py) over a Unix domain socket, so it lets the
Python side (RealEnvBase.setup_insta360, Insta360InputDevice) and the MuJoCo
teleop wiring be exercised end-to-end without any real Insta360 hardware.

Sends a synthetic test-pattern frame and a synthetic pose to every connected
client at --fps. The pose holds still at the origin for --settle_hold_sec
(so Insta360InputDevice's settle-and-anchor state machine can actually reach
enabled_teleop=True -- see that class's MIN_ANCHOR_DELAY/POSE_SETTLE_TIME;
with the defaults, teleop will not enable until a bit after 10s even if this
process has been holding still since t=0), then moves in a small circle
(translation in the XY plane, rotation about Z) so a connected
Insta360InputDevice has real, continuous relative motion to track.

Usage:
    python ./misc/MockInsta360Bridge.py --socket /tmp/insta360_hand.sock
"""

import argparse
import os
import socket
import threading
import time

import numpy as np

from robo_manip_baselines.common import encode_insta360_message


class MockInsta360Bridge:
    def __init__(
        self,
        socket_path,
        fps,
        image_size,
        settle_hold_sec,
        pos_amplitude,
        rot_amplitude,
        period,
    ):
        self.socket_path = socket_path
        self.fps = fps
        self.width, self.height = image_size
        self.settle_hold_sec = settle_hold_sec
        self.pos_amplitude = pos_amplitude
        self.rot_amplitude = rot_amplitude
        self.period = period

        self._clients = []
        self._clients_lock = threading.Lock()

    def _make_frame(self, t):
        # A simple moving gradient, just so a connected viewer can visually
        # confirm frames are actually updating.
        hue = int((t * 40) % 180)
        frame = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
        frame[:, :, 0] = hue
        frame[:, :, 1] = 200
        frame[:, :, 2] = 200
        return frame

    def _make_pose(self, t):
        if t < self.settle_hold_sec:
            pos = [0.0, 0.0, 0.0]
            quat = [1.0, 0.0, 0.0, 0.0]
        else:
            theta = 2.0 * np.pi * (t - self.settle_hold_sec) / self.period
            pos = [
                self.pos_amplitude * np.cos(theta) - self.pos_amplitude,
                self.pos_amplitude * np.sin(theta),
                0.0,
            ]
            half_angle = 0.5 * self.rot_amplitude * np.sin(theta)
            quat = [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
        return pos, quat

    def _accept_loop(self, server_socket):
        while True:
            connection, _ = server_socket.accept()
            with self._clients_lock:
                self._clients.append(connection)
            print(f"[{self.__class__.__name__}] Client connected.")

    def _broadcast(self, message_bytes):
        with self._clients_lock:
            remaining = []
            for connection in self._clients:
                try:
                    connection.sendall(message_bytes)
                    remaining.append(connection)
                except OSError:
                    connection.close()
            self._clients = remaining

    def run(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(self.socket_path)
        server_socket.listen()
        print(f"[{self.__class__.__name__}] Listening on {self.socket_path}")

        accept_thread = threading.Thread(
            target=self._accept_loop, args=(server_socket,), daemon=True
        )
        accept_thread.start()

        start_time = time.time()
        period_sec = 1.0 / self.fps
        try:
            while True:
                tick_start = time.time()
                t = tick_start - start_time

                frame = self._make_frame(t)
                frame_header = {
                    "type": "frame",
                    "t": tick_start,
                    "w": self.width,
                    "h": self.height,
                }
                self._broadcast(
                    encode_insta360_message(frame_header, frame.tobytes())
                )

                pos, quat = self._make_pose(t)
                pose_header = {
                    "type": "pose",
                    "t": tick_start,
                    "pos": pos,
                    "quat": quat,
                    "tracking_state": "OK",
                }
                self._broadcast(encode_insta360_message(pose_header))

                elapsed = time.time() - tick_start
                if elapsed < period_sec:
                    time.sleep(period_sec - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            with self._clients_lock:
                for connection in self._clients:
                    connection.close()
            server_socket.close()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket", type=str, required=True, help="Unix domain socket path to serve"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--image_size", type=str, default="640,480", help="'width,height'"
    )
    parser.add_argument(
        "--settle_hold_sec",
        type=float,
        default=1.0,
        help="how long to hold a stationary pose before starting circular motion",
    )
    parser.add_argument("--pos_amplitude", type=float, default=0.05, help="[m]")
    parser.add_argument(
        "--rot_amplitude", type=float, default=0.3, help="peak rotation about Z [rad]"
    )
    parser.add_argument(
        "--period", type=float, default=8.0, help="circular motion period [s]"
    )
    args = parser.parse_args()

    bridge = MockInsta360Bridge(
        socket_path=args.socket,
        fps=args.fps,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        settle_hold_sec=args.settle_hold_sec,
        pos_amplitude=args.pos_amplitude,
        rot_amplitude=args.rot_amplitude,
        period=args.period,
    )
    bridge.run()
