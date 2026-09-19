"""Wire protocol shared between the Insta360 bridge (envs/real/insta360_bridge, a
separate C++ process that talks to the Insta360 CameraSDK and ORB-SLAM3) and this
repo's Python side (RealEnvBase.setup_insta360, Insta360InputDevice). Also
implemented by misc/MockInsta360Bridge.py, a pure-Python stand-in server used to
test the Python side end-to-end without real Insta360 hardware.

Framing, sent over a Unix domain socket:
    4 bytes            big-endian uint32 header_len
    header_len bytes   UTF-8 JSON header
    (if header["type"] == "frame")
        header["w"] * header["h"] * 3 bytes   raw RGB, row-major, uint8

Header for a "frame" message: {"type": "frame", "t": <unix timestamp>, "w": ..., "h": ...}
Header for a "pose" message: {"type": "pose", "t": <unix timestamp>,
    "pos": [x, y, z], "quat": [w, x, y, z], "tracking_state": "OK"|"LOST"|"INIT"}

A single bridge process serves both message types on one socket; a frame-only
consumer (RealEnvBase) and a pose-only consumer (Insta360InputDevice) each open
their own client connection and ignore the message types they don't need.
"""

import json
import struct

import numpy as np

_HEADER_LEN_STRUCT = struct.Struct(">I")


def recv_exact(sock, num_bytes):
    """Read exactly num_bytes from sock, or raise ConnectionError on EOF."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                "Insta360 bridge connection closed while reading "
                f"{num_bytes} bytes ({remaining} remaining)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(sock):
    """Read one framed message from sock. Returns the JSON header dict, with a
    "frame" key added (an (h, w, 3) uint8 np.ndarray) when header["type"] ==
    "frame". Raises ConnectionError on a clean EOF (bridge disconnected)."""
    (header_len,) = _HEADER_LEN_STRUCT.unpack(recv_exact(sock, 4))
    header = json.loads(recv_exact(sock, header_len).decode("utf-8"))

    if header["type"] == "frame":
        width = header["w"]
        height = header["h"]
        payload = recv_exact(sock, width * height * 3)
        header["frame"] = np.frombuffer(payload, dtype=np.uint8).reshape(
            (height, width, 3)
        )

    return header


def encode_message(header, payload=None):
    """Encode one framed message for sending. `header` is a JSON-serializable
    dict (must not itself contain "frame"); `payload` is the raw bytes to
    append after the header for a "frame" message (omit for "pose")."""
    header_bytes = json.dumps(header).encode("utf-8")
    parts = [_HEADER_LEN_STRUCT.pack(len(header_bytes)), header_bytes]
    if payload is not None:
        parts.append(payload)
    return b"".join(parts)
