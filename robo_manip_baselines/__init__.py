# Load the system libxcb into the global symbol namespace before anything imports
# PyAV ("av", pulled in via torchvision -> robo_manip_baselines.common). The PyAV
# manylinux wheel bundles its own copies of libxcb/libxcb-shm; once those are
# loaded first, OpenCV's Qt xcb platform plugin binds against them and its very
# first GUI call (cv2.imshow/namedWindow) deadlocks inside
# QXcbBasicConnection::initializeShm -> xcb_shm_query_version ->
# xcb_wait_for_reply, hanging the main thread in native code where even Ctrl+C
# (KeyboardInterrupt) cannot be delivered. Preloading the system libraries here
# makes Qt resolve the xcb symbols to consistent copies, while PyAV keeps using
# its own bundled ones. This must run before "import av"; preloading after av is
# already loaded does not help.
import ctypes as _ctypes
import platform as _platform

if _platform.system() == "Linux":
    for _lib in ("libxcb.so.1", "libxcb-shm.so.0"):
        try:
            _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            # Headless system without X libraries; cv2 GUI would not work anyway.
            pass

from .version import __version__
from . import envs
