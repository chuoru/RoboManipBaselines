import os
import shutil
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import videoio

from .DataKey import DataKey

# torchcodec decodes RGB videos slightly faster than videoio, but it dlopens
# FFmpeg's shared libs at import time and each wheel ships loaders for only
# some FFmpeg major versions -- torchcodec 0.1.1 (the build that pairs with
# torch 2.5.1, which in turn is pinned for older NVIDIA drivers) ships
# libtorchcodec{5,6,7}.so only, while Ubuntu 22.04 provides FFmpeg 4
# (libavutil.so.56). That mismatch makes `import torchcodec` raise, which
# previously took the whole package down with it -- training could not even
# start. videoio is already a hard dependency here (the depth path and all
# video writing use it) and reads the same files, so treat torchcodec as an
# optional accelerator and fall back when it is unusable.
try:
    import torchcodec

    _TORCHCODEC_AVAILABLE = True
except Exception:
    torchcodec = None
    _TORCHCODEC_AVAILABLE = False


def to_hashable(path, idx, image_size):
    if isinstance(idx, slice):
        idx_key = (idx.start, idx.stop, idx.step)
    else:
        idx_key = idx
    return (path, idx_key, image_size)


class RmbData:
    """Data in RoboManipBaselines format."""

    class RmbVideo:
        # Note that this cache is not shared among processes in the multi-process data loader.
        # Bounded LRU (not a plain dict): each entry can be a full horizon-length
        # window of frames (tens of MB at typical camera resolution), and this
        # cache never used to evict anything -- with enough distinct (path, idx)
        # windows touched over an epoch (e.g. many episodes x many sliding
        # windows x multiple cameras), it grew large enough to exhaust system
        # RAM and crash training (observed: free RAM dropping from ~22GB to
        # ~1.5GB within 2 epochs on a 28-episode dataset). MAX_CACHE_ENTRIES is
        # a blunt cap (not size-aware), chosen so worst-case entries (16-frame
        # windows at 640x480x3 uint8, ~14.7MB each) stay around a few GB.
        MAX_CACHE_ENTRIES = 200
        cache = OrderedDict()
        # Keyed by path only (not by idx like `cache` above), so the container
        # open/parse cost is paid once per video per worker process instead of
        # once per __getitem__ call. Reusing the decoder for repeated random
        # access is still slower than sequential decode (each access can
        # require a fresh seek), but avoids re-parsing the file header every
        # single time -- worth ~20% in practice, measured by re-reading 200
        # random 16-frame windows from the same file with vs. without reuse.
        _decoder_cache = {}

        def __init__(self, path, enable_cache=False, image_size=None):
            self.path = path
            self.enable_cache = enable_cache
            self.image_size = None if image_size is None else tuple(image_size)

        def video_metadata(self):
            """(num_frames, height, width) of this video.

            Uses torchcodec's header parse when available, otherwise decodes
            with videoio -- correct either way, just slower on the fallback
            path, and only paid on shape/len queries rather than per frame.
            """
            if _TORCHCODEC_AVAILABLE:
                metadata = torchcodec.decoders.VideoDecoder(self.path).metadata
                return metadata.num_frames, metadata.height, metadata.width
            frames = np.asarray(list(videoio.videoread(self.path)))
            return frames.shape[0], frames.shape[1], frames.shape[2]

        def __len__(self):
            return self.video_metadata()[0]

        def __getitem__(self, idx):
            if self.enable_cache:
                hashable = to_hashable(self.path, idx, self.image_size)
                if hashable in self.cache:
                    self.cache.move_to_end(hashable)
                else:
                    self.cache[hashable] = self._get_data(idx)
                    if len(self.cache) > self.MAX_CACHE_ENTRIES:
                        self.cache.popitem(last=False)
                return self.cache[hashable]
            else:
                return self._get_data(idx)

    class RmbRgbVideo(RmbVideo):
        def _get_data(self, idx):
            # torchcodec's VideoDecoder is slightly faster, so prefer it and
            # fall back to videoio when it could not be imported (see the
            # import guard at the top of this module).
            if _TORCHCODEC_AVAILABLE:
                if self.path not in RmbData.RmbVideo._decoder_cache:
                    RmbData.RmbVideo._decoder_cache[self.path] = (
                        torchcodec.decoders.VideoDecoder(
                            self.path, dimension_order="NHWC"
                        )
                    )
                decoder = RmbData.RmbVideo._decoder_cache[self.path]
                data = decoder[idx].numpy()
            else:
                # videoio has no seek/random-access API, so the whole clip is
                # decoded and indexed. Cache the decoded array rather than
                # re-decoding per access -- without it this is far slower than
                # torchcodec, not just "slightly".
                if self.path not in RmbData.RmbVideo._decoder_cache:
                    RmbData.RmbVideo._decoder_cache[self.path] = np.asarray(
                        list(videoio.videoread(self.path))
                    )
                data = RmbData.RmbVideo._decoder_cache[self.path][idx]
            if self.image_size is not None:
                if data.ndim == 3:  # (H, W, C)
                    data = cv2.resize(
                        data,
                        self.image_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                elif data.ndim == 4:  # (T, H, W, C)
                    data = np.stack(
                        [
                            cv2.resize(
                                frame,
                                self.image_size,
                                interpolation=cv2.INTER_LINEAR,
                            )
                            for frame in data
                        ],
                        axis=0,
                    )
                else:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Unexpected video data ndim: {data.ndim}, expected 3 or 4"
                    )
            return data

        @property
        def shape(self):
            num_frames, height, width = self.video_metadata()
            if self.image_size is not None:
                width, height = self.image_size
            return (
                num_frames,
                height,
                width,
                3,
            )

        @property
        def dtype(self):
            return np.uint8

    class RmbDepthVideo(RmbVideo):
        def _get_data(self, idx):
            data = (1e-3 * videoio.uint16read(self.path)[idx]).astype(np.float32)
            if self.image_size is not None:
                if data.ndim == 2:  # (H, W)
                    data = cv2.resize(
                        data,
                        self.image_size,
                        interpolation=cv2.INTER_LINEAR,
                    )

                elif data.ndim == 3:  # (N, H, W)
                    data = np.stack(
                        [
                            cv2.resize(
                                frame,
                                self.image_size,
                                interpolation=cv2.INTER_LINEAR,
                            )
                            for frame in data
                        ],
                        axis=0,
                    )
                else:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Unexpected video data ndim: {data.ndim}, expected 2 or 3"
                    )
            return data

        @property
        def shape(self):
            num_frames, height, width = self.video_metadata()
            if self.image_size is not None:
                width, height = self.image_size
            return (
                num_frames,
                height,
                width,
            )

        @property
        def dtype(self):
            return np.float32

    def __init__(self, path, enable_cache=False, mode="r", image_size=None):
        self.path = path
        self.enable_cache = enable_cache
        self.mode = mode
        self.image_size = None if image_size is None else tuple(image_size)

        _, ext = os.path.splitext(self.path.rstrip("/"))
        if ext.lower() == ".hdf5":
            self.is_single_hdf5 = True
        elif ext.lower() == ".rmb":
            self.is_single_hdf5 = False
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid file extension '{ext}'. Expected '.hdf5' or '.rmb': {self.path}"
            )

        self.h5file = None

    def open(self):
        if self.path is None:
            raise ValueError(f"[{self.__class__.__name__}] The file path is not set.")

        if self.h5file is not None:
            self.close()

        if self.is_single_hdf5:
            path = self.path
        else:
            path = os.path.join(self.path, "main.rmb.hdf5")
        self.h5file = h5py.File(path, self.mode)

    def close(self):
        if self.h5file is None:
            return

        self.h5file.close()
        self.h5file = None

    @property
    def closed(self):
        return self.h5file is None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, key):
        if self.is_single_hdf5:
            return self.h5file[key]
        elif DataKey.is_rgb_image_key(key):
            return self.RmbRgbVideo(
                os.path.join(self.path, f"{key}.rmb.mp4"),
                enable_cache=self.enable_cache,
                image_size=self.image_size,
            )
        elif DataKey.is_depth_image_key(key):
            return self.RmbDepthVideo(
                os.path.join(self.path, f"{key}.rmb.mp4"),
                enable_cache=self.enable_cache,
                image_size=self.image_size,
            )
        else:
            return self.h5file[key]

    def keys(self):
        if self.is_single_hdf5:
            return self.h5file.keys()
        else:
            ret = list(self.h5file.keys())

            for filename in os.listdir(self.path):
                if not filename.endswith(".rmb.mp4"):
                    continue
                key = filename[: -len(".rmb.mp4")]
                if DataKey.is_rgb_image_key(key) or DataKey.is_depth_image_key(key):
                    ret.append(key)

            return ret

    def __contains__(self, key):
        return key in self.keys()

    @property
    def attrs(self):
        return self.h5file.attrs

    def dump_to_hdf5(self, dst_path, force_overwrite=False):
        _, dst_ext = os.path.splitext(dst_path)
        if dst_ext.lower() != ".hdf5":
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid file extension '{dst_ext}'. Expected '.hdf5': {dst_path}"
            )

        self._check_file_existence(dst_path, force_overwrite)

        with h5py.File(dst_path, "w") as dst_h5file:
            for key in self.keys():
                if DataKey.is_rgb_image_key(key) or DataKey.is_depth_image_key(key):
                    dst_h5file.create_dataset(key, data=self[key][:])
                else:
                    self.h5file.copy(key, dst_h5file)

            for key in self.attrs.keys():
                dst_h5file.attrs[key] = self.attrs[key]
            dst_h5file.attrs["format"] = "RmbData-SingleHDF5"

        print(f"[{self.__class__.__name__}] Succeeded to dump a HDF5 file: {dst_path}")

    def dump_to_rmb(self, dst_path, force_overwrite=False):
        _, dst_ext = os.path.splitext(dst_path.rstrip("/"))
        if dst_ext.lower() != ".rmb":
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid file extension '{dst_ext}'. Expected '.rmb': {dst_path}"
            )

        self._check_file_existence(dst_path, force_overwrite)

        os.makedirs(dst_path, exist_ok=True)

        dst_hdf5_path = os.path.join(dst_path, "main.rmb.hdf5")
        with h5py.File(dst_hdf5_path, "w") as dst_h5file:
            for key in self.keys():
                if DataKey.is_rgb_image_key(key):
                    dst_video_path = os.path.join(dst_path, f"{key}.rmb.mp4")
                    images = self[key][:]
                    videoio.videosave(dst_video_path, images)
                elif DataKey.is_depth_image_key(key):
                    dst_video_path = os.path.join(dst_path, f"{key}.rmb.mp4")
                    images = (1e3 * self[key][:]).astype(np.uint16)
                    videoio.uint16save(dst_video_path, images)
                else:
                    self.h5file.copy(key, dst_h5file)

            for key in self.attrs.keys():
                dst_h5file.attrs[key] = self.attrs[key]
            dst_h5file.attrs["format"] = "RmbData-Compact"

        print(f"[{self.__class__.__name__}] Succeeded to dump RMB files: {dst_path}")

    def _check_file_existence(self, path, force_overwrite):
        if not os.path.exists(path):
            return

        if force_overwrite:
            will_remove = True
        else:
            print(f"[{self.__class__.__name__}] A file already exists: {path}")
            answer = input(
                f"[{self.__class__.__name__}] Do you want to overwrite it? (y/n): "
            )
            will_remove = answer.strip().lower() == "y"

        if will_remove:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
