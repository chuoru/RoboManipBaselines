# The base image bakes in NVIDIA_REQUIRE_CUDA=cuda>=<this version>, which the
# container runtime enforces at startup regardless of what CUDA version the
# Python packages below actually use. If the host driver doesn't natively
# meet it, the runtime falls back to CUDA forward compatibility -- which
# only supports data-center GPUs, so on a consumer/workstation card (e.g.
# TITAN RTX) it fails outright with "forward compatibility was attempted on
# non supported HW", even if the installed torch build itself would have
# run fine natively. Override to match the host driver's real ceiling, e.g.
# nvidia/cuda:12.2.2-devel-ubuntu22.04 for a driver capped at CUDA 12.2 (see
# the matching TORCH_VERSION override below -- both need to move together).
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Points at the Windows host's X server via Docker Desktop's built-in
    # DNS alias. Requires an X server (e.g. VcXsrv) running on Windows with
    # access control disabled (XLaunch "Disable access control").
    DISPLAY=host.docker.internal:0.0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Python 3.10 toolchain + GLX/EGL/GLFW libs for the MuJoCo viewer window,
# Tk for matplotlib's point-cloud plot window, and OpenCV's cv2.imshow deps.
# ffmpeg provides both the CLI binary (used by the ffmpeg-python dependency)
# and the libav*.so.56 shared libs that torchcodec dlopens at import time
# (Ubuntu 22.04's ffmpeg is 4.4.2, matching torchcodec's "FFmpeg version 4"
# build -- without it, `import torchcodec` fails with libavutil.so.* missing).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip python3-tk \
      git cmake build-essential pkg-config \
      libgl1 libglx-mesa0 libegl1 libosmesa6-dev libglfw3 \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      x11-apps mesa-utils ffmpeg \
      patchelf ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace/RoboManipBaselines

# Own layer so the large torch download is cached independently of source
# changes. Override TORCH_INDEX_URL to switch CUDA version or go CPU-only
# (e.g. --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu).
# cu128 is the lowest CUDA-wheel index that actually carries torch==2.10.0
# (verified: the cu124 index only goes up to torch 2.6.0). Use --index-url
# (not --extra-index-url): PyPI's own default index also carries
# CUDA-enabled torch wheels, and --extra-index-url only supplements rather
# than replaces it, so pip can silently mix in a different CUDA build than
# the one requested here.
#
# TORCH_VERSION/TORCHVISION_VERSION/TORCHCODEC_VERSION default to
# pyproject.toml's pins. Override all three together (with a matching
# TORCH_INDEX_URL) to target an older driver that can't run the default
# build -- e.g. torch==2.5.1/torchvision==0.20.1/torchcodec==0.1.1 from the
# cu121 index for a host capped at CUDA 12.2. Whatever is set here is
# reasserted below in its own step, since pip would otherwise resolve
# pyproject.toml's exact torch==2.10.0 pin during the ".[act]" install and
# silently undo an override.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHCODEC_VERSION=0.10.0
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
         torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchcodec==${TORCHCODEC_VERSION}

# .dockerignore trims this to what the common install + ACT extra need
# (drops unused vendored third_party/* submodules to keep the build fast).
COPY . .

RUN python3 -m pip install --no-cache-dir -e ".[act]" \
    && cd third_party/act/detr && python3 -m pip install --no-cache-dir -e .

# Reassert the torch/torchvision/torchcodec versions above: pyproject.toml
# pins torch==2.10.0 as a base dependency, so if TORCH_VERSION was
# overridden, the ".[act]" install just silently pulled torch==2.10.0 back
# in (from plain PyPI, no less) to satisfy that pin. --no-deps is safe here
# because download.pytorch.org wheels bundle their own CUDA runtime libs
# rather than depending on separate nvidia-*-cu12 packages. No-op (fast) when
# TORCH_VERSION matches pyproject.toml's pin, as in the default build.
RUN python3 -m pip install --no-cache-dir --no-deps --index-url ${TORCH_INDEX_URL} \
         torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchcodec==${TORCHCODEC_VERSION}

CMD ["/bin/bash"]
