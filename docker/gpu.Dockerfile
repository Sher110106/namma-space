# Namma Space GPU image: gsplat 1.5.3 (Apache-2.0) on PyTorch 2.4.1 / CUDA 12.4.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=32
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 build-essential ninja-build curl \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git /opt/gsplat
RUN pip install --no-cache-dir /opt/gsplat
RUN pip install --no-cache-dir -r /opt/gsplat/examples/requirements.txt
RUN pip install --no-cache-dir open3d trimesh plyfile scipy scikit-image shapely "opencv-python-headless<4.11" "numpy<2"
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgomp1 libx11-6 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
