# COLMAP 4.2.1 (BSD-3) plus cuDNN so ONNX Runtime can run ALIKED/LightGlue on the GPU.
FROM colmap/colmap:latest
RUN apt-get update && (apt-get install -y --no-install-recommends libcudnn9-cuda-12 || apt-get install -y --no-install-recommends cudnn9-cuda-12) && rm -rf /var/lib/apt/lists/*
