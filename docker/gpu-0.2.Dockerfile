# namma-space-gpu:0.2 = 0.1 + Node.js 22 + @playcanvas/splat-transform + MoGe-2 (code only).
# MoGe-2 weights are NOT baked in: they live in the host cache /home/sher/namma-space/cache/hf
# (HF_HOME=/w/cache/hf, HF_HUB_OFFLINE=1). Keeps the new layers small (plan A5).
FROM namma-space-gpu:0.1

# --- Node.js 22 (MIT), official tarball, checksum pinned -------------------------------------
ARG NODE_VERSION=22.23.4
ARG NODE_SHA256=b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406
RUN curl -fsSL -o /tmp/node.tgz https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz \
 && echo "${NODE_SHA256}  /tmp/node.tgz" | sha256sum -c - \
 && tar -xzf /tmp/node.tgz -C /usr/local --strip-components=1 --no-same-owner \
      --exclude='*/CHANGELOG.md' --exclude='*/README.md' --exclude='*/share/doc' --exclude='*/share/man' \
 && rm -f /tmp/node.tgz \
 && node --version && npm --version

# --- splat-transform (MIT): PLY <-> SPZ etc. ---------------------------------------------------
ARG SPLAT_TRANSFORM_VERSION=3.10.3
RUN npm install -g --no-fund --no-audit @playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION} \
 && npm cache clean --force \
 && splat-transform --help | head -5

# --- MoGe-2 (MIT code + weights), pinned commits; --no-deps avoids gradio and a second OpenCV ---
ARG MOGE_SHA=07444410f1e33f402353b99d6ccd26bd31e469e8
ARG UTILS3D_SHA=3fab839f0be9931dac7c8488eb0e1600c236e183
RUN pip install --no-cache-dir --no-deps \
      "git+https://github.com/microsoft/MoGe.git@${MOGE_SHA}" \
      "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_SHA}" \
 && pip install --no-cache-dir "huggingface_hub<1" click "numpy<2" \
 && python -c "import moge, utils3d; from moge.model.v2 import MoGeModel; print('moge import ok')"
ENV HF_HOME=/w/cache/hf
