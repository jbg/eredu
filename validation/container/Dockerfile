# syntax=docker/dockerfile:1.7

ARG TARGETPLATFORM=linux/amd64

# Rust is mounted into the build step and is not retained in the final image.
FROM --platform=${TARGETPLATFORM} docker.io/library/rust:1.89.0-bookworm@sha256:948f9b08a66e7fe01b03a98ef1c7568292e07ec2e4fe90d88c07bb14563c84ff AS rust-toolchain

# linux/amd64 manifest for nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04.
FROM --platform=${TARGETPLATFORM} docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04@sha256:0447c3b139d17436d399cc9d5070062c5013bb1b1a35187c5020a644f642a1ef AS checkpoint-probe-builder

ENV DEBIAN_FRONTEND=noninteractive \
    SAFEMLX_CUDA_ARCHITECTURES="80;89;90" \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    CUDAToolkit_ROOT=/usr/local/cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libblas-dev \
        liblapack-dev \
        liblapacke-dev \
        ninja-build \
        pkg-config \
        python3-pip \
    && python3 -m pip install --no-cache-dir cmake==3.31.6 \
    && rm -rf /var/lib/apt/lists/*

# Build from a read-only context mount so repository sources and Cargo output
# never become image layers. The linkage check permits only libcuda.so.1,
# which the NVIDIA container runtime injects from the GPU host driver.
RUN --mount=type=bind,from=rust-toolchain,source=/usr/local/cargo/bin,target=/opt/rust/bin,readonly \
    --mount=type=bind,from=rust-toolchain,source=/usr/local/rustup,target=/opt/rustup,readonly \
    --mount=type=bind,source=.,target=/src,readonly \
    --mount=type=cache,id=safemlx-cargo-registry-amd64,target=/tmp/safemlx-cargo/registry \
    --mount=type=cache,id=safemlx-cargo-git-amd64,target=/tmp/safemlx-cargo/git \
    --mount=type=cache,id=safemlx-cargo-target-amd64,target=/tmp/safemlx-target \
    set -eux; \
    export PATH="/opt/rust/bin:${PATH}"; \
    export RUSTUP_HOME=/opt/rustup; \
    export RUSTUP_TOOLCHAIN=1.89.0-x86_64-unknown-linux-gnu; \
    export CARGO_HOME=/tmp/safemlx-cargo; \
    export CARGO_TARGET_DIR=/tmp/safemlx-target; \
    cd /src; \
    rustc --version | grep -F 'rustc 1.89.0'; \
    cargo build --locked --release \
        -p safemlx-lm \
        --no-default-features \
        --features cuda,fancy-regex \
        --example checkpoint_probe; \
    install -m 0755 \
        /tmp/safemlx-target/release/examples/checkpoint_probe \
        /usr/local/bin/checkpoint_probe; \
    mlx_cute_header="$(find /tmp/safemlx-target/release/build \
        -path '*/include/cute/numeric/numeric_types.hpp' \
        -print -quit)"; \
    test -n "${mlx_cute_header}"; \
    mlx_include_root="$(dirname "$(dirname "$(dirname "${mlx_cute_header}")")")"; \
    mlx_cccl_include_root="$(find /tmp/safemlx-target/release/build \
        -type d -path '*/_deps/cccl-src/include' \
        -print -quit)"; \
    test -n "${mlx_cccl_include_root}"; \
    install -d /usr/local/include; \
    cp -a "${mlx_include_root}/." /usr/local/include/; \
    install -d /usr/local/include/cccl; \
    cp -a "${mlx_cccl_include_root}/." /usr/local/include/cccl/; \
    ldd /usr/local/bin/checkpoint_probe > /tmp/checkpoint-probe.ldd; \
    unexpected_missing="$(awk '/not found/ && $1 != "libcuda.so.1" { print $1 }' /tmp/checkpoint-probe.ldd)"; \
    if [ -n "${unexpected_missing}" ]; then \
        printf 'unexpected missing libraries:\n%s\n' "${unexpected_missing}" >&2; \
        exit 1; \
    fi

# linux/amd64 manifest for pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime.
FROM --platform=${TARGETPLATFORM} docker.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime@sha256:e05438443ae3c407e8d04447091a959dbb6757b6290b128770c3c787d4bd442b

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=cuda12.9.1-rust1.89.0-torch2.8.0-v1

LABEL org.opencontainers.image.title="SafeMLX CUDA checkpoint validation" \
      org.opencontainers.image.description="Pinned CUDA environment for SafeMLX checkpoint correctness and performance probes" \
      org.opencontainers.image.source="https://github.com/jbg/safemlx" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/data/huggingface \
    PYTHONPATH=/opt/safemlx \
    SAFEMLX_RESULTS=/data/results \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# Keep only runtime libraries in the final filesystem. The CUDA/cuDNN runtime
# and PyTorch 2.8.0 are supplied by the digest-pinned parent image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libblas3 \
        libgomp1 \
        liblapack3 \
        liblapacke \
    && rm -rf /var/lib/apt/lists/*

# Pin the direct Python validation dependencies. The complete resolved set is
# recorded in /opt/safemlx/python-packages.lock.txt in the published image.
RUN python -m pip install --no-cache-dir \
        accelerate==1.14.0 \
        huggingface-hub==1.24.0 \
        PyYAML==6.0.3 \
        safetensors==0.8.0 \
        transformers==5.14.1 \
    && python -m pip freeze | LC_ALL=C sort > /tmp/python-packages.lock.txt

ENV SAFEMLX_CUDA_ARCHITECTURES="80;89;90" \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    CUDAToolkit_ROOT=/usr/local/cuda

# MLX's CUDA backend compiles some execution plans with NVRTC on first use.
# Keep the pinned toolkit headers available to those runtime compilations.
COPY --from=checkpoint-probe-builder /usr/local/cuda/include /usr/local/cuda/include
COPY --from=checkpoint-probe-builder /usr/local/include /usr/local/include
COPY --from=checkpoint-probe-builder /usr/local/bin/checkpoint_probe /usr/local/bin/checkpoint_probe

COPY validation /opt/safemlx/validation
RUN set -eux; \
    mv /tmp/python-packages.lock.txt /opt/safemlx/python-packages.lock.txt; \
    test -f "${CUDA_HOME}/include/cuda_runtime.h"; \
    test -f /usr/local/include/cute/numeric/numeric_types.hpp; \
    test -f /usr/local/include/cutlass/numeric_conversion.h; \
    test -d /usr/local/include/cccl/cuda; \
    find /opt/conda/lib/python3.11/site-packages/nvidia \
        -type d -name lib -print \
        | LC_ALL=C sort > /etc/ld.so.conf.d/pytorch-nvidia.conf; \
    test -s /etc/ld.so.conf.d/pytorch-nvidia.conf; \
    ldconfig; \
    ldd /usr/local/bin/checkpoint_probe > /tmp/checkpoint-probe.ldd; \
    unexpected_missing="$(awk '/not found/ && $1 != "libcuda.so.1" { print $1 }' /tmp/checkpoint-probe.ldd)"; \
    if [ -n "${unexpected_missing}" ]; then \
        printf 'unexpected missing libraries:\n%s\n' "${unexpected_missing}" >&2; \
        exit 1; \
    fi; \
    rm /tmp/checkpoint-probe.ldd; \
    python -m unittest discover \
        -s /opt/safemlx/validation \
        -p 'test_*.py' \
        -q; \
    python -c 'import torch, transformers; assert torch.__version__.startswith("2.8.0"); assert torch.version.cuda == "12.9"; assert transformers.__version__ == "5.14.1"'

WORKDIR /workspace
CMD ["/bin/bash"]
