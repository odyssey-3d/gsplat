FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Download micromamba tarball and extract the binary
RUN wget -qO /tmp/micromamba.tar.bz2 \
     "https://micromamba.snakepit.net/api/micromamba/linux-64/latest" \
 && mkdir -p /tmp/micromamba_extract \
 && tar -xvjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba_extract \
 && mv /tmp/micromamba_extract/bin/micromamba /usr/local/bin/micromamba \
 && chmod +x /usr/local/bin/micromamba \
 && rm -rf /tmp/micromamba.tar.bz2 /tmp/micromamba_extract

COPY environment.yml /tmp/environment.yml

RUN micromamba create -f /tmp/environment.yml -y && \
    micromamba clean --all --yes

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

SHELL ["micromamba", "run", "-n", "gsplat", "/bin/bash", "-c"]

WORKDIR /workspace/gsplat
COPY . /workspace/gsplat

RUN pip install -r examples/requirements.txt
RUN pip install -e .

CMD ["/bin/bash"]
