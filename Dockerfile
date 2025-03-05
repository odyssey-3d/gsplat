FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    ninja-build \
    libx11-6 \
    libgl1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Download and install micromamba
RUN wget -qO /tmp/micromamba.tar.bz2 \
     "https://micromamba.snakepit.net/api/micromamba/linux-64/latest" \
 && mkdir -p /tmp/micromamba_extract \
 && tar -xvjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba_extract \
 && mv /tmp/micromamba_extract/bin/micromamba /usr/local/bin/micromamba \
 && chmod +x /usr/local/bin/micromamba \
 && rm -rf /tmp/micromamba.tar.bz2 /tmp/micromamba_extract

# Copy and create the micromamba environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -f /tmp/environment.yml -y && \
    micromamba clean --all --yes

# Set CUDA architecture list for PyTorch
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Set shell to use micromamba environment
SHELL ["micromamba", "run", "-n", "gsplat", "/bin/bash", "-c"]

# Set working directory and copy project files
WORKDIR /workspace/gsplat
COPY . /workspace/gsplat

# Install Python dependencies
RUN pip install -r examples/requirements.txt
RUN pip install -e .

# Set default command
CMD ["/bin/bash"]