FROM nvidia/cudagl:11.1-devel-ubuntu20.04

LABEL author="Marcelo Garcia"

# The following lines are needed to avoid errors with CUDA and Docker in WSL2
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install additional dependencies if needed
RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
    cmake \
    ffmpeg \
    git \
    pkg-config \
    python3-pip \
    unzip \
    vim \
    wget \
    zip \
    zlib1g-dev \
    libcurl4 \
    curl \
    libgl1-mesa-dev\
    libgl1-mesa-glx\
    libglew-dev\
    libosmesa6-dev\
    software-properties-common\
    net-tools\
    xpra\
    xserver-xorg-dev\
    libglfw3-dev\
    patchelf\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# # Upgrade pip for Python
RUN pip install --upgrade pip

# # Clone the RoboMimic repository (You might want to specify a particular branch or tag)
RUN git clone https://github.com/ARISE-Initiative/robomimic.git /robomimic

# # Install RoboMimic
RUN cd /robomimic && pip install -e .

# Set the working directory
WORKDIR /project

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies not covered by the base image
RUN pip install -r requirements.txt

# Install PyTorch and torchvision compatible with CUDA 11.1
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

