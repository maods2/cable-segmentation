# docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="2.0.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel



# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libsm6 libxrender-dev libxext6 libgl1-mesa-dev build-essential gdb unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
COPY requirements.txt . 
RUN pip install -r requirements.txt

WORKDIR /workspaces/cable-segmentation

RUN git config --global --add safe.directory /workspaces/cable-segmentation