#!/bin/bash

#buid docker image
docker build -t seg-gpu-train .

# run docker image
docker run -it --rm --runtime=nvidia --gpus all \
  --shm-size=8g \
  -v "$(pwd)":/ \
  seg-gpu-train