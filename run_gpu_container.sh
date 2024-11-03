#!/bin/bash

#buid docker image
docker build -t seg-gpu-train .

# run docker image
docker run -it --rm --runtime=nvidia --gpus all \
  -v "$(pwd)":/ \
  seg-gpu-train