#!/bin/bash
# Get the absolute path of the directory containing this script
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -it --rm \
    --gpus=all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOST_HOME=$HOME \
    -e HOST_PWD=$(pwd) \
    -v $(pwd):/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    summary_from_human_feedback