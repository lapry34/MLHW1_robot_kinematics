#!/bin/bash

NVIDIA_STR=""
#NVIDIA_STR="--runtime nvidia --gpus all"

docker run -it \
    -u $(id -u):$(id -g)  \
    --name mlrobotgymenvs --rm \
    $NVIDIA_STR \
    --privileged \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v `pwd`:/opt/MLHW1_robot_kinematics \
    -w /opt/MLHW1_robot_kinematics \
    mlrobotgymenvs
    

