#!/bin/bash

image_name="yolov3-ros"
tag_name="noetic"
root_path=$(pwd)

# /media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21
# is ssd path in author's environment

xhost +
docker run -it --rm \
	--gpus all \
	--privileged \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --net=host \
    -v /media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21:/home/ssd_dir \
	-v /home/amsl/pycode/yolov3-ros:/home/ros_catkin_ws/src/yolov3-ros \
	$image_name:$tag_name