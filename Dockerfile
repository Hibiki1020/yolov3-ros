# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.03-py3
#Ubuntu 20.04

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

#Timezone
ENV DEBIAN_FRONTEND=noninteractive
########## basis ##########
RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
    tzdata \
	build-essential \
    python3-pip \
    python3-empy

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook

########## ROS Noetic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && apt-get install -y lsb-release &&\
	sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
	apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
	apt-get update && apt-get install -y ros-noetic-desktop-full

#RUN apt-get install -y ros-noetic-catkin && \
#    apt-get install -y python-catkin-tools 

########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc

## cmk
RUN echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n cd /home/ros_catkin_ws \n catkin_make \$@ \n cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc

ARG CACHEBUST=1

# Create working directory
WORKDIR /home

# Copy contents
COPY . /home/ros_catkin_ws/src

RUN     cd /home/ros_catkin_ws && \
		/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3" && \
        cd /home

# Set environment variables
ENV HOME=/home

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    source ~/.bashrc

RUN source /opt/ros/noetic/setup.bash && \
    source /home/ros_catkin_ws/devel/setup.bash

RUN mkdir -p /home/pretrained_models && \
	cd /home/pretrained_models && \
	wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt && \
	wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-spp.pt && \
	wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-tiny.pt

RUN cd /home

# https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt
# https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-spp.pt
# https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-tiny.pt

# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov3:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov3:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov3:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/train/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
