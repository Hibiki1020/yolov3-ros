#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError

import argparse
import time
from pathlib import Path
import os
import sys
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class ROS_Detect:
    def __init__(self):
        print("Start ROS Detect in YOLO-V3")

        self.subscribe_topic_name = rospy.get_param('~subscribe_topic_name')
        self.advertise_topic_name = rospy.get_param('~advertise_topic_name')
        self.view_img = rospy.get_param('~view-img')
        self.save_txt = rospy.get_param('~save-txt')
        self.save_conf = rospy.get_param('~save-conf')
        self.no_save = rospy.get_param('~no-save')
        self.agnostic_nms = rospy.get_param('~agnostic-nms')
        self.argment = rospy.get_param('~argment')
        self.update = rospy.get_param('~update')
        self.project = rospy.get_param('~project')
        self.name = rospy.get_param('~name')
        self.exist_ok = rospy.get_param('~exist-ok')
        self.hide_labels = rospy.get_param('~hide-labels')
        self.hide_conf = rospy.get_param('~hide-conf')
        
        self.weights_saved_top_directory = rospy.get_param('~weights_saved_top_directory')
        self.weight_type = rospy.get_param('~weight_type')
        self.weight_path = os.path.join(self.weights_saved_top_directory, self.weight_type)
        
        self.device = rospy.get_param('~device')

        self.img_size = rospy.get_param('~img-size')
        self.conf_thres = rospy.get_param('~conf-thres')
        self.iou_thres = rospy.get_param('~iou-thres')
        self.max_det = rospy.get_param('~max-det')
        self.line_thickness = rospy.get_param('~line-thickness')

        self.color_img_cv = np.empty(0)
        self.inferenced_image = np.empty(0)
        self.bridge = CvBridge()
        self.sub_color_img = rospy.Subscriber(self.subscribe_topic_name, ImageMsg, self.callbackColorImage, queue_size=1, buff_size=2**24)
        self.pub_inferenced_image = rospy.Publisher(self.advertise_topic_name, ImageMsg, queue_size=1)

        self.detect()

    def detect(self):
        #Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weight_type, self.weight_path, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("msg.encoding = ", msg.encoding)
            print("self.color_img_cv.shape = ", self.color_img_cv.shape)

            self.inferenced_image = self.inference(self.color_img_cv)
            imgMsg = self.bridge.cv2_to_imgmsg(self.inferenced_image)
            self.pub_inferenced_image.publish(imgMsg)

        except CvBridgeError as e:
            print(e)

    def inference(self, inference_image):
        print("Inference Function")

if __name__ == '__main__':
    #Set Up in ROS node
    rospy.init_node('yolov3_ros_infer', anonymous=True)
    ros_detect = ROS_Detect()
    ros_detect.spin()