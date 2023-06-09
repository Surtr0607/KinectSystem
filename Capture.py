import pykinect2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectRuntime import *

import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import cv2
from Body import Body
import utils as util
import matplotlib.pyplot as plt

kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
body_estimation = Body('./body_pose_model.pth')


# 开始循环
while True:
    # fetch depth frame

    depth_frame = kinect.get_last_depth_frame()

    # convert to depth image
    depth_image = np.reshape(depth_frame, (424, 512)).astype(np.uint8)
    depth_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

    # show depth image
    cv2.imshow('Kinect Depth Map', depth_color_image)


    # read RGB image for skeleton tracking
    color_frame = kinect.get_last_color_frame()
    color_image = np.reshape(color_frame, (1080, 1920, 4)).astype(np.uint8)
    new_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    new_image = cv2.resize(new_image, (480,270))
    cv2.imshow("RGB image", new_image)


    candidate, subset = body_estimation(new_image)
    canvas = util.draw_bodypose(new_image, candidate, subset)
    cv2.imshow("skeleton tracking", canvas)


    # 等待退出
    if cv2.waitKey(1) == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()





