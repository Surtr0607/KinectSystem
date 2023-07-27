import time

import pykinect2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectRuntime import *
from PIL import Image

import ctypes
import _ctypes
import sys
import numpy as np
import cv2

import utils
from Body import Body
import utils as util
import matplotlib.pyplot as plt

kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
body_estimation = Body('./body_pose_model.pth')

temp = [0, 0]
# 开始循环
while True:


    # # convert it into colorful depth image
    # depth_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

    # # show depth image
    # cv2.imshow('Kinect Depth Map', depth_color_image)

    color2depth = util.color_2_depth_space(kinect, _ColorSpacePoint, kinect._depth_frame_data, show=False, return_aligned_image=True)
    color2depth = cv2.cvtColor(color2depth, cv2.COLOR_BGRA2BGR)

    color2depth = cv2.resize(color2depth, (128, 106))

    candidate, subset = body_estimation(color2depth)


    depth_frame = kinect.get_last_depth_frame()

    # convert to depth image
    depth_image = np.reshape(depth_frame, (424, 512)).astype(np.uint8)
    depth_image = cv2.resize(depth_image, (128, 106))

    # draw the body
    # canvas1 = util.draw_bodypose(color2depth, candidate, subset)

    if candidate.any():
        temp = utils.pack_data(candidate, subset, depth_image, int(time.time()))
        print(temp)
    # cv2.imshow("temp", canvas1)













    # read RGB image for skeleton tracking
    # color_frame = kinect.get_last_color_frame()
    # color_image = np.reshape(color_frame, (1080, 1920, 4)).astype(np.uint8)
    # new_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    # new_image = cv2.resize(new_image, (960, 540))
    # cv2.imshow("RGB image", new_image)



    # candidate, subset = body_estimation(new_image)


    # temp = []
    # if candidate.any():
    #     for i in range(len(candidate)):
    #         x, y = candidate[i][0:2]
    #         temp.append([depth_image[(int(x), int(y))]])
    #     arr = np.append(candidate, temp, 1)
    #     print(arr)







    # canvas1 = util.draw_bodypose(new_image, candidate, subset)
    # cv2.imshow("skeleton tracking", canvas1)
    # canvas2 = util.draw_bodypose_on_depth(kinect, depth_color_image, candidate, subset)
    # cv2.imshow("another", canvas2)


    # 等待退出
    if cv2.waitKey(1) == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()





