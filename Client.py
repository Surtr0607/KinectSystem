from socket import *
import os
import sys
import pykinect2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectRuntime import *
import json
import pickle


import ctypes
import _ctypes
import sys
import numpy as np
import cv2
from Body import Body
import utils as util
import matplotlib.pyplot as plt

kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
body_estimation = Body('./body_pose_model.pth')


# This is the client-side of data synchronization


serverHost = '10.14.147.126'
serverPort = 2456


#建立一个tcp/ip套接字对象
sockobj = socket(AF_INET, SOCK_STREAM)
#连接至服务器及端口
sockobj.connect((serverHost, serverPort))
while 1:
    color_frame = kinect.get_last_color_frame()
    color_image = np.reshape(color_frame, (1080, 1920, 4)).astype(np.uint8)
    new_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    new_image = cv2.resize(new_image, (480, 270))
    candidate, subset = body_estimation(new_image)

    if candidate.any():
        print(candidate)
        serialized_data = pickle.dumps(candidate)
        sockobj.send(serialized_data)




#关闭套接字
sockobj.close( )