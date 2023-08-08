import time

import pykinect2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectRuntime import *
from PIL import Image

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation

import ctypes
import _ctypes
import sys
import numpy as np
import cv2

import utils
from Body import Body

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
body_estimation = Body('./body_pose_model.pth')

temp = [0, 0]
packed = [[0, 0, 0] for _ in range(18)]


fig = plt.figure()
# 开始循环
while True:


    # # convert it into colorful depth image
    # depth_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

    # # show depth image
    # cv2.imshow('Kinect Depth Map', depth_color_image)

    color2depth = utils.color_2_depth_space(kinect, _ColorSpacePoint, kinect._depth_frame_data, show=False, return_aligned_image=True)
    color2depth = cv2.cvtColor(color2depth, cv2.COLOR_BGRA2BGR)

    color2depth = cv2.resize(color2depth, (512, 424))

    candidate, subset = body_estimation(color2depth)


    depth_frame = kinect.get_last_depth_frame()

    # convert to depth image
    depth_image = np.reshape(depth_frame, (424, 512)).astype(np.uint8)
    # depth_image = cv2.resize(depth_image, (128, 106))

    # draw the body

    canvas1 = utils.draw_bodypose(color2depth, candidate, subset)
    cv2.imshow("temp", canvas1)
    plt.ion()
    if candidate.any():
        temp = utils.pack_data(candidate, subset, depth_image, int(time.time()), packed)
        print(temp)

        ax = plt.axes(projection='3d')
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 424)
        ax.set_zlim(0, 300)
        for coordinate in temp:
            if coordinate[1] != 0:
                print(coordinate[1])
                ax.scatter(coordinate[1][0], coordinate[1][1], coordinate[1][2])


        # draw the line
        for i in range(17):
            for n in range(len(subset)):

                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                new = np.array(limbSeq[i]) - 1
                first = int(new[0])
                second = int(new[1])
                ax.plot([packed[first][1][0], packed[second][1][0]],
                        [packed[first][1][1], packed[second][1][1]],
                        [packed[first][1][2], packed[second][1][2]],
                        c='r')


        plt.ioff()

        plt.draw()


        plt.pause(5)
        # scatter.remove([(100,100,100)])

        plt.delaxes(ax)


















    # read RGB image for skeleton tracking
    # color_frame = kinect.get_last_color_frame()
    # color_image = np.reshape(color_frame, (1080, 1920, 4)).astype(np.uint8)
    # new_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    # new_image = cv2.resize(new_image, (960, 540))
    # cv2.imshow("RGB image", new_image)





    # temp = []
    # if candidate.any():
    #     for i in range(len(candidate)):
    #         x, y = candidate[i][0:2]
    #         temp.append([depth_image[(int(x), int(y))]])
    #     arr = np.append(candidate, temp, 1)
    #     print(arr)

    # 等待退出
    if cv2.waitKey(1) == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()





