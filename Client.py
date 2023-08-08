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
import utils
import matplotlib.pyplot as plt

kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
body_estimation = Body('./body_pose_model.pth')


# This is the client-side of data synchronization


class Client:
    def __init__(self):
        self.serverHost = ''
        self.serverPort = 2456
        # Create a tcp/ip socket object
        self.sockobj = socket(AF_INET, SOCK_STREAM)
        self.kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
        self.body_estimation = Body('./body_pose_model.pth')
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    def connect_server(self):
        # Connect to the server and related port
        self.sockobj.connect((self.serverHost, self.serverPort))

    def produce_data_block(self, packed, draw=False, show=False):
        # produce a mapping image between colour space and depth space
        color2depth = utils.color_2_depth_space(self.kinect, _ColorSpacePoint, kinect._depth_frame_data, show=False,
                                                return_aligned_image=True)
        color2depth = cv2.cvtColor(color2depth, cv2.COLOR_BGRA2BGR)

        color2depth = cv2.resize(color2depth, (512, 424))

        # detect the virtual skeleton
        candidate, subset = body_estimation(color2depth)

        # get the depth frame
        depth_frame = kinect.get_last_depth_frame()

        # convert to depth image
        depth_image = np.reshape(depth_frame, (424, 512)).astype(np.uint8)

        if candidate.any():
            packed = utils.pack_data(candidate, subset, depth_image, int(time.time()), packed)

            if draw:
                print(packed)
                canvas1 = utils.draw_bodypose(color2depth, candidate, subset)
                cv2.imshow("temp", canvas1)

            if show:
                plt.ion()
                ax = plt.axes(projection='3d')
                ax.set_xlim(0, 512)
                ax.set_ylim(0, 424)
                ax.set_zlim(0, 300)
                for coordinate in packed:
                    if coordinate[1] != 0:
                        print(coordinate[1])
                        ax.scatter(coordinate[1][0], coordinate[1][1], coordinate[1][2])

                # draw the line
                for i in range(17):
                    for n in range(len(subset)):

                        index = subset[n][np.array(self.limbSeq[i]) - 1]
                        if -1 in index:
                            continue
                        new = np.array(self.limbSeq[i]) - 1
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

        return packed
    def transfer_data_block(self):
        packed = [[0, 0, 0] for _ in range(18)]
        while 1:
            key_points = self.produce_data_block(packed)
            serialized_data = pickle.dumps(key_points)
            self.sockobj.send(serialized_data)
        self.sockobj.close()


if __name__ == "__main__":
    client = Client()
    client.connect_server()
    client.transfer_data_block()