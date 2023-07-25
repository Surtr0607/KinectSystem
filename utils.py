import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pykinect2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectRuntime import *

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


def color_point_2_depth_point(kinect, depth_space_point, depth_frame_data, color_point):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param color_point: color_point pixel location as [x, y]
    :return: depth point of color point
    """
    # Import here to optimize
    import numpy as np
    import ctypes
    # Map Color to Depth Space
    # Make sure that the kinect was able to obtain at least one color and depth frame, else the dept_x and depth_y values will go to infinity
    color2depth_points_type = depth_space_point * int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512*424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    # Where color_point = [xcolor, ycolor]
    depth_x = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].x
    depth_y = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].y
    return [int(depth_x) if depth_x != float('-inf') and depth_x != float('inf') else 0, int(depth_y) if depth_y != float('-inf') and depth_y != float('inf') else 0]

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # candidate就是个包含所有点的大集合，通过subset确定点的归属于哪个人
    for i in range(18):
        # 画点，根据subset每个人，index=-1说明点位没检测到
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    # 连线
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def pack_data(candidate, subset, depth_image):
    # The format of the packed array is like [index of data block, candidate, subset, timestamp]
    packed = [[0,0,0] for _ in range(18)]
    timestamp = int(time.time())
    # The first of packed array is the index of point sequence
    for i in range(18):
        # 画点，根据subset每个人，index=-1说明点位没检测到
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            packed[i][0] = index
            z = depth_image[int(y)][int(x)]
            packed[i][1] = (x, y, z)
            packed[i][2] = timestamp
    return packed

def draw_bodypose_on_depth(kinect, canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # candidate就是个包含所有点的大集合，通过subset确定点的归属于哪个人
    for i in range(18):
        # 画点，根据subset每个人，index=-1说明点位没检测到
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            # print("original")
            # print([x,y])
            # print("after")
            new_point = color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [100, 100])
            # print(new_point)
            cv2.circle(canvas, (new_point[0], new_point[1]), 4, colors[i], thickness=-1)


    return canvas



# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

def get_depth_at_rgb_point(kinect, rgb_x, rgb_y):
    # Get the depth frame
    depth_frame = kinect.get_last_depth_frame()

    # Get the depth frame dimensions
    depth_width = kinect.depth_frame_desc.Width
    depth_height = kinect.depth_frame_desc.Height

    # Get the color frame dimensions
    color_width = kinect.color_frame_desc.Width
    color_height = kinect.color_frame_desc.Height

    # Check if the RGB coordinates are within the valid range
    if 0 <= rgb_x < color_width and 0 <= rgb_y < color_height:
        # Map RGB coordinates to depth coordinates
        depth_space_point = kinect._mapper.MapColorFrameToDepthSpace(PyKinectV2._DepthSpacePoint(rgb_x, rgb_y))

        # Ensure the depth point is within valid range
        if depth_space_point.x >= 0 and depth_space_point.y >= 0 and depth_space_point.x < depth_width and depth_space_point.y < depth_height:
            # Get the depth value at the mapped depth coordinates
            depth_x = int(depth_space_point.x)
            depth_y = int(depth_space_point.y)
            depth_value = depth_frame[depth_y * depth_width + depth_x]

            return depth_value

        # Return None if the RGB coordinates are out of range or mapping is not possible
    return None

# Map Color Space to Depth Space (Image)
def color_2_depth_space(kinect, color_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """

    :param kinect: kinect class
    :param color_space_point: _ColorSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows aligned image with color and depth
    :return: mapped depth to color frame
    """
    import numpy as np
    import ctypes
    import cv2
    # Map Depth to Color Space
    depth2color_points_type = color_space_point * int(512 * 424)
    depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(color_space_point))
    kinect._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(512 * 424), depth_frame_data, kinect._depth_frame_data_capacity, depth2color_points)
    # depth_x = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].x
    # depth_y = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].y
    colorXYs = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(kinect.depth_frame_desc.Height * kinect.depth_frame_desc.Width,)))  # Convert ctype pointer to array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 2).astype(int)
    colorXs = np.clip(colorXYs[:, :, 0], 0, kinect.color_frame_desc.Width - 1)
    colorYs = np.clip(colorXYs[:, :, 1], 0, kinect.color_frame_desc.Height - 1)
    if show or return_aligned_image:
        color_frame = kinect.get_last_color_frame()
        color_img = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        align_color_img = np.zeros((424, 512, 4), dtype=np.uint8)
        align_color_img[:, :] = color_img[colorYs, colorXs, :]



        if show:
            cv2.imshow('img', align_color_img)
            # cv2.waitKey(3000)
        if return_aligned_image:
            return align_color_img
    return colorXs, colorYs