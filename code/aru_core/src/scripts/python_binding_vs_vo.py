# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import google.protobuf
from bagpy import bagreader
import bagpy
import numpy as np

import pandas as pd
import sys

import rosbag
from sensor_msgs.msg import Image

sys.path.insert(0, '/home/paulamayo/code/aru-core/build/lib/')
import cv2
import aru_py_logger
import aru_py_vo
import yaml
from numpy.linalg import inv
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    vo = aru_py_vo.PyVO("/home/paulamayo/data/husky_data/vo/vo_config_zed.yaml")
    logger_vo = aru_py_logger.TransformLogger("/home/paulamayo/data/husky_data/vo/white_lab_vo.monolithic"
                                              , False)

    logger_left = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/white_lab_left.monolithic"
                                                , False)
    logger_right = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/white_lab_right.monolithic"
                                                 , False)
    bag = rosbag.Bag("/home/paulamayo/data/husky_data/bag/2022-07-04-10-25-03.bag", "r")

    with open("/home/paulamayo/code/aru_calib/aru-calibration/ZED/left.yaml") as file:
        left_config = yaml.safe_load(file)
    dist_left = np.array(left_config['distortion_coefficients']['data'])
    mtx_left = np.array(left_config['camera_matrix']['data']).reshape(3, 3)
    P_left = np.array(left_config['projection_matrix']['data']).reshape(3, 4)
    shape_left = (left_config['image_width'], left_config['image_height'])

    with open("/home/paulamayo/code/aru_calib/aru-calibration/ZED/right.yaml") as file:
        right_config = yaml.safe_load(file)
    dist_right = np.array(right_config['distortion_coefficients']['data'])
    mtx_right = np.array(right_config['camera_matrix']['data']).reshape(3, 3)
    P_right = np.array(right_config['projection_matrix']['data']).reshape(3, 4)
    shape_right = (right_config['image_width'], right_config['image_height'])

    count = 0

    x_vo = [0]
    y_vo = [0]

    x_py_vo = [0]
    y_py_vo = [0]

    prev_left = np.zeros((1280, 720, 3), np.uint8)
    prev_right = np.zeros((1280, 720, 3), np.uint8)

    init_position = np.array([[1, 0., 0, 0, ],
                              [0., 1, 0, 0., ],
                              [0., 0, 1, 0., ],
                              [0., 0., 0, 1.]])

    init_py_position = np.array([[1, 0., 0, 0, ],
                                 [0., 1, 0, 0., ],
                                 [0., 0, 1, 0., ],
                                 [0., 0., 0, 1.]])

    # for topic, msg, t in bag.read_messages(topics=["/camera/image_stereo/image_raw"]):
    #     img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    #     h, w, channels = img.shape
    #     if img.shape[2] == 3:
    #         # Read left image
    #         left = img[:, :w // 2]
    #         mapx, mapy = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, P_left, shape_left, 5)
    #         image_left = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
    #         # Read right image
    #         right = img[:, w // 2:]
    #         mapx, mapy = cv2.initUndistortRectifyMap(mtx_right, dist_right, None, P_right, shape_right, 5)
    #         image_right = cv2.remap(right, mapx, mapy, cv2.INTER_LINEAR)
    #         if count > 1:
    #             # Convert to grey scale
    #             img_1_left_gray = cv2.cvtColor(prev_left, cv2.COLOR_BGR2GRAY)
    #             img_1_right_gray = cv2.cvtColor(prev_right, cv2.COLOR_BGR2GRAY)
    #             img_2_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    #             img_2_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    #
    #             transform = vo.stereo_odometry(img_1_left_gray, img_1_right_gray, 0, img_2_left_gray, img_2_right_gray,
    #                                            1)
    #             init_py_position = np.matmul(init_py_position, transform)
    #             x_py_vo.append(init_py_position[2, 3])
    #             y_py_vo.append(init_py_position[0, 3])
    #
    #         count = count + 1
    #         prev_left = image_left
    #         prev_right = image_right

    count = 0
    prev_left = np.zeros((1280, 720, 3), np.uint8)
    prev_right = np.zeros((1280, 720, 3), np.uint8)
    while not logger_left.end_of_file():
        image_left, time_left = logger_left.read_from_file()
        image_right, time_right = logger_right.read_from_file()

        if count > 1:
            # Convert to grey scale
            img_1_left_gray = cv2.cvtColor(prev_left, cv2.COLOR_BGR2GRAY)
            img_1_right_gray = cv2.cvtColor(prev_right, cv2.COLOR_BGR2GRAY)
            img_2_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
            img_2_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

            transform = vo.stereo_odometry(prev_left, prev_right, 0, image_left, image_right,
                                           1)
            init_position = np.matmul(init_position, transform)
            x_vo.append(init_position[2, 3])
            y_vo.append(init_position[0, 3])

        count = count + 1
        prev_left = image_left
        prev_right = image_right


    fig, ax = plt.subplots()
    ax.plot(x_vo, y_vo, linestyle='-', marker='.')
    ax.plot(x_py_vo, y_py_vo, linestyle='-', marker='.')
    plt.show()
