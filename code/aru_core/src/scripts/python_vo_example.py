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
    logger_left = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/white_lab_left.monolithic"
                                                , False)
    logger_right = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/white_lab_right.monolithic"
                                                 , False)

    stereo_logger = aru_py_logger.StereoImageLogger("/home/paulamayo/data/husky_data/dlow/D0.25_TS_L_stereo.monolithic"
                                                    , False)

    count = 0
    for i in range(130):
        stereo_logger.read_from_file()
    image_1_left, image_1_right, time_left = stereo_logger.read_from_file()

    for i in range(1):
        stereo_logger.read_from_file()

    image_2_left, image_2_right, time_left = stereo_logger.read_from_file()
    # Convert to grey scale
    img_1_left_gray = cv2.cvtColor(image_1_left, cv2.COLOR_BGR2GRAY)
    img_1_right_gray = cv2.cvtColor(image_1_right, cv2.COLOR_BGR2GRAY)
    img_2_left_gray = cv2.cvtColor(image_2_left, cv2.COLOR_BGR2GRAY)
    img_2_right_gray = cv2.cvtColor(image_2_right, cv2.COLOR_BGR2GRAY)

    transform, kleft, kright, krleft2 = vo.stereo_odometry(img_1_left_gray, img_1_right_gray, 0, img_2_left_gray,
                                                           img_2_right_gray,
                                                           1)

    print(transform)
    cv2.imshow("Image 1", image_1_left)
    cv2.imshow("Image 2", image_2_left)
    cv2.waitKey(0)

    print(transform)
