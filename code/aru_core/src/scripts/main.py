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
import yaml


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    bag = rosbag.Bag("/home/paulamayo/data/husky_data/bag/2022-08-30-05-35-46.bag", "r")
    logger_left = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/nerf_1_left.monolithic"
                                                , True)
    logger_right = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/nerf_1_right.monolithic"
                                                 , True)

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

    dist_faux = np.array([0, 0, -0.0, 0.0, 0.00000])

    count=1
    for topic, msg, t in bag.read_messages(topics=["/camera/image_stereo/image_raw"]):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        h, w, channels = img.shape
        if img.shape[2] == 3:
            # Read left image
            left = img[:, :w // 2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_faux, shape_left, 1, shape_left)
            left_dis=cv2.undistort(left, mtx_left, dist_faux, None, newcameramtx)
            mapx, mapy = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, P_left, shape_left, 5)
            image_left = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
            # Read right image
            right = img[:, w // 2:]
            mapx, mapy = cv2.initUndistortRectifyMap(mtx_right, dist_right, None, P_right, shape_right, 5)
            image_right = cv2.remap(right, mapx, mapy, cv2.INTER_LINEAR)
            cv2.imshow("Right Image", left_dis)
            cv2.waitKey(5)
            time_out = t.secs * 1000 + t.nsecs // 1000000
            logger_left.write_to_file(image_left, time_out)
            logger_right.write_to_file(image_right, time_out)

    bag.close()
