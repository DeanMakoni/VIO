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
import aru_py_mapping
import yaml


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    logger_rgb = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/keyframe_zoo_2_left.monolithic"
                                               , False)
    logger_depth = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/mesh/keyframe_zoo_2_mesh.monolithic"
                                                 , False)

    with open("/home/paulamayo/code/aru_calib/aru-calibration/ZED/left.yaml") as file:
        left_config = yaml.safe_load(file)
    dist_left = np.array(left_config['distortion_coefficients']['data'])
    mtx_left = np.array(left_config['camera_matrix']['data']).reshape(3, 3)
    P_left = np.array(left_config['projection_matrix']['data']).reshape(3, 4)
    shape_left = (left_config['image_width'], left_config['image_height'])

    mapping = aru_py_mapping.PyMapping(
        "/home/paulamayo/data/husky_data/mapping/viso_mapping_zed.yaml")

    init_position = np.array([[1, 0., 0, 0, ],
                              [0., 1, 0, 0., ],
                              [0., 0, 1, 0., ],
                              [0., 0., 0, 1.]])

    image_rgb, time = logger_rgb.read_from_file()
    image_disp, time = logger_depth.read_channel_from_file()
    image_disp = image_disp.astype(np.float32)

    B = 0.12
    f = mtx_left[0, 0]

    image_depth = B * f / image_disp

    max_depth = 40
    image_depth[image_depth > 40] = 40

    depth_color = cv2.applyColorMap((image_depth * 255 / max_depth).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow('Full_disp', depth_color)
    cv2.waitKey(0)

    mapping.fuse_depth(image_depth, image_rgb, init_position)
