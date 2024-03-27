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
from numpy.linalg import inv
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    logger_vo = aru_py_logger.TransformLogger("/home/paulamayo/data/husky_data/vo/nerf_vo.monolithic"
                                              , False)
    logger_opt = aru_py_logger.TransformLogger("/home/paulamayo/data/husky_data/ba/zoo_loop_opt_poses.monolithic"
                                               , False)
    init_position = np.array([[1, 0., 0, 0, ],
                              [0., 1, 0, 0., ],
                              [0., 0, 1, 0., ],
                              [0., 0., 0, 1.]])

    opt_position = np.array([[1, 0., 0, 0, ],
                             [0., 1, 0, 0., ],
                             [0., 0, 1, 0., ],
                             [0., 0., 0, 1.]])
    x_vo = [0]
    y_vo = [0]
    x_opt = [0]
    y_opt = [0]
    i = 0
    while not logger_vo.end_of_file():
        transform, time_init, time_fin = logger_vo.read_from_file()
        init_position = np.matmul(init_position, (transform))
        x_vo.append(init_position[2, 3])
        y_vo.append(init_position[0, 3])
        i = i + 1

    logger_left = aru_py_logger.MonoImageLogger("/home/paulamayo/data/husky_data/log/nerf_1_left.monolithic"
                                                , False)

    image_1_left, time_init = logger_left.read_index_from_file(300)
    cv2.imshow("Right Image", image_1_left)
    cv2.waitKey(0)

    # i = 0
    # while not logger_left.end_of_file():
    #     image_1_left, time_init = logger_left.read_from_file()
    #     filename = "/home/paulamayo/data/husky_data/nerf/images/ " + str(i) + ".png"
    #     cv2.imshow("Right Image", image_1_left)
    #     cv2.waitKey(5)
    #     i = i + 1

    # while not logger_opt.end_of_file():
    #     transform_opt, time_init, time_fin = logger_opt.read_from_file()
    #     opt_position = np.matmul(opt_position, transform_opt)
    #     x_opt.append(opt_position[2, 3])
    #     y_opt.append(opt_position[0, 3])

    fig, ax = plt.subplots()
    ax.plot(x_vo, y_vo, linestyle='-', marker='.')
    # ax.plot(x_opt, y_opt, linestyle='-', marker='.')
    plt.show()
