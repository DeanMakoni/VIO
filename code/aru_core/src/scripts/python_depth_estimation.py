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
import aru_py_mesh


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    sparse_depth = cv2.imread('/home/paulamayo/data/kitti/training/disp_noc_0/000001_10.png')
    img_left = cv2.imread('/home/paulamayo/data/kitti/training/image_2/000001_10.png')
    img_right = cv2.imread('/home/paulamayo/data/kitti/training/image_3/000001_10.png')
    depth_est = aru_py_mesh.PyDepth("/home/paulamayo/data/kitti/kitti_mesh_depth.yaml")
    sparse_depth = cv2.cvtColor(sparse_depth, cv2.COLOR_BGR2GRAY)
    sparse_depth = np.single(sparse_depth)

    # Make sure sparse_depth is float
    #dense_depth = depth_est.create_dense_depth(sparse_depth)
    sparse_depth_mesh=depth_est.create_sparse_depth(img_left,img_right)

    cv2.imshow("Depth", sparse_depth_mesh)
    cv2.waitKey(0)
