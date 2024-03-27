# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import bagpy
import numpy as np
import pykitti
import cv2
import rosbag
import sys
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

sys.path.insert(0, '/home/paulamayo/code/aru-core/build/lib/')
import aru_py_logger

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger = aru_py_logger.LaserLogger(
        "/home/paulamayo/data/husky_data/log/zoo_2_laser.monolithic",True)


    bag = rosbag.Bag("/home/paulamayo/data/husky_data/bag/2022-05-16-16-35-31.bag", "r")
    for topic, msg, t in bag.read_messages(topics=["/velodyne_points"]):
        cloud_points = np.array(
            list(point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))).astype(np.float64)
        time_out = t.secs * 1000 + t.nsecs // 1000000
        logger.write_to_file(cloud_points, time_out)
