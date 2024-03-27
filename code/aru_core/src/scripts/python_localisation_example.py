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
import aru_py_localisation


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    logger_A = aru_py_logger.MonoImageLogger(
        "/home/paulamayo/data/husky_data/log/keyframe_zoo_2_left.monolithic"
        , False)

    logger_B = aru_py_logger.MonoImageLogger(
        "/home/paulamayo/data/husky_data/log/keyframe_zoo_2_right.monolithic"
        , False)
    logger_C = aru_py_logger.MonoImageLogger(
        "/home/paulamayo/data/husky_data/log/zoo_loop_C_left.monolithic"
        , False)

    localisation = aru_py_localisation.PyLocalisation(
        "/home/paulamayo/data/husky_data/localisation/zoo_vocab.yml",
        "/home/paulamayo/data/husky_data/localisation/zoo_chow_li_tree.yml",
        "/home/paulamayo/data/husky_data/localisation/settings.yml")

    train = False
    save_tree = True

    if train:
        count = 0
        for i in range(200):
            image_sample, time_left = logger_A.read_from_file()
            if (count % 10 == 0):
                localisation.add_vocab_training_image(image_sample)
                cv2.imshow("Local Image", image_sample)
                cv2.waitKey(15)
            count = count + 1
        # Train vocabulary
        localisation.train_save_vocabulary()
    else:
        images = []
        #localisation.initialise_localisation()
        #localisation.add_sample_data("/home/paulamayo/data/husky_data/localisation/sample_descriptors.yml")

        # for i in range(200):
        #     image_query, time_left = logger_A.read_from_file()
        #     images.append(image_query)
        #     #if save_tree:
        #     localisation.add_query_image(image_query)
        #     cv2.imshow("Query Image", image_query)
        #     cv2.waitKey(15)
        if save_tree:
            for i in range(20):
                image_sample, time_left = logger_A.read_from_file()
                localisation.add_sample_image(image_sample)
                cv2.imshow("Image Sample", image_sample)
                cv2.waitKey(15)
            localisation.save_tree()
            #localisation.save_sample_descriptors("/home/paulamayo/data/husky_data/localisation/sample_descriptors.yml")
            #localisation.save_query_descriptors("/home/paulamayo/data/husky_data/localisation/query_descriptors.yml")
        else:

            # Start localising thread
            for i in range(400):
                image_2_left, time_left = logger_B.read_from_file()
                local_id, prob = localisation.localise_against(image_2_left)
                print("Image ", i, " is localised to ", local_id, " with prob ", prob)
                image_prev = images[local_id]
                localised = cv2.hconcat([image_2_left, image_prev])
                localised = cv2.resize(localised, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Localised Image", localised)
                cv2.waitKey(15)

