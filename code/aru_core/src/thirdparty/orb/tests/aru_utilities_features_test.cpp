#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/features/extractors/viso_extractor.h"
#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

using namespace std;
BOOST_AUTO_TEST_CASE(FeatureExtraction) {

  Matcher match;

  // loop through all frames i=0:372
  for (int32_t i = 1; i < 290; i++) {

    // input file names
    char base_name_left[256];
    sprintf(base_name_left, "%d_left.png", i);
    char base_name_right[256];
    sprintf(base_name_right, "%d_right.png", i);
    // string left_img_file_name  = dir + "/I1_" + base_name;
    // string right_img_file_name = dir + "/I2_" + base_name;
    string folder1 = "/home/paulamayo/data/husky_data/images/";
    string folder2 = "/home/paulamayo/data/kitti/2011_09_26"
                     "/2011_09_26_drive_0039_sync/image_01/data/";
    string left_img_file_name = folder1 + base_name_left;
    string right_img_file_name = folder1 + base_name_right;
    std::cout << left_img_file_name << endl;

    cv::Mat img_left = cv::imread(left_img_file_name);
    cv::Mat img_right = cv::imread(right_img_file_name);

    cv::Mat image_1_left_grey, image_1_right_grey;
    cv::cvtColor(img_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_right, image_1_right_grey, cv::COLOR_BGR2GRAY);

    // image dimensions
    int32_t width = image_1_left_grey.rows;
    int32_t height = image_1_right_grey.cols;

    // status
    cout << "Processing: Frame: " << i;

    // compute visual odometry
    int32_t dims[] = {width, height, width};
    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();

    match.pushBack(image_1_left_grey.data, image_1_right_grey.data, dims,
                   false);
    match.matchFeatures(2);

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    cout << "Computing features takes " << elapsed.count() << " seconds"
         << endl;
    cout << "Adding features takes " << 1 / elapsed.count() << " Hz" << endl;
  }
}