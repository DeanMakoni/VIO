#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/image/feature_matcher.h"
#include "aru/core/utilities/image/imageprotocolbufferadaptor.h"
#include "aru/core/utilities/logging/log.h"
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
//BOOST_AUTO_TEST_CASE(FeatureExtraction) {
//
//  Matcher match;
//
//  // loop through all frames i=0:372
//  for (int32_t i = 1; i < 2; i++) {
//
//    // input file names
//    char base_name_left[256];
//    sprintf(base_name_left, "%d_left.png", i);
//    char base_name_right[256];
//    sprintf(base_name_right, "%d_right.png", i);
//    // string left_img_file_name  = dir + "/I1_" + base_name;
//    // string right_img_file_name = dir + "/I2_" + base_name;
//    string folder1 = "/home/paulamayo/data/husky_data/images/";
//    string folder2 = "/home/paulamayo/data/kitti/2011_09_26"
//                     "/2011_09_26_drive_0039_sync/image_01/data/";
//    string left_img_file_name = folder1 + base_name_left;
//    string right_img_file_name = folder1 + base_name_right;
//    std::cout << left_img_file_name << endl;
//
//    cv::Mat img_left = cv::imread(left_img_file_name);
//    cv::Mat img_right = cv::imread(right_img_file_name);
//
//    cv::Mat image_1_left_grey, image_1_right_grey;
//    cv::cvtColor(img_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
//    cv::cvtColor(img_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
//
//    // image dimensions
//    int32_t width = image_1_left_grey.rows;
//    int32_t height = image_1_right_grey.cols;
//
//    // status
//    cout << "Processing: Frame: " << i;
//
//    // compute visual odometry
//    int32_t dims[] = {width, height, width};
//    // Perform the estimation
//    auto estimation_start = std::chrono::high_resolution_clock::now();
//
//    match.pushBack(image_1_left_grey.data, image_1_right_grey.data, dims,
//                   false);
//    match.matchFeatures(2);
//
//    auto estimation_end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
//    cout << "Computing features takes " << elapsed.count() << " seconds"
//         << endl;
//    cout << "Adding features takes " << 1 / elapsed.count() << " Hz" << endl;
//  }
//}

struct MatcherFixture {

  cv::Mat image_1_left;
  cv::Mat image_2_left;

  cv::Mat image_1_disparity;

  cv::Mat image_1_right;
  cv::Mat image_2_right;

  aru::core::utilities::image::MatcherParams matcher_params;
  aru::core::utilities::image::ExtractorParams extractor_params;

  boost::shared_ptr<aru::core::utilities::image::OrbFeatureMatcher> matcher_;

  //  boost::shared_ptr<aru::core::utilities::image::CurvatureMatcher>
  //      curvature_matcher_;

  MatcherFixture() {

    image_1_left =
        cv::imread("/home/paulamayo/data/kitti/training/image_2/000006_10.png");
    image_2_left =
        cv::imread("/home/paulamayo/data/kitti/training/image_2/000006_11.png");

    image_1_right =
        cv::imread("/home/paulamayo/data/kitti/training/image_3/000006_10.png");
    image_2_right =
        cv::imread("/home/paulamayo/data/kitti/training/image_3/000006_11.png");

    image_1_disparity = cv::imread(
        "/home/paulamayo/data/kitti/training/disp_noc_0/000006_10.png");

    image_1_disparity.convertTo(image_1_disparity, CV_8UC1);
    cv::cvtColor(image_1_disparity, image_1_disparity, cv::COLOR_BGR2GRAY);

    matcher_params.stereo_baseline = 0.5372;
    matcher_params.match_threshold_high = 100;
    matcher_params.match_threshold_low = 50;
    matcher_params.focal_length = 718;

    extractor_params.num_levels = 8;
    extractor_params.num_features = 2000;
    extractor_params.minimum_fast_threshold = 7;
    extractor_params.initial_fast_threshold = 20;
    extractor_params.scale_factor = 1.2;
    extractor_params.patch_size = 31;
    extractor_params.edge_threshold = 19;

    matcher_ =
        boost::make_shared<aru::core::utilities::image ::OrbFeatureMatcher>(
            matcher_params, extractor_params,
            "/home/paulamayo/data/husky_data/vocabulary/ORBvoc.txt");

    //    curvature_matcher_ =
    //        boost::make_shared<aru::core::utilities::image
    //        ::CurvatureMatcher>(
    //            matcher_params, extractor_params);
  }
};

// BOOST_FIXTURE_TEST_CASE(DisparityCheck, MatcherFixture) {
//
//   cv::Mat image_1_left_grey;
//   cv::Mat image_1_right_grey;
//
//   cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
//   aru::core::utilities::image::FeatureSPtrVectorSptr features =
//       matcher_->ComputeStereoMatches(image_1_left_grey, image_1_right_grey);
//
//   int num_off = 0;
//   for (const auto &feature : *features) {
//     cv::KeyPoint key_point = feature->GetKeyPoint();
//     int disparity_gnd =
//         (int)image_1_disparity.at<char>(key_point.pt.y, key_point.pt.x);
//     int disparity_match =
//         (int)(feature->GetKeyPoint().pt.x -
//         feature->GetMatchedKeyPoint().pt.x);
//     if (disparity_gnd > 0) {
//       int disp_diff = abs(disparity_gnd - disparity_match);
//       LOG(INFO) << "Disparity error is " << disp_diff << " from [ "
//                 << disparity_match << "," << disparity_gnd << "]";
//       if (disp_diff > 3) {
//         num_off++;
//       }
//     }
//   }
//   LOG(INFO) << "Number of wrong disparities is " << num_off
//             << " out of "
//                ""
//             << features->size();
//   float percentage_error = (float)num_off / (float)features->size();
//   BOOST_CHECK_LE(percentage_error, 0.05);
// }

BOOST_FIXTURE_TEST_CASE(OrbTests, MatcherFixture) {

  cv::Mat image_1_left_grey;
  cv::Mat image_1_right_grey;

  cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
  aru::core::utilities::image::FeatureSPtrVectorSptr features_1 =
      matcher_->ComputeStereoMatches(image_1_left, image_1_right);

  aru::core::utilities::image::FeatureSPtrVectorSptr features =
      matcher_->ComputeMatches(image_1_left, image_1_right, image_2_left,
                               image_2_right);
}