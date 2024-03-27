#define BOOST_TEST_MODULE VO Test
#include "glog/logging.h"
#include <Eigen/Dense>
#include <aru/core/utilities/image/feature_tracker.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/vo/vo.h>
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <opencv2/core.hpp>
#include <pbStereoImage.pb.h>

using namespace datatype::image;
using namespace aru::core::utilities;

struct MonolithicVOFixture {

  boost::shared_ptr<aru::core::vo::VO> vo_;
  boost::shared_ptr<aru::core::utilities::logging::ProtocolLogger<
      datatype::image ::pbStereoImage>>
      image_logger_;
  std::string image_monolithic;
  aru::core::utilities::image::StereoImage stereo_image_1;
  aru::core::utilities::image::StereoImage stereo_image_2;
  aru::core::utilities::image::StereoImage stereo_image_3;
  aru::core::utilities::image::StereoImage stereo_image_4;

  boost::shared_ptr<aru::core::utilities::image::VisoFeatureTracker>
      viso_extractor_;

  void ReadStereoImages() {

    for (int i = 0; i < 15; i++) {
      image_logger_->ReadFromFile();
    }
    // Read previous image
    pbStereoImage image_prev = image_logger_->ReadFromFile();
    stereo_image_1 =
        image::ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(
            image_prev);

    // Read previous image
    image_prev = image_logger_->ReadFromFile();
    stereo_image_2 =
        image::ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(
            image_prev);

    // Read previous image
    image_prev = image_logger_->ReadFromFile();
    stereo_image_3 =
        image::ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(
            image_prev);

    // Read previous image
    image_prev = image_logger_->ReadFromFile();
    stereo_image_4 =
        image::ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(
            image_prev);
  }

  MonolithicVOFixture() {
    vo_ = boost::make_shared<aru::core::vo::VO>("/home/paulamayo/data/husky_data/vocabulary/ORBvoc.txt",
        "/home/paulamayo/data/husky_data/vo/vo_config_zed.yaml");
    image_monolithic = "/home/paulamayo/data/husky_data/log/outdoor_zoo"
                       ".monolithic";
    image_logger_ = boost::make_shared<logging::ProtocolLogger<pbStereoImage>>(
        image_monolithic, false);
    viso_extractor_ =
        boost::make_shared<aru::core::utilities::image::VisoFeatureTracker>();
  }
};

BOOST_FIXTURE_TEST_CASE(ReadSolve, MonolithicVOFixture) {
  ReadStereoImages();

  Eigen::Affine3f R_w_cv;
  R_w_cv.linear() << 0, 0, 1, 0, 1, 0, 1, 0, 0;
  R_w_cv.translation() = Eigen::VectorXf::Zero(3);

  Eigen::Affine3f curr_position;
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);

  cv::Mat image_1_left_grey, image_1_right_grey;
  cv::cvtColor(stereo_image_1.first.GetImage(), image_1_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(stereo_image_1.second.GetImage(), image_1_right_grey,
               cv::COLOR_BGR2GRAY);
  viso_extractor_->FeaturesUntracked(image_1_left_grey, image_1_right_grey);
  cv::Mat image_2_left_grey, image_2_right_grey;
  cv::cvtColor(stereo_image_2.first.GetImage(), image_2_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(stereo_image_2.second.GetImage(), image_2_right_grey,
               cv::COLOR_BGR2GRAY);
  viso_extractor_->FeaturesUntracked(image_2_left_grey, image_2_right_grey);
  image::FeatureSPtrVectorSptr features = viso_extractor_->GetCurrentFeatures();
  transform::Transform T_12 = vo_->EstimateMotion(features);
  curr_position = curr_position * (T_12.GetTransform());

  LOG(INFO) << "The pose T_12 is \n" << ( T_12.GetTransform()).matrix();
  LOG(INFO) << "The current position is \n" << curr_position.matrix();

  cv::Mat image_3_left_grey, image_3_right_grey;
  cv::cvtColor(stereo_image_3.first.GetImage(), image_3_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(stereo_image_3.second.GetImage(), image_3_right_grey,
               cv::COLOR_BGR2GRAY);
  viso_extractor_->FeaturesUntracked(image_3_left_grey, image_3_right_grey);
  features = viso_extractor_->GetCurrentFeatures();
  transform::Transform T_23 = vo_->EstimateMotion(features);
  curr_position = curr_position * ( T_23.GetTransform());

  LOG(INFO) << "The pose T_23 is \n" << ( T_23.GetTransform()).matrix();
  LOG(INFO) << "The current position is \n" << curr_position.matrix();

  cv::Mat image_4_left_grey, image_4_right_grey;
  cv::cvtColor(stereo_image_4.first.GetImage(), image_4_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(stereo_image_4.second.GetImage(), image_4_right_grey,
               cv::COLOR_BGR2GRAY);
  viso_extractor_->FeaturesUntracked(image_4_left_grey, image_4_right_grey);
  features = viso_extractor_->GetCurrentFeatures();
  transform::Transform T_34 = vo_->EstimateMotion(features);
  curr_position = curr_position * ( T_34.GetTransform());

  LOG(INFO) << "The pose T_34 is \n" << ( T_34.GetTransform()).matrix();
  LOG(INFO) << "The current position is \n" << curr_position.matrix();

  cv::imshow("Image_1", stereo_image_1.first.GetImage());
  cv::imshow("Image_4", stereo_image_4.first.GetImage());
  cv::waitKey(0);
}

// struct StereoVOFixture {
//
//   const double baseline = 0.5372;
//   Eigen::MatrixXd K;
//
//   const int n = 1000;
//   const double noise = 3;
//
//   const double uc = 607.1928;
//   const double vc = 185.2157;
//
//   const double fu = 718.856;
//   const double fv = 718.856;
//
//   cv::Mat image_1_left_vo;
//   cv::Mat image_2_left_vo;
//
//   cv::Mat image_1_right_vo;
//   cv::Mat image_2_right_vo;
//
//   cv::Mat stereo_g_truth;
//
//   Eigen::Affine3d T_gnd;
//
//   Eigen::Affine3d T;
//   Eigen::Affine3d T_vo;
//
//   std::vector<cv::Point3d> landmarks;
//   std::vector<cv::Point2d> uv_vector;
//
//   aru::core::utilities::image::MatcherParams matcher_params;
//   aru::core::utilities::image::extractors::ExtractorParams extractor_params;
//
//   aru::core::vo::SolverParams solver_params;
//
//   boost::shared_ptr<aru::core::vo::VO> visual_odometry_;
//
//   static double rand(double min, double max) {
//     return min + (max - min) * double(std::rand()) / RAND_MAX;
//   }
//
//   static Eigen::Affine3d RandomPose() {
//     const double range = 1;
//
//     double phi = rand(0, range * 3.14159 * 2);
//     double theta = rand(0, range * 3.14159);
//     double psi = rand(0, range * 3.14159 * 2);
//
//     Eigen::Matrix3d R;
//     Eigen::Vector3d t;
//
//     R(0, 0) = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
//     R(0, 1) = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
//     R(0, 2) = sin(psi) * sin(theta);
//
//     R(1, 0) = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
//     R(1, 1) = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
//     R(1, 2) = cos(psi) * sin(theta);
//
//     R(2, 0) = sin(theta) * sin(phi);
//     R(2, 1) = -sin(theta) * cos(phi);
//     R(2, 2) = cos(theta);
//
//     t(0) = 5.4f;
//     t(1) = -2.0f;
//     t(2) = 0.8f;
//
//     Eigen::Affine3d Transform;
//     Transform.linear() = R;
//     Transform.translation() = t;
//
//     return Transform;
//   }
//
//   static cv::Point3d RandomPoint() {
//     double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159),
//            R = rand(0, +2);
//
//     cv::Point3d point;
//     point.x = sin(theta) * sin(phi) * R;
//     point.y = -sin(theta) * cos(phi) * R;
//     point.z = cos(theta) * R;
//
//     return point;
//   }
//
//   cv::Point2d ProjectWithNoise(Eigen::Matrix3d R, Eigen::Vector3d t,
//                                const cv::Point3d &point) const {
//     double Xc =
//         R(0, 0) * point.x + R(0, 1) * point.y + R(0, 2) * point.z + t(0);
//     double Yc =
//         R(1, 0) * point.x + R(1, 1) * point.y + R(1, 2) * point.z + t(1);
//     double Zc =
//         R(2, 0) * point.x + R(2, 1) * point.y + R(2, 2) * point.z + t(2);
//
//     double nu = rand(-noise, +noise);
//     double nv = rand(-noise, +noise);
//
//     cv::Point2d uv;
//     uv.x = uc + fu * Xc / Zc + nu;
//     uv.y = vc + fv * Yc / Zc + nv;
//
//     return uv;
//   }
//
//   StereoVOFixture() {
//
//     K = Eigen::MatrixXd::Zero(3, 3);
//     K << fu, 0, uc, 0, fv, vc, 0, 0, 1;
//
//     matcher_params.stereo_baseline = 0.5372;
//     matcher_params.match_threshold_high = 100;
//     matcher_params.match_threshold_low = 50;
//     matcher_params.focal_length = 718;
//
//     extractor_params.num_levels = 8;
//     extractor_params.num_features = 2000;
//     extractor_params.minimum_fast_threshold = 7;
//     extractor_params.initial_fast_threshold = 20;
//     extractor_params.scale_factor = 1.2;
//     extractor_params.patch_size = 31;
//     extractor_params.half_patch_size = 15;
//     extractor_params.edge_threshold = 19;
//
//     solver_params.camera_matrix = K;
//     solver_params.ransac_prob = 0.95;
//     solver_params.threshold = 3.0;
//     solver_params.ransac_max_iterations = 1000;
//
//     visual_odometry_ = boost::make_shared<aru::core::vo::VO>(
//         extractor_params, matcher_params, solver_params);
//
//     image_1_left_vo =
//         cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
//                    "/000000.png");
//     image_2_left_vo =
//         cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
//                    "/000001.png");
//
//     image_1_right_vo =
//         cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
//                    "/000000.png");
//     image_2_right_vo =
//         cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
//                    "/000001.png");
//
//     T = RandomPose();
//
//     Eigen::Matrix3d m;
//     m << 9.999978e-01, 5.272628e-04, -2.066935e-03,
//     -5.296506e-04, 9.999992e-01,
//         -1.154865e-03, 2.066324e-03, 1.155958e-03, 9.999971e-01;
//     Eigen::Vector3d t(-4.690294e-02, -2.839928e-02, 8.586941e-01);
//     T_vo.linear() = m;
//     T_vo.translation() = t;
//
//     Eigen::Matrix3d m_gnd;
//     m_gnd << 9.999978e-01, 5.272628e-04, -2.066935e-03, .296506e-04,
//         9.999992e-01, -1.154865e-03, .066324e-03, 1.155958e-03, 9.999971e-01;
//     Eigen::Vector3d t_gnd(-4.690294e-02, -2.839928e-02, 8.586941e-01);
//     T_gnd.linear() = m_gnd;
//     T_gnd.translation() = t_gnd;
//
//     cv::Mat image_1_left_grey, image_1_right_grey, image_2_left_grey,
//         image_2_right_grey;
//     cv::cvtColor(image_1_left_vo, image_1_left_grey, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(image_1_right_vo, image_1_right_grey, cv::COLOR_BGR2GRAY);
//
//     cv::cvtColor(image_2_left_vo, image_2_left_grey, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(image_2_right_vo, image_2_right_grey, cv::COLOR_BGR2GRAY);
//
//     for (int i = 0; i < n; ++i) {
//       cv::Point3d curr_landmark = RandomPoint();
//       landmarks.push_back(curr_landmark);
//       uv_vector.push_back(
//           ProjectWithNoise(T.linear(), T.translation(), curr_landmark));
//     }
//   }
// };
//
// BOOST_FIXTURE_TEST_CASE(VOInit, StereoVOFixture) {
//   aru::core::vo::VOSolver solver(solver_params, extractor_params,
//                                  matcher_params);
// }
//
// BOOST_FIXTURE_TEST_CASE(VOComputation, StereoVOFixture) {
//
//   cv::Mat image_1_left_grey, image_1_right_grey, image_2_left_grey,
//       image_2_right_grey;
//   cv::cvtColor(image_1_left_vo, image_1_left_grey, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(image_1_right_vo, image_1_right_grey, cv::COLOR_BGR2GRAY);
//
//   cv::cvtColor(image_2_left_vo, image_2_left_grey, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(image_2_right_vo, image_2_right_grey, cv::COLOR_BGR2GRAY);
//
//   aru::core::utilities::image::StereoImage image_1_;
//   image_1_.first = aru::core::utilities::image::Image(1,
//   (image_1_left_grey)); image_1_.second =
//   aru::core::utilities::image::Image(1, (image_1_right_grey));
//   aru::core::utilities::image::StereoImage image_2_;
//   image_2_.first = aru::core::utilities::image::Image(2,
//   (image_2_left_grey)); image_2_.second =
//   aru::core::utilities::image::Image(2, (image_2_right_grey));
//
//   aru::core::utilities::transform::Transform transform =
//       visual_odometry_->EstimateMotion(image_1_, image_2_);
//
//   Eigen::MatrixXd transform_actual = T_gnd.matrix();
//   Eigen::MatrixXf transform_calc = transform.GetTransform().matrix();
//   for (int i = 0; i < 4; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       BOOST_CHECK_LE(abs(transform_actual(i, j) - transform_calc(i, j)),
//       0.2);
//     }
//   }
// }
//
// BOOST_FIXTURE_TEST_CASE(VOsolver, StereoVOFixture) {
//
//   // calculate disparity
//   aru::core::vo::VOSolver vo_solver(solver_params, extractor_params,
//                                     matcher_params);
//
//   Eigen::Affine3d calculated_transform =
//       vo_solver.SolveRansac(landmarks, uv_vector);
//   Eigen::MatrixXd transform_actual = T.matrix();
//   Eigen::MatrixXd transform_calc = calculated_transform.matrix();
//   for (int i = 0; i < 4; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       BOOST_CHECK_LE(abs(transform_actual(i, j) - transform_calc(i, j)),
//       0.1);
//     }
//   }
// }
//
// BOOST_AUTO_TEST_CASE(VOconfig) {
//   aru::core::vo::VO vo(
//       "/home/paulamayo/code/aru-core/src/vo/config/vo_config.yaml");
//   cv::Mat image_1_left =
//       cv::imread("/home/paulamayo/data/kitti/training/image_2/000006_10.png");
//   cv::Mat image_2_left =
//       cv::imread("/home/paulamayo/data/kitti/training/image_2/000006_11.png");
//
//   cv::Mat image_1_right =
//       cv::imread("/home/paulamayo/data/kitti/training/image_3/000006_10.png");
//   cv::Mat image_2_right =
//       cv::imread("/home/paulamayo/data/kitti/training/image_3/000006_11.png");
//
//   cv::Mat image_1_left_grey, image_1_right_grey, image_2_left_grey,
//       image_2_right_grey;
//   cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
//
//   cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_BGR2GRAY);
//
//   aru::core::utilities::image::StereoImage image_1_;
//   image_1_.first = aru::core::utilities::image::Image(1,
//   (image_1_left_grey)); image_1_.second =
//   aru::core::utilities::image::Image(1, (image_1_right_grey));
//   aru::core::utilities::image::StereoImage image_2_;
//   image_2_.first = aru::core::utilities::image::Image(2,
//   (image_2_left_grey)); image_2_.second =
//   aru::core::utilities::image::Image(2, (image_2_right_grey));
//
//   aru::core::utilities::transform::Transform transform =
//       vo.EstimateMotion(image_1_, image_2_);
// }
