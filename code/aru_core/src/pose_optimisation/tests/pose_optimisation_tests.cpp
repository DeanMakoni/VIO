#define BOOST_TEST_MODULE My Test

#include "glog/logging.h"
#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

#include "aru/core/pose_optimisation/pose_optimisation.h"
#include "aru/core/vo/vo.h"
#include <Eigen/Dense>

using namespace aru::core::utilities::image;
using namespace aru::core::utilities::transform;
using namespace aru::core::pose_optimisation;
struct BAFixture {

  const double baseline = 0.5372;
  Eigen::MatrixXd K;

  StereoImage image_1;
  StereoImage image_2;
  StereoImage image_3;
  StereoImage image_4;

  boost::shared_ptr<aru::core::pose_optimisation::PoseOptimisation> po_;
  boost::shared_ptr<aru::core::vo::VO> vo_;

  BAFixture() {

    image_1.first = Image(
        0, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                      "/000000.png"));
    image_2.first = Image(
        1, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                      "/000002.png"));
    image_3.first = Image(
        2, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                      "/000004.png"));
    image_4.first = Image(
        3, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                      "/000006.png"));

    image_1.second = Image(
        0, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                      "/000000.png"));
    image_2.second = Image(
        1, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                      "/000002.png"));
    image_3.second = Image(
        2, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                      "/000004.png"));
    image_4.second = Image(
        3, cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                      "/000006.png"));

    po_ = boost::make_shared<aru::core::pose_optimisation::PoseOptimisation>(
        "/home/paulamayo/data/husky_data/ba/ba_config.yaml");
    vo_ = boost::make_shared<aru::core::vo::VO>(
        "/home/paulamayo/data/husky_data/vo/vo_config_zed.yaml");
  }
};

BOOST_FIXTURE_TEST_CASE(SingleFrame, BAFixture) {

  Eigen::Affine3f curr_position;
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);

  po_->AddPose(boost::make_shared<Transform>(0, 0, curr_position));
  //  ba_->AddPose(
  //      boost::make_shared<Transform>(1,0,curr_position));
  //  ba_->AddPose(
  //      boost::make_shared<Transform>(2,1,curr_position));
  po_->AddPose(
      boost::make_shared<Transform>(vo_->EstimateMotion(image_1, image_2)));
  po_->AddPose(
      boost::make_shared<Transform>(vo_->EstimateMotion(image_2, image_3)));
  po_->AddPose(
      boost::make_shared<Transform>(vo_->EstimateMotion(image_3, image_4)));

  // Add Frames
  po_->InitialFrame(image_1, image_2);
  po_->AddFrame(image_3);
  //  ba_->AddFrame(image_3);
  //  ba_->AddFrame(image_4);

  // Solve
  //po_->Optimise();
  //  LOG(INFO) << "Output to file";
  //  // TODO: correct timestamps and make relative
  //  for (auto pose : ba_poses) {
  //    LOG(INFO)<<"Pose is \n"<<pose.matrix();
  //  }
}
