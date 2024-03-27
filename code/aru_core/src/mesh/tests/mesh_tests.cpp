#define BOOST_TEST_MODULE My Test
#include <aru/core/mesh/mesh.h>

#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

struct MeshFixture {

  cv::Mat image_1_left;
  cv::Mat image_1_right;
  cv::Mat stereo_g_truth;
  boost::shared_ptr<aru::core::mesh::Mesh> mesh_;

  MeshFixture() {
    image_1_left =
        cv::imread("/home/paulamayo/data/kitti/training/image_2/000006_10.png");
    image_1_right =
        cv::imread("/home/paulamayo/data/kitti/training/image_3/000006_10.png");
    stereo_g_truth = cv::imread(
        "/home/paulamayo/data/kitti/training/disp_occ_0/000006_10.png");
    mesh_ = boost::make_shared<aru::core::mesh::Mesh>(
        "/home/paulamayo/code/aru-core/src/mesh/config/mesh_depth.yaml");
  }
};

BOOST_AUTO_TEST_CASE(MeshInitialise) {
  boost::shared_ptr<aru::core::mesh::Mesh> mesh_ =
      boost::make_shared<aru::core::mesh::Mesh>(
          "/home/paulamayo/code/aru-core/src/mesh/config/mesh_depth.yaml");
}

BOOST_FIXTURE_TEST_CASE(Mesher, MeshFixture) {
  cv::Mat image_1_left_grey;
  cv::Mat image_1_right_grey;

  cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);

  mesh_->EstimateDepthGnd(image_1_left, image_1_right,stereo_g_truth);
}
