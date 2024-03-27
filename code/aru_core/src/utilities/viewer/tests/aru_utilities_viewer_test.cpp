#define BOOST_TEST_MODULE My Test

#include "../include/aru/core/utilities/viewer/viewer.h"

#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>



BOOST_AUTO_TEST_CASE(Pangolin){
  aru::core::utilities::viewer::Viewer viewer;

  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 100; ++i) {
    points.emplace_back(i,i,i);
  }
  viewer.ViewPointCloud(points);
}


