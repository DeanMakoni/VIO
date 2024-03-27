#ifndef ARU_CORE_DEPTH_BINDINGS_H_
#define ARU_CORE_DEPTH_BINDINGS_H_

//#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
//#include <include/aru/core/utilities/logging/log.h>
#include <aru/core/bindings/conversions.h>
#include <aru/core/mesh/mesh.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pbStereoImage.pb.h>
#include <utility>

#include <opencv2/opencv.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace aru {
namespace core {
namespace bindings {

class PyMesh {
public:
  explicit PyMesh(const std::string &filename);

  virtual ~PyMesh() = default;

  void EstimateDepth(pybind11::array_t<unsigned char> &image_left,
                     pybind11::array_t<unsigned char> &image_right);

  pybind11::array_t<float>
  CreateSparseDepth(pybind11::array_t<unsigned char> &image_left,
                    pybind11::array_t<unsigned char> &image_right);

  std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<float>>
  CreateDenseDepth(pybind11::array_t<float> sparse_depth);

  pybind11::array_t<unsigned char>
  DrawWireFrame(pybind11::array_t<unsigned char> &image_left,
                pybind11::array_t<float> &sparse_features);

private:
  boost::shared_ptr<aru::core::mesh::Mesh> mesh_estimator_;
};
//------------------------------------------------------------------------------
PyMesh::PyMesh(const std::string &filename) {

  mesh_estimator_ = boost::make_shared<aru::core::mesh::Mesh>(filename);
}
//------------------------------------------------------------------------------
void PyMesh::EstimateDepth(pybind11::array_t<unsigned char> &image_left,
                           pybind11::array_t<unsigned char> &image_right) {

  mesh_estimator_->EstimateMesh(numpy_uint8_3c_to_cv_mat(image_left),
                                      numpy_uint8_3c_to_cv_mat(image_right));
}
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<float>>
PyMesh::CreateDenseDepth(pybind11::array_t<float> sparse_depth) {
  std::pair<cv::Mat, cv::Mat> dense_depth =
      mesh_estimator_->CreateDenseDepthMap(
          numpy_float_1c_to_cv_mat(sparse_depth));

  return std::make_tuple(cv_mat_uint8_3c_to_numpy(dense_depth.first),
                         cv_mat_float_1c_to_numpy(dense_depth.second));
}
//------------------------------------------------------------------------------
pybind11::array_t<unsigned char>
PyMesh::DrawWireFrame(pybind11::array_t<unsigned char> &image_left,
                      pybind11::array_t<float> &sparse_features) {
  cv::Mat wireframe =
      mesh_estimator_->DrawWireframe(numpy_uint8_3c_to_cv_mat(image_left),
                                     numpy_float_1c_to_cv_mat(sparse_features));

  return cv_mat_uint8_3c_to_numpy(wireframe);
}
//------------------------------------------------------------------------------
pybind11::array_t<float>
PyMesh::CreateSparseDepth(pybind11::array_t<unsigned char> &image_left,
                          pybind11::array_t<unsigned char> &image_right) {
  mesh_estimator_->EstimateMesh(numpy_uint8_3c_to_cv_mat(image_left),
                                      numpy_uint8_3c_to_cv_mat(image_right));

  cv::Mat dense_depth = mesh_estimator_->GetInterpolatedDepth();

  cv::Mat sparse_depth =
      cv::Mat(dense_depth.rows, dense_depth.cols, CV_32FC1, cv::Scalar(0));

  std::vector<double> vertice_depths = mesh_estimator_->GetVerticeDepths();
  auto vertice_keypoints = mesh_estimator_->GetVerticeKeypoints();

  LOG(INFO) << "Number of keypoints is " << vertice_keypoints.size();
  int index = 0;
  for (auto keypoint : vertice_keypoints) {
    sparse_depth.at<float>(keypoint.pt.y, keypoint.pt.x) =
        vertice_depths[index];
    index++;
  }
  return cv_mat_float_1c_to_numpy(sparse_depth);
}
} // namespace bindings
} // namespace core
} // namespace aru

#endif // ARU_CORE_DEPTH_BINDINGS_H_
