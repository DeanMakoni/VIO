#ifndef ARU_CORE_MAPPING_BINDINGS_H_
#define ARU_CORE_MAPPING_BINDINGS_H_

#include <aru/core/bindings/conversions.h>
#include <aru/core/mapping/mesh_mapping/mesh_map.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include "aru/core/utilities/transforms/transformprotocolbufferadaptor.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace aru {
namespace core {
namespace bindings {

class PyMapping {
public:
  explicit PyMapping(const std::string &settings_filename, const std::string &vocab_filename);

  virtual ~PyMapping() = default;

  void
  FuseColourDepth(pybind11::array_t<float> &image_depth,
                  pybind11::array_t<unsigned char> &image_colour,
                  const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix);

  void AddImageAtTime(pybind11::array_t<float> &image_depth,
                      pybind11::array_t<unsigned char> &image_colour,
                      long time);

  void
  AddTransform(const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix,
               long source_timestamp, long dest_timestamp);

  void SavePly(const std::string &filename);

private:
  boost::shared_ptr<aru::core::mapping::mesh_map::MeshMap> mapper_;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;
};
//------------------------------------------------------------------------------
PyMapping::PyMapping(const std::string &settings_filename,  const std::string &vocab_filename) {
  mapper_ = boost::make_shared<aru::core::mapping::mesh_map::MeshMap>(settings_filename, vocab_filename);

  transform_map_ = boost::make_shared<utilities::transform::TransformMap>();
}
//------------------------------------------------------------------------------
void PyMapping::AddTransform(
    const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix,
    long source_timestamp, long dest_timestamp) {

  Eigen::Affine3f transform_affine;
  transform_affine.matrix() = transform_matrix.cast<float>();

  utilities::transform::TransformSPtr init_transform =
      boost::make_shared<utilities ::transform::Transform>(
          source_timestamp, dest_timestamp, transform_affine);
  transform_map_->AddTransform(init_transform);
}
//------------------------------------------------------------------------------
void PyMapping::FuseColourDepth(
    pybind11::array_t<float> &image_depth,
    pybind11::array_t<unsigned char> &image_colour,
    const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix) {

  Eigen::Affine3f transform_affine;
  transform_affine.matrix() = transform_matrix.cast<float>();

  mapper_->InsertDepthImage(numpy_float_1c_to_cv_mat(image_depth),
                            numpy_uint8_3c_to_cv_mat(image_colour),
                            transform_affine);
  // mapper_->DrawCurrentTsdf();
}
//------------------------------------------------------------------------------
void PyMapping::AddImageAtTime(pybind11::array_t<float> &image_depth,
                               pybind11::array_t<unsigned char> &image_colour,
                               long timestamp) {
  // Get position
  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(timestamp);
  if (curr_position) {
    mapper_->InsertDepthImage(numpy_float_1c_to_cv_mat(image_depth),
                              numpy_uint8_3c_to_cv_mat(image_colour),
                              curr_position->GetTransform());
//    mapper_->DrawCurrentTsdf();
  }
}

//------------------------------------------------------------------------------
void PyMapping::SavePly(const std::string &filename) {
  mapper_->SaveCurrentTsdf(filename);
}

} // namespace bindings
} // namespace core
} // namespace aru

#endif // ARU_CORE_MAPPING_BINDINGS_H_
