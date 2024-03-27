//
// Created by paulamayo on 2021/05/28.
//

#ifndef ARU_CORE_CONVERSIONS_H
#define ARU_CORE_CONVERSIONS_H
//#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
//#include <include/aru/core/utilities/logging/log.h>
#include <aru/core/utilities/logging/log.h>

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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace aru {
namespace core {
namespace bindings {

pybind11::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat &input) {
  pybind11::array_t<unsigned char> dst =
      pybind11::array_t<unsigned char>({input.rows, input.cols}, input.data);
  return dst;
}

pybind11::array_t<float> cv_mat_float_1c_to_numpy(cv::Mat &input) {
  pybind11::array_t<float> dst =
      pybind11::array_t<float>({input.rows, input.cols}, (float *)input.data);
  return dst;
}

pybind11::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat &input) {
  pybind11::array_t<unsigned char> dst =
      pybind11::array_t<unsigned char>({input.rows, input.cols, 3}, input.data);
  return dst;
}

cv::Mat numpy_uint8_1c_to_cv_mat(pybind11::array_t<unsigned char> &input) {
  if (input.ndim() != 2)
    throw std::runtime_error("1-channel image must be 2 dims ");

  pybind11::buffer_info buf = input.request();

  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char *)buf.ptr);

  return mat;
}

cv::Mat numpy_float_1c_to_cv_mat(pybind11::array_t<float> &input) {
  if (input.ndim() == 2 && input.dtype().is(pybind11::dtype::of<float>())) {

    pybind11::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_32FC1, (float *)buf.ptr);

    return mat;
  } else {
    throw std::runtime_error("1-channel image must be 2 dims and a float ");
  }
}

cv::Mat numpy_uint8_3c_to_cv_mat(pybind11::array_t<unsigned char> &input) {
  if (input.ndim() != 3)
    throw std::runtime_error("3-channel image must be 3 dims ");

  pybind11::buffer_info buf = input.request();

  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

  return mat;
}

} // namespace bindings
} // namespace core
} // namespace aru
#endif // ARU_CORE_CONVERSIONS_H
