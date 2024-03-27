#ifndef ARU_DEPTH_COLOUR_MAP_H_
#define ARU_DEPTH_COLOUR_MAP_H_

#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace aru {
namespace core {
namespace mesh {


static cv::Mat LinSpace(float x0, float x1, int n);
static cv::Mat ArgSort(cv::InputArray &_src, bool ascending = true);
static void SortMatrixRowsByIndices(cv::InputArray &_src,
                                    cv::InputArray &_indices,
                                    cv::OutputArray &_dst);
static cv::Mat SortMatrixRowsByIndices(cv::InputArray &src,
                                       cv::InputArray &indices);

template <typename _Tp>
static cv::Mat Interp1_(const cv::Mat &X_, const cv::Mat &Y_,
                        const cv::Mat &XI);
static cv::Mat Interp1(cv::InputArray &_x, cv::InputArray &_Y,
                       cv::InputArray &_xi);

class ColourMap {

private:
  cv::Mat _lut;

public:
  ColourMap();
  ColourMap(const int n);
  static cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                          cv::InputArray &g, cv::InputArray &b,
                          cv::InputArray &xi);
  cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                          cv::InputArray &g, cv::InputArray &b, const int n);
  cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                          cv::InputArray &g, cv::InputArray &b,
                          const float begin, const float end, const float n);

  void ApplyColourMap(cv::InputArray &src, cv::OutputArray &dst) const;
  void LookUp(int lookup_value, cv::Vec3b &colour_out) const;
  void LookUp2(const float v, cv::Vec3b &c) const;
  void LookUpAlt(int lookup_value, cv::Vec3b &colour_out) const;
};

static ColourMap
    COLOUR_MAP; // instantiate a static variable as only need one instance





} // namespace depth
} // namespace core
} // namespace aru

#endif // ARU_DEPTH_COLOUR_MAP_H_
