#ifndef ARU_CORE_UTILITIES_IMAGE_MATCHER_H_
#define ARU_CORE_UTILITIES_IMAGE_MATCHER_H_

#include "aru/core/utilities/image/point_feature.h"


#include <ORBmatcher.h>
#include <Atlas.h>
#include <viso.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/mat.hpp>

namespace aru {
namespace core {
namespace utilities {
namespace image {

struct MatcherParams {
  int match_threshold_high;
  int match_threshold_low;
  float stereo_baseline;
  float focal_length;
};

struct ExtractorParams {
  int num_features;
  int num_levels;
  float scale_factor;
  int initial_fast_threshold;
  int minimum_fast_threshold;
  int patch_size;
  int half_patch_size;
  int edge_threshold;

  std::string vocab_filename_;
};


class OrbFeatureMatcher {
public:
  OrbFeatureMatcher(MatcherParams matcher_params,
                    ExtractorParams extractor_params);

  OrbFeatureMatcher(MatcherParams matcher_params,
                    ExtractorParams extractor_params,
                    const std::string& vocab_filename);

  FeatureSPtrVectorSptr ComputeMatches(const cv::Mat &image_prev_left,
                                       const cv::Mat &image_prev_right,
                                       const cv::Mat &image_curr_left,
                                       const cv::Mat &image_curr_right);

  ~OrbFeatureMatcher() = default;


  FeatureSPtrVectorSptr ComputeStereoMatches(const cv ::Mat &image_left,
                                             const cv::Mat &image_right);

private:
  MatcherParams matcher_params_;
  ExtractorParams extractor_params_;

  // ORB vocabulary used for place recognition and feature matching.
  ORB_SLAM3::ORBVocabulary* mpVocabulary;

  ORB_SLAM3::GeometricCamera* mpCamera;

  float baseline_;

  //Calibration matrix
  cv::Mat mK;
  cv::Mat mDistCoef;
  float mbf,fx,fy,cx,cy,mThDepth;


  //Atlas
  ORB_SLAM3::Atlas* mpAtlas;

  // KeyFrame database for place recognition (relocalization and loop detection).
  ORB_SLAM3::KeyFrameDatabase* mpKeyFrameDatabase;




};

// class CurvatureMatcher {
// public:
//   CurvatureMatcher(MatcherParams matcher_params,
//                    extractors::ExtractorParams extractor_params);
//
//   ~CurvatureMatcher() = default;
//
//   FeatureSPtrVectorSptr
//   ComputeSequentialMatches(const cv::Mat &image_1, const cv::Mat &image_2,
//                            FeatureSPtrVectorSptr prev_features);
//
//   FeatureSPtrVectorSptr ComputeStereoMatches(const cv ::Mat &image_left,
//                                              const cv::Mat &image_right);
//
// private:
//   MatcherParams matcher_params_;
//   boost::shared_ptr<extractors::CurvatureExtractor> feature_extractor_1_;
//   boost::shared_ptr<extractors::CurvatureExtractor> feature_extractor_2_;
//
//   FeatureSPtrVectorSptr features_left_;
// };

class VisoMatcher {
public:
  VisoMatcher(MatcherParams matcher_params,
              ExtractorParams extractor_params);

  ~VisoMatcher() = default;

  FeatureSPtrVectorSptr ComputeSequentialMatches(const cv::Mat &image_1_left,
                                                 const cv::Mat &image_2_left,
                                                 const cv::Mat &image_1_right,
                                                 const cv::Mat &image_2_right);

  FeatureSPtrVectorSptr ComputeStereoMatches(const cv ::Mat &image_left,
                                             const cv::Mat &image_right);

private:
  MatcherParams matcher_params_;
  ExtractorParams extractor_params_;
  boost::shared_ptr<Matcher> viso_extractor_;

  FeatureSPtrVectorSptr features_left_;
};
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_CORE_UTILITIES_IMAGE_MATCHER_H_
