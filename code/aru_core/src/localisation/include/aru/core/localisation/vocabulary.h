//
// Created by paulamayo on 2021/05/30.
//

#ifndef ARU_CORE_VOCABULARY_H
#define ARU_CORE_VOCABULARY_H

#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "openfabmap.hpp"

namespace aru {
namespace core {
namespace localisation {

class FabMapVocabulary {
public:
  FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                   cv::Ptr<cv::DescriptorExtractor> extractor, cv::Mat vocab);

  FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                   cv::Ptr<cv::DescriptorExtractor> extractor);

  virtual ~FabMapVocabulary() = default;

  bool AddTrainingImage(const cv::Mat &frame);

  void IncludeVocabulary(cv::Mat vocab){vocabulary_=vocab;}

  cv::Mat getVocabulary() const;
  cv::Mat generateBOWImageDescs(const cv::Mat &frame) const;
  cv::Mat generateBOWImageDescsInternal(cv::Mat desc) const;

  void compute(cv::Ptr<cv::DescriptorMatcher> dmatcher,
               cv::Mat keypointDescriptors, cv::Mat &_imgDescriptor) const;

  void convert();

  cv::Mat TrainVocab(double cluster_radius);

  cv::Mat GetTrainingData(){return vocabulary_train_data_;}

  void save(cv::FileStorage fileStorage) const;
  void load(cv::FileStorage fileStorage);

private:
  void addTrainingDescs(const cv::Mat &descs);

  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  cv::Mat vocabulary_;
  cv::Mat vocabulary_train_data_;
};
} // namespace localisation
} // namespace core
} // namespace aru

#endif // ARU_CORE_VOCABULARY_H
