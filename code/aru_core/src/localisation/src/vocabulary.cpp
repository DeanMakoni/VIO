//
// Created by paulamayo on 2021/05/30.
//
#include "aru/core/localisation/vocabulary.h"
#include <Eigen/Dense>

namespace aru {
namespace core {
namespace localisation {
//------------------------------------------------------------------------------
FabMapVocabulary::FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                                   cv::Ptr<cv::DescriptorExtractor> extractor,
                                   cv::Mat vocab)
    : detector_(std::move(detector)), extractor_(std::move(extractor)),
      vocabulary_(std::move(vocab)) {}
//------------------------------------------------------------------------------
FabMapVocabulary::FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                                   cv::Ptr<cv::DescriptorExtractor> extractor)
    : detector_(std::move(detector)), extractor_(std::move(extractor)) {}
//------------------------------------------------------------------------------
cv::Mat FabMapVocabulary::getVocabulary() const { return vocabulary_; }
//------------------------------------------------------------------------------
cv::Mat FabMapVocabulary::generateBOWImageDescs(const cv::Mat &frame) const {
  // use a FLANN matcher to generate bag-of-words representations
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("FlannBased");
  cv::BOWImgDescriptorExtractor bide(extractor_, matcher);
  bide.setVocabulary(vocabulary_);

  cv::Mat bow;
  std::vector<cv::KeyPoint> kpts;

  detector_->detect(frame, kpts);
  bide.compute(frame, kpts, bow);
  return bow;
}
//------------------------------------------------------------------------------
cv::Mat FabMapVocabulary::generateBOWImageDescsInternal(cv::Mat desc) const {
  // use a FLANN matcher to generate bag-of-words representations
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("FlannBased");

  cv::Mat bow;

  compute(matcher, desc, bow);

  return bow;
}

cv::Mat FabMapVocabulary::TrainVocab(double cluster_radius) {
  of2::BOWMSCTrainer trainer(cluster_radius);
  trainer.add(vocabulary_train_data_);
  return trainer.cluster();
}
//------------------------------------------------------------------------------
void FabMapVocabulary::compute(cv::Ptr<cv::DescriptorMatcher> dmatcher,
                               cv::Mat keypointDescriptors,
                               cv::Mat &_imgDescriptor) const {
  CV_Assert(!vocabulary_.empty());
  CV_Assert(!keypointDescriptors.empty());

  int clusterCount = vocabulary_.rows; // = vocabulary.rows

  // Match keypoint descriptors to cluster center (to vocabulary)
  std::vector<cv::DMatch> matches;
  dmatcher->match(keypointDescriptors, vocabulary_, matches);

  // Compute image descriptor

  _imgDescriptor.create(1, clusterCount, CV_32FC1);
  _imgDescriptor.setTo(cv::Scalar::all(0));

  float *dptr = _imgDescriptor.ptr<float>();
  for (size_t i = 0; i < matches.size(); i++) {
    int queryIdx = matches[i].queryIdx;
    int trainIdx = matches[i].trainIdx; // cluster index
    CV_Assert(queryIdx == (int)i);

    dptr[trainIdx] = dptr[trainIdx] + 1.f;
  }

  // Normalize image descriptor.
  _imgDescriptor /= keypointDescriptors.size().height;
}
//------------------------------------------------------------------------------
void FabMapVocabulary::convert() {
  cv::Mat vocab_;
  vocabulary_.convertTo(vocab_, CV_32F);
  vocabulary_ = vocab_;
}
//------------------------------------------------------------------------------
void FabMapVocabulary::save(cv::FileStorage fileStorage) const {
  // Note that this is a partial save, assume that the settings are saved
  // elsewhere.
  fileStorage << "Vocabulary" << vocabulary_;
}
//------------------------------------------------------------------------------
void FabMapVocabulary::load(cv::FileStorage fileStorage) {
  cv::Mat vocab;
  fileStorage["Vocabulary"] >> vocab;
}
//------------------------------------------------------------------------------

bool FabMapVocabulary::AddTrainingImage(const cv::Mat &frame) {
  cv::Mat descs, feats;
  std::vector<cv::KeyPoint> kpts;

  if (frame.data) {
    // detect & extract features
    detector_->detect(frame, kpts);
    extractor_->compute(frame, kpts, descs);

    // add all descriptors to the training data
    addTrainingDescs(descs);
    return true;
  }
  return false;
}

void FabMapVocabulary::addTrainingDescs(const cv::Mat &descs) {
  vocabulary_train_data_.push_back(descs);
}
} // namespace localisation
} // namespace core
} // namespace aru
