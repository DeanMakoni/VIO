#ifndef ARU_LOCALISATION_H_
#define ARU_LOCALISATION_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "aru/core/localisation/vocabulary.h"
#include "openfabmap.hpp"

namespace aru {
namespace core {
namespace localisation {

class Localisation {

public:
  Localisation(const std::string &vocabulary_file,
               const std::string &chow_liu_tree_file,
               const std::string &settings_file);

  ~Localisation() = default;

  void AddVocabularyTrainingImage(cv::Mat image);

  cv::Mat AddQueryImage(const cv::Mat& image);

  void AddSampleImage(const cv::Mat& image);

  void TrainAndSaveTree();

  void InitLocalisation();

  void ClearQueries();

  void AddQuerySubset(int num_queries);

  void SaveQueryDescriptors(const std::string &query_descriptors_file);

  void SaveSampleDescriptors(const std::string &sample_descriptors_file);

  void AddSampleData(const std::string &sample_descriptors_file);

  void AddQueryData(const std::string &query_descriptors_file);

  void AddBowData(const cv::Mat &query_bow_desc);

  void TrainAndSaveVocabulary();

  std::pair<int, float> FindClosestImage(cv::Mat &test_image);

  std::pair<int, float> FindLoopClosure(cv::Mat &test_image, int max_index);

  static cv::Ptr<cv::FeatureDetector> GenerateDetector(cv::FileStorage &fs);

  static cv::Ptr<cv::FeatureDetector> GenerateExtractor(cv::FileStorage &fs);

  static boost::shared_ptr<of2::FabMap> GenerateFabmap(cv::Mat chow_liu_tree,
                                                       cv::FileStorage &fs);

private:
  std::string vocab_filename_;
  std::string chow_liu_filename_;
  std::string settings_filename_;

  boost::shared_ptr<FabMapVocabulary> vocabulary_;
  boost::shared_ptr<of2::FabMap> fabmap_;

  bool vocab_exists_;
  bool chow_liu_tree_exists_;

  cv::Mat cl_tree_;

  cv::Mat query_image_bow_data_;

  cv::Mat sample_image_bow_data_;
};
} // namespace localisation
} // namespace core
} // namespace aru

#endif // ARU_LOCALISATION_H_
