#include "aru/core/localisation/localisation.h"
#include <iostream>
#include <opencv4/opencv2/features2d.hpp>
#include "opencv4/opencv2/xfeatures2d/nonfree.hpp"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <fstream>
#include <opencv4/opencv2/core/persistence.hpp>
#include <opencv4/opencv2/features2d.hpp>

namespace aru {
namespace core {
namespace localisation {
//------------------------------------------------------------------------------
Localisation::Localisation(const std::string &vocab_file,
                           const std::string &chow_liu_tree_file,
                           const std::string &settings_file)
    : vocab_filename_(std::move(vocab_file)),
      chow_liu_filename_(std::move(chow_liu_tree_file)),
      settings_filename_(std::move(settings_file)), vocab_exists_(false),
      chow_liu_tree_exists_(false) {
  cv::FileStorage fs;
  fs.open(settings_file, cv::FileStorage::READ);

  cv::FileStorage fs_vocab;
  LOG(INFO) << "Loading Vocabulary" << std::endl;
  fs_vocab.open(vocab_file, cv::FileStorage::READ);
  cv::Mat vocab;
  fs_vocab["Vocabulary"] >> vocab;
  if (vocab.empty()) {
    vocabulary_ = boost::make_shared<aru::core::localisation::FabMapVocabulary>(
        aru::core::localisation::Localisation::GenerateDetector(fs),
        aru::core::localisation::Localisation::GenerateExtractor(fs));

  } else {
    LOG(INFO) << "Vocabulary read succesfully";
    vocab_exists_ = true;
    vocabulary_ = boost::make_shared<aru::core::localisation::FabMapVocabulary>(
        aru::core::localisation::Localisation::GenerateDetector(fs),
        aru::core::localisation::Localisation::GenerateExtractor(fs), vocab);
  }
  fs_vocab.release();

  std::ifstream checker;
  checker.open(chow_liu_tree_file.c_str());
  cv::Mat clTree;
  if (checker.is_open()) {
    chow_liu_tree_exists_ = true;
    cv::FileStorage tree_fs;
    LOG(INFO) << "Loading Chow-Liu Tree" << std::endl;
    tree_fs.open(chow_liu_tree_file, cv::FileStorage::READ);
    tree_fs["ChowLiuTree"] >> clTree;
    cl_tree_ = clTree;
  }

  if (chow_liu_tree_exists_ && vocab_exists_) {
    fabmap_ =
        aru::core::localisation::Localisation::GenerateFabmap(cl_tree_, fs);
    LOG(INFO) << "FABMAP Initialised";
  }
  fs.release();
}

//------------------------------------------------------------------------------
void Localisation::AddVocabularyTrainingImage(cv::Mat image_mat) {
  vocabulary_->AddTrainingImage(image_mat);
}

//------------------------------------------------------------------------------
void Localisation::TrainAndSaveVocabulary() {
  cv::Mat vocab_train_data = vocabulary_->GetTrainingData();

  cv::FileStorage fs;
  fs.open(settings_filename_, cv::FileStorage::READ);
  double cluster_radius = fs["VocabTrainOptions"]["ClusterSize"];
  of2::BOWMSCTrainer trainer(cluster_radius);
  trainer.add(vocab_train_data);
  cv::Mat vocab = trainer.cluster();

  // save the vocabulary
  cv::FileStorage vocab_fs;
  LOG(INFO) << "Saving vocabulary";
  vocab_fs.open(vocab_filename_, cv::FileStorage::WRITE);
  vocab_fs << "Vocabulary" << vocab;
  vocab_fs.release();

  // Update the vocabulary
  vocabulary_->IncludeVocabulary(vocab);
  vocab_exists_ = true;
}

//------------------------------------------------------------------------------
cv::Mat Localisation::AddQueryImage(const cv::Mat &query_image) {
  cv::Mat bow = vocabulary_->generateBOWImageDescs(query_image);
  fabmap_->add(bow);
  query_image_bow_data_.push_back(bow);
  return bow;
}

//------------------------------------------------------------------------------
void Localisation::AddSampleImage(const cv::Mat &sample_image) {
  cv::Mat bow = vocabulary_->generateBOWImageDescs(sample_image);
  sample_image_bow_data_.push_back(bow);
}

//------------------------------------------------------------------------------
void Localisation::TrainAndSaveTree() {

  cv::FileStorage fs;
  fs.open(settings_filename_, cv::FileStorage::READ);

  // generate the tree from the data
  cv::FileStorage tree_fs;
  std::cout << "Making Chow-Liu Tree" << std::endl;
  of2::ChowLiuTree tree;
  tree.add(sample_image_bow_data_);
  cv::Mat clTree = tree.make(fs["ChowLiuOptions"]["LowerInfoBound"]);

  // save the resulting tree
  std::cout << "Saving Chow-Liu Tree" << std::endl;
  tree_fs.open(chow_liu_filename_, cv::FileStorage::WRITE);
  tree_fs << "ChowLiuTree" << clTree;
  tree_fs.release();
  cl_tree_ = clTree;
  vocab_exists_ = true;
  if (chow_liu_tree_exists_ && vocab_exists_) {
    fabmap_ =
        aru::core::localisation::Localisation::GenerateFabmap(cl_tree_, fs);
  }
}

//------------------------------------------------------------------------------
void Localisation::InitLocalisation() {
  cv::FileStorage fs;
  fs.open(settings_filename_, cv::FileStorage::READ);

  // generate the tree from the data
  if (!chow_liu_tree_exists_) {
    cv::FileStorage tree_fs;
    LOG(INFO) << "Making Chow-Liu Tree";
    of2::ChowLiuTree tree;
    tree.add(sample_image_bow_data_);
    cv::Mat clTree = tree.make(fs["ChowLiuOptions"]["LowerInfoBound"]);
    cl_tree_ = clTree;
  }

  fabmap_ = aru::core::localisation::Localisation::GenerateFabmap(cl_tree_, fs);
  //  fabmap_->addTraining(query_image_bow_data_);
  //  fabmap_->add(query_image_bow_data_);
}
//------------------------------------------------------------------------------
void Localisation::SaveSampleDescriptors(
    const std::string &sample_descriptors_file) {
  cv::FileStorage fs;
  fs.open(sample_descriptors_file, cv::FileStorage::WRITE);
  fs << "BOWImageDescs" << sample_image_bow_data_;
  fs.release();
}
//------------------------------------------------------------------------------
void Localisation::SaveQueryDescriptors(
    const std::string &query_descriptors_file) {
  cv::FileStorage fs;
  fs.open(query_descriptors_file, cv::FileStorage::WRITE);
  fs << "BOWImageDescs" << query_image_bow_data_;
  fs.release();
}
//------------------------------------------------------------------------------
void Localisation::ClearQueries() { fabmap_->ClearTestImgDescriptors(); }
//------------------------------------------------------------------------------
void Localisation::AddQuerySubset(int num_queries) {
  std::vector<cv::Mat> queryImgDescriptors;
  for (int i = 0; i < num_queries; i++) {
    queryImgDescriptors.push_back(query_image_bow_data_.row(i));
  }
  fabmap_->add(queryImgDescriptors);
}
//------------------------------------------------------------------------------
void Localisation::AddSampleData(const std::string &sample_descriptors_file) {
  cv::FileStorage fs_sample;
  fs_sample.open(sample_descriptors_file, cv::FileStorage::READ);
  cv::Mat sample_image_bow_data;
  fs_sample["BOWImageDescs"] >> sample_image_bow_data;
  sample_image_bow_data_ = sample_image_bow_data;
  fabmap_->addTraining(sample_image_bow_data_);
}

//------------------------------------------------------------------------------
void Localisation::AddQueryData(const std::string &query_descriptors_file) {
  cv::FileStorage fs_query;
  fs_query.open(query_descriptors_file, cv::FileStorage::READ);
  cv::Mat query_image_bow_data;
  fs_query["BOWImageDescs"] >> query_image_bow_data;
  query_image_bow_data_ = query_image_bow_data;
  fabmap_->add(query_image_bow_data_);
}

//-------------------------------------------------------------------------------
void Localisation::AddBowData(const cv::Mat &query_bow_desc) {
    fabmap_->add(query_bow_desc);
}
//------------------------------------------------------------------------------
std::pair<int, float> Localisation::FindClosestImage(cv::Mat &test_image) {

  cv::Mat bow = vocabulary_->generateBOWImageDescs(test_image);
  std::vector<of2::IMatch> matches;
  fabmap_->localize(bow, matches, false);

  double best_prob = 0.0;
  int bestMatchIndex = -1;
  for (auto &match : matches) {
//        LOG(INFO)<<"Match "<<match.imgIdx<<" has probability "<<match.match;
    if (match.match > best_prob) {
      best_prob = match.match;
      bestMatchIndex = match.imgIdx;
    }
  }
  return std::make_pair(bestMatchIndex, best_prob);
}

//------------------------------------------------------------------------------
std::pair<int, float> Localisation::FindLoopClosure(cv::Mat &test_image,
                                                    int max_index) {

  // filter queries
  ClearQueries();
  AddQuerySubset(max_index);

  cv::Mat bow = vocabulary_->generateBOWImageDescs(test_image);
  std::vector<of2::IMatch> matches;
  fabmap_->localize(bow, matches, false);

  double best_prob = 0.0;
  int bestMatchIndex = -1;
  int count = 0;
  for (auto &match : matches) {
    int match_index = match.imgIdx;
    if (match.match > best_prob) {
      best_prob = match.match;
      bestMatchIndex = match.imgIdx;
    }
    count++;
  }
  return std::make_pair(bestMatchIndex, best_prob);
}
//------------------------------------------------------------------------------
cv::Ptr<cv::FeatureDetector>
Localisation::GenerateDetector(cv::FileStorage &fs) {

  // create common feature detector and descriptor extractor
  std::string detectorType = fs["FeatureOptions"]["DetectorType"];

  LOG(INFO) << "Detector type is " << detectorType;
  if (detectorType == "BRISK") {
    return cv::BRISK::create(fs["FeatureOptions"]["BRISK"]["Threshold"],
                             fs["FeatureOptions"]["BRISK"]["Octaves"],
                             fs["FeatureOptions"]["BRISK"]["PatternScale"]);
  } else if (detectorType == "ORB") {
    return cv::ORB::create(fs["FeatureOptions"]["ORB"]["nFeatures"],
                           fs["FeatureOptions"]["ORB"]["scaleFactor"],
                           fs["FeatureOptions"]["ORB"]["nLevels"],
                           fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                           fs["FeatureOptions"]["ORB"]["firstLevel"], 2,
                           cv::ORB::HARRIS_SCORE,
                           fs["FeatureOptions"]["ORB"]["patchSize"]);

  } else if (detectorType == "MSER") {
    return cv::MSER::create(
        fs["FeatureOptions"]["MSERDetector"]["Delta"],
        fs["FeatureOptions"]["MSERDetector"]["MinArea"],
        fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
        fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
        fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
        fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
        fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
        fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
        fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
  } else if (detectorType == "FAST") {
    return cv::FastFeatureDetector::create(
        fs["FeatureOptions"]["FastDetector"]["Threshold"],
        (int)fs["FeatureOptions"]["FastDetector"]["NonMaxSuppression"] > 0);
  } else if (detectorType == "AGAST") {
    return cv::AgastFeatureDetector::create(
        fs["FeatureOptions"]["AGAST"]["Threshold"],
        (int)fs["FeatureOptions"]["AGAST"]["NonMaxSuppression"] > 0);

  } else if (detectorType == "SURF") {
    return cv::xfeatures2d::SURF::create(
        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
        fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
        fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
        (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
        (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

  } else if (detectorType == "SIFT") {
    return cv::xfeatures2d::SIFT::create(
        fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
        fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
        fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
        fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
  } else {
    std::cerr << "Could not create detector class. Specify detector "
                 "mode (static/adaptive) in the settings file"
              << std::endl;
  }

  return cv::Ptr<cv::FeatureDetector>(); // return the nullptr
}
//------------------------------------------------------------------------------
cv::Ptr<cv::FeatureDetector>
Localisation::GenerateExtractor(cv::FileStorage &fs) {
  std::string extractorType = fs["FeatureOptions"]["ExtractorType"];

  if (extractorType == "BRISK") {
    return cv::BRISK::create(fs["FeatureOptions"]["BRISK"]["Threshold"],
                             fs["FeatureOptions"]["BRISK"]["Octaves"],
                             fs["FeatureOptions"]["BRISK"]["PatternScale"]);
  } else if (extractorType == "ORB") {
    return cv::ORB::create(fs["FeatureOptions"]["ORB"]["nFeatures"],
                           fs["FeatureOptions"]["ORB"]["scaleFactor"],
                           fs["FeatureOptions"]["ORB"]["nLevels"],
                           fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                           fs["FeatureOptions"]["ORB"]["firstLevel"], 2,
                           cv::ORB::HARRIS_SCORE,
                           fs["FeatureOptions"]["ORB"]["patchSize"]);
  } else if (extractorType == "SURF") {
    return cv::xfeatures2d::SURF::create(
        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
        fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
        fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
        (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
        (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

  } else if (extractorType == "SIFT") {
    return cv::xfeatures2d::SIFT::create(
        fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
        fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
        fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
        fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
  } else {
    std::cerr << "Could not create Descriptor Extractor. Please specify "
                 "extractor type in settings file"
              << std::endl;
  }

  return cv::Ptr<cv::DescriptorExtractor>();
}
//------------------------------------------------------------------------------
boost::shared_ptr<of2::FabMap>
Localisation::GenerateFabmap(cv::Mat chow_liu_tree, cv::FileStorage &fs) {
  // create options flags
  std::string newPlaceMethod = fs["openFabMapOptions"]["NewPlaceMethod"];
  std::string bayesMethod = fs["openFabMapOptions"]["BayesMethod"];
  int simpleMotionModel = fs["openFabMapOptions"]["SimpleMotion"];
  int options = 0;
  if (newPlaceMethod == "Sampled") {
    options |= of2::FabMap::SAMPLED;
  } else {
    options |= of2::FabMap::MEAN_FIELD;
  }
  if (bayesMethod == "ChowLiu") {
    options |= of2::FabMap::CHOW_LIU;
  } else {
    options |= of2::FabMap::NAIVE_BAYES;
  }
  if (simpleMotionModel) {
    options |= of2::FabMap::MOTION_MODEL;
  }

  boost::shared_ptr<of2::FabMap> fabmap;
  // create an instance of the desired type of FabMap
  std::string fabMapVersion = fs["openFabMapOptions"]["FabMapVersion"];
  if (fabMapVersion == "FABMAP1") {
    fabmap = boost::make_shared<of2::FabMap1>(
        chow_liu_tree, fs["openFabMapOptions"]["PzGe"],
        fs["openFabMapOptions"]["PzGne"], options,
        fs["openFabMapOptions"]["NumSamples"]);
  } else if (fabMapVersion == "FABMAPLUT") {
    fabmap = boost::make_shared<of2::FabMapLUT>(
        chow_liu_tree, fs["openFabMapOptions"]["PzGe"],
        fs["openFabMapOptions"]["PzGne"], options,
        fs["openFabMapOptions"]["NumSamples"],
        fs["openFabMapOptions"]["FabMapLUT"]["Precision"]);
  } else if (fabMapVersion == "FABMAPFBO") {
    fabmap = boost::make_shared<of2::FabMapFBO>(
        chow_liu_tree, fs["openFabMapOptions"]["PzGe"],
        fs["openFabMapOptions"]["PzGne"], options,
        fs["openFabMapOptions"]["NumSamples"],
        fs["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
        fs["openFabMapOptions"]["FabMapFBO"]["PsGd"],
        fs["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
        fs["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
  } else if (fabMapVersion == "FABMAP2") {
    fabmap = boost::make_shared<of2::FabMap2>(
        chow_liu_tree, fs["openFabMapOptions"]["PzGe"],
        fs["openFabMapOptions"]["PzGne"], options);
  } else {
    LOG(INFO) << "Could not identify openFABMAPVersion from settings"
                 " file";
    throw std::exception();
    return NULL;
  }
  return fabmap;
}

//------------------------------------------------------------------------------
} // namespace localisation
} // namespace core
} // namespace aru
