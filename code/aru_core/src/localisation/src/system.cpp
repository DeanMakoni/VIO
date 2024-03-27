#include "aru/core/localisation/system.h"
#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <include/aru/core/utilities/logging/log.h>
#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <pbStereoImage.pb.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>

using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace datatype::image;

namespace aru {
namespace core {
namespace localisation {
//------------------------------------------------------------------------------
Localisation::Localisation(const std::string &training_monolithic,
                           const bool train_vocabulary,
                           const std::string &vocab_file,
                           const bool regenerate_tree,
                           const std::string &chow_liu_tree_file,
                           const std::string &settings_file) {
  cv::FileStorage fs;
  fs.open(settings_file, cv::FileStorage::READ);

  if (train_vocabulary) {
    vocabulary_ = boost::make_shared<FabMapVocabulary>(GenerateDetector(fs),
                                                       GenerateExtractor(fs));

    LOG(INFO) << "Generating vocabulary data";
    cv::Mat vocab_train_data = GenerateVocabData(training_monolithic);

    double cluster_radius = fs["VocabTrainOptions"]["ClusterSize"];
    of2::BOWMSCTrainer trainer(cluster_radius);
    trainer.add(vocab_train_data);
    cv::Mat vocab = trainer.cluster();

    // save the vocabulary
    cv::FileStorage vocab_fs;
    std::cout << "Saving vocabulary" << std::endl;
    vocab_fs.open(vocab_file, cv::FileStorage::WRITE);
    vocab_fs << "Vocabulary" << vocab;
    vocab_fs.release();

    // Update the vocabulary
    vocabulary_->IncludeVocabulary(vocab);

  } else {
    LOG(INFO) << "Loading Vocabulary" << std::endl;
    fs.open(vocab_file, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;

    LOG(INFO) << "Initialise the feature descriptors and extractors";
    fs.open(settings_file, cv::FileStorage::READ);

    vocabulary_ = boost::make_shared<FabMapVocabulary>(
        GenerateDetector(fs), GenerateExtractor(fs), vocab);
  }

  // Generate the BoW data from training monolithic
  cv::Mat bow_training_data = GenerateBoWData(training_monolithic);

  if (regenerate_tree) {
    // generate the tree from the data
    cv::FileStorage tree_fs;
    std::cout << "Making Chow-Liu Tree" << std::endl;
    of2::ChowLiuTree tree;
    tree.add(bow_training_data);
    cv::Mat clTree = tree.make(fs["ChowLiuOptions"]["LowerInfoBound"]);

    // save the resulting tree
    std::cout << "Saving Chow-Liu Tree" << std::endl;
    tree_fs.open(chow_liu_tree_file, cv::FileStorage::WRITE);
    tree_fs << "ChowLiuTree" << clTree;
    tree_fs.release();
    LOG(INFO) << " Generate the fabmap pointer";
    fabmap_ = GenerateFabmap(clTree, fs);

  } else {
    cv::FileStorage tree_fs;
    LOG(INFO) << "Loading Chow-Liu Tree" << std::endl;
    tree_fs.open(chow_liu_tree_file, cv::FileStorage::READ);
    cv::Mat clTree;
    tree_fs["ChowLiuTree"] >> clTree;
    LOG(INFO) << " Generate the fabmap pointer";
    fabmap_ = GenerateFabmap(clTree, fs);
  }

  LOG(INFO) << "Adding the test descriptors";
  fabmap_->addTraining(bow_training_data);
  fabmap_->add(bow_training_data);


}
//------------------------------------------------------------------------------
cv::Mat
Localisation::GenerateVocabData(const std::string &training_monolithic) {
  ProtocolLogger<datatype::image::pbStereoImage> logger(training_monolithic,
                                                        false);
  cv::Mat vocab_train_data;
  while (!logger.EndOfFile()) {
    pbStereoImage pb_image = logger.ReadFromFile();
    if (pb_image.has_image_left()) {
      aru::core::utilities::image::StereoImage image_out =
          ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(pb_image);
      cv::imshow("Image read", image_out.first.GetImage());
      cv::waitKey(5);

      vocabulary_->AddTrainingImage(image_out.first.GetImage());
    }
  }
  LOG(INFO) << "Reached end of file";
  return vocabulary_->GetTrainingData();
}
//------------------------------------------------------------------------------
cv::Mat Localisation::GenerateBoWData(const std::string &training_monolithic) {
  ProtocolLogger<datatype::image::pbStereoImage> logger(training_monolithic,
                                                        false);
  cv::Mat BowData;
  int down_sample=1;
  int i=0;
  while (!logger.EndOfFile()) {
    pbStereoImage pb_image = logger.ReadFromFile();
    if (pb_image.has_image_left()&& i%down_sample==0) {
      aru::core::utilities::image::StereoImage image_out =
          ImageProtocolBufferAdaptor::ReadStereoFromProtocolBuffer(pb_image);
      cv::imshow("Image read bow", image_out.first.GetImage());
      cv::waitKey(5);
      cv::Mat bow =
          vocabulary_->generateBOWImageDescs(image_out.first.GetImage());
      BowData.push_back(bow);
      i++;
    }
  }
  return BowData;
}
//------------------------------------------------------------------------------
int Localisation::FindClosestImage(cv::Mat &test_image) {

  cv::Mat bow = vocabulary_->generateBOWImageDescs(test_image);
  std::vector<of2::IMatch> matches;
  fabmap_->localize(bow, matches, false);

  double bestLikelihood = 0.0;
  int bestMatchIndex = -1;
  for (auto &match : matches) {
    //    LOG(INFO)<<"Match "<<match.imgIdx<<" has likelihood "<<match.likelihood;
    if (match.likelihood > bestLikelihood) {
      bestLikelihood = match.likelihood;
      bestMatchIndex = match.imgIdx;
    }
  }
  return bestMatchIndex;
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

  } else if (detectorType == "STAR") {

    return cv::xfeatures2d::StarDetector::create(
        fs["FeatureOptions"]["StarDetector"]["MaxSize"],
        fs["FeatureOptions"]["StarDetector"]["Response"],
        fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
        fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
        fs["FeatureOptions"]["StarDetector"]["Suppression"]);

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
    std::cerr << "Could not identify openFABMAPVersion from settings"
                 " file"
              << std::endl;
  }
  return fabmap;
}

//------------------------------------------------------------------------------
} // namespace localisation
} // namespace core
} // namespace aru

