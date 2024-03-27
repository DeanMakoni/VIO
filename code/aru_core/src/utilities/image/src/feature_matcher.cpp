
#include "Pinhole.h"
#include <Eigen/Dense>
#include <aru/core/utilities/image/feature_matcher.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/features2d.hpp>
#include <utility>
#include <iostream>

namespace aru {
namespace core {
namespace utilities {
namespace image {

using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
VisoMatcher::VisoMatcher(MatcherParams matcher_params,
                         ExtractorParams extractor_params)
    : matcher_params_(matcher_params), extractor_params_(extractor_params) {

  // Create a feature vector
  features_left_ = boost::make_shared<FeatureSPtrVector>();
}
//------------------------------------------------------------------------------
FeatureSPtrVectorSptr
VisoMatcher::ComputeStereoMatches(const cv::Mat &image_left,
                                  const cv::Mat &image_right) {
  int32_t width = image_left.cols;
  int32_t height = image_left.rows;
  // Create feature extractor
  viso_extractor_ = boost::make_shared<Matcher>();

  // compute visual odometry
  int32_t dims[] = {width, height, width};

  viso_extractor_->pushBack(image_left.data, image_right.data, dims, false);
  viso_extractor_->matchFeatures(1);

  std::vector<Matcher::p_match> p_matched = viso_extractor_->getMatches();
  LOG(INFO) << "Number of features is " << p_matched.size();

  FeatureSPtrVectorSptr curr_features = boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr tracked_features =
      boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr new_features = boost::make_shared<FeatureSPtrVector>();
  for (auto match : p_matched) {
    FeatureSPtr curr_feature =
        boost::make_shared<Feature>(Eigen::Vector2d(match.u1c, match.v1c));
    curr_feature->SetKeyPoint(KeyPoint(match.u1c, match.v1c, 1));
    curr_feature->SetMatchedKeypoint(KeyPoint(match.u2c, match.v2c, 1));
    curr_features->push_back(curr_feature);
  }
  return curr_features;
}
//------------------------------------------------------------------------------
FeatureSPtrVectorSptr VisoMatcher::ComputeSequentialMatches(
    const cv::Mat &image_1_left, const cv::Mat &image_2_left,
    const cv::Mat &image_1_right, const cv::Mat &image_2_right) {

  int32_t width = image_1_left.cols;
  int32_t height = image_1_left.rows;

  // compute visual odometry
  int32_t dims[] = {width, height, width};
  // Create feature extractor
  viso_extractor_ = boost::make_shared<Matcher>();

  // Do the first pair
  viso_extractor_->pushBack(image_1_left.data, image_1_right.data, dims, false);
  viso_extractor_->matchFeatures(2);
  // Do the second pair
  viso_extractor_->pushBack(image_2_left.data, image_2_right.data, dims, false);
  viso_extractor_->matchFeatures(2);
  std::vector<Matcher::p_match> p_matched = viso_extractor_->getMatches();
  LOG(INFO) << "Number of features is " << p_matched.size();

  FeatureSPtrVectorSptr curr_features = boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr tracked_features =
      boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr new_features = boost::make_shared<FeatureSPtrVector>();
  
  cv::Mat diff=image_1_left-image_1_right;
  
  for (auto match : p_matched) {
    FeatureSPtr curr_feature =
        boost::make_shared<Feature>(Eigen::Vector2d(match.u1c, match.v1c));
    curr_feature->SetKeyPoint(KeyPoint(match.u1c, match.v1c, 1));
    //LOG(INFO)<<"Key point 1  left is " << match.u1c<<","<<match.v1c;
    //LOG(INFO)<<"Key point 1 right is " << match.u1p<<","<<match.v1p;
    //LOG(INFO)<<"Key point 2  left is " << match.u2c<<","<<match.v2c;
    curr_feature->SetMatchedKeypoint(KeyPoint(match.u2c, match.v2c, 1));
    curr_feature->SetSequentialKeyPoint(KeyPoint(match.u1p, match.v1p, 1));
    curr_features->push_back(curr_feature);
  }
  return curr_features;
}
//------------------------------------------------------------------------------
OrbFeatureMatcher::OrbFeatureMatcher(MatcherParams matcher_params,
                                     ExtractorParams extractor_params)
    : matcher_params_(matcher_params), extractor_params_(extractor_params) {}
//------------------------------------------------------------------------------
OrbFeatureMatcher::OrbFeatureMatcher(
    aru::core::utilities::image::MatcherParams matcher_params,
    aru::core::utilities::image::ExtractorParams extractor_params,
    const std::string &vocab_filename)
    : matcher_params_(matcher_params), extractor_params_(extractor_params) {

  mpVocabulary = new ORB_SLAM3::ORBVocabulary();
  bool bVocLoad = mpVocabulary->loadFromTextFile(vocab_filename);
  cout<<"Filename is "<<vocab_filename<<endl;
  if (!bVocLoad) {
    cerr << "Wrong path to vocabulary. " << endl;
    cerr << "Falied to open at: " << vocab_filename << endl;
    exit(-1);
  }
//  cout << "Vocabulary loaded!" << endl << endl;

  // Create KeyFrame Database
  mpKeyFrameDatabase = new ORB_SLAM3::KeyFrameDatabase(*mpVocabulary);

  // Create the Atlas
  mpAtlas = new ORB_SLAM3::Atlas(0);

  mDistCoef = cv::Mat::zeros(4, 1, CV_32F);

  // TODO : Change hard coding of K
  float data[9] = {527.1638 ,    0.     ,  338.73506,0.     ,  526.92709,  241.73964, 0.     ,    0.     ,    1.     };
  cv::Mat K = cv::Mat(3, 3, CV_32F, data);

  mK = K.clone();

  fx = 527.1638;
  fy = 526.92709;
  cx = 338.73506;
  cy = 241.73964;

  baseline_ = 0.12;

  mbf = 63.24;
  mThDepth = 35;

  std::vector<float> vCamCalib{fx, fy, cx, cy};

  mpCamera = new ORB_SLAM3::Pinhole(vCamCalib);

  mpAtlas->AddCamera(mpCamera);
}
//------------------------------------------------------------------------------
FeatureSPtrVectorSptr OrbFeatureMatcher::ComputeMatches(
    const cv::Mat &image_1_left, const cv::Mat &image_1_right,
    const cv::Mat &image_2_left, const cv::Mat &image_2_right) {

  // Check for greyscale
  cv::Mat image_1_left_grey = image_1_left.clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_1_right_grey = image_1_right.clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_left_grey = image_2_left.clone();
  if (image_2_left_grey.channels() > 1) {
    cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_right_grey = image_2_right.clone();
  if (image_2_right_grey.channels() > 1) {
    cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_BGR2GRAY);
  }

  // Initialise ORB extractors
  ORB_SLAM3::ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

  mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);

  mpORBextractorRight = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);


  ORB_SLAM3::Frame Frame_1 =
      ORB_SLAM3::Frame(image_1_left_grey, image_1_right_grey, 0,
                       mpORBextractorLeft, mpORBextractorRight, mpVocabulary,
                       mK, mDistCoef, mbf, mThDepth, mpCamera);

//  LOG(INFO) << "Number of features is " << Frame_1.N;

  ORB_SLAM3::Frame Frame_2 =
      ORB_SLAM3::Frame(image_2_left_grey, image_2_right_grey, 0,
                       mpORBextractorLeft, mpORBextractorRight, mpVocabulary,
                       mK, mDistCoef, mbf, mThDepth, mpCamera);

//  LOG(INFO) << "Number of features is " << Frame_2.N;

  Frame_1.SetPose(cv::Mat::eye(4, 4, CV_32F));
  ORB_SLAM3::Map *pCurrentMap = mpAtlas->GetCurrentMap();

  // Create KeyFrame
  ORB_SLAM3::KeyFrame *pKFini = new ORB_SLAM3::KeyFrame(
      Frame_1, mpAtlas->GetCurrentMap(), mpKeyFrameDatabase);

  // Insert KeyFrame in the map
  mpAtlas->AddKeyFrame(pKFini);

  // Create MapPoints and asscoiate to KeyFrame
  for (int i = 0; i < Frame_1.N; i++) {
    float z = Frame_1.mvDepth[i];
    if (z > 0) {
      cv::Mat x3D = Frame_1.UnprojectStereo(i);
      auto *pNewMP =
          new ORB_SLAM3::MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
      pNewMP->AddObservation(pKFini, i);
      pKFini->AddMapPoint(pNewMP, i);
      pNewMP->ComputeDistinctiveDescriptors();
      pNewMP->UpdateNormalAndDepth();
      mpAtlas->AddMapPoint(pNewMP);
      Frame_1.mvpMapPoints[i] = pNewMP;
    }
  }


  // We perform first an ORB matching with each candidate
  ORB_SLAM3::ORBmatcher matcher(0.75, true);
  vector<ORB_SLAM3::MapPoint *> vpMapPointMatches;

  // Compute Bag of Words Vector
  Frame_2.ComputeBoW();
  pKFini->ComputeBoW();

  int nmatches = matcher.SearchByBoW(pKFini, Frame_2, vpMapPointMatches);

  FeatureSPtrVectorSptr curr_features = boost::make_shared<FeatureSPtrVector>();
  int idx = 0;
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    ORB_SLAM3::MapPoint *pMP = vpMapPointMatches[i];

    if (pMP) {
      if (!pMP->isBad()) {
        if (i >= Frame_2.mvKeysUn.size())
          continue;
        const cv::KeyPoint &kp = Frame_2.mvKeysUn[i];
        // 3D coordinates
        cv::Mat cv_pos = pMP->GetWorldPos();

        auto observation = pMP->GetObservations()[pKFini];
        cv::KeyPoint kp_prev = pKFini->mvKeysUn[std::get<0>(observation)];
        double depth = Frame_1.mvDepth[std::get<0>(observation)];
        double disp = mbf / depth;
        cv::KeyPoint kp_right =
            cv::KeyPoint(kp_prev.pt.x - disp, kp_prev.pt.y, 1);
        cv::Mat x3D = Frame_1.UnprojectStereo(std::get<0>(observation));

        FeatureSPtr curr_feature = boost::make_shared<Feature>(
            Eigen::Vector2d(kp_prev.pt.x, kp_prev.pt.y));
        curr_feature->SetKeyPoint(kp_prev);
        curr_feature->SetMatchedKeypoint(kp_right);
        curr_feature->SetSequentialKeyPoint(kp);
        curr_features->push_back(curr_feature);
      }
    }
  }

  LOG(INFO) << "Number of features is " << curr_features->size();
  return curr_features;
}

//------------------------------------------------------------------------------
FeatureSPtrVectorSptr
OrbFeatureMatcher::ComputeStereoMatches(const cv ::Mat &image_left,
                                        const cv::Mat &image_right) {
  // Check for greyscale
  cv::Mat image_1_left_grey = image_left.clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(image_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
    std::cout << "Fail 1" << std::endl;
  }
  cv::Mat image_1_right_grey = image_right.clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(image_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
    std::cout << "Fail 2" << std::endl;
  }

  // Initialise ORB extractors
  ORB_SLAM3::ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
  std::cout << "Fail 3" << std::endl;
  mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);
 std::cout << "Fail 4" << std::endl;
  mpORBextractorRight = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);
 std::cout << "Fail 5" << std::endl;
  ORB_SLAM3::Frame Frame_1 =
      ORB_SLAM3::Frame(image_1_left_grey, image_1_right_grey, 0,
                       mpORBextractorLeft, mpORBextractorRight, mpVocabulary,
                       mK, mDistCoef, mbf, mThDepth, mpCamera);
  std::cout << "Fail 6" << std::endl;
  // Create MapPoints and associate to KeyFrame
  FeatureSPtrVectorSptr curr_features = boost::make_shared<FeatureSPtrVector>();
  std::cout << "Fail 4" << std::endl;
  for (int i = 0; i < Frame_1.N; i++) {
    float z = Frame_1.mvDepth[i];
    if (z > 0) {
      cv::KeyPoint kp_prev = Frame_1.mvKeysUn[i];
      double disp = mbf / z;
      cv::KeyPoint kp_right =
          cv::KeyPoint(kp_prev.pt.x - disp, kp_prev.pt.y, 1);

      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(kp_prev.pt.x, kp_prev.pt.y));
      curr_feature->SetKeyPoint(kp_prev);
      curr_feature->SetMatchedKeypoint(kp_right);
      curr_features->push_back(curr_feature);
    }
  }
  return curr_features;
}

} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru
