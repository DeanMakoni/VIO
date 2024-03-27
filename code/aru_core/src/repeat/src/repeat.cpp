#include "aru/core/repeat/repeat.h"
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>

namespace aru {
namespace core {
namespace repeat {

using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
// Constructor
Repeat::Repeat(std::string repeat_config_file, std::string vocab_file) {

  map_vocabulary_ = new ORB_SLAM3::ORBVocabulary();
  bool bVocLoad = map_vocabulary_->loadFromTextFile(vocab_file);
  LOG(INFO) << "Filename is " << vocab_file << endl;
  if (!bVocLoad) {
    LOG(ERROR) << "Wrong path to vocabulary. " << endl;
    LOG(ERROR) << "Failed to open at: " << vocab_file << endl;
    exit(-1);
  }
  LOG(INFO) << "Vocabulary loaded!" << endl;
  // Create KeyFrame Database
  map_key_frame_database_ = new ORB_SLAM3::KeyFrameDatabase(*map_vocabulary_);
  LOG(INFO) << "Database created!" << endl;

  // Create the Atlas
  map_atlas_ = new ORB_SLAM3::Atlas(0);
  LOG(INFO) << "Atlas created!";
  // Create the transform map
  transform_map_ = std::make_shared<utilities::transform::TransformMap>();
  LOG(INFO) << "Transform map created!";

  cv::FileStorage fs;
  fs.open(repeat_config_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open vo model file: ";
  }

  // Extractor Parameters
  extractor_params_.num_features = fs["FeatureExtractor"]["num_features"];
  extractor_params_.num_levels = fs["FeatureExtractor"]["num_levels"];
  extractor_params_.scale_factor = fs["FeatureExtractor"]["scale_factor"];
  extractor_params_.initial_fast_threshold =
      fs["FeatureExtractor"]["intitial_fast_threshold"];
  extractor_params_.minimum_fast_threshold = fs["FeatureExtractor"
                                                ""]["minimum_fast_threshold"];
  extractor_params_.patch_size = fs["FeatureExtractor"]["patch_size"];
  extractor_params_.half_patch_size = fs["FeatureExtractor"]["half_patch_size"];
  extractor_params_.edge_threshold = fs["FeatureExtractor"]["edge_threshold"];

  // Camera params
  cv::Mat camera_mat;
  fs["FeatureSolver"]["CameraMatrix"] >> camera_mat;

  // Create the extractors
  mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);

  mpORBextractorRight = new ORB_SLAM3::ORBextractor(
      extractor_params_.num_features, extractor_params_.scale_factor,
      extractor_params_.num_levels, extractor_params_.initial_fast_threshold,
      extractor_params_.minimum_fast_threshold);

  LOG(INFO) << "Creating extractors";

  // TODO: Change hard coding of K
  float data[9] = {527.1638,  0., 338.73506, 0., 526.92709,
                   241.73964, 0., 0.,        1.};
  cv::Mat K = cv::Mat(3, 3, CV_32F, data);

  mK = K.clone();

  fx = 527.1638;
  fy = 526.92709;
  cx = 338.73506;
  cy = 241.73964;

  baseline_ = 0.12;
  mDistCoef = cv::Mat::zeros(4, 1, CV_32F);

  mbf = 63.24;
  mThDepth = 35;

  std::vector<float> vCamCalib{fx, fy, cx, cy};

  LOG(INFO) << "Camera creating";

  map_camera_ = new ORB_SLAM3::Pinhole(vCamCalib);
  LOG(INFO) << "Camera created!";

  // Create Visual Odometry
  vo_ = std::make_shared<aru::core::vo::VO>(repeat_config_file, vocab_file);
  
  // Set the Pose Chains
  teach_pose_chain_ = boost::make_shared<aru::core::utilities::transform::TransformSPtrVector>();
  repeat_pose_chain_ = boost::make_shared<aru::core::utilities::transform::TransformSPtrVector>();
}
//------------------------------------------------------------------------------
void Repeat::InitialiseMap(utilities::image::StereoImage image) {

  // Check for greyscale
  cv::Mat image_1_left_grey = image.first.GetImage().clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(image.first.GetImage().clone(), image_1_left_grey,
                 cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_1_right_grey = image.second.GetImage().clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(image.second.GetImage().clone(), image_1_right_grey,
                 cv::COLOR_BGR2GRAY);
  }

  ORB_SLAM3::Frame Frame_1 =
      ORB_SLAM3::Frame(image_1_left_grey, image_1_right_grey, 0,
                       mpORBextractorLeft, mpORBextractorRight, map_vocabulary_,
                       mK, mDistCoef, mbf, mThDepth, map_camera_);

  LOG(INFO) << "Number of features is " << Frame_1.N;

  Frame_1.SetPose(cv::Mat::eye(4, 4, CV_32F));

  // Create KeyFrame
  ORB_SLAM3::KeyFrame *pKFini = new ORB_SLAM3::KeyFrame(
      Frame_1, map_atlas_->GetCurrentMap(), map_key_frame_database_);

  // Compute Bag of Words Vector
  pKFini->ComputeBoW();
  // Insert KeyFrame in the map
  map_atlas_->AddKeyFrame(pKFini);

  // Create MapPoints and associate to KeyFrame
  for (int i = 0; i < Frame_1.N; i++) {
    float z = Frame_1.mvDepth[i];
    if (z > 0) {
      cv::Mat x3D = Frame_1.UnprojectStereo(i);
      auto *pNewMP =
          new ORB_SLAM3::MapPoint(x3D, pKFini, map_atlas_->GetCurrentMap());
      pNewMP->AddObservation(pKFini, i);
      pKFini->AddMapPoint(pNewMP, i);
      pNewMP->ComputeDistinctiveDescriptors();
      pNewMP->UpdateNormalAndDepth();
      map_atlas_->AddMapPoint(pNewMP);
      Frame_1.mvpMapPoints[i] = pNewMP;
    }
  }
  LOG(INFO) << "Number of map points is "
            << map_atlas_->GetCurrentMap()->MapPointsInMap();

  image_vector_.push_back(image);
}
//------------------------------------------------------------------------------
void Repeat::AddTeachKeyframe(utilities::image::StereoImage teach_keyframe) {
  // Find position of teach keyframe
  // pose is T_prev_curr. Source is curr_image dest is prev_image
  utilities::transform::TransformSPtr pose =
      transform_map_->Interpolate(teach_keyframe.first.GetTimeStamp());
  if (pose) {
    LOG(INFO) << "Added teach keyframe";
    // Check for greyscale
    cv::Mat image_1_left_grey = teach_keyframe.first.GetImage().clone();
    if (image_1_left_grey.channels() > 1) {
      cv::cvtColor(teach_keyframe.first.GetImage().clone(), image_1_left_grey,
                   cv::COLOR_BGR2GRAY);
    }
    cv::Mat image_1_right_grey = teach_keyframe.second.GetImage().clone();
    if (image_1_right_grey.channels() > 1) {
      cv::cvtColor(teach_keyframe.second.GetImage().clone(), image_1_right_grey,
                   cv::COLOR_BGR2GRAY);
    }

    ORB_SLAM3::Frame Frame_1 = ORB_SLAM3::Frame(
        image_1_left_grey, image_1_right_grey, 0, mpORBextractorLeft,
        mpORBextractorRight, map_vocabulary_, mK, mDistCoef, mbf, mThDepth,
        map_camera_);

    cv::Mat pose_cv;
    cv::eigen2cv(pose->GetTransform().matrix(), pose_cv);
    Frame_1.SetPose(pose_cv);

    // Create KeyFrame
    ORB_SLAM3::KeyFrame *key_frame_teach = new ORB_SLAM3::KeyFrame(
        Frame_1, map_atlas_->GetCurrentMap(), map_key_frame_database_);

    // Compute Bag of Words Vector
    key_frame_teach->ComputeBoW();
    // Insert KeyFrame in the map
    map_atlas_->AddKeyFrame(key_frame_teach);
    // Insert Keyframe in the database
    map_key_frame_database_->add(key_frame_teach);

    // Create MapPoints and associate to KeyFrame
    for (int i = 0; i < Frame_1.N; i++) {
      float z = Frame_1.mvDepth[i];

      if (z > 0) {
        cv::Mat x3D = Frame_1.UnprojectStereo(i);
        auto *pNewMP = new ORB_SLAM3::MapPoint(x3D, key_frame_teach,
                                               map_atlas_->GetCurrentMap());
        pNewMP->AddObservation(key_frame_teach, i);
        key_frame_teach->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        map_atlas_->AddMapPoint(pNewMP);
        Frame_1.mvpMapPoints[i] = pNewMP;
      }
    }
    image_vector_.push_back(teach_keyframe);
  }
}
//------------------------------------------------------------------------------
void Repeat::QueryRepeatframe(utilities::image::StereoImage repeat_frame) {

  // Check for greyscale
  cv::Mat image_1_left_grey = repeat_frame.first.GetImage().clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(repeat_frame.first.GetImage().clone(), image_1_left_grey,
                 cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_1_right_grey = repeat_frame.second.GetImage().clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(repeat_frame.second.GetImage().clone(), image_1_right_grey,
                 cv::COLOR_BGR2GRAY);
  }

  ORB_SLAM3::Frame Frame_curr =
      ORB_SLAM3::Frame(image_1_left_grey, image_1_right_grey, 0,
                       mpORBextractorLeft, mpORBextractorRight, map_vocabulary_,
                       mK, mDistCoef, mbf, mThDepth, map_camera_);

  Frame_curr.ComputeBoW();

  // Query Keyframe Database for keyframe candidates for relocalisation
  std::vector<ORB_SLAM3::KeyFrame *> vpCandidateKFs =
      map_key_frame_database_->DetectRelocalizationCandidates(
          &Frame_curr, map_atlas_->GetCurrentMap());

  const int nKFs = vpCandidateKFs.size();

  LOG(INFO) << "Number of potential keyframes is " << nKFs;

  // Perform ORB matching with each candidate to reduce the numbers
  ORB_SLAM3::ORBmatcher matcher(0.75, true);
  vector<vector<ORB_SLAM3::MapPoint *>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);


  int nCandidates = 0;
  int nbestmatches = -1;
  int bestindex = 0;

  for (int i = 0; i < nKFs; i++) {
    ORB_SLAM3::KeyFrame *pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      int nmatches =
          matcher.SearchByBoW(pKF, Frame_curr, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        // Do a pose estimation with matches
        //auto curr_features =
        //    FeaturesFromMapPoints(pKF, &Frame_curr, vvpMapPointMatches[i]);
        //auto pose_diff = vo_->EstimateMotion(curr_features);
        //LOG(INFO) << "Transform ORB is " << pose_diff.GetTransform().matrix();
        LOG(INFO)<<"Number of matches is "<<nmatches;
        auto pose_viso = vo_->EstimateMotion(image_vector_[pKF->mnId],repeat_frame);
        LOG(INFO) << "Transform Viso is " << pose_viso.GetTransform().matrix();

        if (vo_->NumInliers() > nbestmatches) {
          nbestmatches = vo_->NumInliers();
          bestindex = i;
        }
        nCandidates++;
      }
    }
  }
  LOG(INFO) << "Number of candidates after matching is " << nCandidates;

  if (!vpCandidateKFs.empty() && nCandidates > 0) {
    ORB_SLAM3::KeyFrame *pKF = vpCandidateKFs[bestindex];
    LOG(INFO) << "Best candidate is " << pKF->mnId;

    
    auto pose_viso = vo_->EstimateMotion(image_vector_[pKF->mnId],repeat_frame);
    LOG(INFO) << "Transform is " << pose_viso.GetTransform().matrix();
    repeat_pose_chain_->clear();
    auto pose = transform_map_->Interpolate(image_vector_[pKF->mnId].first.GetTimeStamp());   
    pose->GetTransform()=pose_viso.GetTransform()*pose->GetTransform();
    repeat_pose_chain_->push_back(pose);
    
    

    cv::Mat image_cat;
    cv::hconcat(image_vector_[pKF->mnId].first.GetImage(),
                repeat_frame.first.GetImage(), image_cat);
    cv::resize(image_cat, image_cat, cv::Size(), 0.5, 0.5);
    cv::imshow("Localiser output", image_cat);
    cv::waitKey(15);
  }
  LOG(INFO) << "Queried frame";
}
//------------------------------------------------------------------------------
aru::core::utilities::image::FeatureSPtrVectorSptr
Repeat::FeaturesFromMapPoints(ORB_SLAM3::KeyFrame *key_frame,
                              ORB_SLAM3::Frame *frame,
                              std::vector<ORB_SLAM3::MapPoint *> map_matches) {

  utilities::image::FeatureSPtrVectorSptr curr_features =
      boost::make_shared<utilities::image::FeatureSPtrVector>();
  for (size_t i = 0, iend = map_matches.size(); i < iend; i++) {
    ORB_SLAM3::MapPoint *pMP = map_matches[i];

    if (pMP) {
      if (!pMP->isBad()) {
        if (i >= frame->mvKeysUn.size())
          continue;
        const cv::KeyPoint &kp = frame->mvKeysUn[i];
        // 3D coordinates
        cv::Mat cv_pos = pMP->GetWorldPos();
        auto observation = pMP->GetObservations()[key_frame];
        cv::KeyPoint kp_prev = key_frame->mvKeysUn[std::get<0>(observation)];
        double depth = frame->mvDepth[std::get<0>(observation)];
        double disp = mbf / depth;
        cv::KeyPoint kp_right =
            cv::KeyPoint(kp_prev.pt.x - disp, kp_prev.pt.y, 1);

        utilities::image::FeatureSPtr curr_feature =
            boost::make_shared<utilities::image::Feature>(
                Eigen::Vector2d(kp_prev.pt.x, kp_prev.pt.y));
        curr_feature->SetKeyPoint(kp_prev);
        curr_feature->SetMatchedKeypoint(kp_right);
        curr_feature->SetSequentialKeyPoint(kp);
        curr_features->push_back(curr_feature);
      }
    }
  }
  return curr_features;
}
//------------------------------------------------------------------------------
void Repeat::AddTeachTransform(
    utilities::transform::TransformSPtr teach_transform) {

  LOG(INFO) << "Time transform is " << teach_transform->GetSourceTimestamp();
  transform_map_->AddTransform(teach_transform);
  auto pose = transform_map_->Interpolate(teach_transform->GetSourceTimestamp());
  if(pose) teach_pose_chain_->push_back(pose);
  
}
} // namespace repeat
} // namespace core
} // namespace aru
