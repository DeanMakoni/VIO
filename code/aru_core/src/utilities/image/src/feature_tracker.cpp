#include <aru/core/utilities/image/feature_tracker.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/features2d.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace image {

using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
OrbFeatureTracker::OrbFeatureTracker(MatcherParams matcher_params,
                                     ExtractorParams extractor_params)
    : matcher_params_(matcher_params) {

  // Create the feature vectors
  curr_features_ = boost::make_shared<FeatureSPtrVector>();
  matched_features_ = boost::make_shared<FeatureSPtrVector>();
  // Create feature matcher
  feature_matcher_ = boost::make_shared<OrbFeatureMatcher>(
      matcher_params, extractor_params,
      "/home/paulamayo/data/husky_data/vocabulary/ORBvoc.txt");
}
//------------------------------------------------------------------------------
void OrbFeatureTracker::InitialiseFeatures(const cv::Mat &image_left,
                                           const cv::Mat &image_right) {
  curr_features_ =
      feature_matcher_->ComputeStereoMatches(image_left, image_right);
  prev_image_left = image_left;
}
//------------------------------------------------------------------------------
void OrbFeatureTracker::UpdateFeatures(const cv::Mat &image_left,
                                       const cv::Mat &image_right) {

  matched_features_ =
      feature_matcher_->ComputeStereoMatches(prev_image_left, image_left);

  curr_features_ =
      feature_matcher_->ComputeStereoMatches(image_left, image_right);

  prev_image_left = image_left;
}

//------------------------------------------------------------------------------
VisoFeatureTracker::VisoFeatureTracker() {

  // Create the feature vectors
  curr_features_ = boost::make_shared<FeatureSPtrVector>();
  tracked_features_ = boost::make_shared<FeatureSPtrVector>();
  matched_features_ = boost::make_shared<FeatureSPtrVector>();
  // Create feature matcher
  viso_extractor_ = boost::make_shared<Matcher>();
}

//------------------------------------------------------------------------------
VisoFeatureTracker::VisoFeatureTracker(MatcherParams matcher_params,
                                       ExtractorParams extractor_params)
    : matcher_params_(matcher_params) {

  // Create the feature vectors
  curr_features_ = boost::make_shared<FeatureSPtrVector>();
  tracked_features_ = boost::make_shared<FeatureSPtrVector>();
  matched_features_ = boost::make_shared<FeatureSPtrVector>();
  // Create feature matcher
  viso_extractor_ = boost::make_shared<Matcher>();
}
//------------------------------------------------------------------------------
void VisoFeatureTracker::InitialiseFeatures(const cv::Mat &image_dest_left,
                                            const cv::Mat &image_dest_right,
                                            const cv::Mat &image_src_left,
                                            const cv::Mat &image_src_right) {

  int32_t width = image_dest_left.cols;
  int32_t height = image_dest_left.rows;

  // compute visual odometry
  int32_t dims[] = {width, height, width};
  prev_image_left = image_dest_left.clone();

  viso_extractor_->pushBack(image_dest_left.data, image_dest_right.data, dims,
                            false);
  viso_extractor_->matchFeatures(2);

  viso_extractor_->pushBack(image_src_left.data, image_src_right.data, dims,
                            false);
  viso_extractor_->matchFeatures(2);

  std::vector<Matcher::p_match> p_matched = viso_extractor_->getMatches();

  active_tracks_.clear();
  terminated_tracks_.clear();
  frame_index_ = 0;

  curr_features_->clear();
  FeatureSPtrVectorSptr tracked_features =
      boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr new_features = boost::make_shared<FeatureSPtrVector>();
  for (auto match : p_matched) {
    float disp_p = match.u1p - match.u2p;
    float disp_c = match.u1c - match.u2c;
    if (disp_p > 0 && disp_c > 0) {
      FeatureSPtr dest_feature =
          boost::make_shared<Feature>(Eigen::Vector2d(match.u1p, match.v1p));
      dest_feature->SetKeyPoint(KeyPoint(match.u1p, match.v1p, 1));
      dest_feature->SetMatchedKeypoint(KeyPoint(match.u2p, match.v2p, 1));

      FeatureSPtr src_feature =
          boost::make_shared<Feature>(Eigen::Vector2d(match.u1c, match.v1c));
      src_feature->SetKeyPoint(KeyPoint(match.u1c, match.v1c, 1));
      src_feature->SetMatchedKeypoint(KeyPoint(match.u2c, match.v2c, 1));
      // Create new tracklets

      FeatureTrack new_track;
      new_track.feature_track_->push_back(dest_feature);
      new_track.feature_track_->push_back(src_feature);
      new_track.frame_track_->push_back(0);
      new_track.frame_track_->push_back(1);
      new_track.match_track_->push_back(match);
      new_track.match_track_->push_back(match);
      new_track.curr_feature_ = src_feature;
      new_track.previous_index = match.i1c;

      active_tracks_.push_back(new_track);

      curr_features_->push_back(src_feature);
    }
  }
  int min_track_length = 4;
  frame_index_ = 1;
  UpdateTracks(min_track_length);
  frame_index_ = 2;
  LOG(INFO) << "Number of active tracks is " << active_tracks_.size();
  matched_features_ = curr_features_;
}
//------------------------------------------------------------------------------
void VisoFeatureTracker::FeaturesUntracked(const cv::Mat &image_left,
                                           const cv::Mat &image_right) {

  int32_t width = image_left.cols;
  int32_t height = image_left.rows;

  // compute visual odometry
  int32_t dims[] = {width, height, width};
  auto estimation_start = std::chrono::high_resolution_clock::now();
  viso_extractor_->pushBack(image_left.data, image_right.data, dims, false);
  viso_extractor_->matchFeatures(2);

  auto estimation_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = estimation_end - estimation_start;
  VLOG(2) << "Feature extraction runs at " << 1 / elapsed.count() << " Hz";
  viso_extractor_->bucketFeatures(1, 15, 15);

  std::vector<Matcher::p_match> p_matched = viso_extractor_->getMatches();

  VLOG(2) << "Number of features is " << p_matched.size();

  curr_features_->clear();
  FeatureSPtrVectorSptr tracked_features =
      boost::make_shared<FeatureSPtrVector>();
  FeatureSPtrVectorSptr new_features = boost::make_shared<FeatureSPtrVector>();
  for (auto match : p_matched) {
    FeatureSPtr curr_feature =
        boost::make_shared<Feature>(Eigen::Vector2d(match.u1c, match.v1c));
    curr_feature->SetKeyPoint(KeyPoint(match.u1c, match.v1c, 1));
    curr_feature->SetMatchedKeypoint(KeyPoint(match.u2c, match.v2c, 1));
    curr_feature->SetSequentialKeyPoint(KeyPoint(match.u1p, match.v1p, 1));
    if (match.u1c - match.u2c > 0)
      curr_features_->push_back(curr_feature);
  }
  prev_image_left = image_left;
}

//------------------------------------------------------------------------------
void VisoFeatureTracker::UpdateFeatures(const cv::Mat &image_left,
                                        const cv::Mat &image_right) {

  int32_t width = image_left.cols;
  int32_t height = image_left.rows;

  cv::Mat curr_image = image_left.clone();
   
  // compute visual odometry
  int32_t dims[] = {width, height, width};

  viso_extractor_->pushBack(image_left.data, image_right.data, dims, false);
  
  viso_extractor_->matchFeatures(2);
  // viso_extractor_->bucketFeatures(2, 15, 15);

  std::vector<Matcher::p_match> p_matched = viso_extractor_->getMatches();
  
  std::cout << "Viso test 1" << std::endl;
  curr_features_->clear();
   std::cout << "Viso test 2" << std::endl;
  tracked_features_->clear();
   std::cout << "Viso test 3" << std::endl;
  landmark_Ids.clear();
  
  FeatureSPtrVectorSptr tracked_features =
      boost::make_shared<FeatureSPtrVector>();
  std::cout << "Viso test 2" << std::endl;
  FeatureSPtrVectorSptr new_features = boost::make_shared<FeatureSPtrVector>();
  
  for (auto match : p_matched) {
    float disp = match.u1c - match.u2c;
    if (disp > 0) {
      FeatureSPtr curr_feature =
          boost::make_shared<Feature>(Eigen::Vector2d(match.u1c, match.v1c));
      curr_feature->SetKeyPoint(KeyPoint(match.u1c, match.v1c, 1));
      curr_feature->SetMatchedKeypoint(KeyPoint(match.u2c, match.v2c, 1));
      curr_feature->SetSequentialKeyPoint(KeyPoint(match.u1p, match.v1p, 1));
      // Disparity check
      // Create new tracklets

      // Start with a search for tracks in the previous frame
      auto it = find(active_track_indexes_.begin(), active_track_indexes_.end(),
                     match.i1p);
      if (it != active_track_indexes_.end()) {
        int index = it - active_track_indexes_.begin();
        active_tracks_[index].feature_track_->push_back(curr_feature);
        active_tracks_[index].match_track_->push_back(match);
        active_tracks_[index].curr_feature_ = curr_feature;
        active_tracks_[index].frame_track_->push_back(frame_index_);
        active_tracks_[index].previous_index = match.i1c;
        tracked_features->push_back(curr_feature);
        tracked_features_->push_back(curr_feature);
        landmark_Ids.push_back(index);
      } else {
        FeatureTrack new_track;
        new_track.feature_track_->push_back(curr_feature);
        new_track.match_track_->push_back(match);
        new_track.curr_feature_ = curr_feature;
        new_track.previous_index = match.i1c;
        new_track.frame_track_->push_back(frame_index_);
        active_tracks_.push_back(new_track);
        new_features->push_back(curr_feature);
      }

      curr_features_->push_back(curr_feature);
    }
  }

  cv::cvtColor(prev_image_left, curr_image, cv::COLOR_GRAY2BGR);
  for (const auto &feat : *tracked_features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(curr_image, keypoint.pt, 1, cv::Scalar(0, 255, 0), -1,
               cv::FILLED);
  }
  for (const auto &feat : *new_features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(curr_image, keypoint.pt, 1, cv::Scalar(255, 0, 0), -1,
               cv::FILLED);
  }
  cv::imshow("Features obtained", curr_image);
  cv::waitKey(15);

  LOG(INFO) << "Number of active tracks is " << active_tracks_.size();
  matched_features_ = curr_features_;
  int min_track_length = 3;
  UpdateTracks(min_track_length);
  LOG(INFO) << "Number of terminated tracks is " << terminated_tracks_.size();
  prev_image_left = image_left.clone();
  frame_index_++;
}

//------------------------------------------------------------------------------
void VisoFeatureTracker::UpdateTracks(int min_track_length) {

  active_track_indexes_.clear();
  std::vector<FeatureTrack> tracked_features;
  for (const auto &tracks : active_tracks_) {
    if (tracks.frame_track_->back() == frame_index_) {
      tracked_features.push_back(tracks);
      active_track_indexes_.push_back(tracks.match_track_->back().i1c);
    } else {
      // Only retain tracks that terminate with a minimum length
      if (tracks.frame_track_->size() > min_track_length) {
        terminated_tracks_.push_back(tracks);
      }
    }
  }
  active_tracks_ = tracked_features;
}
//------------------------------------------------------------------------------
FeatureSPtrVectorSptr VisoFeatureTracker::GetActiveFeatures() {
  FeatureSPtrVectorSptr active_features =
      boost::make_shared<FeatureSPtrVector>();
  for (const auto &tracks : active_tracks_) {

    FeatureSPtr curr_feature = boost::make_shared<Feature>(Eigen::Vector2d(
        tracks.match_track_->back().u1c, tracks.match_track_->back().v1c));
    curr_feature->SetKeyPoint(KeyPoint(tracks.match_track_->back().u1c,
                                       tracks.match_track_->back().v1c, 1));
    curr_feature->SetMatchedKeypoint(KeyPoint(
        tracks.match_track_->back().u2c, tracks.match_track_->back().v2c, 1));
    curr_feature->SetSequentialKeyPoint(KeyPoint(
        tracks.match_track_->back().u1p, tracks.match_track_->back().v1p, 1));
    active_features->push_back(curr_feature);
  }
  return active_features;
}
//------------------------------------------------------------------------------
FeatureSPtrVectorSptr
VisoFeatureTracker::GetWindowActiveFeatures(int min_track_window) {
  FeatureSPtrVectorSptr active_features =
      boost::make_shared<FeatureSPtrVector>();
  for (const auto &tracks : active_tracks_) {
    if (tracks.frame_track_->back() == frame_index_ - 1 &&
        tracks.frame_track_->size() > min_track_window) {

      FeatureSPtr curr_feature = boost::make_shared<Feature>(Eigen::Vector2d(
          tracks.match_track_->back().u1c, tracks.match_track_->back().v1c));
      curr_feature->SetKeyPoint(KeyPoint(tracks.match_track_->back().u1c,
                                         tracks.match_track_->back().v1c, 1));
      curr_feature->SetMatchedKeypoint(KeyPoint(
          tracks.match_track_->back().u2c, tracks.match_track_->back().v2c, 1));
      curr_feature->SetSequentialKeyPoint(KeyPoint(
          tracks.match_track_->back().u1p, tracks.match_track_->back().v1p, 1));
      active_features->push_back(curr_feature);
    }
  }
  return active_features;
}
//------------------------------------------------------------------------------
std::vector<FeatureTrack> VisoFeatureTracker::GetAllTracks() {
  std::vector<FeatureTrack> all_tracks = terminated_tracks_;
  all_tracks.insert(all_tracks.end(), active_tracks_.begin(),
                    active_tracks_.end());
  return all_tracks;
}
//------------------------------------------------------------------------------
std::vector<FeatureTrack>
VisoFeatureTracker::GetWindowActiveTracks(int min_track_window) {

  std::vector<FeatureTrack> feature_triplet;
  for (const auto &tracks : active_tracks_) {
    if (tracks.frame_track_->back() == frame_index_ - 1 &&
        tracks.frame_track_->size() > min_track_window) {
      feature_triplet.push_back(tracks);
    }
  }
  return feature_triplet;
}
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru
