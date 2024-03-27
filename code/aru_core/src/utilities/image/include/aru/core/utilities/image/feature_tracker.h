//
// Created by paulamayo on 2021/08/31.
//

#ifndef ARU_CORE_FEATURE_TRACKER_H
#define ARU_CORE_FEATURE_TRACKER_H

#include "feature_matcher.h"
#include <viso.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/mat.hpp>

#define TERMINATED_TRACKLET 1e7
namespace aru {
namespace core {
namespace utilities {
namespace image {

struct FeatureTrack {
  FeatureSPtrVectorSptr feature_track_;
  FeatureSPtr curr_feature_;
  // std::vector<int> frame_track_;
  boost::shared_ptr<std::vector<int>> frame_track_;
  boost::shared_ptr<std::vector<Matcher::p_match>> match_track_;
  int previous_index;

  FeatureTrack() {
    feature_track_ = boost::make_shared<FeatureSPtrVector>();
    curr_feature_ = boost::make_shared<Feature>();
    frame_track_ = boost::make_shared<std::vector<int>>();
    match_track_ = boost::make_shared<std::vector<Matcher ::p_match>>();
  }
};

class OrbFeatureTracker {
public:
  OrbFeatureTracker(MatcherParams matcher_params,
                    ExtractorParams extractor_params);

  void InitialiseFeatures(const cv::Mat &image_dest_left,
                          const cv::Mat &image_dest_right);

  void UpdateFeatures(const cv::Mat &image_left, const cv::Mat &image_right);

  FeatureSPtrVectorSptr GetMatchedFeatures() { return matched_features_; }

  FeatureSPtrVectorSptr GetCurrentFeatures() { return curr_features_; }
  ~OrbFeatureTracker() = default;

private:
  MatcherParams matcher_params_;
  boost::shared_ptr<OrbFeatureMatcher> feature_matcher_;

  FeatureSPtrVectorSptr curr_features_;
  FeatureSPtrVectorSptr matched_features_;
  cv::Mat prev_image_left;
};


class VisoFeatureTracker {
public:
  VisoFeatureTracker();

  VisoFeatureTracker(MatcherParams matcher_params,
                     ExtractorParams extractor_params);

  void InitialiseFeatures(const cv::Mat &image_dest_left,
                          const cv::Mat &image_dest_right,
                          const cv::Mat &image_src_left,
                          const cv::Mat &image_src_right);

  void FeaturesUntracked(const cv::Mat &image_left, const cv::Mat &image_right);

  void UpdateFeatures(const cv::Mat &image_left, const cv::Mat &image_right);

  FeatureSPtrVectorSptr GetMatchedFeatures() { return matched_features_; }

  void UpdateTracks(int min_track_length);

  std::vector<FeatureTrack> GetAllTracks();

  std::vector<FeatureTrack> GetActiveTracks() { return active_tracks_; };

  std::vector<FeatureTrack> GetWindowActiveTracks(int min_track_window);
  FeatureSPtrVectorSptr GetWindowActiveFeatures(int min_track_window);

  FeatureSPtrVectorSptr GetCurrentFeatures() { return curr_features_; }
  // my own function that I am addig to access features that appear 
  // in an image and are already being tracked
  // these features are added to factor graph only not Values
  FeatureSPtrVectorSptr GetTrackedFeatures() { return tracked_features_; }  
  std::vector<int> GetTrackedIDs(){return landmark_Ids;}
  FeatureSPtrVectorSptr GetActiveFeatures();
  ~VisoFeatureTracker() = default;

private:
  MatcherParams matcher_params_;
  boost::shared_ptr<Matcher> viso_extractor_;

  FeatureSPtrVectorSptr curr_features_;
  FeatureSPtrVectorSptr matched_features_;
  // My owner private variable  that I am adding
  // It stores feautures that are akready being tracked from the new image
  FeatureSPtrVectorSptr tracked_features_;
  // store IDs of the features that are already existing
  std::vector<int> landmark_Ids;
  
  cv::Mat prev_image_left;
  std::vector<FeatureTrack> active_tracks_;
  std::vector<int> active_track_indexes_;
  std::vector<FeatureTrack> terminated_tracks_;
  int frame_index_;
};

} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_CORE_FEATURE_TRACKER_H
