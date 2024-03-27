//
// Created by paulamayo on 2022/08/14.
//

#ifndef ARU_CORE_LOOP_CLOSER_H
#define ARU_CORE_LOOP_CLOSER_H

#include <DBoW2/DBoW2.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

namespace aru {
namespace core {
namespace localisation {

typedef cv::Mat OrbDescriptor;
typedef std::vector<OrbDescriptor> OrbDescriptorVec;

using FrameId = std::uint64_t;

struct LCDFrame {
  LCDFrame() {}
  LCDFrame(const FrameId &id, const std::vector<cv::KeyPoint> &keypoints,
           const OrbDescriptorVec &descriptors_vec,
           const OrbDescriptor &descriptors_mat)
      : id_(id), keypoints_(keypoints), descriptors_vec_(descriptors_vec),
        descriptors_mat_(descriptors_mat) {}

  FrameId id_;
  std::vector<cv::KeyPoint> keypoints_;
  OrbDescriptorVec descriptors_vec_;
  OrbDescriptor descriptors_mat_;
}; // struct LCDFrame

struct MatchIsland {
  MatchIsland()
      : start_id_(0),
        end_id_(0),
        island_score_(0),
        best_id_(0),
        best_score_(0) {}

  MatchIsland(const FrameId& start, const FrameId& end)
      : start_id_(start),
        end_id_(end),
        island_score_(0),
        best_id_(0),
        best_score_(0) {}

  MatchIsland(const FrameId& start, const FrameId& end, const double& score)
      : start_id_(start),
        end_id_(end),
        island_score_(score),
        best_id_(0),
        best_score_(0) {}

  inline bool operator<(const MatchIsland& other) const {
    return island_score_ < other.island_score_;
  }

  inline bool operator>(const MatchIsland& other) const {
    return island_score_ > other.island_score_;
  }

  inline size_t size() const { return end_id_ - start_id_ + 1; }

  inline void clear() {
    start_id_ = 0;
    end_id_ = 0;
    island_score_ = 0;
    best_id_ = 0;
    best_score_ = 0;
  }

  FrameId start_id_;
  FrameId end_id_;
  double island_score_;
  FrameId best_id_;
  double best_score_;
};  // struct MatchIsland

class LoopCloser {

public:
  LoopCloser(const std::string &vocabulary_file,
             const std::string loop_closure_settings_file);

  ~LoopCloser() = default;

  int AddKeyFrame(const cv::Mat &image_left, const cv::Mat &image_right);

  bool LoopDetect(const cv::Mat &image_left, const cv::Mat &image_right);

  bool checkTemporalConstraint(const FrameId& id, const MatchIsland& island);

  /* ------------------------------------------------------------------------ */
  /** @brief Computes the various islands created by a QueryResult, which is
   *  given by the OrbDatabase.
   * @param[in] q A QueryResults object containing all the resulting possible
   *  matches with a frame.
   * @param[out] A vector of MatchIslands, each of which is an island of
   *  nearby possible matches with the frame being queried.
   */
  // TODO(marcus): unit tests
  void computeIslands(DBoW2::QueryResults* q,
                      std::vector<MatchIsland>* islands) const;

private:
  /* ------------------------------------------------------------------------ */
  /** @brief Compute the overall score of an island.
   * @param[in] q A QueryResults object containing all the possible matches
   *  with a frame.
   * @param[in] start_id The frame ID that starts the island.
   * @param[in] end_id The frame ID that ends the island.
   * @reutrn The score of the island.
   */
  double computeIslandScore(const DBoW2::QueryResults& q,
                            const FrameId& start_id,
                            const FrameId& end_id) const;

private:
  // ORB extraction and matching members
  cv::Ptr<cv::ORB> orb_feature_detector_;
  cv::Ptr<cv::DescriptorMatcher> orb_feature_matcher_;

  // BoW and Loop Detection database and members
  std::unique_ptr<OrbDatabase> db_BoW_;
  std::vector<LCDFrame> db_frames_;
  std::string vocab_filename_;

  DBoW2::BowVector latest_bowvec_;
  int frame_id_;


  int temporal_entries_;
  MatchIsland latest_matched_island_;
  FrameId latest_query_id_;
};
} // namespace localisation
} // namespace core
} // namespace aru

#endif // ARU_CORE_LOOP_CLOSER_H
