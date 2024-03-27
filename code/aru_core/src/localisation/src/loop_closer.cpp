//
// Created by paulamayo on 2022/08/14.
//

#include "aru/core/localisation/loop_closer.h"
#include <boost/make_shared.hpp>
#include <opencv2/features2d.hpp>

namespace aru {
namespace core {
namespace localisation {
//------------------------------------------------------------------------------
LoopCloser::LoopCloser(const std::string &vocabulary_file,
                       const std::string loop_closure_settings_file)
    : db_frames_(), latest_bowvec_() {

  // Initialize the ORB feature detector object:
 // orb_feature_detector_ = cv::ORB::create(1000, 1.2, 8, 31, 0, 2, 0, 31, 20);

  // Initialize our feature matching object:
  //orb_feature_matcher_ = cv::DescriptorMatcher::create(3);

  // Load ORB vocabulary:

  OrbVocabulary vocab;
  LOG(INFO) << "LoopClosureDetector:: Loading vocabulary from "
            << vocabulary_file;
  vocab.load(vocabulary_file);
  LOG(INFO) << "Loaded vocabulary with " << vocab.size() << " visual words.";

  // Initialize db_BoW_:
  db_BoW_ = std::make_unique<OrbDatabase>(vocab);

  frame_id_ = 0;
}
//------------------------------------------------------------------------------
int LoopCloser::AddKeyFrame(const cv::Mat &image_left,
                            const cv::Mat &image_right) {

  // Check for greyscale
  cv::Mat image_left_grey = image_left.clone();
  if (image_left_grey.channels() > 1) {
    cv::cvtColor(image_left, image_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_right_grey = image_right.clone();
  if (image_right_grey.channels() > 1) {
    cv::cvtColor(image_right, image_right_grey, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::KeyPoint> keypoints;
  OrbDescriptor descriptors_mat;
  OrbDescriptorVec descriptors_vec;

  // Extract ORB features and construct descriptors_vec.
  orb_feature_detector_->detectAndCompute(image_left, cv::Mat(), keypoints,
                                          descriptors_mat);

  int L = orb_feature_detector_->descriptorSize();
  descriptors_vec.resize(descriptors_mat.size().height);

  for (size_t i = 0; i < descriptors_vec.size(); i++) {
    descriptors_vec[i] = cv::Mat(1, L, descriptors_mat.type()); // one row only
    descriptors_mat.row(i).copyTo(descriptors_vec[i].row(0));
  }
  // Build and store LCDFrame object.
  db_frames_.push_back(
      LCDFrame(db_frames_.size(), keypoints, descriptors_vec, descriptors_mat));
  CHECK(!db_frames_.empty());
  return db_frames_.back().id_;
}
//------------------------------------------------------------------------------
bool LoopCloser::LoopDetect(const cv::Mat &image_left,
                            const cv::Mat &image_right) {
  FrameId frame_id = AddKeyFrame(image_left, image_right);

  // Create BOW representation of descriptors.
  DBoW2::BowVector bow_vec;
  DCHECK(db_BoW_);
  OrbDescriptorVec descriptors_vec = db_frames_[frame_id].descriptors_vec_;
  db_BoW_->getVocabulary()->transform(descriptors_vec, bow_vec);

  int recent_frames_window = 500;
  int max_db_results = 50;
  float min_nss_factor = 0.05;
  int min_matches_per_island = 1;
  int max_intraisland_gap = 3;
  int max_nrFrames_between_islands = 3;
  int max_nrFrames_between_queries = 2;

  int max_possible_match_id = frame_id - recent_frames_window;
  if (max_possible_match_id < 0)
    max_possible_match_id = 0;

  // Query for BoW vector matches in database.
  DBoW2::QueryResults query_result;
  db_BoW_->query(bow_vec, query_result, max_db_results, max_possible_match_id);

  // Add current BoW vector to database.
  db_BoW_->add(bow_vec);

  if (query_result.empty()) {
    LOG(INFO) << "Query empty";
  } else {
    double nss_factor = 1.0;
    nss_factor = db_BoW_->getVocabulary()->score(bow_vec, latest_bowvec_);

    if (nss_factor < min_nss_factor) {
      LOG(INFO)<<"Nss fail";
    } else {
      //      FrameId match_id_ = query_result[0].Id;
      //      LOG(INFO) << "Match id of frame " << frame_id << " is " <<
      //      match_id_
      //                << " before nss";
      float alpha = 0.1;
      // Remove low scores from the QueryResults based on nss.
      DBoW2::QueryResults::iterator query_it =
          lower_bound(query_result.begin(), query_result.end(),
                      DBoW2::Result(0, alpha * nss_factor), DBoW2::Result::geq);
      if (query_it != query_result.end()) {
        query_result.resize(query_it - query_result.begin());
      }

      // Begin grouping and checking matches.
      if (query_result.empty()) {
        LOG(INFO) << "Grouping fail";
      } else {
        // Set best candidate to highest scorer.
        FrameId match_id_ = query_result[0].Id;
        std::vector<MatchIsland> islands;
        computeIslands(&query_result, &islands);

        if (islands.empty()) {
          LOG(INFO) << "Island fail";
        } else {
          // Find the best island grouping using MatchIsland sorting.
          const MatchIsland &best_island =
              *std::max_element(islands.begin(), islands.end());

          // Run temporal constraint check on this best island.
          bool pass_temporal_constraint =
              checkTemporalConstraint(frame_id, best_island);

          if (!pass_temporal_constraint) {
            LOG(INFO) << "Temporal fail";
          } else {
            LOG(INFO) << "Match id of frame " << frame_id << " is "
                      << match_id_;
            // cv::waitKey(0);
          }
        }
      }
    }
  }

  // Update latest bowvec for normalized similarity scoring (NSS).
  if (static_cast<int>(frame_id + 1) > recent_frames_window) {
    latest_bowvec_ = bow_vec;
  } else {
    LOG(INFO) << "LoopClosureDetector: Not enough frames processed.";
  }
  return true;
}

/* ------------------------------------------------------------------------ */
bool LoopCloser::checkTemporalConstraint(const FrameId &id,
                                         const MatchIsland &island) {

  int max_nrFrames_between_queries = 2;
  // temporal_entries_ starts at zero and counts the number of
  if (temporal_entries_ == 0 ||
      static_cast<int>(id - latest_query_id_) > max_nrFrames_between_queries) {
    temporal_entries_ = 1;
  } else {
    int a1 = static_cast<int>(latest_matched_island_.start_id_);
    int a2 = static_cast<int>(latest_matched_island_.end_id_);
    int b1 = static_cast<int>(island.start_id_);
    int b2 = static_cast<int>(island.end_id_);

    // Check that segments (a1, a2) and (b1, b2) have some overlap
    bool overlap = (b1 <= a1 && a1 <= b2) || (a1 <= b1 && b1 <= a2);
    bool gap_is_small = false;
    if (!overlap) {
      // Compute gap between segments (one of the two is negative)
      int d1 = static_cast<int>(latest_matched_island_.start_id_) -
               static_cast<int>(island.end_id_);
      int d2 = static_cast<int>(island.start_id_) -
               static_cast<int>(latest_matched_island_.end_id_);

      int gap = (d1 > d2 ? d1 : d2); // Choose positive gap

      int max_nrFrames_between_islands = 3;
      gap_is_small = gap <= max_nrFrames_between_islands;
    }

    if (overlap || gap_is_small) {
      temporal_entries_++;
    } else {
      temporal_entries_ = 1;
    }
  }

  latest_matched_island_ = island;
  latest_query_id_ = id;
  int min_temporal_matches_ = 3;
  return temporal_entries_ > min_temporal_matches_;
}

/* ------------------------------------------------------------------------ */
void LoopCloser::computeIslands(DBoW2::QueryResults *q,
                                std::vector<MatchIsland> *islands) const {
  CHECK_NOTNULL(q);
  CHECK_NOTNULL(islands);
  islands->clear();

  int min_matches_per_island = 1;
  int max_intraisland_gap = 3;

  // The case of one island is easy to compute and is done separately
  if (q->size() == 1) {
    const DBoW2::Result &result = (*q)[0];
    const DBoW2::EntryId &result_id = result.Id;
    MatchIsland island(result_id, result_id, result.Score);
    island.best_id_ = result_id;
    island.best_score_ = result.Score;
    islands->push_back(island);
  } else if (!q->empty()) {
    // sort query results in ascending order of frame ids
    std::sort(q->begin(), q->end(), DBoW2::Result::ltId);

    // create long enough islands
    DBoW2::QueryResults::const_iterator dit = q->begin();
    int first_island_entry = static_cast<int>(dit->Id);
    int last_island_entry = static_cast<int>(dit->Id);

    // these are indices of q
    FrameId i_first = 0;
    FrameId i_last = 0;

    double best_score = dit->Score;
    DBoW2::EntryId best_entry = dit->Id;

    ++dit;
    for (FrameId idx = 1; dit != q->end(); ++dit, ++idx) {
      if (static_cast<int>(dit->Id) - last_island_entry < max_intraisland_gap) {
        last_island_entry = dit->Id;
        i_last = idx;
        if (dit->Score > best_score) {
          best_score = dit->Score;
          best_entry = dit->Id;
        }
      } else {
        // end of island reached
        int length = last_island_entry - first_island_entry + 1;
        if (length >= min_matches_per_island) {
          MatchIsland island =
              MatchIsland(first_island_entry, last_island_entry,
                          computeIslandScore(*q, i_first, i_last));

          islands->push_back(island);
          islands->back().best_score_ = best_score;
          islands->back().best_id_ = best_entry;
        }

        // prepare next island
        first_island_entry = last_island_entry = dit->Id;
        i_first = i_last = idx;
        best_score = dit->Score;
        best_entry = dit->Id;
      }
    }
    // add last island
    // TODO: do we need this? why isn't it handled in prev for loop?
    if (last_island_entry - first_island_entry + 1 >= min_matches_per_island) {
      MatchIsland island = MatchIsland(first_island_entry, last_island_entry,
                                       computeIslandScore(*q, i_first, i_last));

      islands->push_back(island);
      islands->back().best_score_ = best_score;
      islands->back().best_id_ = static_cast<FrameId>(best_entry);
    }
  }
}

/* ------------------------------------------------------------------------ */
double LoopCloser::computeIslandScore(const DBoW2::QueryResults &q,
                                      const FrameId &start_id,
                                      const FrameId &end_id) const {
  CHECK_GT(q.size(), start_id);
  CHECK_GT(q.size(), end_id);
  double score_sum = 0.0;
  for (FrameId id = start_id; id <= end_id; id++) {
    score_sum += q.at(id).Score;
  }

  return score_sum;
}

} // namespace localisation
} // namespace core
} // namespace aru
