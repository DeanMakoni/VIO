#include "aru/core/mesh/mesh_estimation.h"

#include <Eigen/Dense>
#include <aru/core/mesh/delaunay.h>
#include <aru/core/mesh/mesh_regularisers.h>
#include <aru/core/utilities/image/feature_matcher.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <utility>

namespace aru {
namespace core {
namespace mesh {
//------------------------------------------------------------------------------
MeshEstimation::MeshEstimation(
    utilities::camera::CameraParams camera_params, RegularisationParams reg_params,
    utilities::image::MatcherParams matcher_params,
    utilities::image::ExtractorParams extractor_params)
    : cam_params_(std::move(camera_params)), reg_params_(reg_params),
      matcher_params_(matcher_params), extractor_params_(extractor_params) {

  point_features_ = boost::make_shared<utilities::image::FeatureSPtrVector>();
  feature_matcher_ = boost::make_shared<Matcher>();
}
//------------------------------------------------------------------------------
void MeshEstimation::EstimateMesh(
    utilities::image::FeatureSPtrVectorSptr features,
    bool triangulate_features) {
  point_features_ = (features);
  /* ------------------ STAGE 1: DELAUNAY TRIANGULATION ------------------*/

  if (triangulate_features) {
    for (const auto &feature : *point_features_) {
      feature->TriangulatePoint(cam_params_.K, cam_params_.baseline);
    }
  }
  Delaunay delaunay;
  triangles_ = delaunay.Triangulate(point_features_);

  /* ------------------ STAGE 2: VARIATIONAL SMOOTHING  ------------------*/
  std::vector<Edge> edges = delaunay.edges();

  if (triangulate_features) {
    MeshRegulariser regulariser(reg_params_);
    regulariser.Regularise(point_features_, edges);
  }
}
//------------------------------------------------------------------------------
void MeshEstimation::EstimateMesh(const cv::Mat &image_left,
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

  // Clear the vectors
  point_features_->clear();
  triangles_.clear();

  // Do the stereo matching
  fx_ = cam_params_.K(0, 0);
  int32_t width = image_left.cols;
  int32_t height = image_left.rows;

  // compute stereo matches
  auto matching_start = std::chrono::high_resolution_clock::now();
  int32_t dims[] = {width, height, width};
  feature_matcher_->pushBack(image_left_grey.data, image_right_grey.data, dims,
                             false);
  feature_matcher_->matchFeatures(1);
  feature_matcher_->bucketFeatures(1, 15, 15);
  std::vector<Matcher::p_match> p_matched = feature_matcher_->getMatches();
  for (auto match : p_matched) {
    utilities::image::FeatureSPtr curr_feature =
        boost::make_shared<utilities::image::Feature>(
            Eigen::Vector2d(match.u1c, match.v1c));
    curr_feature->SetKeyPoint(cv::KeyPoint(match.u1c, match.v1c, 1));
    curr_feature->SetMatchedKeypoint(cv::KeyPoint(match.u2c, match.v2c, 1));
    if (match.u1c - match.u2c > 0)
      point_features_->push_back(curr_feature);
  }
  VLOG(2)<<"Number of features is "<<point_features_->size();
  auto matching_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = matching_end - matching_start;
  VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
  VLOG(2) << "Stereo matching runs at " << 1 / elapsed.count() << " Hz";
  //  aru::core::utilities::viewer::Viewer::ViewStereoImageFeatures(
  //      image_left, image_right, 0.6, point_features_);

  /* ------------------ STAGE 1: DELAUNAY TRIANGULATION ------------------*/
  VLOG(2) << "Delauanay triangulation";
  auto triangulation_start = std::chrono::high_resolution_clock::now();
  for (const auto &feature : *point_features_) {
    feature->TriangulatePoint(cam_params_.K, cam_params_.baseline);
  }
  Delaunay delaunay;
  triangles_ = delaunay.Triangulate(point_features_);
  auto triangulation_end = std::chrono::high_resolution_clock::now();
  elapsed = triangulation_end - triangulation_start;
  VLOG(2) << "Triangulation takes " << elapsed.count() << " seconds";
  VLOG(2) << "Triangulation runs at " << 1 / elapsed.count() << " Hz";

  // cv::Mat delaunay_unsmooth_img = image_left.clone();
  //  triangles_ = RemoveLongTriangles();
  //   Delaunay::DrawWireframe(point_features_, triangles_,
  //   delaunay_unsmooth_img,
  //                           50);
  //  cv::imshow("Filtered Image", delaunay_unsmooth_img);

  /* ------------------ STAGE 2: VARIATIONAL SMOOTHING  ------------------*/
  VLOG(2) << "Start regularisation";
  auto regularisation_start = std::chrono::high_resolution_clock::now();
  std::vector<Edge> edges = delaunay.edges();
  MeshRegulariser regulariser(reg_params_);
  regulariser.Regularise(point_features_, edges);
  auto regularisation_end = std::chrono::high_resolution_clock::now();
  elapsed = regularisation_end - regularisation_start;
  VLOG(2) << "Regularisation takes " << elapsed.count() << " seconds";
  VLOG(2) << "Regularisation runs at " << 1 / elapsed.count() << " Hz";
}
//------------------------------------------------------------------------------
std::vector<Triangle> MeshEstimation::RemoveLongTriangles() {
  std::vector<Triangle> filtered_triangles_;
  float edge_length_thresh = 0.1;
  float dist_thresh2 = edge_length_thresh * cam_params_.image_width;
  dist_thresh2 *= dist_thresh2;
  for (auto &triangle : triangles_) {

    cv::Point2f vtx0 = point_features_->at(triangle[0])->GetKeyPoint().pt;
    cv::Point2f vtx1 = point_features_->at(triangle[1])->GetKeyPoint().pt;
    cv::Point2f vtx2 = point_features_->at(triangle[2])->GetKeyPoint().pt;

    cv::Point2f diff01(vtx0 - vtx1);
    cv::Point2f diff02(vtx0 - vtx2);
    cv::Point2f diff12(vtx1 - vtx2);

    float dist01 = diff01.x * diff01.x + diff01.y * diff01.y;
    float dist02 = diff02.x * diff02.x + diff02.y * diff02.y;
    float dist12 = diff12.x * diff12.x + diff12.y * diff12.y;

    if ((dist01 < dist_thresh2) && (dist02 < dist_thresh2) &&
        (dist12 < dist_thresh2)) {
      filtered_triangles_.push_back(triangle);
    }
  }
  return filtered_triangles_;
}
//------------------------------------------------------------------------------
std::vector<Triangle> MeshEstimation::RemoveObtuseTriangles() {
  std::vector<Triangle> filtered_triangles_;
  float oblique_normal_thresh = 2.7626;
  for (auto &triangle : triangles_) {
    // NOTE: Triangle spits out points in clock-wise order.
    Eigen::Vector3f p0(point_features_->at(triangle[0])->GetKeyPoint().pt.x,
                       point_features_->at(triangle[0])->GetKeyPoint().pt.y,
                       1.0f);
    p0 = cam_params_.K.inverse().cast<float>() * p0 *
         point_features_->at(triangle[0])->GetDepth();

    Eigen::Vector3f p1(point_features_->at(triangle[1])->GetKeyPoint().pt.x,
                       point_features_->at(triangle[1])->GetKeyPoint().pt.x,
                       1.0f);
    p1 = cam_params_.K.inverse().cast<float>() * p1 *
         point_features_->at(triangle[1])->GetDepth();

    Eigen::Vector3f p2(point_features_->at(triangle[2])->GetKeyPoint().pt.x,
                       point_features_->at(triangle[2])->GetKeyPoint().pt.x,
                       1.0f);
    p2 = cam_params_.K.inverse().cast<float>() * p2 *
         point_features_->at(triangle[2])->GetDepth();

    // Inward-facing normal.
    Eigen::Vector3f delta1(p1 - p0);
    Eigen::Vector3f delta2(p2 - p0);
    Eigen::Vector3f normal(delta1.cross(delta2));
    normal.normalize();

    // Compute angle diff between inward normal and viewing ray through center
    // of triangle.
    Eigen::Vector3f ray((p0 + p1 + p2) / 3);
    ray.normalize();

    float angle = fabs(acos(ray.dot(normal)));
    if (angle > oblique_normal_thresh) {
      filtered_triangles_.push_back(triangle);
    }
  }
  return filtered_triangles_;
}
} // namespace mesh
} // namespace core
} // namespace aru
