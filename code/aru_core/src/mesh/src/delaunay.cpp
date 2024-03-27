#include "aru/core/mesh/delaunay.h"

#include "aru/core/mesh/mesh.h"
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

namespace aru {
namespace core {
namespace mesh {

/* ADAPTED FROM FLAME */
//------------------------------------------------------------------------------
std::vector<Triangle> Delaunay::Triangulate(
    const utilities::image::FeatureSPtrVectorSptr &sparse_supports) {

  std::vector<Triangle> triangles;

  // input/output structure for triangulation
  struct triangulateio in{};
  int32_t k;

  // inputs
  in.numberofpoints = sparse_supports->size();
  in.pointlist =
      (float *)malloc(in.numberofpoints * 2 * sizeof(float)); // NOLINT
  k = 0;
  for (int32_t i = 0; i < sparse_supports->size(); i++) {
    in.pointlist[k++] = sparse_supports->at(i)->GetKeyPoint().pt.x;
    in.pointlist[k++] = sparse_supports->at(i)->GetKeyPoint().pt.y;
  }
  in.numberofpointattributes = 0;
  in.pointattributelist = NULL;
  in.pointmarkerlist = NULL;
  in.numberofsegments = 0;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.regionlist = NULL;

  // outputs
  out.pointlist = NULL;
  out.pointattributelist = NULL;
  out.pointmarkerlist = NULL;
  out.trianglelist = NULL;
  out.triangleattributelist = NULL;
  out.neighborlist = NULL;
  out.segmentlist = NULL;
  out.segmentmarkerlist = NULL;
  out.edgelist = NULL;
  out.edgemarkerlist = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary
  // markers)
  char parameters[] = "zneQB";
  ::triangulate(parameters, &in, &out, NULL);
  free(in.pointlist);

  GetTriangles(&triangles);
  GetNeighbors();
  GetEdges();
  Cleanup();

  return triangles;
}

//------------------------------------------------------------------------------
std::vector<Triangle>
Delaunay::Triangulate(const std::vector<cv::Point2f>& points) {

  std::vector<Triangle> triangles;
  // input/output structure for triangulation
  struct triangulateio in;
  int32_t k;

  // inputs
  in.numberofpoints = points.size();
  in.pointlist =
      (float *)malloc(in.numberofpoints * 2 * sizeof(float)); // NOLINT
  k = 0;

  for (const auto &point : points) {
    in.pointlist[k++] = point.x;
    in.pointlist[k++] = point.y;
  }

  in.numberofpointattributes = 0;
  in.pointattributelist = NULL;
  in.pointmarkerlist = NULL;
  in.numberofsegments = 0;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.regionlist = NULL;

  // outputs
  out.pointlist = NULL;
  out.pointattributelist = NULL;
  out.pointmarkerlist = NULL;
  out.trianglelist = NULL;
  out.triangleattributelist = NULL;
  out.neighborlist = NULL;
  out.segmentlist = NULL;
  out.segmentmarkerlist = NULL;
  out.edgelist = NULL;
  out.edgemarkerlist = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary
  // markers)
  char parameters[] = "zneQB";
  ::triangulate(parameters, &in, &out, NULL);
  free(in.pointlist);

  GetTriangles(&triangles);
  GetNeighbors();
  GetEdges();
  Cleanup();

  return triangles;
}
//------------------------------------------------------------------------------
void Delaunay::Cleanup() {
  // free memory used for triangulation
  free(out.pointlist);
  free(out.trianglelist);
  free(out.edgelist);
  free(out.neighborlist);

  out.pointlist = NULL;
  out.trianglelist = NULL;
  out.edgelist = NULL;
  out.neighborlist = NULL;

  return;
}
//------------------------------------------------------------------------------
void Delaunay::GetTriangles(std::vector<Triangle> *triangles) {
  // put resulting triangles into vector tri
  triangles->resize(out.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out.numberoftriangles; i++) {
    (*triangles)[i] = Triangle(out.trianglelist[k], out.trianglelist[k + 1],
                               out.trianglelist[k + 2]);
    k += 3;
  }
  return;
}
//------------------------------------------------------------------------------
void Delaunay::GetNeighbors() {
  // put neighboring triangles into vector tri
  neighbours_.resize(out.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out.numberoftriangles; i++) {
    neighbours_[i] = Triangle(out.neighborlist[k], out.neighborlist[k + 1],
                              out.neighborlist[k + 2]);
    k += 3;
  }
  return;
}
//------------------------------------------------------------------------------
void Delaunay::GetEdges() {
  // put resulting edges into vector
  edges_.resize(out.numberofedges);
  int k = 0;
  for (int32_t i = 0; i < out.numberofedges; i++) {
    edges_[i] = Edge(out.edgelist[k], out.edgelist[k + 1]);
    k += 2;
  }
  return;
}
//------------------------------------------------------------------------------
void Delaunay::drawLineColourMap(cv::Mat &img, const cv::Point &start,
                       const cv::Point &end, const int depth_norm1,
                       const int depth_norm2) {
  cv::Vec3b c1;
  cv::LineIterator iter(img, start, end, cv::LINE_8);

  for (int i = 0; i < iter.count; i++, iter++) {
    double alpha = double(i) / iter.count;

    COLOUR_MAP.LookUp(depth_norm1 * (1.0 - alpha) + depth_norm2 * alpha, c1);

    (*iter)[0] = (uint8_t)(c1[0]);
    (*iter)[1] = (uint8_t)(c1[1]);
    (*iter)[2] = (uint8_t)(c1[2]);
  }
}
//------------------------------------------------------------------------------
void Delaunay::DrawWireframe(
    const utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Triangle> &triangles, cv::Mat &img, int max_depth) {

  float COLOUR_SCALE = 1.0; // 1.12

  int thickness = 1;
  int lineType = cv::LINE_8;

  cv::Mat depths(1, sparse_supports->size(), CV_32FC1);
  cv::Mat depths_norm(1, sparse_supports->size(), CV_32FC1);

  std::transform(sparse_supports->begin(), sparse_supports->end(),
                 depths.begin<float>(),
                 [](utilities::image::FeatureSPtr const &p) -> float {
                   return p->GetDepth();
                 });

  // cv::normalize(depths,depths_norm,0, 255, cv::NORM_MINMAX, -1);

  depths = (depths / (float)max_depth * COLOUR_SCALE) * 255.0f;
  std::transform(depths.begin<float>(), depths.end<float>(),
                 depths_norm.begin<float>(), [](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255

  float radius = 2;

  for (auto t : triangles) {

    cv::Point2f vert1 = sparse_supports->at(t[0])->GetKeyPoint().pt;
    cv::Point2f vert2 = sparse_supports->at(t[1])->GetKeyPoint().pt;
    cv::Point2f vert3 = sparse_supports->at(t[2])->GetKeyPoint().pt;

    int depth1_norm = static_cast<int>(depths_norm.at<float>(0, t[0]));
    int depth2_norm = static_cast<int>(depths_norm.at<float>(0, t[1]));
    int depth3_norm = static_cast<int>(depths_norm.at<float>(0, t[2]));

    cv::Vec3b c1, c2, c3;
    COLOUR_MAP.LookUp(depth1_norm, c1);
    COLOUR_MAP.LookUp(depth2_norm, c2);
    COLOUR_MAP.LookUp(depth3_norm, c3);

    drawLineColourMap(img, vert1, vert2, depth1_norm, depth2_norm);
    drawLineColourMap(img, vert2, vert3, depth2_norm, depth3_norm);
    drawLineColourMap(img, vert3, vert1, depth3_norm, depth1_norm);
  }
}

} // namespace mesh
} // namespace core
} // namespace aru
