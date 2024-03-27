#ifndef ARU_CORE_DELAUNAY_H_
#define ARU_CORE_DELAUNAY_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/utilities/image/point_feature.h"
#include "triangle/triangle.h"

namespace aru {
namespace core {
namespace mesh {

using Vertex = cv::Point2f;
using Triangle = cv::Vec3i;
using Edge = cv::Vec2i;

// wrapper class for Triangle library
class Delaunay {
private:
  struct triangulateio out;
  std::vector<Triangle> triangles_;
  std::vector<Triangle> neighbours_;
  std::vector<Edge> edges_;

public:
  std::vector<Triangle>
  Triangulate(const utilities::image::FeatureSPtrVectorSptr &sparse_supports);
  std::vector<Triangle>
  Triangulate(const std::vector<cv::Point2f>& points);
  void Cleanup();
  void GetTriangles(std::vector<Triangle> *triangles);
  void GetNeighbors();
  void GetEdges();

  static void
  DrawWireframe(const utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                const std::vector<Triangle> &triangles, cv::Mat &img,
                int max_depth);

  static void drawLineColourMap(cv::Mat &img, const cv::Point &start,
                                const cv::Point &end, const int depth_norm1,
                                const int depth_norm2);


  // Accessors.
  const std::vector<Triangle> &triangles() const { return triangles_; }
  const std::vector<Edge> &edges() const { return edges_; }
  const std::vector<Triangle> &neighbors() const { return neighbours_; }
};

} // namespace mesh
} // namespace core
} // namespace aru

#endif // ARU_CORE_DELAUNAY_H_
