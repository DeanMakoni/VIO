#ifndef ARU_DEPTH_REGULARISERS_H_
#define ARU_DEPTH_REGULARISERS_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/mesh/delaunay.h"
#include "aru/core/mesh/mesh.h"
#include "mesh_estimation.h"

namespace aru {
namespace core {
namespace mesh {

class MeshRegulariser {
public:
  MeshRegulariser(RegularisationParams params);

  ~MeshRegulariser() = default;

  void Regularise(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                  const std::vector<Edge> &edges);
  void
  RegulariseVoronoi(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                    int num_delaunay, const std::vector<Edge> &edges);

private:
  void run_TV(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
              const std::vector<Edge> &edges) const;
  void run_TGV(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
               const std::vector<Edge> &edges) const;
  void run_logTV(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                 const std::vector<Edge> &edges) const;
  void run_logTGV(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                  const std::vector<Edge> &edges);
  void
  run_voronoi_logTGV(utilities::image::FeatureSPtrVectorSptr &sparse_supports,
                     const std::vector<Edge> &edges, int num_delaunay);

private:
  RegularisationParams params_;
};

} // namespace mesh
} // namespace core
} // namespace aru

#endif // ARU_DEPTH_REGULARISERS_H_
