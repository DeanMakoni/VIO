#ifndef CUDA_OPTIMISER_H_
#define CUDA_OPTIMISER_H_

#include <aru/core/coral/coral.h>
#include "cuda_matrix.h"
#include <iostream>

namespace aru {
namespace core {
namespace coral {
namespace optimiser {
namespace cuda {
const size_t N_THREADS_BLOCK = 16;

struct CudaOptimiserParams {

  int num_neighbours;

  double outlier_threshold;

  double lambda;

  double beta;
  double tau;

  double alpha;

  double nu;

  int num_features;

  int num_labels;
  int num_loops;

  int num_iterations;

  uint max_neighbours;

  int height_image;
  int width_image;

  bool use_label_dual;

  bool use_pyramid;

  bool update_models;

  double pyramid_scale;

  int pyramid_levels;
};

class CudaOptimiser {
public:
  CudaOptimiser(const cv::Mat &model_costs, const cv::Mat &nabla,
                const cv::Mat &inverse_neighbour_index_,
                CudaOptimiserParams params);

  ~CudaOptimiser() = default;

  matrix::CudaMatrix<float> Optimise();

  void UpdateModelCosts(cv::Mat model_cost);

  void UpdateNumLabels(int num_labels);

  size_t BlocksPerSide(size_t side, size_t threads) {
    return ((side + threads - 1) / threads);
  }

private:
  void PrimalOptimisation();

  void SmoothnessDualOptimisation();

  void CompactnessDualOptimisation();

  matrix::CudaMatrix<float> primal_;
  matrix::CudaMatrix<float> primal_relaxed_;
  matrix::CudaMatrix<float> smoothness_dual_;
  matrix::CudaMatrix<float> compactness_dual_;
  matrix::CudaMatrix<float> nabla_;
  matrix::CudaMatrix<float> inverse_neighbour_index_;
  matrix::CudaMatrix<float> model_costs_;

  CudaOptimiserParams params_;
};
} // namespace cuda
} // namespace optimiser
} // namespace coral
} // namespace core
} // namespace aru
#endif // CUDA_OPTIMISER_H_
