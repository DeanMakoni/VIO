#ifndef CORAL_H_
#define CORAL_H_

#include "aru/core/utilities/image/point_feature.h"
#include "models/coral_model.h"
#include "nanoflann.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include <iostream>
#include <utility>

namespace aru {
namespace core {
namespace coral {
namespace optimiser {

struct CoralOptimiserParams {
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

struct EnergyMinimisationResult {
  Eigen::MatrixXd SoftLabel;
  Eigen::MatrixXd DiscreteLabel;
};

template <typename Model> class CoralOptimiser {

  typedef Eigen::MatrixXd Dual;
  typedef Eigen::MatrixXd Primal;
  typedef Eigen::SparseMatrix<double> Gradient;
  typedef Eigen::MatrixXd Label;

public:
  CoralOptimiser(const CoralOptimiserParams coral_optimiser_params);

  ~CoralOptimiser();

public:
  virtual EnergyMinimisationResult
  EnergyMinimisation(const utilities::image::FeatureSPtrVectorSptr &features,
                     models::ModelVectorSPtr models);

  virtual EnergyMinimisationResult
  EnergyMinimisation(const Eigen::MatrixXd feature_costs,
                     Gradient neighbour_index);

  Eigen::MatrixXd
  EvaluateModelCost(const utilities::image::FeatureSPtrVectorSptr &features,
                    const models::ModelVectorSPtr &models);

  void UpdateNumFeatures(int num_features) {
    coral_optimiser_params_.num_features = num_features;
    InitialiseVariables();
  }

  void UpdateNumLabels(int num_labels) {
    coral_optimiser_params_.num_labels = num_labels;
    InitialiseVariables();
  }

  void UpdateLambda(float lambda) { coral_optimiser_params_.lambda = lambda; }

  void UpdateBeta(float beta) { coral_optimiser_params_.beta = beta; }

  static Eigen::MatrixXd SimplexProjectionVector(Eigen::MatrixXd matrix);

  Gradient GetGradient() { return neighbour_index_; };

  void FindNearestNeighbours(
      const utilities::image::FeatureSPtrVectorSptr &features);

  void LabelsFromPrimal();

  void LabelsFromModelCosts();

  void PrimalFromLabels();

  Primal GetPrimal() { return primal_; };

  Label GetLabel() { return label_; };

  void SetPrimal(Primal primal) { primal_ = std::move(primal); };

  void UpdateModels(utilities::image::FeatureSPtrVectorSptr features,
                    models::ModelVectorSPtr models);

  void ReduceModels(models::ModelVectorSPtr models);

  void ReduceModels();

  int NumLabels() { return coral_optimiser_params_.num_labels; }

private:
  void InitialiseVariables();

  void UpdateCompactnessDual();

  void UpdateSmoothnessDual();

  void UpdatePrimal();

  Dual GetClampedDualNorm(Dual dual, double clamp_value);

  static void ClampVariable(Primal &primal, double clamp_value);

  void SimplexProjection();

  static Eigen::MatrixXd SortMatrix(Primal primal_matrix);

private:
  CoralOptimiserParams coral_optimiser_params_;

  Primal primal_;
  Primal primal_relaxed_;

  Dual smoothness_dual_;
  Dual compactness_dual_;

  Label label_;
  Gradient neighbour_index_;

  Eigen::MatrixXd model_costs_;
};

//------------------------------------------------------------------------------
template <typename InputType>
CoralOptimiser<InputType>::CoralOptimiser(
    const CoralOptimiserParams coral_optimiser_params)
    : coral_optimiser_params_(coral_optimiser_params) {}

//------------------------------------------------------------------------------
template <typename InputType>
CoralOptimiser<InputType>::~CoralOptimiser() = default;

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::InitialiseVariables() {
  // Set up the optimisation variables
  model_costs_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                       coral_optimiser_params_.num_labels);
  primal_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                  coral_optimiser_params_.num_labels);
  primal_relaxed_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                          coral_optimiser_params_.num_labels);
  smoothness_dual_ =
      Eigen::MatrixXd::Zero(coral_optimiser_params_.num_neighbours *
                                coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  compactness_dual_ = Eigen::MatrixXd::Zero(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels);

  label_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features, 1);
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::FindNearestNeighbours(
    const aru::core::utilities::image::FeatureSPtrVectorSptr &features) {

  int nn = coral_optimiser_params_.num_neighbours + 1;
  int dim = 2;

  Eigen::MatrixXd target_features(features->size(), dim);

  int feat_no = 0;
  std::vector<std::vector<double>> query_points;
  for (auto feature : *features) {
    Eigen::VectorXd feat_point = feature->GetValue();
    std::vector<double> current_point;
    current_point.push_back(feat_point(0));
    current_point.push_back(feat_point(1));
    query_points.push_back(current_point);
    target_features.row(feat_no) = feat_point;
    feat_no++;
  }

  // ------------------------------------------------------------
  // construct a kd-tree index:
  //    Some of the different possibilities (uncomment just one)
  // ------------------------------------------------------------
  // Dimensionality set at run-time (default: L2)
  typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> my_kd_tree_t;

  my_kd_tree_t mat_index(dim, std::cref(target_features), 10 /* max leaf */);
  mat_index.index->buildIndex();

  // Create the gradient variable

  neighbour_index_ = Gradient(coral_optimiser_params_.num_neighbours *
                                  coral_optimiser_params_.num_features,
                              coral_optimiser_params_.num_features);
  neighbour_index_.reserve(coral_optimiser_params_.num_neighbours *
                           coral_optimiser_params_.num_features);

  uint cont = 0;
  for (int i = 0; i < features->size(); ++i) {
    std::vector<size_t> ret_indexes(nn);
    std::vector<double> out_dists_sqr(nn);

    nanoflann::KNNResultSet<double> resultSet(nn);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &query_points[i][0],
                                   nanoflann::SearchParams(10));

    for (int k = 1; k < nn; ++k) {
      neighbour_index_.coeffRef(cont, i) = -1;
      neighbour_index_.coeffRef(cont, ret_indexes[k]) = 1;
      cont++;
    }
  }
}

//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd CoralOptimiser<InputType>::EvaluateModelCost(
    const utilities::image::FeatureSPtrVectorSptr &features,
    const models::ModelVectorSPtr &models) {
  Eigen::MatrixXd ModelMatrix = Eigen::MatrixXd::Constant(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels,
      coral_optimiser_params_.outlier_threshold);

  double e = 0.001;
  for (int i = 0; i < coral_optimiser_params_.num_labels - 1; ++i) {
    ModelMatrix.col(i) = (*models)[i]->EvaluateCost(features);
  }
  //  uint num_inliers = (ModelMatrix.col(0).array() < 3).count();
  //  LOG(INFO) << "Num inliers is " << num_inliers;
  return ModelMatrix;
}

//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd
CoralOptimiser<InputType>::GetClampedDualNorm(Dual dual, double clamp_value) {

  Dual l2_norm = Dual::Zero(coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  uint counter = 0;
  for (int feat_no = 0; feat_no < coral_optimiser_params_.num_features;
       ++feat_no) {
    for (int neighbour_no = 0;
         neighbour_no < coral_optimiser_params_.num_neighbours;
         ++neighbour_no) {
      l2_norm.row(feat_no) =
          l2_norm.row(feat_no) + (dual.row(counter)).array().square().matrix();
      counter++;
    }
  }
  l2_norm = l2_norm.cwiseSqrt();
  Dual replicated_l2_norm = Dual::Zero(coral_optimiser_params_.num_neighbours *
                                           coral_optimiser_params_.num_features,
                                       coral_optimiser_params_.num_labels);
  counter = 0;
  for (int feat_no = 0; feat_no < coral_optimiser_params_.num_features;
       ++feat_no) {
    for (int neighbour_no = 0;
         neighbour_no < coral_optimiser_params_.num_neighbours;
         ++neighbour_no) {
      replicated_l2_norm.row(counter) = l2_norm.row(feat_no);
      counter++;
    }
  }

  // Clamp the norm

  for (int i = 0; i < replicated_l2_norm.rows(); ++i) {
    for (int j = 0; j < replicated_l2_norm.cols(); ++j) {
      if (replicated_l2_norm(i, j) < clamp_value)
        replicated_l2_norm(i, j) = clamp_value;
    }
  }
  return replicated_l2_norm;
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::UpdateSmoothnessDual() {

  Eigen::MatrixXd intermediate_dual, l2_norm_dual;

  intermediate_dual = smoothness_dual_ + coral_optimiser_params_.lambda *
                                             coral_optimiser_params_.alpha *
                                             neighbour_index_ * primal_relaxed_;
  l2_norm_dual = GetClampedDualNorm(intermediate_dual, 1);

  smoothness_dual_ =
      (intermediate_dual.array() / l2_norm_dual.array()).matrix();
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::UpdateCompactnessDual() {
  Eigen::MatrixXd intermediate_dual, l2_norm_dual;
  compactness_dual_ = compactness_dual_ + coral_optimiser_params_.beta *
                                              coral_optimiser_params_.nu *
                                              primal_relaxed_;
  compactness_dual_ =
      SimplexProjectionVector(compactness_dual_.transpose()).transpose();
}

//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd CoralOptimiser<InputType>::SortMatrix(Primal primal_matrix) {
  Primal sorted_matrix = primal_matrix;
  for (int i = 0; i < primal_matrix.rows(); ++i) {
    Eigen::VectorXd x = primal_matrix.row(i);
    std::sort(x.data(), x.data() + x.size());
    sorted_matrix.row(i) = x;
  }
  return sorted_matrix;
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::ClampVariable(Primal &primal,
                                              double clamp_value) {
  for (int i = 0; i < primal.rows(); ++i) {
    for (int j = 0; j < primal.cols(); ++j) {
      if (primal(i, j) < clamp_value)
        primal(i, j) = clamp_value;
    }
  }
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::SimplexProjection() {
  Primal primal_sorted = SortMatrix(primal_);

  // create x zero
  int final_column_no = primal_sorted.cols() - 1;
  Eigen::VectorXd x_zero = primal_sorted.col(final_column_no) -
                           Primal::Constant(primal_sorted.rows(), 1, 1);

  Primal primal_new(primal_sorted.rows(), primal_sorted.cols() + x_zero.cols());
  primal_new << primal_sorted, x_zero;
  Primal primal_new_sorted = SortMatrix(primal_new);

  // Create the f matrix

  int num_rows = primal_new_sorted.rows();
  int num_cols = primal_new_sorted.cols();

  Primal f = Primal::Zero(num_rows, num_cols);

  for (int i = 0; i < num_cols; ++i) {
    Primal curr_delta = primal_new_sorted.col(i);
    Primal tau_vec = primal_new_sorted - curr_delta.replicate(1, num_cols);
    ClampVariable(tau_vec, 0);
    f.col(i) = tau_vec.rowwise().sum();
  }

  // calculate the indices
  Eigen::MatrixXi minimum_matrix = (f.array() >= 1).cast<int>();
  Eigen::MatrixXi index(num_rows, 1);

  for (int i = 0; i < num_rows; ++i) {
    Eigen::MatrixXi::Index col_no;
    minimum_matrix.row(i).minCoeff(&col_no);
    index(i) = col_no - 1;
  }
  // Calculate the optimal value of v
  Primal v(num_rows, 1);
  for (int i = 0; i < num_rows; ++i) {
    int curr_index = index(i);
    v(i) = primal_new_sorted(i, curr_index) +
           (1 - f(i, curr_index)) *
               (primal_new_sorted(i, curr_index) -
                primal_new_sorted(i, curr_index + 1)) /
               (f(i, curr_index) - f(i, curr_index + 1));
  }
  // Calculate the new primal variable
  Primal updated_primal = primal_ - v.replicate(1, primal_.cols());
  ClampVariable(updated_primal, 0);
  primal_ = updated_primal;
}

//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd
CoralOptimiser<InputType>::SimplexProjectionVector(Eigen::MatrixXd matrix) {
  Primal matrix_sorted = SortMatrix(matrix);

  // create x zero
  int final_column_no = matrix_sorted.cols() - 1;
  Eigen::VectorXd x_zero = matrix_sorted.col(final_column_no) -
                           Primal::Constant(matrix_sorted.rows(), 1, 1);

  Primal matrix_new(matrix_sorted.rows(), matrix_sorted.cols() + x_zero.cols());
  matrix_new << matrix_sorted, x_zero;
  Primal matrix_new_sorted = SortMatrix(matrix_new);

  // Create the f matrix

  int num_rows = matrix_new_sorted.rows();
  int num_cols = matrix_new_sorted.cols();

  Primal f = Primal::Zero(num_rows, num_cols);

  for (int i = 0; i < num_cols; ++i) {
    Primal curr_delta = matrix_new_sorted.col(i);
    Primal tau_vec = matrix_new_sorted - curr_delta.replicate(1, num_cols);
    ClampVariable(tau_vec, 0);
    f.col(i) = tau_vec.rowwise().sum();
  }

  // calculate the indices
  Eigen::MatrixXi minimum_matrix = (f.array() >= 1).cast<int>();
  Eigen::MatrixXi index(num_rows, 1);

  for (int i = 0; i < num_rows; ++i) {
    Eigen::MatrixXi::Index col_no;
    minimum_matrix.row(i).minCoeff(&col_no);
    index(i) = col_no - 1;
  }
  // Calculate the optimal value of v
  Primal v(num_rows, 1);
  for (int i = 0; i < num_rows; ++i) {
    int curr_index = index(i);
    v(i) = matrix_new_sorted(i, curr_index) +
           (1 - f(i, curr_index)) *
               (matrix_new_sorted(i, curr_index) -
                matrix_new_sorted(i, curr_index + 1)) /
               (f(i, curr_index) - f(i, curr_index + 1));
  }
  // Calculate the new primal variable
  Primal updated_matrix = matrix - v.replicate(1, matrix.cols());
  ClampVariable(updated_matrix, 0);
  return updated_matrix;
}

//------------------------------------------------------------------------------
template <typename InputType> void CoralOptimiser<InputType>::UpdatePrimal() {
  Primal intermediate_primal, prev_primal;

  intermediate_primal = model_costs_ +
                        coral_optimiser_params_.lambda *
                            neighbour_index_.transpose() * smoothness_dual_ +
                        coral_optimiser_params_.beta * compactness_dual_;
  prev_primal = primal_;
  // Update primal
  primal_ = prev_primal - coral_optimiser_params_.tau * intermediate_primal;
  SimplexProjection();
  primal_relaxed_ = 2 * primal_ - prev_primal;
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::LabelsFromPrimal() {
  for (int i = 0; i < coral_optimiser_params_.num_features; ++i) {
    Eigen::MatrixXi::Index index = 0;
    primal_.row(i).maxCoeff(&index);
    label_(i, 0) = index;
  }
}

//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::LabelsFromModelCosts() {
  for (int i = 0; i < coral_optimiser_params_.num_features; ++i) {
    Eigen::MatrixXi::Index index = 0;
    model_costs_.row(i).minCoeff(&index);
    label_(i, 0) = index;
  }
  PrimalFromLabels();
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::PrimalFromLabels() {
  primal_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                  coral_optimiser_params_.num_labels);
  for (int j = 0; j < coral_optimiser_params_.num_labels; ++j) {
    for (int i = 0; i < coral_optimiser_params_.num_features; ++i) {
      if (label_(i, 0) == j) {
        primal_(i, j) = 1;
      }
    }
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::ReduceModels(models::ModelVectorSPtr models) {
  std::vector<int> model_merged_indices;
  int i = 0;
  while (i < coral_optimiser_params_.num_labels - 1) {
    int numbers = (label_.array() == i).count();
    i++;
    if (numbers < 3) {
      for (int j = (i - 1); j < coral_optimiser_params_.num_labels - 2; j++) {
        label_ = (label_.array() == (j + 1)).select(j, label_);
      }
      coral_optimiser_params_.num_labels--;
      models->pop_back();
      i = 0;
    }
  }
  // UpdateNumLabels(coral_optimiser_params_.num_labels);

  primal_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                  coral_optimiser_params_.num_labels);
  primal_relaxed_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                          coral_optimiser_params_.num_labels);
  smoothness_dual_ =
      Eigen::MatrixXd::Zero(coral_optimiser_params_.num_neighbours *
                                coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  compactness_dual_ = Eigen::MatrixXd::Zero(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels);

  PrimalFromLabels();
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::ReduceModels() {
  std::vector<int> model_merged_indices;
  int i = 0;
  while (i < coral_optimiser_params_.num_labels - 1) {
    int numbers = (label_.array() == i).count();
    i++;
    if (numbers < 3) {
      for (int j = (i - 1); j < coral_optimiser_params_.num_labels - 2; j++) {
        label_ = (label_.array() == (j + 1)).select(j, label_);
      }
      coral_optimiser_params_.num_labels--;
      i = 0;
    }
  }

  primal_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                  coral_optimiser_params_.num_labels);
  primal_relaxed_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                          coral_optimiser_params_.num_labels);
  smoothness_dual_ =
      Eigen::MatrixXd::Zero(coral_optimiser_params_.num_neighbours *
                            coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  compactness_dual_ = Eigen::MatrixXd::Zero(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels);

  PrimalFromLabels();
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::UpdateModels(
    utilities::image::FeatureSPtrVectorSptr features,
    models::ModelVectorSPtr models) {
  for (int i = 0; i < coral_optimiser_params_.num_labels - 1; ++i) {
    utilities::image::FeatureSPtrVectorSptr model_update_features(
        new utilities::image::FeatureSPtrVector);

    for (int j = 0; j < coral_optimiser_params_.num_features; ++j) {
      if (label_(j) == i) {
        model_update_features->push_back((*features)[j]);
      }
    }
    // Update the models
    if (model_update_features->size() >= (*models)[i]->ModelDegreesOfFreedom())
      (*models)[i]->UpdateModel(model_update_features);
  }
}

//------------------------------------------------------------------------------
template <typename InputType>
EnergyMinimisationResult CoralOptimiser<InputType>::EnergyMinimisation(
    const Eigen::MatrixXd feature_costs, const Gradient neighbour_index) {

  coral_optimiser_params_.num_features = feature_costs.rows();
  coral_optimiser_params_.num_labels = feature_costs.cols();

  InitialiseVariables();
  model_costs_ = feature_costs;
  neighbour_index_ = neighbour_index;

  for (int iter = 0; iter < coral_optimiser_params_.num_iterations; ++iter) {
    Eigen::MatrixXd prev_row;
    // Update dual and primal
    UpdateCompactnessDual();
    UpdateSmoothnessDual();
    UpdatePrimal();
  }

  LabelsFromPrimal();
  EnergyMinimisationResult result;
  result.SoftLabel = primal_;
  result.DiscreteLabel = label_;
  return result;
}

//------------------------------------------------------------------------------
template <typename InputType>
EnergyMinimisationResult CoralOptimiser<InputType>::EnergyMinimisation(
    const utilities::image::FeatureSPtrVectorSptr &features,
    models::ModelVectorSPtr models) {
  // Update the params
  coral_optimiser_params_.num_features = features->size();
  coral_optimiser_params_.num_labels = models->size() + 1;
  LOG(INFO) << "Number of labels is " << coral_optimiser_params_.num_labels
            << std::endl;

  InitialiseVariables();
  LOG(INFO) << "Variables initialised" << std::endl;
  // Update the gradient  and model cost variables
  FindNearestNeighbours(features);
  LOG(INFO) << "Nearest neighbours found" << std::endl;
  model_costs_ = EvaluateModelCost(features, models);
  LOG(INFO) << "Model costs evaluated" << std::endl;
  // LOG(INFO) << model_costs_;
  //  LabelsFromModelCosts();
  LOG(INFO) << "Initial models computed" << std::endl;

  for (int curr_loop = 0; curr_loop < coral_optimiser_params_.num_loops;
       ++curr_loop) {
    for (int iter = 0; iter < coral_optimiser_params_.num_iterations; ++iter) {
      LOG(INFO) << "Iteration number " << iter;
      // Update dual and primal
      UpdateSmoothnessDual();
      // UpdateCompactnessDual();
      UpdatePrimal();
    }

    // get the labels
    LabelsFromPrimal();
    ReduceModels(models);
    if (coral_optimiser_params_.update_models)
      UpdateModels(features, models);

    model_costs_ = EvaluateModelCost(features, models);
  }
  LOG(INFO) << "Model assignment is " << primal_.colwise().sum();
  EnergyMinimisationResult result;
  result.SoftLabel = primal_;
  result.DiscreteLabel = label_;
  return result;
}

} // namespace optimiser
} // namespace coral
} // namespace core
} // namespace aru
#endif // CORAL_H_
