#define BOOST_TEST_MODULE My Test
#include "aru/core/coral/coral.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace aru::core;
using namespace aru::core::coral;
//------------------------------------------
struct CoralFixture {
  optimiser::CoralOptimiserParams params;
  boost::shared_ptr<optimiser::CoralOptimiser<int>> optimiser;

  utilities::image::FeatureSPtrVectorSptr point_features;

  Eigen::MatrixXd feature_costs;

  CoralFixture() {
    params.num_neighbours = 2;
    params.max_neighbours = 1;
    params.num_features = 3;
    params.tau = 0.125;
    params.alpha = 0.125;
    params.nu = 0.125;
    params.lambda = 0;
    params.beta = 0;
    params.num_iterations = 1000;
    params.num_features = 9;

    optimiser = boost::make_shared<optimiser::CoralOptimiser<int>>(params);

    point_features = boost::make_shared<utilities::image::FeatureSPtrVector>();
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(0, 0.01)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(0, 1.02)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(0.03, 2)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(1.04, 0)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(1, 1.06)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(1, 2.07)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(2, 0.02)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(2, 1.08)));
    point_features->push_back(boost::make_shared<utilities::image::Feature>(
        Eigen::Vector2d(2.10, 2)));

    feature_costs = Eigen::MatrixXd::Zero(9, 4);

    feature_costs(0, 0) = 1;
    feature_costs(1, 0) = 1;
    feature_costs(2, 0) = 1;
    feature_costs(3, 0) = 1;
    feature_costs(4, 0) = 3;
    feature_costs(5, 0) = 4;
    feature_costs(6, 0) = 5;
    feature_costs(7, 0) = 6;
    feature_costs(8, 0) = 8;

    feature_costs(0, 1) = 2;
    feature_costs(1, 1) = 2;
    feature_costs(2, 1) = 2;
    feature_costs(3, 1) = 1;
    feature_costs(4, 1) = 1;
    feature_costs(5, 1) = 1;
    feature_costs(6, 1) = 4;
    feature_costs(7, 1) = 3;
    feature_costs(8, 1) = 3;

    feature_costs(0, 2) = 3;
    feature_costs(1, 2) = 3;
    feature_costs(2, 2) = 3;
    feature_costs(3, 2) = 3;
    feature_costs(4, 2) = 3;
    feature_costs(5, 2) = 3;
    feature_costs(6, 2) = 3;
    feature_costs(7, 2) = 1;
    feature_costs(8, 2) = 1;

    feature_costs(0, 3) = 3;
    feature_costs(1, 3) = 3;
    feature_costs(2, 3) = 3;
    feature_costs(3, 3) = 3;
    feature_costs(4, 3) = 3;
    feature_costs(5, 3) = 3;
    feature_costs(6, 3) = 3.1;
    feature_costs(7, 3) = 0.9;
    feature_costs(8, 3) = 1.1;
  }
};

BOOST_AUTO_TEST_CASE(simplex_test) {

  int num_points = 10;
  int num_dimensions = 2;

  Eigen::VectorXd x_values(num_points);
  Eigen::Matrix3d K;

  K << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;

  Eigen::Matrix3d K_simplex =
      optimiser::CoralOptimiser<int>::SimplexProjectionVector(K);

  BOOST_CHECK_CLOSE(K_simplex(0, 0) + K_simplex(0, 1) + K_simplex(0, 2), 1, 1);
  BOOST_CHECK_CLOSE(K_simplex(1, 0) + K_simplex(1, 1) + K_simplex(1, 2), 1, 1);
  BOOST_CHECK_CLOSE(K_simplex(2, 0) + K_simplex(2, 1) + K_simplex(2, 2), 1, 1);
}

BOOST_FIXTURE_TEST_CASE(neighbourhood_test, CoralFixture) {

  optimiser->FindNearestNeighbours(point_features);
  Eigen::SparseMatrix<double> nabla = optimiser->GetGradient();

  // Check neighbours of point 0, which are 1 and 3
  int point = 0;
  int index1 = point * 2;
  int index2 = point * 2 + 1;
  int neighbour1 = 1;
  int neighbour2 = 3;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 1, which are 2 and 4
  point = 1;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 2;
  neighbour2 = 4;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 2, which are 5 and 1
  point = 2;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 5;
  neighbour2 = 1;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 3, which are 6 and 0
  point = 3;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 6;
  neighbour2 = 0;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 4, which are 7 and 1
  point = 4;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 7;
  neighbour2 = 1;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 5, which are 2 and 4
  point = 5;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 2;
  neighbour2 = 4;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 6, which are 3 and 8
  point = 6;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 3;
  neighbour2 = 7;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 7, which are 8 and 4
  point = 7;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 8;
  neighbour2 = 4;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
  // Check neighbours of point 8, which are 7 and 5
  point = 8;
  index1 = point * 2;
  index2 = point * 2 + 1;
  neighbour1 = 7;
  neighbour2 = 5;
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index1, neighbour1));
  BOOST_CHECK_EQUAL(1, nabla.coeffRef(index2, neighbour2));
}
BOOST_FIXTURE_TEST_CASE(PrimalMinimisation, CoralFixture) {

  optimiser->FindNearestNeighbours(point_features);
  Eigen::SparseMatrix<double> nabla = optimiser->GetGradient();
  optimiser::EnergyMinimisationResult result =
      optimiser->EnergyMinimisation(feature_costs, nabla);

  Eigen::MatrixXd primal = result.SoftLabel;
  // Features 0-2 have label 0
  BOOST_CHECK_CLOSE(primal(0, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(1, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(2, 0), 1, 1);
  // Feature 3 is split between label 0 and 1
  BOOST_CHECK_CLOSE(primal(3, 0), 0.5, 1);
  BOOST_CHECK_CLOSE(primal(3, 1), 0.5, 1);
  // Feature 4 and 5 have label 1
  BOOST_CHECK_CLOSE(primal(4, 1), 1, 1);
  BOOST_CHECK_CLOSE(primal(5, 1), 1, 1);
  // Feature 6 has label 2
  BOOST_CHECK_CLOSE(primal(6, 2), 1, 1);
  // Feature 7 has label 3
  BOOST_CHECK_CLOSE(primal(7, 3), 1, 1);
  // Feature 8 has label 2
  BOOST_CHECK_CLOSE(primal(8, 2), 1, 1);
}

BOOST_FIXTURE_TEST_CASE(PrimalSmoothnessDualMinimisation, CoralFixture) {

  optimiser->FindNearestNeighbours(point_features);
  Eigen::SparseMatrix<double> nabla = optimiser->GetGradient();

  float lambda = 0.1;
  optimiser->UpdateLambda(lambda);
  optimiser::EnergyMinimisationResult result =
      optimiser->EnergyMinimisation(feature_costs, nabla);
  Eigen::MatrixXd primal = result.SoftLabel;
  // Features 0-3 have label 0
  BOOST_CHECK_CLOSE(primal(0, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(1, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(2, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(3, 0), 1, 1);
  // Feature 4 and 5 have label 1
  BOOST_CHECK_CLOSE(primal(4, 1), 1, 1);
  BOOST_CHECK_CLOSE(primal(5, 1), 1, 1);
  // Feature 6-8 has label 2
  BOOST_CHECK_CLOSE(primal(6, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(7, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(8, 2), 1, 1);

  lambda = 0;
  optimiser->UpdateLambda(lambda);
}

BOOST_FIXTURE_TEST_CASE(PrimalCompactnessDualMinimisation, CoralFixture) {

  optimiser->FindNearestNeighbours(point_features);
  Eigen::SparseMatrix<double> nabla = optimiser->GetGradient();

  float beta = 1.5;
  optimiser->UpdateBeta(beta);
  optimiser::EnergyMinimisationResult result =
      optimiser->EnergyMinimisation(feature_costs, nabla);

  Eigen::MatrixXd primal = result.SoftLabel;
  // Features 0-3 have label 0
  BOOST_CHECK_CLOSE(primal(0, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(1, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(2, 0), 1, 1);
  // Feature 3 is split between label 0 and 1
  BOOST_CHECK_CLOSE(primal(3, 0), 0.487, 1);
  BOOST_CHECK_CLOSE(primal(3, 1), 0.512, 1);
  // Feature 4 and 5 have label 1
  BOOST_CHECK_CLOSE(primal(4, 1), 1, 1);
  BOOST_CHECK_CLOSE(primal(5, 1), 1, 1);
  // Feature 6-8 has label 2
  BOOST_CHECK_CLOSE(primal(6, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(7, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(8, 2), 1, 1);

  beta = 0;
  optimiser->UpdateLambda(beta);
}

BOOST_FIXTURE_TEST_CASE(PrimalSmoothnessCompactnessDualMinimisation,
                        CoralFixture) {

  optimiser->FindNearestNeighbours(point_features);
  Eigen::SparseMatrix<double> nabla = optimiser->GetGradient();

  float lambda = 0.1;
  float beta = 1.5;
  optimiser->UpdateLambda(lambda);
  optimiser->UpdateBeta(beta);
  optimiser::EnergyMinimisationResult result =
      optimiser->EnergyMinimisation(feature_costs, nabla);

  Eigen::MatrixXd primal = result.SoftLabel;
  // Features 0-3 have label 0
  BOOST_CHECK_CLOSE(primal(0, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(1, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(2, 0), 1, 1);
  BOOST_CHECK_CLOSE(primal(3, 0), 1, 1);
  // Feature 4 and 5 have label 1
  BOOST_CHECK_CLOSE(primal(4, 1), 1, 1);
  BOOST_CHECK_CLOSE(primal(5, 1), 1, 1);
  // Feature 6-8 has label 2
  BOOST_CHECK_CLOSE(primal(6, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(7, 2), 1, 1);
  BOOST_CHECK_CLOSE(primal(8, 2), 1, 1);
}
BOOST_FIXTURE_TEST_CASE(ReduceLabels, CoralFixture) {

  Eigen::VectorXd label_ = Eigen::MatrixXd::Zero(7, 1);
  label_(0) = 25;
  label_(1) = 25;
  label_(2) = 25;
  label_(3) = 25;
  label_(4) = 25;
  label_(5) = 25;
  label_(6) = 1;
  int i = 0;
  int num_labels = 26;
  while (i < num_labels - 1) {
    int numbers = (label_.array() == i).count();
    i++;
    if (numbers == 0) {
      LOG(INFO) << "Curr label is " << label_.transpose();
      for (int j = (i - 1); j < num_labels - 1; j++) {
        label_ = (label_.array() == (j + 1)).select(j, label_);
        LOG(INFO) << "New label is " << label_.transpose();
      }
      num_labels--;
      i = 0;
    }
  }
  LOG(INFO) << "Number labels is " << num_labels;
}