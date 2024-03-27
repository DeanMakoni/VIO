
#include "aru/core/mesh/mesh_regularisers.h"
#include <Eigen/Dense>

namespace aru {
namespace core {
namespace mesh {
//------------------------------------------------------------------------------
MeshRegulariser::MeshRegulariser(RegularisationParams params)
    : params_(params) {}
//------------------------------------------------------------------------------
void MeshRegulariser::Regularise(
    utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Edge> &edges) {

  if (params_.use_tv) {
    run_TV(sparse_supports, edges);
    return;
  } else if (params_.use_tgv) {
    LOG(INFO) << "TGV used";
    run_TGV(sparse_supports, edges);
    return;
  } else if (params_.use_log_tv) {
    run_logTV(sparse_supports, edges);
    return;
  } else if (params_.use_log_tgv) {
    run_logTGV(sparse_supports, edges);
    return;
  }
}
//------------------------------------------------------------------------------
void MeshRegulariser::run_TV(
    utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Edge> &edges) const {

  float sigma = 0.125f; // 0.025f
  float tau = 0.125f;
  float lambda = 1.0f; // 1.0
  float theta = 1.0f;  // 1
  int L = 2000;        // 50

  std::vector<float> z(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(), z.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> float {
                   return feature->GetDisparity();
                 });

  std::vector<cv::KeyPoint> z_pts(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(),
                 z_pts.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> cv::KeyPoint {
                   return feature->GetKeyPoint();
                 });

  std::vector<float> x(z);
  std::vector<float> p(edges.size(), 0);
  std::vector<float> x_bar(x);

  for (int iter = 0; iter < params_.iterations; iter++) {
    std::vector<float> x_prev(x);

    // primal update
    for (int i = 0; i < edges.size(); i++) {
      float u_p =
          p[i] + params_.sigma * (x_bar[edges[i][1]] - x_bar[edges[i][0]]);
      p[i] = u_p / std::max(std::abs(u_p), 1.0f);
    }

    // dual update
    for (int i = 0; i < edges.size(); i++) {
      x[edges[i][0]] += params_.tau * (p[i]);
      x[edges[i][1]] -= params_.tau * (p[i]);
    }

    // L2 norm
    // std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //              [=](float const& x_i, float const& z_i) -> float { return
    //              (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

    // L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                   [=](float const &x_i, float const &z_i) -> float {
                     return (x_i - z_i > params_.lambda * params_.tau
                                 ? x_i - params_.lambda * params_.tau
                                 : (x_i - z_i < -params_.lambda * params_.tau
                                        ? x_i + params_.lambda * params_.tau
                                        : z_i));
                   });

    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                   [=](float const &x_i, float const &x_prev_i) -> float {
                     return x_i + params_.theta * (x_i - x_prev_i);
                   });
  }

  for (int i = 0; i < sparse_supports->size(); i++) {
    LOG(INFO) << "Init depth is " << sparse_supports->at(i)->GetDisparity()
              << " and new depth is " << x[i];
    sparse_supports->at(i)->UpdateDisparity(x[i]);
  }
}
//------------------------------------------------------------------------------
void MeshRegulariser::run_TGV(
    utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Edge> &edges) {

  // auto start = std::chrono::system_clock::now();

  float sigma = 0.125f;
  float tau = 0.125f;
  float lambda = 0.5f;
  float theta = 1.0f;
  float alpha1 = 0.5f;
  float alpha2 = 0.8f;
  int L = 4000;

  // Initialise variables
  std::vector<float> z(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(), z.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> float {
                   return feature->GetDepth();
                 });

  std::vector<cv::KeyPoint> z_pts(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(),
                 z_pts.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> cv::KeyPoint {
                   return feature->GetKeyPoint();
                 });

  std::vector<float> x(z);
  std::vector<float> p(edges.size(), 0);
  std::vector<float> x_bar(x);

  std::vector<float> y(x.size(), 0);
  std::vector<float> q(edges.size(), 0);
  std::vector<float> y_bar(y);

  for (int iter = 0; iter < params_.iterations; iter++) {
    std::vector<float> x_prev(x);
    std::vector<float> y_prev(y);
    std::vector<float> p_prev(p);
    std::vector<float> q_prev(q);

    for (int i = 0; i < edges.size(); i++) {
      float u_p_1 = p[i] + params_.sigma * params_.alpha_1 *
                               (x_bar[edges[i][1]] - x_bar[edges[i][0]] -
                                y_bar[edges[i][0]]);
      p[i] = u_p_1 / std::max(std::abs(u_p_1), 1.0f);
      float u_p_2 = q[i] + params_.sigma * params_.alpha_2 *
                               (y_bar[edges[i][1]] - y_bar[edges[i][0]]);
      q[i] = u_p_2 / std::max(std::abs(u_p_2), 1.0f);
    }

    for (int i = 0; i < edges.size(); i++) {
      x[edges[i][0]] += params_.tau * params_.alpha_1 * (p[i]);
      x[edges[i][1]] -= params_.tau * params_.alpha_1 * (p[i]);

      y[edges[i][0]] += params_.tau * params_.alpha_1 * (p_prev[i]);
      y[edges[i][1]] += params_.tau * params_.alpha_1 * (p_prev[i]);

      y[edges[i][0]] += params_.tau * params_.alpha_2 * (q_prev[i]);
      y[edges[i][1]] -= params_.tau * params_.alpha_2 * (q_prev[i]);
    }

    // L2 norm
    //    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
    //                   [=](float const &x_i, float const &z_i) -> float {
    //                     return (x_i + lambda * tau * z_i) / (1 + lambda *
    //                     tau);
    //                   });

    // L1 norm
    std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                   [=](float const &x_i, float const &z_i) -> float {
                     return (x_i - z_i > params_.lambda * params_.tau
                                 ? x_i - params_.lambda * params_.tau
                                 : (x_i - z_i < -params_.lambda * params_.tau
                                        ? x_i + params_.lambda * params_.tau
                                        : z_i));
                   });
    //

    // Relaxation step
    std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                   [=](float const &x_i, float const &x_prev_i) -> float {
                     return x_i + params_.theta * (x_i - x_prev_i);
                   });

    std::transform(y.begin(), y.end(), y_prev.begin(), y_bar.begin(),
                   [=](float const &y_i, float const &y_prev_i) -> float {
                     return y_i + params_.theta * (y_i - y_prev_i);
                   });
  }

  for (int i = 0; i < sparse_supports->size(); i++) {
    sparse_supports->at(i)->UpdateDepth(x[i]);
  }
}
//------------------------------------------------------------------------------
void MeshRegulariser::run_logTV(
    utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Edge> &edges) {

  float sigma = 0.125f;
  float tau = 0.125f;
  float lambda = 1.0f;
  float theta = 1.0f;
  float beta = 1;
  int L = 25;
  int F = 10;

  // Initialise variables
  std::vector<float> z(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(), z.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> float {
                   return feature->GetDisparity();
                 });

  std::vector<cv::KeyPoint> z_pts(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(),
                 z_pts.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> cv::KeyPoint {
                   return feature->GetKeyPoint();
                 });

  std::vector<float> x(z);
  std::vector<float> p(edges.size(), 0);
  std::vector<float> x_bar(x);

  std::vector<float> w(edges.size(), 0);

  for (int outer_iteration = 0; outer_iteration < params_.outer_iterations;
       outer_iteration++) {

    for (int i = 0; i < edges.size(); i++) {
      w[i] = beta / (1 + params_.beta * abs((x[edges[i][1]] - x[edges[i][0]])));
    }

    for (int iteration = 0; iteration < params_.iterations; iteration++) {
      std::vector<float> x_prev(x);

      for (int i = 0; i < edges.size(); i++) {
        float u_p =
            p[i] +
            params_.sigma * (x_bar[edges[i][1]] - x_bar[edges[i][0]]) *
                w[i]; // (z_pts[edges[i][1]].pt.x - z_pts[edges[i][0]].pt.x);
        p[i] = u_p / std::max(std::abs(u_p), 1.0f);
      }

      for (int i = 0; i < edges.size(); i++) {
        x[edges[i][0]] += tau * (p[i]) * w[i]; // (z_pts[edges[i][1]].pt.x -
        // z_pts[edges[i][0]].pt.x);;
        x[edges[i][1]] -= tau * (p[i]) * w[i]; // (z_pts[edges[i][1]].pt.x -
        // z_pts[edges[i][0]].pt.x);;
      }

      // L2 norm
      // std::transform(x.begin(), x.end(), z.begin(), x.begin(),
      //              [=](float const& x_i, float const& z_i) -> float { return
      //              (x_i+ λ * τ * z_i)/(1 + λ * τ); } );

      // L1 norm
      std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                     [=](float const &x_i, float const &z_i) -> float {
                       return (x_i - z_i > params_.lambda * params_.tau
                                   ? x_i - params_.lambda * params_.tau
                                   : (x_i - z_i < -params_.lambda * params_.tau
                                          ? x_i + params_.lambda * params_.tau
                                          : z_i));
                     });

      // Relaxation
      std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                     [=](float const &x_i, float const &x_prev_i) -> float {
                       return x_i + params_.theta * (x_i - x_prev_i);
                     });
    }
  }

  for (int i = 0; i < sparse_supports->size(); i++) {
    sparse_supports->at(i)->UpdateDisparity(x[i]);
  }
}
//------------------------------------------------------------------------------
void MeshRegulariser::run_logTGV(
    utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    const std::vector<Edge> &edges) {

  float sigma = 0.125f;
  float tau = 0.125f;
  float lambda = 1.0f;
  float theta = 1.0f;
  float beta = 1;
  float alpha_1 = 0.5;
  float alpha_2 = 0.8;

  int num_iter_tgv = 200;
  int num_iter_reweighting = 20;

  bool use_l2 = false;

  // Initialise variables
  std::vector<float> z(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(), z.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> float {
                   return feature->GetDisparity();
                 });

  std::vector<cv::KeyPoint> z_pts(sparse_supports->size());
  std::transform(sparse_supports->begin(), sparse_supports->end(),
                 z_pts.begin(),
                 [](utilities::image::FeatureSPtr &feature) -> cv::KeyPoint {
                   return feature->GetKeyPoint();
                 });

  std::vector<float> x(z);
  std::vector<float> p(edges.size(), 0);
  std::vector<float> x_bar(x);

  std::vector<float> y(x.size(), 0);
  std::vector<float> q(edges.size(), 0);
  std::vector<float> y_bar(y);

  std::vector<float> w(edges.size(), 0);
  std::vector<float> w_y(edges.size(), 0);

  for (int iter_outer = 0; iter_outer < params_.outer_iterations;
       iter_outer++) {

    for (int i = 0; i < edges.size(); i++) {
      w[i] = params_.beta /
             (1 + params_.beta *
                      abs((x[edges[i][1]] - x[edges[i][0]]) - y[edges[i][0]]));
      w_y[i] = params_.beta /
               (1 + params_.beta * abs((y[edges[i][1]] - y[edges[i][0]])));
    }

    for (int iter_inner = 0; iter_inner < num_iter_tgv; iter_inner++) {
      std::vector<float> x_prev(x);
      std::vector<float> y_prev(y);
      std::vector<float> p_prev(p);
      std::vector<float> q_prev(q);

      // Dual step
      for (int i = 0; i < edges.size(); i++) {
        float u_p = p[i] + params_.sigma * params_.alpha_1 *
                               (x_bar[edges[i][1]] - x_bar[edges[i][0]] -
                                y_bar[edges[i][0]]) *
                               w[i];
        p[i] = u_p / std::max(std::abs(u_p), 1.0f);
        float u_q = q[i] + params_.sigma * params_.alpha_2 *
                               (y_bar[edges[i][1]] - y_bar[edges[i][0]]) *
                               w_y[i];
        q[i] = u_q / std::max(std::abs(u_q), 1.0f);
      }

      // Primal step
      for (int i = 0; i < edges.size(); i++) {
        x[edges[i][0]] += params_.tau * (p[i]) * w[i];
        x[edges[i][1]] -= params_.tau * (p[i]) * w[i];

        y[edges[i][0]] += params_.tau * params_.alpha_1 * (p_prev[i]);
        y[edges[i][1]] += params_.tau * params_.alpha_1 * (p_prev[i]);

        y[edges[i][0]] += params_.tau * params_.alpha_2 * (q_prev[i]);
        y[edges[i][1]] -= params_.tau * params_.alpha_2 * (q_prev[i]);
      }

      // L2 norm
      // L2 norm
      //        std::transform(x.begin(), x.end(), z.begin(), x.begin(),
      //                       [=](float const &x_i, float const &z_i) -> float
      //                       {
      //                         return (x_i + lambda * tau * z_i) / (1 + lambda
      //                         * tau);
      //                       });
      //      } else {
      // L1 norm
      std::transform(x.begin(), x.end(), z.begin(), x.begin(),
                     [=](float const &x_i, float const &z_i) -> float {
                       return (x_i - z_i > params_.lambda * params_.tau
                                   ? x_i - params_.lambda * params_.tau
                                   : (x_i - z_i < -params_.lambda * params_.tau
                                          ? x_i + params_.lambda * params_.tau
                                          : z_i));
                     });

      // relaxation step
      std::transform(x.begin(), x.end(), x_prev.begin(), x_bar.begin(),
                     [=](float const &x_i, float const &x_prev_i) -> float {
                       return x_i + params_.theta * (x_i - x_prev_i);
                     });
      std::transform(y.begin(), y.end(), y_prev.begin(), y_bar.begin(),
                     [=](float const &y_i, float const &y_prev_i) -> float {
                       return y_i + params_.theta * (y_i - y_prev_i);
                     });
    }
  }

  for (int i = 0; i < sparse_supports->size(); i++) {
    sparse_supports->at(i)->UpdateDisparity(x[i]);
  }
}

} // namespace mesh
} // namespace core
} // namespace aru
