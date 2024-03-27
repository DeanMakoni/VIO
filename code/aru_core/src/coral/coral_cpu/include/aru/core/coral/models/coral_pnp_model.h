#ifndef CORAL_PNP_MODEL_H_
#define CORAL_PNP_MODEL_H_

#include "coral_model.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace aru {
namespace core {
namespace coral {

namespace models {

typedef Eigen::Matrix3d Rot;
typedef Eigen::Vector3d Trans;

class CoralPNPModel : public CoralModelBase {
public:
  CoralPNPModel(Eigen::Matrix3d K);

  virtual ~CoralPNPModel() = default;

public:
  Eigen::MatrixXd EvaluateCost(const utilities::image::FeatureSPtrVectorSptr &features);

  void UpdateModel(const utilities::image::FeatureSPtrVectorSptr &features);

  int ModelDegreesOfFreedom();

  Eigen::MatrixXd ModelEquation();

  void SetCameraParams(Eigen::MatrixXd K);

  Eigen::MatrixXd GetCameraParams() const;

  Eigen::Matrix3d GetRotation() { return R_curr_; }

  Eigen::Vector3d GetTranslation() { return t_curr_; }

  void SetRotation(Eigen::Matrix3d rot) {  R_curr_=rot; }

  void SetTranslation(Eigen::Vector3d trans) { t_curr_=trans; }

  void RelativeError(double &rot_err, double &transl_err, const Rot &R_true,
                     Trans T_true, const Rot &R_est, Trans T_est);

  void MatToQuat(const Rot R, Eigen::Vector4d &q);

private:
  // camera intrinsics
  double u_c_, v_c_, f_u_, f_v_;

  // Current Estimation
  Rot R_curr_;
  Trans t_curr_;
};

} // namespace models
} // namespace coral
} // namespace core
} // namespace aru
#endif // CORAL_PNP_MODEL_H_