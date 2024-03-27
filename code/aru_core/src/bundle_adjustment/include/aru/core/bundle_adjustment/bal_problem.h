#ifndef ARU_CORE_BUNDLE_ADJUSTMENT_BAL_PROBLEM_H_
#define ARU_CORE_BUNDLE_ADJUSTMENT_BAL_PROBLEM_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <opencv2/opencv.hpp>

namespace aru {
namespace core {
namespace bundle_adjustment {

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  T *residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T &l1 = camera[7];
    const T &l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // Compute final projected point position.
    const T &focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
struct SnavelyReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  T *residuals) const {
    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without a local parameterization.
    T p[3];
    ceres::QuaternionRotatePoint(camera, point, p);
    p[0] += camera[4];
    p[1] += camera[5];
    p[2] += camera[6];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T &l1 = camera[8];
    const T &l2 = camera[9];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // Compute final projected point position.
    const T &focal = camera[7];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<
            SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
        new SnavelyReprojectionErrorWithQuaternions(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};

class BALProblem {
public:
  explicit BALProblem(const std::string &filename, bool use_quaternions);
  ~BALProblem();
  void WriteToFile(const std::string &filename) const;
  void WriteToPLYFile(const std::string &filename) const;
  // Move the "center" of the reconstruction to the origin, where the
  // center is determined by computing the marginal median of the
  // points. The reconstruction is then scaled so that the median
  // absolute deviation of the points measured from the origin is
  // 100.0.
  //
  // The reprojection error of the problem remains the same.
  void Normalize();
  // Perturb the camera pose and the geometry with random normal
  // numbers with corresponding standard deviations.
  void Perturb(const double rotation_sigma, const double translation_sigma,
               const double point_sigma);
  // clang-format off
  int camera_block_size()      const { return use_quaternions_ ? 10 : 9; }
  int point_block_size()       const { return 3;                         }
  int num_cameras()            const { return num_cameras_;              }
  int num_points()             const { return num_points_;               }
  int num_observations()       const { return num_observations_;         }
  int num_parameters()         const { return num_parameters_;           }
  const int* point_index()     const { return point_index_;              }
  const int* camera_index()    const { return camera_index_;             }
  const double* observations() const { return observations_;             }
  const double* parameters()   const { return parameters_;               }
  const double* cameras()      const { return parameters_;               }
  double* mutable_cameras()          { return parameters_;               }
  // clang-format on
  double *mutable_points() {
    return parameters_ + camera_block_size() * num_cameras_;
  }

private:
  void CameraToAngleAxisAndCenter(const double *camera, double *angle_axis,
                                  double *center) const;
  void AngleAxisAndCenterToCamera(const double *angle_axis,
                                  const double *center, double *camera) const;
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  bool use_quaternions_;
  int *point_index_;
  int *camera_index_;
  double *observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double *parameters_;
};
} // namespace bundle_adjustment
} // namespace core
} // namespace aru

#endif // ARU_CORE_BUNDLE_ADJUSTMENT_BAL_PROBLEM_H_
