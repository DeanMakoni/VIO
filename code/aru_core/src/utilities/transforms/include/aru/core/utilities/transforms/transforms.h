#ifndef ARU_UTILITIES_TRANSFORM_H_
#define ARU_UTILITIES_TRANSFORM_H_

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace transform {

class Transform {
public:
  Transform() = default;
//
//  Transform(Transform const &transform_in);

  Transform(std::string source, std::string destination,
            const Eigen::Affine3f &transform);

  Transform(int64_t source_timestamp, int64_t dest_timestamp,
            const Eigen::Affine3f &transform);

  Transform(std::string source, std::string destination,
            int64_t source_timestamp, int64_t dest_timestamp,
            const Eigen::Affine3f &transform);

  virtual ~Transform() = default;

  std::string GetSourceFrame() { return source_frame_; }
  std::string GetDestinationFrame() { return destination_frame_; }

  int64_t GetSourceTimestamp() const { return source_timestamp_; }
  int64_t GetDestinationTimestamp() const { return destination_timestamp_; }

  void SetSourceTimestamp(int64_t source_timestamp) {
    source_timestamp_ = source_timestamp;
  }

  void SetDestTimestamp(int64_t dest_timestamp) {
    destination_timestamp_ = dest_timestamp;
  }

  void RightCompose(Transform transform);

  Eigen::Affine3f GetTransform() { return transform_; }

  Eigen::Matrix3f GetRotation() { return transform_.linear(); };

  Eigen::Vector3f GetTranslation() { return transform_.translation(); };

  static Eigen::Vector3f RPYFromRotationMatrix(Eigen::Matrix3f rot);

  static Eigen::Matrix3f RotationMatrixFromRPY(Eigen::Vector3f rpy);

private:
  Eigen::Affine3f transform_;
  std::string source_frame_;
  std::string destination_frame_;
  int64_t source_timestamp_;
  int64_t destination_timestamp_;
};
Transform inline TransformInverse(Transform input) {
  Transform inverse(input.GetDestinationFrame(), input.GetSourceFrame(),
                    input.GetSourceTimestamp(), input.GetDestinationTimestamp(),
                    input.GetTransform().inverse());
  return inverse;
}

typedef boost::shared_ptr<Transform> TransformSPtr;
typedef std::vector<TransformSPtr> TransformSPtrVector;
typedef boost::shared_ptr<TransformSPtrVector> TransformSPtrVectorSptr;

} // namespace transform
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_IMAGE_H_
