//
// Created by paulamayo on 2022/05/04.
//

#ifndef ARU_CORE_TRANSFORM_BUFFER_H
#define ARU_CORE_TRANSFORM_BUFFER_H

#include "transforms.h"
#include <map>

namespace aru {
namespace core {
namespace utilities {
namespace transform {

class TransformMap {

  /**
   * Mapping from timestamps to Transforms useful for interpolation
   */
  typedef std::map<int64_t, TransformSPtr> TimeTransformMap;

public:
  TransformMap() = default;

  virtual ~TransformMap() = default;

  void AddTransform(TransformSPtr transform);

  TransformSPtr Interpolate(int64_t t_start, int64_t t_end);

  TransformSPtr Interpolate(int64_t t_end);

  bool IsWindowValid(int64_t t_start,int64_t t_end);

private:
  TimeTransformMap time_transform_map_;
};
} // namespace transform
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_CORE_TRANSFORM_BUFFER_H
