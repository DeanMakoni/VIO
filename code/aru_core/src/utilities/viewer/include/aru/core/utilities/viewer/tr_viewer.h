//
// Created by paulamayo on 2022/04/14.
//

#ifndef ARU_CORE_TR_VIEWER_H
#define ARU_CORE_TR_VIEWER_H

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include <aru/core/utilities/image/point_feature.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

namespace aru {
namespace core {
namespace utilities {
namespace viewer {

class TRViewer {
public:
  TRViewer(int height, int width,
           utilities::transform::TransformSPtrVectorSptr teach_pose_chain,
           utilities::transform::TransformSPtrVectorSptr repeat_pose_chain);

  TRViewer() = default;

  ~TRViewer() = default;

  void Run();

private:
  std::shared_ptr<PoseViewer> teach_pose_handler_;
  std::shared_ptr<PoseViewer> repeat_pose_handler_;
  int height_;
  int width_;
};
} // namespace viewer
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_VIEWER_H
