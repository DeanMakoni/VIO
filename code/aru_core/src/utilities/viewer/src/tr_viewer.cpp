#include <Eigen/Dense>
#include <aru/core/utilities/viewer/tr_viewer.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <pangolin/gl/gl.h>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace viewer {
//------------------------------------------------------------------------------
TRViewer::TRViewer(
    int height, int width,
    utilities::transform::TransformSPtrVectorSptr teach_pose_chain,
    utilities::transform::TransformSPtrVectorSptr repeat_pose_chain)
    : height_(height), width_(width) {

  teach_pose_handler_ = std::make_shared<PoseViewer>(teach_pose_chain);
  repeat_pose_handler_ = std::make_shared<PoseViewer>(repeat_pose_chain);
}
//------------------------------------------------------------------------------
void TRViewer::Run() {

  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

  pangolin::Renderable tree;
  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View &view) {
    view.Activate(s_cam);
    tree.Render();
  });

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Twc = teach_pose_handler_->GetCurrentPose();
    s_cam.Follow(Twc);

    // Draw Teach Poses
    teach_pose_handler_->DrawPoses();

    Eigen::Vector3f color(1, 0, 0);
    repeat_pose_handler_->DrawPoses(color);

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow("Main");
}
//------------------------------------------------------------------------------

} // namespace viewer
} // namespace utilities
} // namespace core
} // namespace aru
