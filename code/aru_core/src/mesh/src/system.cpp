
#include "aru/core/mesh/system.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
using namespace datatype::image;
using namespace datatype::transform;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace mesh {
//------------------------------------------------------------------------------
System::System(std::string mesh_config_file, std::string image_left_monolithic,
               std::string image_right_monolithic,
               std::string mesh_depth_monolithic) {

  mesh_ = boost::make_shared<Mesh>(mesh_config_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
  mesh_depth_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      mesh_depth_monolithic, true);
}
//------------------------------------------------------------------------------
void System::Run() {
  // Read left and right images
  pbImage image_left_curr = image_left_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_logger_->ReadFromFile();

  int num = 0;

  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {

    // Perform the estimation
    image::Image curr_image_left =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);
    image::Image curr_image_right =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);

    auto estimation_start = std::chrono::high_resolution_clock::now();
    mesh_->EstimateMesh(curr_image_left.GetImage(),
                        curr_image_right.GetImage());
    cv::Mat depth = mesh_->GetInterpolatedDepth();
    cv::Mat disparity = mesh_->DepthToDisparity(depth);
    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
    LOG(INFO) << "Depth estimation runs at " << 1 / elapsed.count() << " Hz";

    std::vector<cv::KeyPoint> keypoints = mesh_->GetVerticeKeypoints();
    std::vector<double> depths = mesh_->GetVerticeDepths();
    std::vector<Eigen::Vector3i> triangles = mesh_->GetMeshTriangles();

    cv::Mat image_clone = curr_image_left.GetImage().clone();
    float max_depth = 50;
    viewer::Viewer::ViewMeshWireFrame(image_clone, keypoints, depths, triangles,
                                      max_depth);
    viewer::Viewer::ViewInterpolatedMesh(image_clone, depth, max_depth);
    viewer::Viewer::ViewDisparity(disparity, 50);
    cv::waitKey(1);

    pbImage pb_disparity =
        utilities::image ::ImageProtocolBufferAdaptor::ReadToProtocolBuffer(
            image::Image(curr_image_left.GetTimeStamp(), disparity));
    mesh_depth_logger_->WriteToFile(pb_disparity);

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();

    LOG(INFO) << "Num is " << num;
    num++;
  }
}

} // namespace mesh
} // namespace core
} // namespace aru