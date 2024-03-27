
#include <Eigen/Dense>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "aru/core/mesh/mesh.h"

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  aru::core::utilities::logging::ProtocolLogger<datatype::image::pbStereoImage>
      logger("/home/paulamayo/data/husky_data/log/outdoor_zoo.monolithic",
             false);
  for (int i = 1; i < 30; ++i) {
    datatype::image::pbStereoImage pb_image = logger.ReadFromFile();
    if (pb_image.has_image_left() && pb_image.has_image_right()) {

      aru::core::utilities::image::StereoImage stereo_image =
          aru::core::utilities::image::ImageProtocolBufferAdaptor ::
              ReadStereoFromProtocolBuffer(pb_image);

      cv::Mat image_left = stereo_image.first.GetImage();
      cv::Mat image_right = stereo_image.second.GetImage();
      //
      aru::core::mesh::Mesh mesh_estimator(
          "/home/paulamayo/data/husky_data/mesh/husky_mesh_depth_zed.yaml");
      LOG(INFO) << "Start Depth Estimate";
      mesh_estimator.EstimateMesh(image_left, image_right);

      cv::Mat disparity = mesh_estimator.GetInterpolatedDepth();
      std::vector<Eigen::Vector3d> vertices = mesh_estimator.GetMeshFeatures();
      std::vector<Eigen::Vector3i> triangles =
          mesh_estimator.GetMeshTriangles();
      //      aru::core::utilities::viewer::MeshViewer mesh_viewer(
      //          image_left.rows, image_left.cols,
      //          mesh_estimator.GetCameraIntrinsic());
      //
      //      mesh_viewer.ViewMesh(image_left, disparity, vertices, triangles);
    }
  }

  return 0;
}
