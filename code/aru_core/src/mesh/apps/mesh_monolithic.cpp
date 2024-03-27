
#include <aru/core/mesh/system.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
DEFINE_string(MESH_CONFIG, "", "path to Mesh Config file");
DEFINE_string(IMAGE_LEFT, "", "path to Image left monolithic file");
DEFINE_string(IMAGE_RIGHT, "", "path to Image right monolithic file");
DEFINE_string(MESH_DEPTH_MONO, "", "path to Mesh Depth output monolithic file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // TODO: Check that all files exist otherwise throw an error

  LOG(INFO) << "This is an info  message";

  mesh::System MeshDepthEstimator(FLAGS_MESH_CONFIG, FLAGS_IMAGE_LEFT,
                                  FLAGS_IMAGE_RIGHT, FLAGS_MESH_DEPTH_MONO);

  MeshDepthEstimator.Run();
  return 0;
}
