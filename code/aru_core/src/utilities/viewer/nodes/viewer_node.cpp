// /////////////////////////////////////////////////////////////////////////

//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// /////////////////////////////////////////////////////////////////////////

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/transform_broadcaster.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/vo_viewer.h>
#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/int64.hpp>
#include <thread>
#include <boost/make_shared.hpp>

using namespace aru::core;
using namespace aru::core::utilities;

class Ros_Vo_Viewer : public rclcpp::Node {

public:
  explicit Ros_Vo_Viewer(const rclcpp::NodeOptions &options);

  virtual ~Ros_Vo_Viewer();

protected:
  // callback functions
  void tf_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {

    Eigen::Affine3d transform_eigen = tf2::transformToEigen(*msg);
    Eigen::Affine3f transform_eigen_f = transform_eigen.cast<float>();

    // Get timestamp
    std_msgs::msg::Header h = msg->header;
    uint64_t seconds = h.stamp.sec;
    uint64_t nanoseconds = h.stamp.nanosec;
    uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);

    aru::core::utilities::transform::Transform pose(time_out, time_out,
                                                    transform_eigen_f);
                                                    
    curr_position = curr_position * pose.GetTransform();

    transform::TransformSPtr curr_transform =
        boost::make_shared<utilities ::transform::Transform>(0, 0,
                                                             curr_position);

    // Add to chain
    pose_chain_->push_back(curr_transform);
  }

private:
  // subscribers
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
      transform_subscriber_;

  // Pose Chain
  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  std::shared_ptr<aru::core::utilities::viewer::VOViewer> vo_viewer_;
  
  Eigen::Affine3f curr_position;
};

Ros_Vo_Viewer::Ros_Vo_Viewer(const rclcpp::NodeOptions &options)
    : Node("repeat_node", options) {
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS REPEAT ");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  // Set up the parameters for the logging
  this->declare_parameter("vocab_file", "/home/administrator/data/ORBvoc.txt");
  this->declare_parameter(
      "image_left_teach_monolithic",
      "/home/administrator/data/image_left_teach.monolithic");
  this->declare_parameter(
      "image_right_teach_monolithic",
      "/home/administrator/data/image_right_teach.monolithic");
  this->declare_parameter("transform_monolithic",
                          "/home/administrator/data/transform.monolithic");
  this->declare_parameter("stereo_topic", "camera/bumblebee/image_keyframe");
  this->declare_parameter("transform_topic", "kf/tf2");

  // retrieve ROS parameters
  std::string transform_monolithic = this->get_parameter("transform_monolithic")
                                         .get_parameter_value()
                                         .get<std::string>();
  std::string stereo_topic = this->get_parameter("stereo_topic")
                                 .get_parameter_value()
                                 .get<std::string>();
  std::string transform_topic = this->get_parameter("transform_topic")
                                    .get_parameter_value()
                                    .get<std::string>();

  pose_chain_ = boost::make_shared<aru::core::utilities::transform::TransformSPtrVector>();

  vo_viewer_ = std::make_shared<aru::core::utilities::viewer::VOViewer>(
      640, 480, pose_chain_);
      
  // Add initial transform
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);
  
  transform::TransformSPtr curr_transform =
        boost::make_shared<utilities ::transform::Transform>(0, 0,
                                                             curr_position);

    pose_chain_->push_back(curr_transform);

  // Initialise subscribers
  transform_subscriber_ =
      this->create_subscription<geometry_msgs::msg ::TransformStamped>(
          transform_topic, 10,
          std::bind(&Ros_Vo_Viewer::tf_callback, this, std::placeholders::_1));

  // start threads
  auto viewer_thread =
      new std::thread(&utilities::viewer::VOViewer::Run, vo_viewer_);
}
Ros_Vo_Viewer::~Ros_Vo_Viewer() {
  RCLCPP_DEBUG(get_logger(), "Destroying node");
}

int main(int argc, char **argv) {

  // Force flush of the stdout buffer.
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);

  // Initialize any global resources needed by the middleware and the client
  // library. This will also parse command line arguments one day (as of Beta 1
  // they are not used). You must call this before using any other part of the
  // ROS system. This should be called once per process.
  rclcpp::init(argc, argv);

  // Create an executor that will be responsible for execution of callbacks for
  // a set of nodes. With this version, all callbacks will be called from within
  // this thread (the main one).
  rclcpp::executors::MultiThreadedExecutor exec;
  rclcpp::NodeOptions options;

  options.use_intra_process_comms(true);

  // Add Teach  node
  auto view_node = std::make_shared<Ros_Vo_Viewer>(options);
  exec.add_node(view_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
