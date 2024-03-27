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
#include <aru/core/utilities/camera/camera.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/vo/vo.h>
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

using namespace aru::core;
using namespace datatype::image;
using namespace datatype::transform;
using namespace aru::core::utilities;

class Ros_Teach : public rclcpp::Node {

public:
  explicit Ros_Teach(const rclcpp::NodeOptions &options);

  virtual ~Ros_Teach();

protected:
  // void runRepeat();

  // callback functions
  void tf_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {

    Eigen::Affine3d transform_eigen= tf2::transformToEigen(*msg);
    Eigen::Affine3f transform_eigen_f=transform_eigen.cast<float>();

    // Get timestamp
    std_msgs::msg::Header h = msg->header;
    uint64_t seconds=h.stamp.sec;
    uint64_t nanoseconds=h.stamp.nanosec;
    uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);

    aru::core::utilities::transform::Transform pose(
        time_out, time_out, transform_eigen_f);

    // Log transform
    pbTransform pb_transform = utilities::transform ::
        TransformProtocolBufferAdaptor::ReadToProtocolBuffer(pose);
    transform_logger_->WriteToFile(pb_transform);
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // Get images from msg
    LOG(INFO)<<"Image callback";
        cv::Mat image_stereo;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat temp_image = cv_ptr->image;
    image_stereo = temp_image;

    // Split Images
    cv::Size s = temp_image.size();
    int width = s.width;
    cv::Mat left_image = temp_image(cv::Rect(0, 0, width / 2, s.height));
    cv::Mat right_image =
        temp_image(cv::Rect(width / 2, 0, width / 2, s.height));

    // Get timestamp
    std_msgs::msg::Header h = msg->header;
    uint64_t seconds=h.stamp.sec;
    uint64_t nanoseconds=h.stamp.nanosec;
    uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);

    // Log images
    image_left_teach_logger_->WriteToFile(
        utilities::image::ImageProtocolBufferAdaptor::ReadToProtocolBuffer(
            utilities::image::Image(time_out, left_image)));
    image_right_teach_logger_->WriteToFile(
        utilities::image::ImageProtocolBufferAdaptor::ReadToProtocolBuffer(
            utilities::image::Image(time_out, right_image)));
  }

private:
  // subscribers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
      transform_subscriber_;

  // Loggers
  std::shared_ptr<utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_teach_logger_;
  std::shared_ptr<utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_teach_logger_;
  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      transform_logger_;
};

Ros_Teach::Ros_Teach(const rclcpp::NodeOptions &options)
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
  std::string image_left_teach_monolithic =
      this->get_parameter("image_left_teach_monolithic")
          .get_parameter_value()
          .get<std::string>();
  std::string image_right_teach_monolithic =
      this->get_parameter("image_right_teach_monolithic")
          .get_parameter_value()
          .get<std::string>();
  std::string transform_monolithic = this->get_parameter("transform_monolithic")
                                         .get_parameter_value()
                                         .get<std::string>();
  std::string stereo_topic = this->get_parameter("stereo_topic")
                                 .get_parameter_value()
                                 .get<std::string>();
  std::string transform_topic = this->get_parameter("transform_topic")
                                    .get_parameter_value()
                                    .get<std::string>();

  // Initlaise loggers
  image_left_teach_logger_ = std::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_teach_monolithic, true);
  image_right_teach_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_right_teach_monolithic, true);
  transform_logger_ = std::make_shared<logging::ProtocolLogger<pbTransform>>(
      transform_monolithic, true);

  // Initialise subscribers
  image_stereo_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic, 10,
      std::bind(&Ros_Teach::image_callback, this, std::placeholders::_1));

  transform_subscriber_ =
      this->create_subscription<geometry_msgs::msg ::TransformStamped>(
          transform_topic, 10,
          std::bind(&Ros_Teach::tf_callback, this, std::placeholders::_1));
}
Ros_Teach::~Ros_Teach() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

/*void ROSREPEAT::runRepeat() {

}*/

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
  auto teach_node = std::make_shared<Ros_Teach>(options);
  exec.add_node(teach_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
