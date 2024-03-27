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

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include <aru/core/utilities/camera/camera.h>
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
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <thread>

// #include <boost/filesystem/operations.hpp>
// #include <boost/filesystem/path.hpp>

class ROSVO : public rclcpp::Node {

public:
  explicit ROSVO(const rclcpp::NodeOptions &options);

  virtual ~ROSVO();

protected:
  bool startCamera();
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

  // ----> Thread functions
  void threadFunc_Video();

  void runVO();
  // callback functions
  void callback_function(const sensor_msgs::msg::Image::SharedPtr msg) {
    // Update the previous image to the current image
    image_stereo_prev_ = image_stereo_;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat temp_image = cv_ptr->image;
    image_stereo_ = temp_image;
    // Undistort and rectify the images

    // todo: use the camera class
  }

private:
  // Shared pointer to Visual Odometry Class
  std::shared_ptr<aru::core::vo::VO> vo_;

  // Shared pointer for camera model
  // std::shared_ptr<aru::core::utilities::camera::CameraModel> cam_model_;

  // subscribers
  std::string stereo_topic_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber;

  cv::Mat image_stereo_;

  cv::Mat image_stereo_prev_;

  // Publisher
  std::shared_ptr<tf2_ros::TransformBroadcaster> vo_broadcaster_;
};

ROSVO::ROSVO(const rclcpp::NodeOptions &options) : Node("vo_node", options) {
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS VO ");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  stereo_topic_ = "camera/image_stereo/image_raw";

  image_stereo_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic_, 10,
      std::bind(&ROSVO::callback_function, this, std::placeholders::_1));

  // Initialise VO params
  auto vo_config_file = "/home/administrator/code/ZED_vo_config.yaml";
  auto vo_vocab_file = "/home/administrator/code/ORBvoc.txt";

  // Initialise VO class
  vo_ = std::make_shared<aru::core::vo::VO>(vo_config_file, vo_vocab_file);

  // Initialise boradcaster
  vo_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  // Define Publisher
  // vo_Pub = this->create_publisher<>("TF2",10);

  // Camera Model params
  // auto
  // cam_model_file="/home/administrator/code/aru_calibration/ZED/left.yaml";

  // Initialise camera model class
  // cam_model_=std::make_shared<aru::core::utilities::camera::CameraModel>(cam_model_file);
}
ROSVO::~ROSVO() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void ROSVO::runVO() {

  aru::core::utilities::image::StereoImage prev_image;

  prev_image.first = aru::core::utilities::image::Image(0, image_stereo_prev_);
  prev_image.second = aru::core::utilities::image::Image(0, image_stereo_prev_);

  aru::core::utilities::image::StereoImage curr_image;
  curr_image.first = aru::core::utilities::image::Image(0, image_stereo_);
  curr_image.second = aru::core::utilities::image::Image(0, image_stereo_);

  auto pose = vo_->EstimateMotion(prev_image, curr_image);

  geometry_msgs::msg::TransformStamped t;

  /***
  // Read message content and assign it to
  // corresponding tf variables
  t.header.stamp = this->get_clock()->now();
  t.header.frame_id = "world";
  t.child_frame_id = turtlename_.c_str();

  // Turtle only exists in 2D, thus we get x and y translation
  // coordinates from the message and set the z coordinate to 0
  t.transform.translation.x = msg->x;
  t.transform.translation.y = msg->y;
  t.transform.translation.z = 0.0;

  // For the same reason, turtle can only rotate around one axis
  // and this why we set rotation in x and y to 0 and obtain
  // rotation in z axis from the message
  tf2::Quaternion q;
  q.setRPY(0, 0, msg->theta);
  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();
   ***/

  // Send the transformation
  vo_broadcaster_->sendTransform(t);
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

  // Add VO Ros node
  auto vo_node = std::make_shared<ROSVO>(options);
  exec.add_node(vo_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
