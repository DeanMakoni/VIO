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
#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <thread>

// #include <boost/filesystem/operations.hpp>
// #include <boost/filesystem/path.hpp>

class ROSCamera : public rclcpp::Node {

public:
  explicit ROSCamera(const rclcpp::NodeOptions &options);

  virtual ~ROSCamera();

protected:
  bool startCamera();
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

  // callback functions
  void callback_function(const sensor_msgs::msg::Image::SharedPtr msg) {
    // Read the message into opencv
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat temp_image = cv_ptr->image;

    // Undistort and rectify the images
    cv::Size s = temp_image.size();
    int width = s.width;

    cv::Mat left_image = temp_image(cv::Rect(0, 0, width / 2, s.height));
    cv::Mat right_image =
        temp_image(cv::Rect(width / 2, 0, width / 2, s.height));

    cv::Mat left_image_rectified =
        stereo_cam_model_->UndistortRectifyLeft(left_image);

    cv::Mat right_image_rectified =
        stereo_cam_model_->UndistortRectifyRight(right_image);

    // Combine the two images
    cv::Mat image_rectified;
    cv::hconcat(left_image_rectified, right_image, image_rectified);

    auto mStereoCamInfoMsg = std::make_shared<sensor_msgs::msg::CameraInfo>();

    // Publish the rectified images
    auto msg_stereo =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_rectified)
            .toImageMsg();
            
    msg_stereo->header.stamp = this->get_clock()->now();
    mPubStereoRectified.publish(msg_stereo, mStereoCamInfoMsg);
  }

private:
  // Shared pointer for camera model
  std::shared_ptr<aru::core::utilities::camera::StereoCameraModel>
      stereo_cam_model_;

  // subscribers
  std::string stereo_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber;

  // Publisher
  std::string rectified_topic_;
  image_transport::CameraPublisher mPubStereoRectified;
};

ROSCamera::ROSCamera(const rclcpp::NodeOptions &options)
    : Node("camera_node", options) {

  using namespace std::placeholders;

  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS Rectification Node");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  stereo_topic_ = "camera/zed/image_raw";
  rectified_topic_ = "camera/zed/image_rectified";

  image_stereo_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic_, 10,
      std::bind(&ROSCamera::callback_function, this, std::placeholders::_1));

  // Define Publisher
  mPubStereoRectified =
      image_transport::create_camera_publisher(this, rectified_topic_);
  RCLCPP_INFO_STREAM(get_logger(),
                     "Advertised on topic: " << mPubStereoRectified.getTopic());

  // Camera Model params
  auto cam_model_left_file =
      "/home/administrator/code/aru_calibration/ZED/left.yaml";
  auto cam_model_right_file =
      "/home/administrator/code/aru_calibration/ZED/right.yaml";

  // Initialise camera model class
  stereo_cam_model_ =
      std::make_shared<aru::core::utilities::camera ::StereoCameraModel>(
          cam_model_left_file, cam_model_right_file);
}
ROSCamera::~ROSCamera() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

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
  auto rectify_node = std::make_shared<ROSCamera>(options);
  exec.add_node(rectify_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
