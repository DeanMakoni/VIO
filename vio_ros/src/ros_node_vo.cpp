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
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/transform_broadcaster.h"
#include <aru/core/utilities/camera/camera.h>
#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/vo/vo.h>
#include <cmath>
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
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/header.hpp>
#include <thread>

using namespace aru::core;

class ROSVO : public rclcpp::Node {

public:
  explicit ROSVO(const rclcpp::NodeOptions &options);

  virtual ~ROSVO();

protected:
  bool startCamera();
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

  // ----> Thread functions
  void callback_function(const sensor_msgs::msg::Image::SharedPtr msg);
  void imu_callback_function(const sensor_msgs::msg::Imu::SharedPtr msg);
  void sonar_callback_function(const sensor_msgs::msg::Range::SharedPtr msg);
  void depth_callback_function(const sensor msgs::msg::FluidPressure::SHaredPtr msg);

private:
  // Shared pointer to Visual Odometry Class
  std::shared_ptr<aru::core::vo::VO> vo_;

  // Shared pointer to transform map for keyframe stuff
  std::shared_ptr<aru::core::utilities::transform::TransformMap> transform_map_;

  aru::core::utilities::image::StereoImage image_stereo_;
  aru::core::utilities::image::StereoImage image_stereo_prev_;

  aru::core::utilities::image::StereoImage image_key_;

  bool prev_image = false;
  uint64_t prev_timestamp;

  // subscribers
  
  // Camera subscriber
  std::string stereo_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber;

  // Imu subscriber
  std::string imu_topic;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr
      imu_subscriber;

  // sonar subscriber
  std::string::sonar_topic;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr
      sonar_subscriber;

  // Pressure subscriber
  std::string depth_topic
  rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr
      depth_subscriber;

  // Publisher
  std::string vo_tf2_topic_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      vo_tf2_publisher_;

  std::string kf_tf2_topic_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      kf_tf2_publisher_;

  std::string kf_image_topic_;
  image_transport::CameraPublisher kf_image_publisher_;

  // keyframe selectors
  float min_distance_;
  float min_rotation_;
  Eigen::Affine3f curr_position_;

  // image_downsample
  int down_sample_;
  uint count;
};

ROSVO::ROSVO(const rclcpp::NodeOptions &options) : Node("vo_node", options) {

  using namespace std::placeholders;
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS VO ");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  stereo_topic_ = "camera/bumblebee/image_rectified";
  imu_topic = "imu/imu";
  sonar_topic = "imagenex831l/range";
  depth_topic = "bar30/depth";
  vo_tf2_topic_ = "vo/tf2";
  kf_tf2_topic_ = "kf/tf2";
  kf_image_topic_ = "camera/bumblebee/image_keyframe";

  min_distance_ = 2.0;
  min_rotation_ = 0.2536;

  count = 0;
  down_sample_ = 15;

  image_stereo_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic_, 10,
      std::bind(&ROSVO::callback_function, this, std::placeholders::_1));

  imu_subsrciber = this->create_subscription<sensor_msgs::msgs::Imu>(
      imu_topic, 10,
      std::bind(&ROSVO::imu_callback_function, this, std::placeholders::_1));

  sonar_subscriber = this->create_subscription<sensor_msgs::msgs::Range>(
      sonar_topic, 10, 
      std::bind(&ROSVO::sonar_callback_function, this, std::placeholders::_1));

  depth_subcriber = this->create_subscription<sensor_msgs::msgs::FluidPressure>(
      depth_topic, 10,
      std::bind(&ROSVO::depth_callback_function, this, std::placeholders::_1));
  // Initialise VO params
  auto vo_config_file = "/home/administrator/code/BB_vo_config.yaml";
  auto vo_vocab_file = "/home/administrator/code/ORBvoc.txt";

  // Initialise VO class
  vo_ = std::make_shared<aru::core::vo::VO>(vo_config_file, vo_vocab_file);

  // Define Publisher
  vo_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
          vo_tf2_topic_, 10);
  kf_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
          kf_tf2_topic_, 10);
  kf_image_publisher_ =
      image_transport::create_camera_publisher(this, kf_image_topic_);
  RCLCPP_INFO_STREAM(get_logger(),
                     "Advertised on topic: " << kf_image_publisher_.getTopic());

  // Initialise transform map
  transform_map_ =
      std::make_shared<aru::core::utilities::transform::TransformMap>();
}

ROSVO::~ROSVO() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void ROSVO::callback_function(const sensor_msgs::msg::Image::SharedPtr msg) {

  // Get images from msg
  cv::Mat image_stereo;
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  cv::Mat temp_image = cv_ptr->image;
  image_stereo = temp_image;

  // Split Images
  cv::Size s = temp_image.size();
  int width = s.width;
  cv::Mat left_image = temp_image(cv::Rect(0, 0, width / 2, s.height));
  cv::Mat right_image = temp_image(cv::Rect(width / 2, 0, width / 2, s.height));

  // Get timestamp
  std_msgs::msg::Header h = msg->header;
  uint64_t seconds = h.stamp.sec;
  uint64_t nanoseconds = h.stamp.nanosec;
  uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);



  // Detect and match features in the left and right images
  matcher_ = boost::make_shared<aru::core::utilities::image::OrbFeatureMatcher>(
             match_params, extractor_params);
  aru::core::utilities::image::FeatureSPtrVectorSptr feature = 
              matcher->ComputeStereoMatches(left_image,right_image);


  // Add StereoImage
  image_stereo_.first =
      aru::core::utilities::image::Image(time_out, left_image);
  image_stereo_.second =
      aru::core::utilities::image::Image(time_out, right_image);

  if (count > 0) {

    // cv::imshow("Left Image", left_image);
    // cv::waitKey(0);

    // EstimateMotion
    aru::core::utilities::transform::Transform transform =
        vo_->EstimateMotion(image_stereo_prev_, image_stereo_);
    Eigen::Affine3d transform_eigen = transform.GetTransform().cast<double>();
    // Add to the transform map
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            transform);
    transform_map_->AddTransform(curr_transform_sptr);

    // Check the distance moved from the last keyframe
    // pose is T_prev_curr. Source is curr_image dest is prev_image
    utilities::transform::TransformSPtr pose =
        transform_map_->Interpolate(prev_timestamp, time_out);
    if (pose) {
      float dist = pose->GetTransform().translation().norm();
      cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
      cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
      // Read Rotation matrix and convert to vector
      cv::eigen2cv(pose->GetRotation(), R_matrix);
      cv::Rodrigues(R_matrix, rvec);
      Eigen::Vector3f rpy;
      cv::cv2eigen(rvec, rpy);
      float rotation = rpy.norm();
      if (dist > min_distance_ || rotation > min_rotation_) {
        prev_timestamp = time_out;
        image_key_ = image_stereo_;

        // Publish key frame transform
        Eigen::Affine3d pose_eigen = pose->GetTransform().cast<double>();
        geometry_msgs::msg::TransformStamped t =
            tf2::eigenToTransform(pose_eigen);
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = "camera";
        t.child_frame_id = "camera";
        kf_tf2_publisher_->publish(t);

        // Publish key frame image
        auto mStereoCamInfoMsg =
            std::make_shared<sensor_msgs::msg::CameraInfo>();

        // Publish the rectified images
        auto msg_kf =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_stereo)
                .toImageMsg();

        msg_kf->header = h;

        kf_image_publisher_.publish(msg_kf, mStereoCamInfoMsg);
      }
    }

    // Read message content and assign it to
    // corresponding tf variables
    geometry_msgs::msg::TransformStamped t =
        tf2::eigenToTransform(transform_eigen);
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "camera";
    t.child_frame_id = "camera";

    vo_tf2_publisher_->publish(t);

  } else {
    prev_image = true;
    prev_timestamp = time_out;
    LOG(INFO) << "Time out is updated";

    // Add an identity transform to the map for the first image
    Eigen::Affine3f curr_position;
    curr_position_.linear() = Eigen::MatrixXf::Identity(3, 3);
    curr_position_.translation() = Eigen::VectorXf::Zero(3);

    utilities::transform::TransformSPtr init_transform =
        boost::make_shared<utilities ::transform::Transform>(time_out, time_out,
                                                             curr_position_);
    transform_map_->AddTransform(init_transform);
    image_key_ = image_stereo_;

    // output the first image
    // Publish key frame image
    auto mStereoCamInfoMsg = std::make_shared<sensor_msgs::msg::CameraInfo>();

    // Publish the rectified images
    auto msg_kf =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_stereo)
            .toImageMsg();

    msg_kf->header = h;

    kf_image_publisher_.publish(msg_kf, mStereoCamInfoMsg);
  }
  // Update the previous image to the current image
  image_stereo_prev_ = image_stereo_;
  count++;
}

void imu_callback_function(const sensor_msgs::msg::Imu::SharePtr msg){
 
 Vector3 measuredAcc (msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
 Vector3 measuredOmega (msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

 // Check on how to add the correct dt

 preintegrated->integrateMeasurement(measuredAcc,measuredOmega, dt);

}

void sonar_callback_function(const sensor_msgs::msg::Range::SharedPtr msg){

}

void depth_callback_function(const sensor_msgs::msg::FluidPressure::SharedPtr msg){

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
