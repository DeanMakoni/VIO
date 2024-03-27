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
#include <aru/core/repeat/repeat.h>
#include <aru/core/utilities/viewer/tr_viewer.h>
#include <aru/core/utilities/camera/camera.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <boost/make_shared.hpp>
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

// #include <boost/filesystem/operations.hpp>
// #include <boost/filesystem/path.hpp>

using namespace datatype::image;
using namespace datatype::transform;
using namespace aru::core::utilities;
using namespace aru::core;

class Ros_Repeat : public rclcpp::Node {

public:
  explicit Ros_Repeat(const rclcpp::NodeOptions &options);

  virtual ~Ros_Repeat();

protected:
  // void runRepeat();

  // callback functions
  void tf_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {

    Eigen::Affine3d transform_eigen= tf2::transformToEigen(*msg);
    Eigen::Affine3f transform_eigen_f=transform_eigen.cast<float>();

    // Get timestamp
    std_msgs::msg::Header h = msg->header;
    int64 time_out = h.stamp.nanosec;

    aru::core::utilities::transform::Transform pose(
        time_out, time_out, transform_eigen_f);

    // Find current position transform Map maybe?
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


    // Add the repeat keyframes to the map
    image::StereoImage curr_image;
    curr_image.first =utilities::image::Image(time_out, left_image);
    curr_image.second =utilities::image::Image(time_out, right_image);
    repeat_->QueryRepeatframe(curr_image);
    LOG(INFO)<<"Queried image";

  }


  void CreateTeachMap();

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

  // repeat shared_ptr
  std::shared_ptr<aru::core::repeat::Repeat> repeat_;
  
  //pointer to viewer
  std::shared_ptr<aru::core::utilities::viewer::TRViewer> tr_viewer_;
  
  bool init_localisation;
};

Ros_Repeat::Ros_Repeat(const rclcpp::NodeOptions &options)
    : Node("repeat_node", options) {
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS REPEAT ");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  // Set up the parameters for the logging
  this->declare_parameter("vocab_file", "/home/administrator/data/ORBvoc.txt");
  this->declare_parameter("config_file",
                          "/home/administrator/data/BB_vo_config.yaml");
  this->declare_parameter(
      "image_left_teach_monolithic",
      "/home/administrator/data/image_left_teach.monolithic");
  this->declare_parameter(
      "image_right_teach_monolithic",
      "/home/administrator/data/image_right_teach.monolithic");
  this->declare_parameter("transform_monolithic",
                          "/home/administrator/data/transform.monolithic");
  this->declare_parameter("stereo_topic", "camera/bumblebee/image_rectified");
  this->declare_parameter("transform_topic", "vo/tf2");

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

  std::string vocab_file = this->get_parameter("vocab_file")
                               .get_parameter_value()
                               .get<std::string>();

  std::string config_file = this->get_parameter("config_file")
                                .get_parameter_value()
                                .get<std::string>();

  // Initialise repeat
  repeat_ =
      std::make_shared<aru::core::repeat::Repeat>(config_file, vocab_file);
  init_localisation=false;
      

  // Initlaise loggers
  image_left_teach_logger_ = std::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_teach_monolithic, false);
  image_right_teach_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_right_teach_monolithic, false);
  transform_logger_ = std::make_shared<logging::ProtocolLogger<pbTransform>>(
      transform_monolithic, false);

  // Create the teach map
  CreateTeachMap();
  
  LOG(INFO)<<"Initialising viewer";
  // Initialise viewer
  tr_viewer_=std::make_shared<aru::core::utilities::viewer::TRViewer>(
      640, 480, repeat_->TeachPoseChain(),repeat_->RepeatPoseChain());
     

  // Initialise subscribers
  image_stereo_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic, 10,
      std::bind(&Ros_Repeat::image_callback, this, std::placeholders::_1));

  transform_subscriber_ =
      this->create_subscription<geometry_msgs::msg ::TransformStamped>(
          transform_topic, 10,
          std::bind(&Ros_Repeat::tf_callback, this, std::placeholders::_1));
          
   // start threads
  auto viewer_thread =
      new std::thread(&utilities::viewer::TRViewer::Run, tr_viewer_);
}
Ros_Repeat::~Ros_Repeat() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void Ros_Repeat::CreateTeachMap() {
  // Create the teach map

  LOG(INFO) << "Running Teach Transforms";
  // Read the Vo monolithic into the Repeat Map
  pbTransform pb_transform = transform_logger_->ReadFromFile();
  while (!transform_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    repeat_->AddTeachTransform(curr_transform_sptr);
    pb_transform = transform_logger_->ReadFromFile();
  }

  LOG(INFO) << "Running Teach Keyframes";
  // Read previous image
  image::StereoImage init_image;
  pbImage image_left_prev = image_left_teach_logger_->ReadFromFile();
  init_image.first = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_left_prev);
  pbImage image_right_prev = image_right_teach_logger_->ReadFromFile();
  init_image.second = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_right_prev);

  // check whether the timestamp for this is after the left image
  if (init_image.second.GetTimeStamp() < init_image.first.GetTimeStamp()) {
    image_right_prev = image_right_teach_logger_->ReadFromFile();
    init_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_prev);
  }
  repeat_->InitialiseMap(init_image);
  LOG(INFO) << "Initialised Map";

  pbImage image_left_curr = image_left_teach_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_teach_logger_->ReadFromFile();

  int num = 0;

  while (!image_left_teach_logger_->EndOfFile() &&
         !image_right_teach_logger_->EndOfFile()) {

    // Add the teach keyframes to the map
    image::StereoImage curr_image;
    curr_image.first =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);
    curr_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);
    
    repeat_->AddTeachKeyframe(curr_image);
    image_left_curr = image_left_teach_logger_->ReadFromFile();
    image_right_curr = image_right_teach_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }

  LOG(INFO) << "Read all the teach keyframes";
}

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

  // Add VO Ros node
  auto repeat_node = std::make_shared<Ros_Repeat>(options);
  exec.add_node(repeat_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
