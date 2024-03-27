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

#include <aru/core/localisation/localisation.h>
#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <include/aru/core/utilities/image/image.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/int64.hpp>
#include <thread>
// #include <boost/filesystem/operations.hpp>
// #include <boost/filesystem/path.hpp>

class ROSLOC : public rclcpp::Node {

public:
  explicit ROSLOC(const rclcpp::NodeOptions &options);

  virtual ~ROSLOC();

protected:
  void runLOC();
  // callback functions
  void callback_function(const sensor_msgs::msg::Image::SharedPtr msg);

private:
  // Shared pointer to Localisation Class
  std::shared_ptr<aru::core::localisation::Localisation> loc_;

  // subscribers
  std::string stereo_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber_;

  cv::Mat image_stereo_;

  // publishers
  std::string match_index_;
  rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr image_index_pub_;
};

ROSLOC::ROSLOC(const rclcpp::NodeOptions &options) : Node("loc_node", options) {
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), "      ROS LOC ");
  RCLCPP_INFO(get_logger(), "********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "********************************");

  this->declare_parameter("vocab_file", "");
  this->declare_parameter("chow_liu_tree", "");
  this->declare_parameter("settings", "");

  stereo_topic_ = "camera/image_stereo/image_raw";
  auto loc_publisher = "localisation/index";

  image_stereo_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic_, 10,
      std::bind(&ROSLOC::callback_function, this, std::placeholders::_1));
  image_index_pub_ =
      this->create_publisher<std_msgs::msg::Int64>(loc_publisher, 10);

  // Initialise Loclaisation Class

  // retrieve ROS parameters
  std::string vocab_file = this->get_parameter("vocab_file")
                               .get_parameter_value()
                               .get<std::string>();
  std::string chow_liu_tree = this->get_parameter("chow_liu_tree")
                                  .get_parameter_value()
                                  .get<std::string>();
  std::string settings =
      this->get_parameter("settings").get_parameter_value().get<std::string>();

  // Initialise Localiser class
  loc_ = std::make_shared<aru::core::localisation::Localisation>(
      vocab_file, chow_liu_tree, settings);
  loc_->InitLocalisation();
}
ROSLOC::~ROSLOC() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void ROSLOC::callback_function(const sensor_msgs::msg::Image::SharedPtr msg) {
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  cv::Mat temp_image = cv_ptr->image;
  image_stereo_ = temp_image;
  // Undistort and rectify the images

  // Run the Localiser
  runLOC();
}

void ROSLOC::runLOC() {

  aru::core::utilities::image::StereoImage image;
  // image = aru::core::utilities::image::Image(0,image_stereo_);

  auto pair = loc_->FindClosestImage(image_stereo_);
  int index = pair.first;

  // image_index_pub_->publish(index);
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

  // Add Localisation Ros node
  auto loc_node = std::make_shared<ROSLOC>(options);
  exec.add_node(loc_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
