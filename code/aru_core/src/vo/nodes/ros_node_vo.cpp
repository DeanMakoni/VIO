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
#include "aru/core/utilities/image/feature_tracker.h"
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
#include "geometry_msgs/msg/pose.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
//nclude "sensor_msgs/msg/PointField.h"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <std_msgs/msg/header.hpp>
#include "sensor_msgs/msg/range.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include <thread>
//GT-SAM includes
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/navigation/CombinedImuFactor.h>
//#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/StereoFactor.h>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

// GTSAM related includes.
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/BarometricFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
//#include <gtsam/noiseModel/Gaussian.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>

using namespace std;
using namespace gtsam;

using namespace aru::core;
using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)//
using gtsam::symbol_shorthand::P;  // Pressure bias
using gtsam::symbol_shorthand::L;  // Landmark point

// Stereo Camera node
class ROSCamera : public rclcpp::Node {

public:
  explicit ROSCamera(const rclcpp::NodeOptions &options);

  virtual ~ROSCamera();
 
 typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CompressedImage,

       sensor_msgs::msg::CompressedImage>
        StereoApprxTimeSyncPolicy;
 typedef message_filters::Synchronizer<StereoApprxTimeSyncPolicy> StereoApprxTimeSyncer;
           
            message_filters::Subscriber<sensor_msgs::msg::CompressedImage> image_subscriber_1;
            message_filters::Subscriber<sensor_msgs::msg::CompressedImage> image_subscriber_2;
 std::shared_ptr<StereoApprxTimeSyncer> stereo_approx_time_syncher_;
  
protected:
  bool startCamera();
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

  // callback functions
  void callback_function(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg_right, 
      const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg_left) {
      
       
    // Get images from msg
   cv::Mat left_image = cv::imdecode(cv::Mat(msg_right->data), cv::IMREAD_COLOR);
   cv::Mat right_image = cv::imdecode(cv::Mat(msg_left->data), cv::IMREAD_COLOR);
   RCLCPP_INFO(get_logger(), "Rectified Stereo Topic Received "); 
  
    

    // Combine the two images
    cv::Mat image_rectified;
    cv::hconcat(left_image, right_image, image_rectified);

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

 // image_stereo_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
 //     stereo_topic_, 10,
 //     std::bind(&ROSCamera::callback_function, this, std::placeholders::_1));

  
  //stereo_subsriber_right.subscribe(this, stereo_topic_right);
 // stereo_subscriber_left.subscribe(this, stereo_topic_left);
 
 // sync = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::CompressedImage,sensor_msgs::msg::CompressedImage>>(stereo_subsriber_right,stereo_subscriber_left ,3);

image_subscriber_1.subscribe(
      this, "slave1/image_raw/compressed",
      rmw_qos_profile_sensor_data);
image_subscriber_2.subscribe(
      this, "slave2/image_raw/compressed",
      rmw_qos_profile_sensor_data);
stereo_approx_time_syncher_.reset(
                new StereoApprxTimeSyncer(
                  StereoApprxTimeSyncPolicy(10),
                  image_subscriber_1,
                  image_subscriber_2));
                                  
stereo_approx_time_syncher_->registerCallback(
                std::bind(
                  &ROSCamera::callback_function, this, std::placeholders::_1,
                                    std::placeholders::_2));

  // Define Publisher
  mPubStereoRectified =
      image_transport::create_camera_publisher(this, rectified_topic_);
  RCLCPP_INFO_STREAM(get_logger(),
                     "Advertised on topic: " << mPubStereoRectified.getTopic());

}
ROSCamera::~ROSCamera() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }



//using gtsam::symbol_shorthand::B_P; // barometric bias that will be constrained
class ROSVO : public rclcpp::Node {

public:
explicit ROSVO(const rclcpp::NodeOptions &options);

virtual ~ROSVO();
std::shared_ptr<gtsam::PreintegrationType> preintegrated;


int pressure_bias =0; //barometric bias that will be constrained
int pressure_count = 0;

imuBias::ConstantBias prior_imu_bias;
imuBias::ConstantBias prev_bias;
std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>> sync;

// Thread Management
mutable std::mutex stereo_mutex;
mutable std::mutex imu_mutex;
mutable std::mutex depth_mutex;


// Create iSAM2 object
//std::unique_ptr<ISAM2> ISAM;
ISAM2* ISAM = 0;
Key biasKey = Symbol('b', 0);
int p = 1; // ISAM counter
int b = 0; // barometer count
// Initialize factor graph and values estimates on nodes (continually updated by isam.update()) 
NonlinearFactorGraph* graph = new NonlinearFactorGraph();
Values newNodes;
Values result;       // current estimate of values
Pose3 prev_camera_pose; 
gtsam::Pose3 prior_pose;
Vector3 prior_velocity;
double prior_pressure; // change this value accordingly  
bool start; // used to initialise visoextractor_
int  landmark_id;  // landmark ID represents the index in the FeatureSPtrVectorSptr 
int frame_index_; //indexing frames for initialisation
// Initialize VIO Variables
double fx;  
double fy;                   // Camera calibration intrinsics
double cx;
double cy;
double resolution_x;          // Image distortion intrinsics
double resolution_y;
double Tx;           // Camera calibration extrinsic: distance from cam0 to cam1
gtsam::Matrix4 T_cam_imu_mat; // Transform to get to camera IMU frame from camera frame


uint64_t prev_imu_timestamp;

protected:

/**
  *Optimise using ISAM2 and publish
  *
  */
  void Optimise_and_publish();
  
 /**
   * @brief Structure to hold information about a named key. Named keys are used to keep track of the index for which
   * data should be added to the graph.
   *
   */
  struct NamedKeyInfo {
      NamedKeyInfo(const int key = 0, const unsigned int priority = 0) : key(key), priority(priority) {}

      // Key/index used to acquire gtsam::Key in graph
        int key;

      // Priority of the named key, used in various operations. 0 is maximum priority.
      unsigned int priority;
    };
  
  // associative container to store keys
  std::map<std::string, NamedKeyInfo> keys;
  
  // All Timestamps (key -> time)
  std::map<int, rclcpp::Time> timestamps_;
  
  
  // Graph and key management functions
  
/**
 * @brief Get the timestamp associated with a particular key value.
 *
 * @param key
 * @return const rclcpp::Time&
 */
 const rclcpp::Time& timestamp(const int key) const;

/**
 * @brief Get the timestamp associated with a named key.
 *
 * @param name
 * @return const rclcpp::Time&
 */
 const rclcpp::Time& timestamp(const std::string& key, const int offset = 0) const;
 // Set time stamps
 void set_timestamp(const int key, const rclcpp::Time& timestamp);
 void set_timestamp(const std::string& key, const rclcpp::Time& timestamp, const int offset = 0);
  /**
   * @brief Increment a named key.
   *
   * @param name
   */
 
  void increment(const std::string& name);

  /**
  * @brief Get the key value for a named key.
  *
  * @param name
  * @return int
   */
  int key(const std::string& name, const int offset = 0) const;

  /**
  * @brief Return the smallest key value of the named keys with priority <= priority arg (highest = 0).
  *
  * @return int maximum priority of named keys used in the evaluation of the minimum key.
  */
  int minimum_key(const unsigned int priority = std::numeric_limits<unsigned int>::max()) const;
  
   /**
     * @brief Set/create a named key.
     *
     * @param name
     * @param value
     */
   void set_named_key(const std::string& name, const int key = 0, const unsigned int priority = 0);

    
  bool startCamera();
  void featureToLandmark(const cv::Mat &image_lef, const cv::Mat &image_right); //,std::pair<Eigen::MatrixXf, Eigen::MatrixXf> coordinates);
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

       // ----> Thread functions
  void test_callback_function(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void callback_function(const sensor_msgs::msg::Image::SharedPtr msg);
  void imu_callback_function(const sensor_msgs::msg::Imu::SharedPtr msg);
  void sonar_callback_function(const sensor_msgs::msg::Range::SharedPtr msg);
  void depth_callback_function(const sensor_msgs::msg::FluidPressure::SharedPtr msg);

   // IMU biases
   
private:

  //initial pose ID
  //int pose_id = 0;
       // current estimate of previous pose
   

  // Noise models
  noiseModel::Isotropic::shared_ptr prior_landmark_noise = noiseModel::Isotropic::Sigma(3, 0.1);
  //noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.3),Vector3::Constant(0.1)).finished()); // 30cm std on x,y,z 0.1 rad on      roll,pitch,yaw 
 noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas(
    (Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished() // (roll,pitch,yaw in rad; std on x,y,z in meters)
  );
  noiseModel::Isotropic::shared_ptr pose_landmark_noise = noiseModel::Isotropic::Sigma(3, 10.0); // one pixel in u and v

   // Noise models for IMU 
  noiseModel::Isotropic::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1);  // m/s
  noiseModel::Isotropic::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6, 1e-3);
  
     
  
  // Shared pointer to Visual Odometry Class
  std::shared_ptr<aru::core::vo::VO> vo_;
  
  // shared pointer feature_tracker class
  std::shared_ptr<aru::core::utilities::image::VisoFeatureTracker> viso_extractor_;
  
  // Shared pointer to transform map for keyframe stuff
  std::shared_ptr<aru::core::utilities::transform::TransformMap> transform_map_;

  aru::core::utilities::image::StereoImage image_stereo_;
  aru::core::utilities::image::StereoImage image_stereo_prev_;

  aru::core::utilities::image::StereoImage image_key_;
  aru::core::utilities::image::Frames  frames_;

  bool prev_image = false;
  uint64_t prev_timestamp;

  // subscribers
  std::string stereo_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_stereo_subscriber;
 

  // Imu subscriber
  std::string imu_topic;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr
      imu_subscriber;

  // sonar subscriber
  std::string sonar_topic;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr
      sonar_subscriber;

  // Pressure subscriber
  std::string depth_topic;
  rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr
      depth_subscriber;

  // Publisher
  std::string vo_tf2_topic_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      vo_tf2_publisher_;

  std::string pose_topic_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr
      pose_publisher_;

  std::string macthed_points_topic_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      macthed_points_publisher_;

  std::string kf_tf2_topic_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      kf_tf2_publisher_;

  std::string kf_image_topic_;
  image_transport::CameraPublisher kf_image_publisher_;
  
  std::string optimised_odometry_topic;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
  optimised_odometry_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
  path_publisher_;

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

  start = true;
 
  stereo_topic_ = "camera/zed/image_rectified";
  imu_topic = "imu/imu";
  sonar_topic = "imagenex/range";
  depth_topic = "bar30/pressure";
  vo_tf2_topic_ = "vo/tf2";
  kf_tf2_topic_ = "kf/tf2";
  kf_image_topic_ = "camera/bumblebee/image_keyframe";
  RCLCPP_INFO(get_logger(), "Dean1");

  min_distance_ = 2.0;
  min_rotation_ = 0.2536;

  count = 0;
  down_sample_ = 15;
  
  // Initialise factor graph keys
  this->set_named_key("barometer", 0, 1);
  this->set_named_key("pose", 0, 1);
  this->set_named_key("velocity", 0, 1);
  this->set_named_key("imu_bias", 0, 1);
  this->set_named_key("landmark", 0, 1);
  
  // intialise key for Timestamp
  this->set_timestamp(0, rclcpp::Time(0, 0));
 
image_stereo_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_topic_, 10,
      std::bind(&ROSVO::callback_function, this, std::placeholders::_1));
         
imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, 10,
     std::bind(&ROSVO::imu_callback_function, this, std::placeholders::_1));
      
sonar_subscriber = this->create_subscription<sensor_msgs::msg::Range>(
     sonar_topic, 10, 
      std::bind(&ROSVO::sonar_callback_function, this, std::placeholders::_1));

depth_subscriber = this->create_subscription<sensor_msgs::msg::FluidPressure>(
    depth_topic, 10,
     std::bind(&ROSVO::depth_callback_function, this, std::placeholders::_1));
  
  
 RCLCPP_INFO(get_logger(), "Dean 2");

  // Initialise VO params
  auto vo_config_file = "/home/dean/vo_config.yaml";
  auto vo_vocab_file = "/home/jetson/Downloads/Dean/code/aru_core/ORBvoc.txt";
  // initialise ISAM2  parameters
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  parameters.optimizationParams = ISAM2DoglegParams();
  parameters.factorization = gtsam::ISAM2Params::QR;
  ISAM = new ISAM2(parameters);
  // Initialise VO class
  //vo_ = std::make_shared<aru::core::vo::VO>(vo_config_file, vo_vocab_file);
  aru::core::utilities::image::MatcherParams matcher_params;
  aru::core::utilities::image::ExtractorParams extractor_params;
  aru::core::vo::SolverParams solver_params;

  matcher_params.stereo_baseline = 0.5372;
  matcher_params.match_threshold_high = 100;
  matcher_params.match_threshold_low = 50;
  matcher_params.focal_length = 718;

  extractor_params.num_levels = 8;
  extractor_params.num_features = 2000;
  extractor_params.minimum_fast_threshold = 7;
  extractor_params.initial_fast_threshold = 20;
  extractor_params.scale_factor = 1.2;
  extractor_params.patch_size = 31;
  extractor_params.edge_threshold = 19;

  solver_params.ransac_prob = 0.95;
  solver_params.ransac_max_iterations = 100;
  solver_params.threshold = 3;

  
  
  
  vo_ = std::make_shared<aru::core::vo::VO>(vo_vocab_file,extractor_params,matcher_params, solver_params);
  
  // Track features for the incoming images 
  viso_extractor_ =  std::make_shared <utilities::image::VisoFeatureTracker>(matcher_params,extractor_params);
 
  // Initialise camera and transformation from .yaml configuration files.
   // Load YAML file
   // YAML::Node config = YAML::LoadFile("/home/jetson/Downloads/VIO/code/aru_core/src/vo/config/camchain-imucam-Speedo1.yaml");
   auto config_file = "/home/jetson/Downloads/VIO/code/aru_core/src/vo/config/camchain-imucam-Speedo1.yaml";
   cv::FileStorage fs;
   fs.open(config_file, cv::FileStorage::READ);

   if (!fs.isOpened()) {
     LOG(ERROR) << "Could not open vo model file: " << config_file;
  // Handle the error here, maybe exit the program or use default values
   } 
    // Extract T_cam_imu matrices for cam0 and cam1
   
    //for (size_t i = 0; i < 4; ++i) {
      //  for (size_t j = 0; j < 4; ++j) {
        //    T_cam_imu_mat(i, j) = config["cam0"]["T_cam_imu"][i][j].as<float>();
       // }
    //}
   // Extract T_cam_imu from cam0
    cv::Mat T_cam_imu_mat_cv;
    fs["cam0"]["T_cam_imu"] >> T_cam_imu_mat_cv;

 // Convert T_cam_imu from cv::Mat to GTSAM Matrix4
    for (int i = 0; i < T_cam_imu_mat_cv.rows; ++i) {
        for (int j = 0; j < T_cam_imu_mat_cv.cols; ++j) {
            T_cam_imu_mat(i, j) = T_cam_imu_mat_cv.at<double>(i, j);
        }
    }

   // Extract camera intrinsics
  fx = fs["cam0"]["intrinsics"][0];
  fy = fs["cam0"]["intrinsics"][1];
  cx = fs["cam0"]["intrinsics"][2];
  cy = fs["cam0"]["intrinsics"][3];

  // Extract distortion coefficients (assuming radtan model)

  // Extract image resolution
  resolution_x = fs["cam0"]["resolution"][0];
  resolution_y = fs["cam0"]["resolution"][1];

  // Extract extrinsic parameter (assuming Tx from T_cn_cnm1)
  Tx = fs["cam1"]["T_cn_cnm1"][0][3];
  
  // Define Publisher
  vo_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
           "vo/tf2", 10); // vo_tf2_topic_
  
  kf_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
          "kf/tf2", 10); // kf_tf2_topic_
  pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::Pose>(
         "gtsam/pose", 10); // pose_topic_
   macthed_points_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "macthed/points", 10); //macthed_points_topic_
   optimised_odometry_publisher_ =
      this->create_publisher<nav_msgs::msg::Odometry>(
      "optimisation/odometry", 10);
    path_publisher_ = 
            this->create_publisher<nav_msgs::msg::Path>(
      "output/path", 10);
   
  kf_image_publisher_ =
      image_transport::create_camera_publisher(this, kf_image_topic_);
  RCLCPP_INFO_STREAM(get_logger(),
                    "Advertised on topic: " << kf_image_publisher_.getTopic());
  RCLCPP_INFO(get_logger(), "Dean 5");
  // Initialise transform map
  transform_map_ =
      std::make_shared<aru::core::utilities::transform::TransformMap>();
 
 // IMU parameters
  // We use the sensor specs to build the noise model for the IMU factor.
  double accel_noise_sigma = 0.0003924;
  double gyro_noise_sigma = 0.000205689024915;
  double accel_bias_rw_sigma = 0.004905;
  double gyro_bias_rw_sigma = 0.000001454441043;
  Matrix33 measured_acc_cov = I_3x3 * pow(accel_noise_sigma, 2);
  Matrix33 measured_omega_cov = I_3x3 * pow(gyro_noise_sigma, 2);
  Matrix33 integration_error_cov = I_3x3 * 1e-8;  // error committed in integrating position from velocities
  Matrix33 bias_acc_cov = I_3x3 * pow(accel_bias_rw_sigma, 2);
  Matrix33 bias_omega_cov = I_3x3 * pow(gyro_bias_rw_sigma, 2);
  Matrix66 bias_acc_omega_init = I_6x6 * 1e-5;  // error in the bias used for preintegration

  auto p = PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);
  // PreintegrationBase params:
  p->accelerometerCovariance =
      measured_acc_cov;  // acc white noise in continuous
  p->integrationCovariance =
      integration_error_cov;  // integration uncertainty continuous
  // should be using 2nd order integration
  // PreintegratedRotation params:
  p->gyroscopeCovariance =
      measured_omega_cov;  // gyro white noise in continuous
  // PreintegrationCombinedMeasurements params:
  p->biasAccCovariance = bias_acc_cov;      // acc bias in continuous
  p->biasOmegaCovariance = bias_omega_cov;  // gyro bias in continuous
  p->biasAccOmegaInt = bias_acc_omega_init;

  
 
  prev_bias = prior_imu_bias;
  auto p1 = p;
 // preintegrated = std::make_shared<PreintegratedImuMeasurements>(p1, prior_imu_bias);
  preintegrated = std::make_shared<PreintegratedCombinedMeasurements>(p1, prior_imu_bias);
  assert(preintegrated);
}

ROSVO::~ROSVO() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void ROSVO::test_callback_function(const sensor_msgs::msg::CompressedImage::SharedPtr msg){

  RCLCPP_INFO(get_logger(), "Stereo Topic Received");
}

void ROSVO::callback_function(const sensor_msgs::msg::Image::SharedPtr msg) {

  // lock the thread
  stereo_mutex.lock();
   RCLCPP_INFO(get_logger(), "Filtered Stereo Topic Received "); 
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
  
  cv::Mat left_gray, right_gray;
  cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
  //cv_bridge::CvImagePtr cv_ptr;
  //cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  //cv::Mat temp_image = cv_ptr->image;
  //image_stereo = temp_image;

  // Split Images
  //cv::Size s = temp_image.size();
  //int width = s.width;
  //cv::Mat left_image = temp_image(cv::Rect(0, 0, width / 2, s.height));
  //cv::Mat right_image = temp_image(cv::Rect(width / 2, 0, width / 2, s.height));

  // Get timestamp
  std_msgs::msg::Header h = msg->header;
  uint64_t seconds = h.stamp.sec;
  uint64_t nanoseconds = h.stamp.nanosec;
  uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);
  
  // change these values accordingly
  aru::core::utilities::image::MatcherParams matcher_params;
  aru::core::utilities::image::ExtractorParams extractor_params;

    matcher_params.stereo_baseline = 0.5372;
    matcher_params.match_threshold_high = 100;
    matcher_params.match_threshold_low = 50;
    matcher_params.focal_length = 718;

    extractor_params.num_levels = 8;
    extractor_params.num_features = 2000;
    extractor_params.minimum_fast_threshold = 7;
    extractor_params.initial_fast_threshold = 20;
    extractor_params.scale_factor = 1.2;
    extractor_params.patch_size = 31;
    extractor_params.edge_threshold = 19;

  // Initialise VO params
  auto vo_config_file = "/home/dean/vo_config.yaml";
  auto vo_vocab_file = "/home/jetson/Downloads/Dean/code/aru_core/ORBvoc.txt";
  
  // Detect and match features in the left and right images and return an array of features
  boost::shared_ptr<aru::core::utilities::image::VisoMatcher> matcher_ = boost::make_shared<aru::core::utilities::image::VisoMatcher>(
             matcher_params,extractor_params);
  
  aru::core::utilities::image::FeatureSPtrVectorSptr features = 
              matcher_->ComputeStereoMatches(left_image,right_image);
    RCLCPP_INFO(get_logger(), "Stereo images Parameters set 1"); 
    
   
  // Convert features to landmark and add them to the graph  
  featureToLandmark(left_gray, right_gray);  
           
  // publish every feature in features   
   for(auto feat : *features){

       aru::core::utilities::image::Feature& feature = *feat;
       sensor_msgs::msg::PointCloud2 pcl_msg;

       //Modifier to describe what the fields are.
       sensor_msgs::PointCloud2Modifier modifier(pcl_msg);

       modifier.setPointCloud2Fields(4,
           "x", 1, sensor_msgs::msg::PointField::FLOAT32,
           "y", 1, sensor_msgs::msg::PointField::FLOAT32,
           "z", 1, sensor_msgs::msg::PointField::FLOAT32,
           "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);

       //Msg header
       pcl_msg.header = std_msgs::msg::Header();
       pcl_msg.header.stamp = this->now();
       pcl_msg.header.frame_id = "frame";

       pcl_msg.height = 1; // this is jus a single feature with x,y,z. TODO: Check if this correcg
       pcl_msg.width = 1;
       pcl_msg.is_dense = true;

       //Total number of bytes per point
       pcl_msg.point_step = 16;
       pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width;
       pcl_msg.data.resize(pcl_msg.row_step);

       // Populate the feature
       Eigen::Vector3d point = feature.GetTriangulatedPoint();
       cv::KeyPoint keyPoint = feature.GetKeyPoint();

       sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");
       sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");
       sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");
       sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");

       *iterX = point.x();
       *iterY = point.y();
       *iterZ = point.z();
       *iterIntensity = 10; //int.response(); // this not intensity, change this

      macthed_points_publisher_->publish(pcl_msg);
   }
  // Get the pixel coordinates of the right and left images. 
  // the pixel cordinates are used to make SterioPoint2 for GenericSterionFactors
  // Stereo Point measurement (u_l, u_r, v). v will be identical for left & right for rectified stereo pair
   
   std::pair<Eigen::MatrixXf, Eigen::MatrixXf> cordinates_ = vo_->ObtainStereoPoints(left_image,right_image);

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
    // Change these values to correct initial values
   //gtsam::Rot3 prior_rotation = gtsam::Rot3::Quaternion(1, 1,1,1);
   //Point3 prior_point; 
   //gtsam::Pose3 prior_pose(gtsam::Rot3::Quaternion(1, 1,1,1), Point3(0.05, -0.10, 0.20));
   //Vector3 prior_velocity;

  //gtsam::Pose3 prior_pose;
 //gtsam::Vector3 prior_velocity;
  
 

    // Check the distance moved from the last keyframe
    // pose is T_prev_curr. Source is curr_image dest is prev_image
    utilities::transform::TransformSPtr pose =
        transform_map_->Interpolate(prev_timestamp, time_out);
    if (pose) {
     /// if (pose_id == 0){
           // add prior factors for stereo cameras and IMU
        ///   newNodes.insert(X(pose_id), prior_pose);
         //  newNodes.insert(V(pose_id), prior_velocity);
           //newNodes.insert(B(pose_id), prior_imu_bias);
 
           //graph->addPrior(X(pose_id), prior_pose, pose_noise);
           //graph->addPrior(V(pose_id), prior_velocity, velocity_noise_model);
           //graph->addPrior(B(pose_id), prior_imu_bias, bias_noise_model);
           //pose_id = pose_id + 1;
      //}
      float dist = pose->GetTransform().translation().norm();
      cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
      cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
      // Read Rotation matrix and convert to vector
      cv::eigen2cv(pose->GetRotation(), R_matrix);
      cv::Rodrigues(R_matrix, rvec);
      Eigen::Vector3f rpy;
      cv::cv2eigen(rvec, rpy);
      float rotation = rpy.norm();

     // Get translation and rotation from the current pose

     Eigen::Matrix3f rotation_matrix = pose->GetRotation();

    // Add node value for current pose with initial estimate being previous pose
    // count is the pose_id

   
    //  Checking if it is a key frame
      if (dist > min_distance_ || rotation > min_rotation_) {
        prev_timestamp = time_out;
        image_key_ = image_stereo_;
        
        
        // track features of the stereo images
        // intialise viso_extraactor on first flag
        viso_extractor_->UpdateFeatures(image_left, image_right);
        
        // Get a track of the features
        std::vector<FeatureTrack> active_tracks_->GetActiveTracks();
        
        //  from FeatureTrack iterate through every feature
        // check for sufficient parallax for current frame and previous frames
        // start with the latest two frames
        for (int i =0 ; i< frame_index_; i++) {
             //pair that will contain corresponding features in frames we are comparing
             typedef std::pair<FeatureSPtr, FeatureSPtr> FeaturePair;
             std::vector<FeaturePair> corres;
             
             for( const auto& track : active_tracks_){
                 //check if the two frame index exist in current FeatureTrack
                 // FeatureTrack is a feature tracked in multiple frames
                 // start by comparing the latest two frames
                  auto it1 = std::find(track.frame_track_->begin(), track.frame_track_->end(), frame_index_);
                  auto it2 = std::find(track.frame_track_->begin(), track.frame_track_->end(), frame_index_-i);

                 if (it1 != track.frame_track_->end() && it2 != track.frame_track_->end()) {
                 	// if the feature appears in oth frames there is a correspondance
                       // Get the indexes of num1 and num2 in the frame_track_ vector
                      int index1 = std::distance(track.frame_track_->begin(), it1);
                      int index2 = std::distance(track.frame_track_->begin(), it2);
                      
                      FeatureSPtr feature1 = (*track.feature_track_)[index1];
                      FeatureSPtr feature2 = (*track.feature_track_)[index2];
                      // Store feature1 and feature2 in a std::pair and push it into the vector
                      corres.push_back(std::make_pair(feature1, feature2));
                     //std::cout << "Both " << num1 << " and " << num2 << " exist in frame_track_" << std::endl;
                  
                }
             }
             
            if (corres.size() > 20){
            
              double sumparallax = 0;
              double average_parallax;
              for (const auto& pair : featurePairs) {
                 FeatureSPtr feature1 = pair.first;
                 FeatureSPtr feature2 = pair.second;
                 
                 aru::core::utilities::image::Feature& feature_1 = *feature1;
                 aru::core::utilities::image::Feature& feature_2 = *feature2;
                 Eigen::Vector3d camera_point1 = feature_1.GetTriangulatedPoint();
                 Eigen::Vector3d camera_point2 = feature_2.GetTriangulatedPoint();
                 
                 // Extract Eigen::Vector2d from Eigen::Vector3d
                 Eigen::Vector2d point1_2d = camera_point1.head<2>();
                 Eigen::Vector2d point2_2d = camera_point2.head<2>();
                 
                 double parallax = (point1_2d - point2_2d).norm();
                 sumparallax = sumparallax + parallax;
                 
                }
            }
            average_parallax = 1 * sumparallax/ int(corres.size());
            
        }
        
        // Store the image_key_ in a vector
        frames.pushback( image_key_);
        
        // iterate through a map of frames using iterator to check for tracked feature and 2D to 2D
        // correspondance between succesive features
        std::map<aru::core::utilities::image::Image, aru::core::utilities::image::StereoImage>::iterator it;
        
        if (frames.size() > 1) {
             for (it = frames.begin(); it != frames.end(); ++it) {
           
               }
        }
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
        frame_index_++;
        // Publish the rectified images
        //auto msg_kf =
      //      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_stereo)
         //       .toImageMsg();

       // msg_kf->header = h;

       // kf_image_publisher_.publish(msg_kf, mStereoCamInfoMsg);
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
    //image_key_ = image_stereo_;

    // output the first image
    // Publish key frame image
   // auto mStereoCamInfoMsg = std::make_shared<sensor_msgs::msg_right::CameraInfo>();

    // Publish the rectified images
    //auto msg_kf =
    //    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_stereo)
     //       .toImageMsg();

   // msg_kf->header = h;

   // kf_image_publisher_.publish(msg_kf, mStereoCamInfoMsg);
  }
  // Update the previous image to the current image
  image_stereo_prev_ = image_stereo_;

  // count is the pose_id
  count++;
  stereo_mutex.unlock();
}

void ROSVO::imu_callback_function(const sensor_msgs::msg::Imu::SharedPtr msg){
// Lock the thread
 imu_mutex.lock();
 
 
 int next_pose_key = this->key("pose");
 int next_velocity_key = this->key("velocity");
 int next_bias_key = this->key("imu_bias");
 auto current_timestamp = msg->header.stamp;
 
if (next_pose_key == 0){
       
       geometry_msgs::msg::Quaternion orient = msg->orientation; 
       prior_pose = Pose3(gtsam::Rot3::Quaternion(orient.x, orient.y,orient.z,orient.w), Point3(0, 0,0));
       //prior_pose = Pose3(gtsam::Rot3::Ypr(0, 0,0), Point3(0, 0,0));
       prior_velocity = Vector3(0,0,0);
       double height = 10;
       //prior_pressure = 1; //change this vlue accordingly
       // add prior factors for stereo cameras and IMU
       newNodes.insert(X(next_pose_key), prior_pose);
       newNodes.insert(V(next_velocity_key), prior_velocity);
       newNodes.insert(B(next_bias_key), imuBias::ConstantBias());
      // newNodes.insert(P(next_barometer_key), height); // CHANGE HEIGHT
      
      // graph->addPrior(X(next_pose_key), prior_pose, pose_noise); 
      // graph->addPrior(V(next_velocity_key), prior_velocity, velocity_noise_model);
      // graph->addPrior(B(next_bias_key),imuBias::ConstantBias() , bias_noise_model);
       
      graph->emplace_shared< PriorFactor<Pose3> >(X(next_pose_key),prior_pose, pose_noise);
      graph->emplace_shared< PriorFactor<Vector3> >(V(next_velocity_key), prior_velocity, velocity_noise_model);
      graph->emplace_shared< PriorFactor<imuBias::ConstantBias> >(B(next_bias_key),imuBias::ConstantBias() , bias_noise_model);
       
       SharedNoiseModel pressure_noise_model = gtsam::noiseModel::Isotropic::Variance(1, 1.0e-6); // change this value accordingly
      // graph->addPrior(P(next_barometer_key),height, pressure_noise_model);
       //pose_id = pose_id + 1;
      // biasKey++;
       RCLCPP_INFO(get_logger(), "Pose_ID zero");
      
     this->increment("pose");
     this->increment("velocity");
     this->increment("imu_bias");      
   
       
    }
    
    
 else{
 
 //RCLCPP_INFO(get_logger(), "IMU topic received");
 Vector3 measuredAcc (msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
 Vector3 measuredOmega (msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
 //Check the time stamp 
 // if it is greater than 10 nano se it is a pose
 gtsam::Point3 curr_accel(measuredAcc);
 gtsam::Point3 curr_angular(measuredOmega);

 
 //std::cout << "Timestamp: " << current_timestamp.nanosec << "nanoseconds" << std::endl;
 //std::cout << "Timestamp: " << prev_timestamp  << "seconds" << std::endl;
 
 // Check on how to add the correct dt
//double dt = current_timestamp.nanosec - prev_imu_timestamp; 

//uint64_t seconds = msg->header.stamp.sec;
//uint64_t nanoseconds = msg->header.stamp.nanosec;
//uint64_t time_out = seconds * 1000 + floor(nanoseconds / 1000000);

//double dt = seconds - prev_imu_timestamp;
double dt = 0.005;
//prev_imu_timestamp = seconds;

//if (dt != 0){
  
   //preintegrated->integrateMeasurement(curr_accel,curr_angular, dt);
   preintegrated->integrateMeasurement(measuredAcc,measuredOmega, dt);
// }
//else{
//  dt = 0.005; 
//  preintegrated->integrateMeasurement(curr_accel,curr_angular, dt);
 // }
RCLCPP_INFO(get_logger(), "Intergrated");
 
//RCLCPP_INFO(get_logger(), "Imu callback done");
 
 
if (current_timestamp.nanosec - prev_timestamp > 117401090){

      //TODO: Set timestamp for this pose that we want to optimise
      this->set_timestamp("pose", current_timestamp);
   
        // add IMU and Bias factors to graph
     //auto preint_imu =
      //   dynamic_cast<const PreintegratedImuMeasurements&>(*preintegrated);
      auto preint_imu =
        dynamic_cast<const PreintegratedCombinedMeasurements&>(*preintegrated);
      //auto preint_imu = std::make_unique<gtsam::PreintegratedImuMeasurements>(
       //                  CastToPreintergratedImuMeasurements(*preintegrated);
                         
     // ImuFactor imu_factor(X(this->key("pose", -1)), X(this->key("pose", -1)),
      //                     X(this->key("pose")), V(this->key("velocity")),
     //                      B(this->key("imu_bias")), preint_imu);
                           
      //graph->add(imu_factor);
      RCLCPP_INFO(get_logger(), "IMU factor added");
      imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
      graph->emplace_shared<CombinedImuFactor>(
       X(this->key("pose", -1)), V(this->key("pose", -1)),
       X(this->key("pose")), V(this->key("velocity")),
       B(this->key("imu_bias", -1)), B(this->key("imu_bias")),
       preint_imu
     );
      //graph->add(BetweenFactor<imuBias::ConstantBias>(
      //   B(this->key("imu_bias", -1)),B(this->key("imu_bias")) ,zero_bias ,
       //  bias_noise_model));
      // RCLCPP_INFO(get_logger(), "Bias factor added");
       
      gtsam::NavState  prev_state = NavState(prior_pose,
                          prior_velocity); 
      gtsam::NavState prop_state = preintegrated->predict(prev_state, prev_bias);
      newNodes.insert(X(this->key("pose")), prop_state.pose());
      newNodes.insert(V(this->key("velocity")), prop_state.v());
      newNodes.insert(B(this->key("imu_bias")), prev_bias);
      

   // Optimise and publish
   this->Optimise_and_publish();
   prev_timestamp = current_timestamp.nanosec;
   
   this->increment("pose");
   this->increment("velocity");
   this->increment("imu_bias");
      
   }
  } 
 // unlock the thread 
 imu_mutex.unlock();
 
}

void ROSVO::sonar_callback_function(const sensor_msgs::msg::Range::SharedPtr msg){

}

void ROSVO::depth_callback_function(const sensor_msgs::msg::FluidPressure::SharedPtr msg){
 
 // Lock the thread  
 depth_mutex.lock();
 RCLCPP_INFO(get_logger(), "Barometer topic received");
 // get latest barometer key
 int next_barometer_key = this->key("barometer");
 
  //Create the Gaussian noise model
  SharedNoiseModel pressure_noise_model = gtsam::noiseModel::Isotropic::Variance(1, 1.0e-6);

 // sensor message's pressure is in pascals, convert it o KPa for gtsam
 const double pressure = (msg->fluid_pressure/ 1000);

 // create barometric factor

  // BarometricFactor pressure_factor(X(this->key("pose")),
  //                                    P(this->key("barometer")),pressure,
   //                                  pressure_noise_model);
   //graph->add(pressure_factor);
   graph->emplace_shared<BarometricFactor>(X(this->key("pose")),
                                      P(this->key("barometer")),pressure,
                                     pressure_noise_model);

   //TODO: Create barometer bias factors up to the current key


   // gtsam::Pose3 pressure_pose = Pose3(gtsam::Rot3::RzRyRx(0, 0,0), Point3(0, 0,1000));
  newNodes.insert(P(this->key("barometer")), pressure);
   
  this->increment("barometer");


  // unlock the thread
  depth_mutex.unlock();
}

void ROSVO::Optimise_and_publish() {
            // ISAM2 solver
       
       try{   
       int max_key = this->key("pose");  
       RCLCPP_INFO(get_logger(), "ISAM solver");
       //Values Testnode;
     // NonlinearFactorGraph* testgraph = new NonlinearFactorGraph();
      ISAM->update(*graph, newNodes);
       RCLCPP_INFO(get_logger(), "ISAM solver Done for ");
       std::cout << p << std::endl;
       p = p+1;
       result = ISAM->calculateEstimate();
      // LevenbergMarquardtOptimizer optimizer(*graph, newNodes);
       //result = optimizer.optimize();
        
        RCLCPP_INFO(get_logger(), "Calculation done.");
       
       prior_pose = result.at<Pose3>(X(this->key("pose")));

       prior_velocity = result.at<Vector3>(V(this->key("velocity")));

        RCLCPP_INFO(get_logger(), "Calculation done 1");
        // reset the graph
        graph->resize(0);
        newNodes.clear();

        // Overwrite the beginning of the preintegration for the next step.
      // prev_state = NavState(result.at<Pose3>(X(pose_id - 1)),
         //                   result.at<Vector3>(V(pose_id - 1)));
       RCLCPP_INFO(get_logger(), "Prev state updated");
       prev_bias = result.at<imuBias::ConstantBias>(B(this->key("imu_bias")));
        RCLCPP_INFO(get_logger(), "Prev bias updated");
       // Reset the preintegration object.
       preintegrated->resetIntegrationAndSetBias(prev_bias);
       
       
       // Puublish poses
      nav_msgs::msg::Odometry optimised_odometry_msg;
      optimised_odometry_msg.header.stamp = this->now();
      optimised_odometry_msg.header.frame_id = "map_frame_id";
      optimised_odometry_msg.child_frame_id = "body_frame_id";
      //optimised_odometry_msg.header.frame_id = std::to_string(max_key);
      //optimised_odometry_msg.child_frame_id = std::to_string(max_key);
        
       gtsam::Pose3 current_pose = result.at<Pose3>(X(this->key("pose")));
       gtsam::Velocity3 current_velocity = result.at<Velocity3>(V(this->key("velocity")));
       RCLCPP_INFO(get_logger(), "Pose  Id correct");
       // Extract position
       gtsam::Vector3 position = current_pose.translation();
       // Extract quaternion
       gtsam::Rot3 rotation = current_pose.rotation();

       gtsam::Quaternion quaternion = rotation.toQuaternion();
       
       // msg for pose_publisher
       geometry_msgs::msg::Pose pose_msg;
       // Set position
       pose_msg.position.x = position.x();
       pose_msg.position.y = position.y();
       pose_msg.position.z = position.z();
       
       

       // Set orientation (example using roll, pitch, yaw)
      //f::Quaternion quat = tf::createQuaternionFromRPY(0.0, M_PI / 4, 0.0); // Set desired roll, pitch, yaw
       pose_msg.orientation.w = quaternion.w();
       pose_msg.orientation.x = quaternion.x();
       pose_msg.orientation.y = quaternion.y();
       pose_msg.orientation.z = quaternion.z();
       

       // Populate the twist message
       geometry_msgs::msg::Twist twist_msg;
       twist_msg.linear.x = current_velocity.x();
       twist_msg.linear.y = current_velocity.y();
       twist_msg.linear.z = current_velocity.z();
       

      // Assign twist and pose message to the odometry message
      optimised_odometry_msg.twist.twist.linear = twist_msg.linear;
      optimised_odometry_msg.pose.pose = pose_msg;
      
       //Odometry msg covariance //
       //TODO: Disable this when using Stereo factors
       gtsam::Matrix poseCovariance = ISAM->marginalCovariance(X(this->key("pose")));
       gtsam::Matrix velocityCovariance = ISAM->marginalCovariance(V(this->key("velocity")));
       
       for (int i = 0; i < 6; ++i) {
         for (int j = 0; j < 6; ++j) {
             optimised_odometry_msg.pose.covariance[i * 6 + j] = poseCovariance(i, j);
             optimised_odometry_msg.twist.covariance[i * 6 + j] = poseCovariance(i, j);
           }   
    
       }
       
       // Publish Optimised Path and change path
       nav_msgs::msg::Path path;
      
       const std::string frame_id_prefix;
       path.header.stamp = this->timestamp(max_key);
       path.header.frame_id = "map_frame_id";
      // path.header.frame_id = std::to_string(max_key);
       
       for (int key = 0; key <= max_key; ++key) {
       
          geometry_msgs::msg::PoseStamped pose_msg;
          pose_msg.header.stamp = this->timestamp(key);
          pose_msg.header.frame_id = frame_id_prefix + "_" + std::to_string(key);
          gtsam::Pose3 current_pose = result.at<gtsam::Pose3>(X(key));
          gtsam::Vector3 position = current_pose.translation();
         // Extract quaternion and translation
         gtsam::Rot3 rotation = current_pose.rotation();
         gtsam::Quaternion quaternion = rotation.toQuaternion();
      
         // Set position
         pose_msg.pose.position.x = position.x();
         pose_msg.pose.position.y = position.y();
         pose_msg.pose.position.z = position.z();
         
       // Set orientation (example using roll, pitch, yaw)
         //f::Quaternion quat = tf::createQuaternionFromRPY(0.0, M_PI / 4, 0.0); // Set desired roll, pitch, yaw
         pose_msg.pose.orientation.w = quaternion.w();
         pose_msg.pose.orientation.x = quaternion.x();
         pose_msg.pose.orientation.y = quaternion.y();
         pose_msg.pose.orientation.z = quaternion.z();
         
         path.poses.push_back(pose_msg);
        }
       
       pose_publisher_->publish(pose_msg);
       optimised_odometry_publisher_->publish(optimised_odometry_msg);
       path_publisher_->publish(path);
       
       }
       catch(const gtsam::IndeterminantLinearSystemException& ex){
            result.print();    
       }
       
      

  }    

// Transform feature from image_processor to landmark with 3D coordinates
// Add landmark to ISAM2 graph if not already there (connect to current pose with a factor)
// Add landmark to point cloud 
void ROSVO::featureToLandmark(const cv::Mat &image_left, const cv::Mat &image_right){  //,std::pair<Eigen::MatrixXf, Eigen::MatrixXf> coordinates){

 RCLCPP_INFO(get_logger(), "Image dimension test 1");

 cv::Mat image_dest_left, image_dest_right;
//  image_dest_left_gray,image_dest_right_gray;
// cv::cvtColor(image_dest_left, image_dest_left_gray, cv::COLOR_BGR2GRAY);
// cv::cvtColor(image_dest_right, image_dest_right_gray, cv::COLOR_BGR2GRAY);
 
 
// iterate through the features pointer vector 
// add each feature to the graph
// check if the feature does not already exist  before adding to grpah


// initialise the visoextarctor if it is the first start
if (start == true){
        //TODO: Ask Paul about how to initialise this function 
        RCLCPP_INFO(get_logger(), "Image dimension test 2");
	 viso_extractor_->InitialiseFeatures(image_left, image_right, image_left, image_right); 
	
 }
//TODO: add untracked features to ISAM optimizer and non linear graph 
viso_extractor_->FeaturesUntracked(image_left, image_right);
aru::core::utilities::image::FeatureSPtrVectorSptr features = viso_extractor_->GetCurrentFeatures();
RCLCPP_INFO(get_logger(), "Image dimension test 3");      

for (auto feat : *features) {

 // Get feature pixel coordinates
 aru::core::utilities::image::Feature& feature = *feat;
 cv::KeyPoint keypoint_left = feature.GetKeyPoint();
 cv::KeyPoint keypoint_right = feature.GetMatchedKeyPoint();
 Eigen::Vector3d camera_point_1 = feature.GetTriangulatedPoint();// feature in Camera frame 
 gtsam::Point3  camera_point  = gtsam::Point3(camera_point_1);

 double uL = keypoint_left.pt.x; //coordinates.first.row(i)(0);  // from example it is being multiplied by resolution, check why - resolution_x is image     distortion intrinsics
 double uR = keypoint_right.pt.x; //coordinates.second.row(num)(0); // from example it is being multiplied by resolution check why? 
 double v  = keypoint_left.pt.y; //coordinates.first.row(num)(1);  // same for both left and right images if the stereo cameras are rectified
 
       

  //TODO:// If landmark is behind camera, don't add to isam2 graph/point cloud
  const Cal3_S2Stereo::shared_ptr K(
	      new Cal3_S2Stereo(this->fx,this->fy, 0,this->cx, this->cy, this->Tx));

    graph->emplace_shared<
      GenericStereoFactor<Pose3, Point3> >(StereoPoint2(uL, uR, v), 
        pose_landmark_noise, X(this->key("pose")), L(this->key("landmark")), K);
  
  //TODO: // Transform landmark coordinates to world frame 
  Pose3 camera_pose = Pose3(T_cam_imu_mat)*newNodes.at<Pose3>(X(this->key("pose")));
  Point3  worldpoint = camera_pose.transformFrom(camera_point);
  //TODO: //Add ISAM2 value for feature/landmark 
   
  
  newNodes.insert(L(this->key("landmark")), worldpoint);

  landmark_id++; 
  this->increment("landmark");
  }
  
       
// Update tracked features with new images 
// add tracked features to graph not to ISAM 

viso_extractor_->UpdateFeatures(image_left, image_right);
aru::core::utilities::image::FeatureSPtrVectorSptr tracked_features = viso_extractor_->GetTrackedFeatures();

// get ids of the tracked features 
std::vector tracked_ids = viso_extractor_->GetTrackedIDs();
// index is used to iterate throught he IDs
// tracked_ids and tracked_features are of the same size
RCLCPP_INFO(get_logger(), "Image dimension test 7");
// skip the following when we are just starting vis_extracor
// nothing has been tracked yet
if ( start == false){

	int index = 0;
	for (auto feat : *tracked_features) {
	 // Get feature coordinates

	 aru::core::utilities::image::Feature& feature = *feat;
	 cv::KeyPoint keypoint_left = feature.GetKeyPoint();
	 cv::KeyPoint keypoint_right = feature.GetMatchedKeyPoint();
	 Eigen::Vector3d camera_point = feature.GetTriangulatedPoint();// feature in Camera frame 
         
         double uL = keypoint_left.pt.x; //coordinates.first.row(i)(0);  // from example it is being multiplied by resolution, check why resolution_x    is       image distortion intrinsics
         double uR = keypoint_right.pt.x; //coordinates.second.row(num)(0); // from example it is being multiplied by resolution check why? 
         double v  = keypoint_left.pt.y; //coordinates.first.row(num)(1);  // same for both left and right images if the stereo cameras are rectified
 

	  //bool bool_new_landmark = !result.exists(Symbol('l', landmark_id));
	  //if (bool_new_landmark) {
	     
	  // Landmark is from  
	  Symbol landmark = Symbol('l', tracked_ids.at(index));   
	   
	  // create stereo camera calibration object with .2m between cameras
	  // change accordingly the K calue depending n the camera parameters
          const Cal3_S2Stereo::shared_ptr K(
	      new Cal3_S2Stereo(this->fx,this->fy, 0,this->cx, this->cy, this->Tx));
	      
	   graph->emplace_shared<
	      GenericStereoFactor<Pose3, Point3> >(StereoPoint2(uL, uR, v), 
		pose_landmark_noise, X(this->key("pose")), landmark, K);
	    index++;
	    
	  }
	    start = false;
}
RCLCPP_INFO(get_logger(), "Image dimension test 6");
}

void ROSVO::increment(const std::string& key_) {
    ++keys.at(key_).key;
}

int ROSVO::key(const std::string& key_, const int offset) const {
    return keys.at(key_).key + offset;
}

int ROSVO::minimum_key(const unsigned int priority_) const {
    bool minimum_key_found{false};
    int minimum_key_ = std::numeric_limits<int>::max();
    for (const auto& [name, key_info] : keys) {
        if (key_info.priority <= priority_) {
            minimum_key_ = std::min(minimum_key_, key_info.key);
            minimum_key_found = true;
        }
    }
    if (!minimum_key_found) {
        throw std::runtime_error("No named keys with requested priority <= " + std::to_string(priority_) + " exist.");
    }
    return minimum_key_;
}

void ROSVO::set_named_key(const std::string& name, const int key_, const unsigned int priority_) {
    if (!keys.emplace(name, NamedKeyInfo{key_, priority_}).second) {
        keys.at(name) = NamedKeyInfo{key_, priority_};
    }
}

const rclcpp::Time& ROSVO::timestamp(const int key_) const {
    assert(key_ >= 0);
    return timestamps_.at(key_);
}

const rclcpp::Time& ROSVO::timestamp(const std::string& key_, const int offset) const {
    return timestamp(key(key_, offset));
 }   
void ROSVO::set_timestamp(const int key_, const rclcpp::Time& timestamp_) {
    assert(key_ >= 0);
    auto emplace_it = timestamps_.emplace(key_, timestamp_);
    emplace_it.first->second = timestamp_;
}

void ROSVO::set_timestamp(const std::string& key_, const rclcpp::Time& timestamp_, const int offset) {
    set_timestamp(key(key_, offset), timestamp_);
}


int main(int argc, char **argv) {

  // Force flush of the stdout buffer.
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);

  // Initialize any global resources needed by the middleware and the clienti
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
  auto rectify_node = std::make_shared<ROSCamera>(options);
  exec.add_node(vo_node);
  exec.add_node(rectify_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
