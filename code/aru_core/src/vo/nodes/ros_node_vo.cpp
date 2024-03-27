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

using namespace std;
using namespace gtsam;

using namespace aru::core;

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::P;  // Pressure bias
//using gtsam::symbol_shorthand::B_P; // barometric bias that will be constrained
class ROSVO : public rclcpp::Node {

public:
  explicit ROSVO(const rclcpp::NodeOptions &options);

  virtual ~ROSVO();
std::shared_ptr<gtsam::PreintegrationType> preintegrated;
int pose_id = 0;
int pressure_bias =0; //barometric bias that will be constrained
int pressure_count = 0;
imuBias::ConstantBias prior_imu_bias;
imuBias::ConstantBias prev_bias;
std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>> sync;

// Create iSAM2 object
//std::unique_ptr<ISAM2> ISAM;
ISAM2* ISAM = 0;
Key biasKey = Symbol('b', 0);
int p = 1;
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
  
 typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CompressedImage,
       sensor_msgs::msg::CompressedImage>
        StereoApprxTimeSyncPolicy;
 typedef message_filters::Synchronizer<StereoApprxTimeSyncPolicy> StereoApprxTimeSyncer;
           
            message_filters::Subscriber<sensor_msgs::msg::CompressedImage> image_subscriber_1;
            message_filters::Subscriber<sensor_msgs::msg::CompressedImage> image_subscriber_2;
 std::shared_ptr<StereoApprxTimeSyncer> stereo_approx_time_syncher_;

 
protected:
  bool startCamera();
  void featureToLandmark(const cv::Mat &image_lef, const cv::Mat &image_right); //,std::pair<Eigen::MatrixXf, Eigen::MatrixXf> coordinates);
  void publishImage(cv::Mat &img, image_transport::CameraPublisher &pubImg,
                    std::string imgFrameId, rclcpp::Time t);

       // ----> Thread functions
  void test_callback_function(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void callback_function(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg_right,const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg_left);
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
  noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.3),Vector3::Constant(0.1)).finished()); // 30cm std on x,y,z 0.1 rad on      roll,pitch,yaw 
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

  bool prev_image = false;
  uint64_t prev_timestamp;

  // subscribers
  
  // Camera subscriber
  std::string stereo_topic_left;
  std::string stereo_topic_right;
  
 rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
    image_stereo_subscriber_right ;
      
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      image_stereo_subscriber_left;
  

  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> stereo_subsriber_right;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> stereo_subscriber_left;
  
 

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
  stereo_topic_right = "slave1/image_raw/compressed";
  stereo_topic_left = "slave2/image_raw/compressed";
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
  

// image_stereo_subscriber_right = this->create_subscription<sensor_msgs::msg::CompressedImage>(
    //  stereo_topic_right, 10,
     // std::bind(&ROSVO::test_callback_function, this, std::placeholders::_1));
 // image_stereo_subscriber_left = this->create_subscription<sensor_msgs::msg::CompressedImage>(
    //  stereo_topic_left, 10,
    //  std::bind(&ROSVO::callback_function, this, std::placeholders::_1));
   

  
  
  stereo_subsriber_right.subscribe(this, stereo_topic_right);
  stereo_subscriber_left.subscribe(this, stereo_topic_left);
 
  sync = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::CompressedImage,sensor_msgs::msg::CompressedImage>>(stereo_subsriber_right,stereo_subscriber_left ,3);
//sync->getPolicy()->setMaxIntervalDuration(rclcpp::Duration(10,0));
//sync->registerCallback(std::bind(&ROSVO::callback_function, this, std::placeholders::_1, std::placeholders::_2));
 
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
                  &ROSVO::callback_function, this, std::placeholders::_1,
                  std::placeholders::_2));
 
  
   
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

  //RCLCPP_INFO(get_logger(), "Dean 3");
  vo_ = std::make_shared<aru::core::vo::VO>(vo_vocab_file,extractor_params,matcher_params, solver_params);
  
  // Track features for the incoming images 
  viso_extractor_ =  std::make_shared <utilities::image::VisoFeatureTracker>(matcher_params,extractor_params);
  RCLCPP_INFO(get_logger(), "Dean 4");
  
  
  // Define Publisher
  vo_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
           "vo/tf2", 10); // vo_tf2_topic_
  
  kf_tf2_publisher_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
          "kf/tf2", 10); // kf_tf2_topic_
  pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::Pose>(
         "gtsm/pose", 10); // pose_topic_
   macthed_points_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "macthed/points", 10); //macthed_points_topic_
   
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
  preintegrated = std::make_shared<PreintegratedImuMeasurements>(p1, prior_imu_bias);
  assert(preintegrated);
}

ROSVO::~ROSVO() { RCLCPP_DEBUG(get_logger(), "Destroying node"); }

void ROSVO::test_callback_function(const sensor_msgs::msg::CompressedImage::SharedPtr msg){

  RCLCPP_INFO(get_logger(), "Stereo Topic Received");
}

void ROSVO::callback_function(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg_right, const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg_left) {
  RCLCPP_INFO(get_logger(), "Filtered Stereo Topic Received");
  // Get images from msg
  cv::Mat left_image = cv::imdecode(cv::Mat(msg_right->data), cv::IMREAD_COLOR);
  cv::Mat right_image = cv::imdecode(cv::Mat(msg_left->data), cv::IMREAD_COLOR);
  
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
  std_msgs::msg::Header h = msg_right->header;
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
    RCLCPP_INFO(get_logger(), "Stereo images Parameters set"); 
    
     
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

    TODO:// Check why is the example using prev_camera pose instead of the current pose
    //if (count == 0 || count == 1) {
    //  prev_camera_pose = Pose3() * Pose3(T_cam_imu_mat);
   // 
    // add Stereo factors to graph and values t
     //, coordinates);
    

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
}

void ROSVO::imu_callback_function(const sensor_msgs::msg::Imu::SharedPtr msg){
 
 RCLCPP_INFO(get_logger(), "IMU topic received");
 Vector3 measuredAcc (msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
 Vector3 measuredOmega (msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
 //Check the time stamp 
 // if it is greater than 10 nano se it is a pose
 gtsam::Point3 curr_accel(measuredAcc);
 gtsam::Point3 curr_angular(measuredOmega);

 auto current_timestamp = msg->header.stamp;
 std::cout << "Timestamp: " << current_timestamp.sec << "seconds" << std::endl;
 std::cout << "Timestamp: " << prev_timestamp  << "seconds" << std::endl;
 
 // Check on how to add the correct dt
 double dt = 0.005; 
 preintegrated->integrateMeasurement(curr_accel,curr_angular, dt);
 RCLCPP_INFO(get_logger(), "Intergrated");
 
 RCLCPP_INFO(get_logger(), "Imu callback done");
 
}

void ROSVO::sonar_callback_function(const sensor_msgs::msg::Range::SharedPtr msg){

}

void ROSVO::depth_callback_function(const sensor_msgs::msg::FluidPressure::SharedPtr msg){
     
 
 
     
   //gtsam::Rot3 prior_rotation = gtsam::Rot3::Quaternion(1, 1,1,1);
   //Point3 prior_point;
  // gtsam::Pose3 prior_pose(gtsam::Rot3::Quaternion(1, 18,16,1), Point3(10, 8,23));
  // Vector3 prior_velocity (1,2,10);

  // gtsam::NavState prev_state(prior_pose, prior_velocity);
    //gtsam::NavState prop_state = prev_state;

    // Conver Imu message  to gtsam::Pose3
    // Extract orientation quaternion from IMU message
   // Eigen::Quaterniond orientation(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

    // Convert orientation quaternion to rotation matrix
    //Eigen::Matrix3d rotationMatrix = orientation.toRotationMatrix();
   // Rot3 rot3(rotationMatrix);
    // Extract translation if available (assuming it's in the IMU message)
 //   double x_translation = msg->linear_acceleration.x;
  //  double y_translation = msg->linear_acceleration.y;
   // double z_translation = msg->linear_acceleration.z;

    // Create Pose3 object
    // use it as prior pose
   // Pose3 gtsam_pose(rot3, Point3(x_translation, y_translation, z_translation));

    if (pose_id == 0){
       
       
     // Prior pressure noise model TODO: change valye accordingly
     //double sigma = 0.1; // Adjust this value as needed
     //gtsam::Matrix1 covariance = gtsam::Matrix::Identity(1,1) * sigma*sigma;
     // Create the Gaussian noise model
     //auto prior_pressure_noise_model = gtsam::noiseModel::Gaussian::Covariance(covariance);
       
       prior_pose = Pose3(gtsam::Rot3::Ypr(0, 0,0), Point3(0, 0,0));
       prior_velocity = Vector3(0,0,0);
       //prior_pressure = 1; //change this vlue accordingly
       // add prior factors for stereo cameras and IMU
       newNodes.insert(X(pose_id), prior_pose);
       newNodes.insert(V(pose_id), prior_velocity);
      // newNodes.insert(P(pose_id), prior_pressure);
       //newNodes.insert(B(pose_id), prior_imu_bias);
       newNodes.insert(biasKey, imuBias::ConstantBias());

       graph->addPrior(X(pose_id), prior_pose, pose_noise); 
       //graph->addPrior(P(pose_id), prior_pressure,prior_pressure_noise_model);
       graph->addPrior(V(pose_id), prior_velocity, velocity_noise_model);
       //graph->addPrior(B(pose_id), prior_imu_bias, bias_noise_model);
       graph->addPrior(biasKey,imuBias::ConstantBias() , bias_noise_model);
       pose_id = pose_id + 1;
       biasKey++;
       RCLCPP_INFO(get_logger(), "Pose_ID zero");
      }
        //TODO: //  Create a Navstate to store pose, velocity and bias
    // add  values to  be optimized
    //Navstate prev_state(prior_pose, prior_velocity);
    else{
    //gtsam::NavState  prev_state = NavState(result.at<Pose3>(X(pose_id,
     //                       result.at<Vector3>(V(pose_id - 1)));
    

     // add IMU and Bias factors to graph
     auto preint_imu =
         dynamic_cast<const PreintegratedImuMeasurements&>(*preintegrated);
      //auto preint_imu = std::make_unique<gtsam::PreintegratedImuMeasurements>(
       //                  CastToPreintergratedImuMeasurements(*preintegrated);
                         
      ImuFactor imu_factor(X(pose_id - 1 ), V(pose_id -1),
                           X(pose_id), V(pose_id),
                           biasKey -1, preint_imu);
      graph->add(imu_factor);
      RCLCPP_INFO(get_logger(), "IMU factor added");
      imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
      graph->add(BetweenFactor<imuBias::ConstantBias>(
         biasKey-1, biasKey, imuBias::ConstantBias(),
        bias_noise_model));
       RCLCPP_INFO(get_logger(), "Bias factor added");
       
      gtsam::NavState  prev_state = NavState(prior_pose,
                          prior_velocity); 
      gtsam::NavState prop_state = preintegrated->predict(prev_state, imuBias::ConstantBias());
      newNodes.insert(X(pose_id), prop_state.pose());
      newNodes.insert(V(pose_id), prop_state.v());
      newNodes.insert(biasKey, imuBias::ConstantBias());
      
      // create Brometric factor and add it to the graph
     // Define the noise variance (sigma^2)
     
     double sigma = 0.1; // Adjust this value as needed

     // Create a 1x1 covariance matrix
     gtsam::Matrix1 covariance = gtsam::Matrix::Identity(1,1) * msg->variance;

    // Create the Gaussian noise model
     //auto pressure_noise_model = gtsam::noiseModel::Gaussian::Covariance(covariance);
     //auto pressure_noise_model = gtsam::noiseModel::Diagonal(gtsam::Vector1(msg->variance);
     SharedNoiseModel pressure_noise_model = noiseModel::Isotropic::Sigma(1, 0.25);
     // Noise model for barometer/pressure TODO: Chnge this to use suitale values
     // noiseModel::Gaussian::shared_ptr pressure_noise_model = noiseModel::Gaussian::Sigma(6, 1e-3);
  
     // sensor message's pressure is in pascals, convert it o KPa for gtsam
     const double pressure = (msg->fluid_pressure/ 1000);
       
     RCLCPP_INFO(get_logger(), "Barometer topic Received");
     // create barometric factor
     BarometricFactor pressure_factor(X(pose_id),
                                     P(pose_id),pressure,
                                     pressure_noise_model);
    graph->add(pressure_factor);
    //gtsam::Pose3 pressure_pose = Pose3(gtsam::Rot3::RzRyRx(0, 0,0), Point3(0, 0,1000));
    newNodes.insert(P(pose_id), pressure);
    double pressurebias = 0;
    //newNodes.insert(Symbol('B_P',pressure_bias),  pressurebias);
       
       
       // ISAM2 solver
       RCLCPP_INFO(get_logger(), "ISAM solver");
       //Values Testnode;
       //NonlinearFactorGraph* testgraph = new NonlinearFactorGraph();
       ISAM->update(*graph, newNodes);
       RCLCPP_INFO(get_logger(), "ISAM solver Done for ");
       std::cout << p << std::endl;
       p = p+1;
       result = ISAM->calculateEstimate();
       prior_pose = result.at<Pose3>(X(pose_id ));

       prior_velocity = result.at<Vector3>(V(pose_id ));
       
       
       RCLCPP_INFO(get_logger(), "Calculation done");
        // reset the graph
        graph->resize(0);
        newNodes.clear();
         
        // Overwrite the beginning of the preintegration for the next step.
      // prev_state = NavState(result.at<Pose3>(X(pose_id - 1)),
         //                   result.at<Vector3>(V(pose_id - 1)));
       RCLCPP_INFO(get_logger(), "Prev state updated");
       prev_bias = result.at<imuBias::ConstantBias>(biasKey);
        RCLCPP_INFO(get_logger(), "Prev bias updated");
       // Reset the preintegration object.
       preintegrated->resetIntegrationAndSetBias(imuBias::ConstantBias());
       // publish poses

       gtsam::Pose3 current_pose = result.at<Pose3>(X(pose_id));
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

       pose_publisher_->publish(pose_msg);
      
       pose_id = pose_id + 1;
       biasKey++;
        // see how to add  barometric bias to graph and values
      pressure_bias = pressure_bias + 1;
      pressure_count = pressure_count + 1;
       }

}

// Transform feature from image_processor to landmark with 3D coordinates
// Add landmark to ISAM2 graph if not already there (connect to current pose with a factor)
// Add landmark to point cloud 
void ROSVO::featureToLandmark(const cv::Mat &image_left, const cv::Mat &image_right){  //,std::pair<Eigen::MatrixXf, Eigen::MatrixXf> coordinates){



 cv::Mat image_dest_left, image_dest_right;

// iterate through the features pointer vector 
// add each feature to the graph
// check if the feature does not already exist  before adding to grpah


// initialise the visoextarctor if it is the first start
if (start == true){
	viso_extractor_->InitialiseFeatures(image_dest_left, image_dest_right, image_left, image_right);         

 }
// add untracked features to ISAM and graph 
viso_extractor_->FeaturesUntracked(image_left, image_right);
// Get the feature that are not macthed yet
// Add them to vlues and graph
// features that are macthed add them to graph only
aru::core::utilities::image::FeatureSPtrVectorSptr features = viso_extractor_->GetCurrentFeatures();

for (auto feat : *features) {
 // Get feature coordinates

 aru::core::utilities::image::Feature& feature = *feat;
 cv::KeyPoint keypoint_left = feature.GetKeyPoint();
 cv::KeyPoint keypoint_right = feature.GetMatchedKeyPoint();
 Eigen::Vector3d worldpoint = feature.GetTriangulatedPoint();// world point
 //TODO: Check how you can use transform from under gtsam so that it is in imu/world coordiantes

 double uL = keypoint_left.pt.x; //coordinates.first.row(i)(0);  // from example it is being multiplied by resolution, check why - resolution_x is image distortion intrinsics
 double uR = keypoint_right.pt.x; //coordinates.second.row(num)(0); // from example it is being multiplied by resolution check why? 
 double v  = keypoint_right.pt.y; //coordinates.first.row(num)(1);  // same for both left and right images if the stereo cameras are rectified
 
 // Estimate feature location in camera frame
  double d = uR-uL;
  double x = uL;
  double y = v;
  double W = d/ 0.2; //distance from cam0 to cam1

 // cv::Mat K_matrix =  cv::Mat::zeros(3, 3, CV_64FC1); // intrinsic camera parameters
 // K_matrix.at<double>(0, 0) = 0;//f_u_;   //      [ fx   0  cx ]
  //K_matrix.at<double>(1, 1) = 0; //f_v_;   //      [  0  fy  cy ]
  //K_matrix.at<double>(0, 2) = 0; //u_c_;   //      [  0   0   1 ]

// Estimate feature location in camera frame
// change camera intrinsics accordingly
  double X_camera =  (x -  637.87114)/W;//(x-cx)/W;
  double Y_camera =  (y - 331.27469)/W;//(x -cy)/w
  double Z_camera =  531.14774/W;  //f/W

 gtsam::Point3 camera_point = gtsam::Point3(X_camera,Y_camera,Z_camera);

  //TODO:// If landmark is behind camera, don't add to isam2 graph/point cloud
  

  //TODO: // Transform landmark coordinates to world frame 

  //TODO: //Add ISAM2 value for feature/landmark if it doesn't already exist

  //bool bool_new_landmark = !result.exists(Symbol('l', landmark_id));
  //if (bool_new_landmark) {
     
      // Landmark is from  
  Symbol landmark = Symbol('l', landmark_id);
  newNodes.insert(landmark, worldpoint);
    //}
    
   //Symbol landmark = Symbol('l', landmark_id);
  // Add ISAM2 factor connecting this frame's pose to the landmark
  // create stereo camera calibration object with .2m between cameras
  // change accordingly the K calue depending n the camera parameters
  const Cal3_S2Stereo::shared_ptr K(
      new Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2));

    graph->emplace_shared<
      GenericStereoFactor<Pose3, Point3> >(StereoPoint2(uL, uR, v), 
        pose_landmark_noise, X(pose_id), landmark, K);
   landmark_id++;
    
  }
// Update tracked features with new images 
// add tracked features to graph not to ISAM 
viso_extractor_->UpdateFeatures(image_left, image_right);
aru::core::utilities::image::FeatureSPtrVectorSptr tracked_features = viso_extractor_->GetTrackedFeatures();

// get ids of the tracked features 
std::vector tracked_ids = viso_extractor_->GetTrackedIDs();
// index is used to iterate throught he IDs
// tracked_ids and tracked_features are of the same size

// skip the following when we are just starting vis_extracor
if ( start == false){

	int index = 0;
	for (auto feat : *tracked_features) {
	 // Get feature coordinates

	 aru::core::utilities::image::Feature& feature = *feat;
	 cv::KeyPoint keypoint_left = feature.GetKeyPoint();
	 cv::KeyPoint keypoint_right = feature.GetMatchedKeyPoint();
	 Eigen::Vector3d worldpoint = feature.GetTriangulatedPoint();// world point
	 //TODO: Check how you can use transform from under gtsam so that it is in imu/world coordiantes

	 double uL = keypoint_left.pt.x; //coordinates.first.row(i)(0);  // from example it is being multiplied by resolution, 
	 //check why - resolution_x is image distortion intrinsics
	 double uR = keypoint_right.pt.x; //coordinates.second.row(num)(0); // from example it is being multiplied by resolution check why? 
	 double v  = keypoint_right.pt.y; //coordinates.first.row(num)(1);  // same for both left and right images if the stereo cameras are rectified
	 
	 // Estimate feature location in camera frame
	  double d = uR-uL;
	  double x = uL;
	  double y = v;
	  double W = d/ 0.2; //distance from cam0 to cam1

	 // cv::Mat K_matrix =  cv::Mat::zeros(3, 3, CV_64FC1); // intrinsic camera parameters
	 // K_matrix.at<double>(0, 0) = 0;//f_u_;   //      [ fx   0  cx ]
	  //K_matrix.at<double>(1, 1) = 0; //f_v_;   //      [  0  fy  cy ]
	  //K_matrix.at<double>(0, 2) = 0; //u_c_;   //      [  0   0   1 ]

	// Estimate feature location in camera frame
	// change camera intrinsics accordingly
	  double X_camera =  (x -  637.87114)/W;//(x-cx)/W;
	  double Y_camera =  (y - 331.27469)/W;//(x -cy)/w
	  double Z_camera =  531.14774/W;  //f/W

	 gtsam::Point3 camera_point = gtsam::Point3(X_camera,Y_camera,Z_camera);

	  //TODO:// If landmark is behind camera, don't add to isam2 graph/point cloud
	  

	  //TODO: // Transform landmark coordinates to world frame 

	  //TODO: //Add ISAM2 value for feature/landmark if it doesn't already exist

	  //bool bool_new_landmark = !result.exists(Symbol('l', landmark_id));
	  //if (bool_new_landmark) {
	     
	  // Landmark is from  
	  Symbol landmark = Symbol('l', tracked_ids.at(index));   
	   
	  // create stereo camera calibration object with .2m between cameras
	  // change accordingly the K calue depending n the camera parameters
	  const Cal3_S2Stereo::shared_ptr K(
	      new Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2));

	    graph->emplace_shared<
	      GenericStereoFactor<Pose3, Point3> >(StereoPoint2(uL, uR, v), 
		pose_landmark_noise, X(pose_id), landmark, K);
	    index++;
	    
	  }
	    start = false;
}

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
  exec.add_node(vo_node);

  // spin will block until work comes in, execute work as it becomes available,
  // and keep blocking. It will only be interrupted by Ctrl-C.
  exec.spin();

  rclcpp::shutdown();

  return 0;
}
