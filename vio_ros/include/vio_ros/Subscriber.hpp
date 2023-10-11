// Dean Makoni

#include "rclcpp/rclcpp.hpp"

namespace aru{

 /**
 * @brief This class handles all the buffering of incoming data.
 */

 class Subscriber{
 	
        public:
	// Destructor 
         ~Subscriber()

	/**
         * @brief Constructor. This will either subscribe to the relevant ROS topics or
         *        start up the sensor and register the callbacks directly there.
         * @param nh The ROS node handle.
         * @param vioInterfacePtr Pointer to the VioInterface.
         * @param param_reader  Parameter reader.
         */

        Subscriber(rclcpp::Node&,,);

       /// @brief Set the node handle. This sets up the callbacks. This is called in the constructor.
        void setNodeHandle(rclcpp::Node& node);  // NOLINT
	
        protected: 
       
        const cv::Mat readRosImage(const sensor_msgs::ImageConstPtr& img_msg) const;
        
        /// @brief The image callback.
        void imageCallback(const sensor_msgs::ImageConstPtr& msg, unsigned int cameraIndex);

       /// @brief The depth image callback.
       void depthImageCallback(const sensor_msgs::ImageConstPtr&, unsigned int);

       /// @brief The IMU callback.
       void imuCallback(const sensor_msgs::ImuConstPtr& msg);

      /// @brief The Sonar Range callback. @Sharmin
       boost::shared_ptr<tf2_ros::Buffer> tfBuffer_;
       boost::shared_ptr<tf2_ros::TransformListener> tfListener_;
       void sonarCallback(const imagenex831l::ProcessedRange::ConstPtr& msg);
 
       rclcpp::Node* node;  // node handle
       rclcpp::Subscription subImu_;  // IMU subscriber node
       rclcpp::Subscription subSonarRange_;  /// The Sonar Range Subscriber 
       rclcpp::Subscription  subDepth_;     ///< The Depth Subscriber       
       rclcpp::Subscription subReloPoints_;  ///< The Relocalization Points Subscriber from pose_graph
       unsigned int imgLeftCounter;                                 // @Sharmin
       unsigned int imgRightCounter;
  }

} //namspace aru
