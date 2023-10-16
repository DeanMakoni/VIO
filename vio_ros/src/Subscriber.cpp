// Dean Makoni
#include <glog/logging.h>

#include <functional>
#include <memory>
#include <include/vio_ros/Subscriber.hpp>
#include <vector>


namespace aru{
 

 Subscriber::~Subscriber(){
  // add code for destructor here
 }

 Subscriber::Subscriber()
 {

 }

 void Subscriber::setNodeHandle(rclpp::Node& node)
 {
 } 
} // namespace aru
