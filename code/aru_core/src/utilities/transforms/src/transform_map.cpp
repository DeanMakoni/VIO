//
// Created by paulamayo on 2022/05/04.
//
#include <aru/core/utilities/transforms/transform_map.h>
#include <boost/make_shared.hpp>
//#include <cv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
namespace aru {
namespace core {
namespace utilities {
namespace transform {
//------------------------------------------------------------------------------
void TransformMap::AddTransform(TransformSPtr transform) {
  time_transform_map_.insert(
      TimeTransformMap::value_type(transform->GetSourceTimestamp(), transform));
}
//------------------------------------------------------------------------------
bool TransformMap::IsWindowValid(int64_t t_start, int64_t t_end) {

  int64_t last_time = (--(time_transform_map_.end()))->first;
  int64_t start_time = time_transform_map_.begin()->first;
  if (t_start >= start_time && t_start <= last_time && t_end > start_time &&
      t_end <= last_time) {
    return true;
  } else {
    return false;
  }
}
//------------------------------------------------------------------------------
TransformSPtr TransformMap::Interpolate(int64_t t_end) {
  int64_t t_start = time_transform_map_.begin()->first;
  if (IsWindowValid(t_start, t_end)) {
    // Create a smaller map bounded by the timestamps
    TimeTransformMap local_time_transport_map;
    TimeTransformMap::const_iterator end_node =
        time_transform_map_.lower_bound(t_end);
    //    TimeTransformMap::const_iterator start_node =
    //        time_transform_map_.upper_bound(t_start);
    //    --start_node;
    //    ++end_node;
    //    local_time_transport_map.insert(start_node, end_node);
    //
    //    // Get the iterators to start and end
    //    end_node = --(local_time_transport_map.end());
    //    start_node = ++(local_time_transport_map.begin());

    // Get the iterators to the nodes right before these
    TimeTransformMap::const_iterator end_node_m1 = end_node;
    --end_node_m1;
    //    TimeTransformMap::const_iterator start_node_m1 = start_node;
    //    --start_node_m1;

    // Get the end interpolation ratio
    double end_ratio = double(end_node->first - t_end) /
                       double(end_node->first - end_node_m1->first);
    cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    // Read Rotation matrix and convert to vector
    cv::eigen2cv(end_node->second->GetRotation(), R_matrix);
    Eigen::Matrix3f rot_eig = end_node->second->GetRotation();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    quat = quat.slerp(end_ratio,
                      Eigen::Quaternionf(end_node->second->GetRotation()));
    Eigen::Matrix3f Rot_t_end_end_node = Eigen::Matrix3f(quat);

    // cv::Rodrigues(R_matrix, rvec);

    // Convert rotation rpy to Matrix
    //    cv::Rodrigues(end_ratio * rvec, R_matrix);
    //    Eigen::Matrix3f Rot_t_end_end_node;
    //    cv::cv2eigen(R_matrix, Rot_t_end_end_node);
    Eigen::Vector3f xyz_t_end_end_node =
        end_ratio * end_node->second->GetTranslation();


    // Get the interpolated transformation between end node m1 and t_end
    Eigen::Affine3f T_t_end_end_node;
    T_t_end_end_node.linear() = Rot_t_end_end_node;
    T_t_end_end_node.translation() = xyz_t_end_end_node;
    Eigen::Affine3f T_end_node_m1_end_node = end_node->second->GetTransform();
    Eigen::Affine3f T_end_node_m1_t_end =
        T_end_node_m1_end_node * T_t_end_end_node.inverse();

    // Chain the transforms between start_node and end_node_m1
    Eigen::Affine3f T_start_node_end_node_m1;
    T_start_node_end_node_m1.linear() = Eigen::MatrixXf::Identity(3, 3);
    T_start_node_end_node_m1.translation() = Eigen::VectorXf::Zero(3);
    for (auto itr = end_node_m1; itr != time_transform_map_.begin(); --itr) {
      T_start_node_end_node_m1 =
          itr->second->GetTransform() * T_start_node_end_node_m1;
    }

    //
    Eigen::Affine3f T_t_start_t_end =
        T_start_node_end_node_m1 * T_end_node_m1_t_end;

    return boost::make_shared<Transform>(t_end, t_start, T_t_start_t_end);
  } else {
    return nullptr;
  }
}
//------------------------------------------------------------------------------
TransformSPtr TransformMap::Interpolate(int64_t t_start, int64_t t_end) {
  if (t_start == t_end || t_start > t_end) {
    return nullptr;
  }

  if (IsWindowValid(t_start, t_end)) {
    // Create a smaller map bounded by the timestamps
    TimeTransformMap local_time_transport_map;
    TimeTransformMap::const_iterator end_node =
        time_transform_map_.lower_bound(t_end);
    TimeTransformMap::const_iterator start_node =
        time_transform_map_.upper_bound(t_start);
    //--start_node;
    //++end_node;
    //local_time_transport_map.insert(start_node, end_node);

    // Get the iterators to start and end
    //end_node = --(local_time_transport_map.end());
    //start_node = ++(local_time_transport_map.begin());

    // Get the iterators to the nodes right before these
    auto end_node_m1 = end_node;
    --end_node_m1;
    auto start_node_m1 = start_node;
    --start_node_m1;
    
    LOG(INFO)<<"Size of map is "<<time_transform_map_.size();

    // Get the end interpolation ratio
    double end_ratio = double(end_node->first - t_end) /
                       double(end_node->first - end_node_m1->first);

    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    quat = quat.slerp(end_ratio,
                      Eigen::Quaternionf(end_node->second->GetRotation()));
    Eigen::Matrix3f Rot_t_end_end_node = Eigen::Matrix3f(quat);
    Eigen::Vector3f xyz_t_end_end_node =
        end_ratio * end_node->second->GetTranslation();
    // Get the interpolated transformation between end node m1 and t_end
    Eigen::Affine3f T_t_end_end_node;
    T_t_end_end_node.linear() = Rot_t_end_end_node;
    T_t_end_end_node.translation() = xyz_t_end_end_node;
    Eigen::Affine3f T_end_node_m1_end_node = end_node->second->GetTransform();
    Eigen::Affine3f T_end_node_m1_t_end;

    if (end_node_m1->first > start_node->first) {
      T_end_node_m1_t_end = T_end_node_m1_end_node * T_t_end_end_node.inverse();
    } else {
      T_end_node_m1_t_end.linear() = Eigen::MatrixXf::Identity(3, 3);
      T_end_node_m1_t_end.translation() = Eigen::VectorXf::Zero(3);
    }

    // Chain the transforms between start_node and end_node_m1
    Eigen::Affine3f T_start_node_end_node_m1;
    T_start_node_end_node_m1.linear() = Eigen::MatrixXf::Identity(3, 3);
    T_start_node_end_node_m1.translation() = Eigen::VectorXf::Zero(3);
    int num_nodes=0;
    if (end_node_m1->first > start_node->first) {
      for (auto itr = end_node_m1; itr != start_node; --itr) {
        T_start_node_end_node_m1 =
            itr->second->GetTransform() * T_start_node_end_node_m1;
            num_nodes++;
            //LOG(INFO)<<"Transforms to be chained are "<<itr->first <<" with dist "<<itr->second->GetTransform().translation().norm();
      }
    }

    // Get the start interpolation ratio
    double start_ratio = double(start_node->first - t_start) /
                         double(start_node->first - start_node_m1->first);
                         
    LOG(INFO)<<"Num nodes is "<<num_nodes; 

    Eigen::Quaternionf quat_start = Eigen::Quaternionf::Identity();
    quat_start = quat_start.slerp(
        start_ratio, Eigen::Quaternionf(start_node->second->GetRotation()));
    Eigen::Matrix3f Rot_t_start_start_node = Eigen::Matrix3f(quat_start);
    Eigen::Vector3f xyz_t_start_start_node =
        start_ratio * start_node->second->GetTranslation();

    // Get the interpolated transformation between end node m1 and t_end
    Eigen::Affine3f T_t_start_start_node;
    T_t_start_start_node.linear() = Rot_t_start_start_node;
    T_t_start_start_node.translation() = xyz_t_start_start_node;

    //
    Eigen::Affine3f T_t_start_t_end =
        T_t_start_start_node * T_start_node_end_node_m1 * T_end_node_m1_t_end;

    return boost::make_shared<Transform>(t_end, t_start, T_t_start_t_end);
  } else {
    return nullptr;
  }
}
} // namespace transform
} // namespace utilities
} // namespace core
} // namespace aru
