#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/logging/log.h"
#include "aru/core/utilities/transforms/transformprotocolbufferadaptor.h"
#include "glog/logging.h"
#include <aru/core/utilities/transforms/transform_map.h>
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace aru::core::utilities::transform;
struct TransformFixture {
  Eigen::Affine3f transform_1;
  Eigen::Affine3f transform_2;

  TransformFixture() {

    transform_1.matrix() = Eigen::MatrixXf::Identity(4, 4);
    transform_2.matrix() = Eigen::MatrixXf::Ones(4, 4);
  }
};

BOOST_FIXTURE_TEST_CASE(WriteToProtocolBuffer, TransformFixture) {

  aru::core::utilities::transform::Transform aru_transform_1("A", "B", 0, 1,
                                                             transform_1);
  aru::core::utilities::transform::Transform aru_transform_2("B", "C", 1, 2,
                                                             transform_2);

  // Initialize the adaptor
  aru::core::utilities::transform::TransformProtocolBufferAdaptor adaptor;

  LOG(INFO) << "Transform 1 is \n" << transform_1.matrix();
  Eigen::MatrixXf extrinsic = transform_1.matrix();

  LOG(INFO) << "Block 1 is \n" << extrinsic.block<1, 3>(0, 0);
  LOG(INFO) << "Block 2 is \n" << extrinsic.block<1, 3>(1, 0);
  LOG(INFO) << "Block 3 is \n" << extrinsic.block<1, 3>(2, 0);
  LOG(INFO) << "R is \n" << extrinsic.block<3, 3>(0, 0);
  LOG(INFO) << "T is \n" << extrinsic.block<3, 1>(0, 3);

  LOG(INFO) << "Initialised the adaptor";
  aru::core::utilities::logging::ProtocolLogger<
      datatype::transform::pbTransform>
      logger("/home/paulamayo/data/kitti/training/transform.monolithic", true);

  datatype::transform::pbTransform pb_transform_1 =
      adaptor.ReadToProtocolBuffer(aru_transform_1);
  datatype::transform::pbTransform pb_transform_2 =
      adaptor.ReadToProtocolBuffer(aru_transform_2);

  LOG(INFO) << "Source pb is " << pb_transform_1.source();

  logger.WriteToFile(pb_transform_1);
  logger.WriteToFile(pb_transform_2);
}

BOOST_FIXTURE_TEST_CASE(ReadFromProtocolBuffer, TransformFixture) {

  // Initialize the adaptor
  aru::core::utilities::transform::TransformProtocolBufferAdaptor adaptor;

  LOG(INFO) << "Transform 1 is " << transform_1.matrix();
  LOG(INFO) << "Initialised the adaptor";
  aru::core::utilities::logging::ProtocolLogger<
      datatype::transform::pbTransform>
      logger("/home/paulamayo/data/husky_data/vo/outdoor_zoo_vo.monolithic",
             false);

  datatype::transform::pbTransform pb_transform_file_1, pb_transform_file_2;

  pb_transform_file_1 = logger.ReadFromFile();
  pb_transform_file_2 = logger.ReadFromFile();

  LOG(INFO) << "Reading the files";
  aru::core::utilities::transform::Transform transform_out_1 =
      adaptor.ReadFromProtocolBuffer(pb_transform_file_1);
  aru::core::utilities::transform::Transform transform_out_2 =
      adaptor.ReadFromProtocolBuffer(pb_transform_file_2);

  LOG(INFO) << "Transform 1 is " << transform_out_1.GetTransform().matrix();
  LOG(INFO) << "Transform 2 is " << transform_out_2.GetTransform().matrix();
}

BOOST_AUTO_TEST_CASE(Interpolation) {
  Eigen::Affine3f transform_1;
  transform_1.matrix() = Eigen::MatrixXf::Identity(4, 4);
  TransformSPtr T_AB = boost::make_shared<Transform>(0, 0, transform_1);
  Eigen::Affine3f transform_2;
  transform_2.matrix() = Eigen::MatrixXf::Identity(4, 4);
  transform_2.translation() = Eigen::Vector3f(1, 0, 1);
  TransformSPtr T_BC = boost::make_shared<Transform>(2, 0, transform_2);
  Eigen::Affine3f transform_3;
  transform_3.matrix() = Eigen::MatrixXf::Identity(4, 4);
  transform_3.translation() = Eigen::Vector3f(1, 0, 0);
  TransformSPtr T_CD = boost::make_shared<Transform>(4, 2, transform_3);

  TransformMap transform_map;
  transform_map.AddTransform(T_AB);
  transform_map.AddTransform(T_BC);
  transform_map.AddTransform(T_CD);

  TransformSPtr T_inter = transform_map.Interpolate(1, 3);
  LOG(INFO) << "Transform out is \n" << T_inter->GetTransform().matrix();
}