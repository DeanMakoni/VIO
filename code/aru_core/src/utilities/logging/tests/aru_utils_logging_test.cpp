#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/logging/index_log.h"
#include "glog/logging.h"
#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>

BOOST_AUTO_TEST_CASE(INDEX_LOG) {

  aru::core::utilities::logging::IndexLogger<datatype::image::pbImage>
      logger("/home/paulamayo/data/husky_data/log/zoo_loop_left.monolithic", false);

  datatype::image::pbImage pb_image=logger.ReadIndex(10);

  LOG(INFO)<<"Image time is "<<pb_image.timestamp();

}
BOOST_AUTO_TEST_CASE(Read_bytes) {

  aru::core::utilities::logging::ProtocolLogger<datatype::laser::pbLaser>
      logger("/home/paulamayo/data/husky_data/log/white_lab_laser"
             ".monolithic",
             false);


  datatype::laser::pbLaser laser_pb=logger.ReadSkipped(14274350);

  LOG(INFO)<<"Pb laser time is "<<laser_pb.timestamp();

  laser_pb=logger.ReadFromFile();

  LOG(INFO)<<"Pb laser time is "<<laser_pb.timestamp();

  for (int i = 0; i < 10; ++i) {

    logger.ReadFromFile();
    int num_bytes = logger.ReadNextBytes();
    LOG(INFO) << "Number of bytes is " << num_bytes;
  }
}
