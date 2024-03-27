#include "aru/core/utilities/navigation/experienceprotocolbufferadaptor.h"
#include "aru/core/utilities/image/imageprotocolbufferadaptor.h"
#include "aru/core/utilities/transforms/transformprotocolbufferadaptor.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/features2d.hpp>
#include <utility>
#include <opencv2/core/eigen.hpp>

namespace aru {
    namespace core {
        namespace utilities {
            namespace navigation {
                using namespace aru::core::utilities::image;
                using namespace aru::core::utilities::transform;

//------------------------------------------------------------------------------
                ExperienceProtocolBufferAdaptor::ExperienceProtocolBufferAdaptor() {}

//------------------------------------------------------------------------------
                datatype::navigation::pbExperience
                ExperienceProtocolBufferAdaptor::ReadToProtocolBuffer(Experience experience) {

                    datatype::navigation::pbExperience pb_experience;

                    pb_experience.set_timestamp(experience.GetTimeStamp());
                    // experience image
                    Image experience_image(0, experience.GetImage());
                    datatype::image::pbImage *pb_exp_image = pb_experience.mutable_image_left();
                    datatype::image::pbImage pb_exp_image_value =
                            ImageProtocolBufferAdaptor::ReadToProtocolBuffer(experience_image);
                    *pb_exp_image = pb_exp_image_value;

                    // experience image
                    Image desc_image(0, experience.GetDescriptors());
                    datatype::image::pbImage *pb_desc_image = pb_experience.mutable_descriptors();
                    datatype::image::pbImage pb_desc_image_value =
                            ImageProtocolBufferAdaptor::ReadToProtocolBuffer(desc_image);
                    *pb_desc_image = pb_desc_image_value;

                    // experience image
                    Image bow_desc_image(0, experience.GetBowDescriptors());
                    datatype::transform::pbMatrix *pb_bow_desc_image =
                            pb_experience.mutable_bow_desc();
                    Eigen::MatrixXf bow_eigen;
                    cv::cv2eigen(experience.GetBowDescriptors(),bow_eigen);
                    datatype::transform::pbMatrix pb_bow_desc__value =
                            TransformProtocolBufferAdaptor::ReadMatrixToProtocolBuffer(
                                    bow_eigen);
                    *pb_bow_desc_image = pb_bow_desc__value;

                    // landmarks
                    datatype::transform::pbMatrix *pb_landmarks =
                            pb_experience.mutable_landmarks();
                    datatype::transform::pbMatrix pb_landmarks_value =
                            TransformProtocolBufferAdaptor::ReadMatrixToProtocolBuffer(
                                    experience.GetLandmarks());
                    *pb_landmarks = pb_landmarks_value;

                    // keypoints
                    datatype::transform::pbMatrix *pb_keypoints =
                            pb_experience.mutable_keypoints();
                    datatype::transform::pbMatrix pb_keypoints_value =
                            TransformProtocolBufferAdaptor::ReadMatrixToProtocolBuffer(
                                    experience.GetKeypoints());
                    *pb_keypoints = pb_keypoints_value;

                    return pb_experience;
                }

//------------------------------------------------------------------------------
                Experience ExperienceProtocolBufferAdaptor::ReadFromProtocolBuffer(
                        const datatype::navigation::pbExperience &pb_experience) {
                    Image experience_image = ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                            pb_experience.image_left());
                    Image desc_image = ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                            pb_experience.descriptors());
                    Eigen::MatrixXf bow_desc_image = TransformProtocolBufferAdaptor::ReadMatrixFromProtocolBuffer(
                            pb_experience.bow_desc());
                    cv::Mat bow_desc;
                    cv::eigen2cv(bow_desc_image,bow_desc);

                    Eigen::MatrixXf landmarks =
                            TransformProtocolBufferAdaptor::ReadMatrixFromProtocolBuffer(
                                    pb_experience.landmarks());
                    Eigen::MatrixXf keypoints =
                            TransformProtocolBufferAdaptor::ReadMatrixFromProtocolBuffer(
                                    pb_experience.keypoints());

                    return Experience(pb_experience.timestamp(), experience_image.GetImage(),
                                      keypoints, landmarks, desc_image.GetImage(),
                                      bow_desc);
                }
//------------------------------------------------------------------------------

            } // namespace navigation
        } // namespace utilities
    } // namespace core
} // namespace aru
