#ifndef ARU_UTILITIES_LOGGING_INDEX_LOG_H_
#define ARU_UTILITIES_LOGGING_INDEX_LOG_H_

#include "pbIndex.pb.h"
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <aru/core/utilities/logging/log.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace logging {

template <typename ProtocolBufferClass> class IndexLogger {
public:
  IndexLogger(std::string filename, bool overwrite);

  IndexLogger(std::string filename, std::string filename_index);

  void CreateIndexFile();

  void WriteToFile(ProtocolBufferClass protobuf);

  ProtocolBufferClass ReadFromFile();

  void OpenIndex();

  virtual ~IndexLogger() = default;

  ProtocolBufferClass ReadIndex(int index);

  bool EndOfFile() { return pb_logger_->EndOfFile(); }

private:
  std::string filename_;
  std::string filename_index_;

  std::vector<int64_t> index_offsets_;

  std::shared_ptr<ProtocolLogger<ProtocolBufferClass>> pb_logger_;
  std::shared_ptr<ProtocolLogger<datatype::logging::pbIndex>> pb_index_logger_;
};
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
IndexLogger<ProtocolBufferClass>::IndexLogger(std::string filename,
                                              bool overwrite)
    : filename_(std::move(filename)) {
  filename_index_ = filename_ + ".index";
  if (!overwrite) {
    CreateIndexFile();

    OpenIndex();
  } else {
    pb_logger_ = std::make_shared<ProtocolLogger<ProtocolBufferClass>>(
        filename_, overwrite);
  }
}

//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
IndexLogger<ProtocolBufferClass>::IndexLogger(std::string filename,
                                              std::string filename_index)
    : filename_(std::move(filename)),
      filename_index_(std::move(filename_index)) {
  OpenIndex();
}

//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
void IndexLogger<ProtocolBufferClass>::OpenIndex() {
  pb_index_logger_ =
      std::make_shared<ProtocolLogger<datatype::logging::pbIndex>>(
          filename_index_, false);
  index_offsets_.clear();
  while (!pb_index_logger_->EndOfFile()) {
    datatype::logging::pbIndex index_proto = pb_index_logger_->ReadFromFile();
    index_offsets_.push_back(index_proto.offset());
  }
}
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
void IndexLogger<ProtocolBufferClass>::CreateIndexFile() {

  pb_index_logger_ =
      std::make_shared<ProtocolLogger<datatype::logging::pbIndex>>(
          filename_index_, true);
  pb_logger_ =
      std::make_shared<ProtocolLogger<ProtocolBufferClass>>(filename_, false);

  while (!pb_logger_->EndOfFile()) {
    ProtocolBufferClass protobuf = pb_logger_->ReadFromFile();
    int num_bytes = pb_logger_->ReadNextBytes();
    datatype::logging::pbIndex index_proto;
    index_proto.set_timestamp(0);
    index_proto.set_offset(num_bytes);
    pb_index_logger_->WriteToFile(index_proto);
  }
}

//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
ProtocolBufferClass IndexLogger<ProtocolBufferClass>::ReadIndex(int index) {
  // TODO: make this more efficient, very very very rough
  pb_logger_ =
      std::make_shared<ProtocolLogger<ProtocolBufferClass>>(filename_, false);

  int64_t offset = index_offsets_[index];
  return pb_logger_->ReadSkipped(offset);
}

//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
void IndexLogger<ProtocolBufferClass>::WriteToFile(
    ProtocolBufferClass protobuf) {
  pb_logger_->WriteToFile(protobuf);
}
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
ProtocolBufferClass IndexLogger<ProtocolBufferClass>::ReadFromFile() {

  ProtocolBufferClass protobuf = pb_logger_->ReadFromFile();
  return protobuf;
}

} // namespace logging
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_LOGGING_INDEX_LOG_H_
