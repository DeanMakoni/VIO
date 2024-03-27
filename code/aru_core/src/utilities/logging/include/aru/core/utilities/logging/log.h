#ifndef ARU_UTILITIES_LOGGING_H_
#define ARU_UTILITIES_LOGGING_H_

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace logging {

template <typename ProtocolBufferClass> class ProtocolLogger {
public:
  ProtocolLogger(const std::string &filename, bool overwrite);

  virtual ~ProtocolLogger() = default;

  void WriteToFile(ProtocolBufferClass protobuf);

  ProtocolBufferClass ReadFromFile();

  ProtocolBufferClass ReadSkipped(int64_t num_bytes);

  int ReadNextBytes();

  bool EndOfFile() { return end_of_file_; }

private:
  bool end_of_file_;
  std::string filename_;
  std::ofstream out_file_;
  std::ifstream in_file_;
  google::protobuf::io::IstreamInputStream in_prot_stream_;
};
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
ProtocolLogger<ProtocolBufferClass>::ProtocolLogger(const std::string &filename,
                                                    bool overwrite)
    : filename_(filename),
      in_file_(std::ifstream(filename_, std::ifstream::binary)),
      in_prot_stream_(&in_file_) {
  end_of_file_ = false;
  if (overwrite)
    out_file_ = std::ofstream(filename_, std::ofstream::binary);
}
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
void ProtocolLogger<ProtocolBufferClass>::WriteToFile(
    ProtocolBufferClass protobuf) {
  // Write delimited to file
  bool written =
      google::protobuf::util::SerializeDelimitedToOstream(protobuf, &out_file_);
}

//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
int ProtocolLogger<ProtocolBufferClass>::ReadNextBytes() {
  return in_prot_stream_.ByteCount();
}
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
ProtocolBufferClass ProtocolLogger<ProtocolBufferClass>::ReadSkipped(int64_t num_bytes) {
  // Read delimited from file

  std::ifstream in_file(std::ifstream(filename_, std::ifstream::binary));
  google::protobuf::io::IstreamInputStream in_prot_stream(&in_file);

  in_prot_stream.Skip(num_bytes);

  ProtocolBufferClass protobuf;
  google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &protobuf, &in_prot_stream, &end_of_file_);
  // LOG(INFO) << "End of file is " << end_of_file_;
  return protobuf;
}
//------------------------------------------------------------------------------
template <typename ProtocolBufferClass>
ProtocolBufferClass ProtocolLogger<ProtocolBufferClass>::ReadFromFile() {
  // Read delimited from file
  ProtocolBufferClass protobuf;
  google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &protobuf, &in_prot_stream_, &end_of_file_);
  // LOG(INFO) << "End of file is " << end_of_file_;
  return protobuf;
}
} // namespace logging
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_LOGGING_H_
