// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbImage.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_pbImage_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_pbImage_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_pbImage_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_pbImage_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbImage_2eproto;
namespace datatype {
namespace image {
class pbImage;
class pbImageDefaultTypeInternal;
extern pbImageDefaultTypeInternal _pbImage_default_instance_;
}  // namespace image
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> ::datatype::image::pbImage* Arena::CreateMaybeMessage<::datatype::image::pbImage>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace datatype {
namespace image {

enum pbImage_ImageType : int {
  pbImage_ImageType_RGB_UINT8 = 0,
  pbImage_ImageType_GREY_UINT8 = 1,
  pbImage_ImageType_RGB_FLOAT = 2,
  pbImage_ImageType_GREY_FLOAT = 3,
  pbImage_ImageType_Depth_FLOAT = 4,
  pbImage_ImageType_pbImage_ImageType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::min(),
  pbImage_ImageType_pbImage_ImageType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::max()
};
bool pbImage_ImageType_IsValid(int value);
constexpr pbImage_ImageType pbImage_ImageType_ImageType_MIN = pbImage_ImageType_RGB_UINT8;
constexpr pbImage_ImageType pbImage_ImageType_ImageType_MAX = pbImage_ImageType_Depth_FLOAT;
constexpr int pbImage_ImageType_ImageType_ARRAYSIZE = pbImage_ImageType_ImageType_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* pbImage_ImageType_descriptor();
template<typename T>
inline const std::string& pbImage_ImageType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, pbImage_ImageType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function pbImage_ImageType_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    pbImage_ImageType_descriptor(), enum_t_value);
}
inline bool pbImage_ImageType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, pbImage_ImageType* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<pbImage_ImageType>(
    pbImage_ImageType_descriptor(), name, value);
}
// ===================================================================

class pbImage PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:datatype.image.pbImage) */ {
 public:
  inline pbImage() : pbImage(nullptr) {}
  virtual ~pbImage();

  pbImage(const pbImage& from);
  pbImage(pbImage&& from) noexcept
    : pbImage() {
    *this = ::std::move(from);
  }

  inline pbImage& operator=(const pbImage& from) {
    CopyFrom(from);
    return *this;
  }
  inline pbImage& operator=(pbImage&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const pbImage& default_instance();

  static inline const pbImage* internal_default_instance() {
    return reinterpret_cast<const pbImage*>(
               &_pbImage_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(pbImage& a, pbImage& b) {
    a.Swap(&b);
  }
  inline void Swap(pbImage* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(pbImage* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline pbImage* New() const final {
    return CreateMaybeMessage<pbImage>(nullptr);
  }

  pbImage* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<pbImage>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const pbImage& from);
  void MergeFrom(const pbImage& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(pbImage* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "datatype.image.pbImage";
  }
  protected:
  explicit pbImage(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_pbImage_2eproto);
    return ::descriptor_table_pbImage_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  typedef pbImage_ImageType ImageType;
  static constexpr ImageType RGB_UINT8 =
    pbImage_ImageType_RGB_UINT8;
  static constexpr ImageType GREY_UINT8 =
    pbImage_ImageType_GREY_UINT8;
  static constexpr ImageType RGB_FLOAT =
    pbImage_ImageType_RGB_FLOAT;
  static constexpr ImageType GREY_FLOAT =
    pbImage_ImageType_GREY_FLOAT;
  static constexpr ImageType Depth_FLOAT =
    pbImage_ImageType_Depth_FLOAT;
  static inline bool ImageType_IsValid(int value) {
    return pbImage_ImageType_IsValid(value);
  }
  static constexpr ImageType ImageType_MIN =
    pbImage_ImageType_ImageType_MIN;
  static constexpr ImageType ImageType_MAX =
    pbImage_ImageType_ImageType_MAX;
  static constexpr int ImageType_ARRAYSIZE =
    pbImage_ImageType_ImageType_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  ImageType_descriptor() {
    return pbImage_ImageType_descriptor();
  }
  template<typename T>
  static inline const std::string& ImageType_Name(T enum_t_value) {
    static_assert(::std::is_same<T, ImageType>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function ImageType_Name.");
    return pbImage_ImageType_Name(enum_t_value);
  }
  static inline bool ImageType_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      ImageType* value) {
    return pbImage_ImageType_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kImageDataFieldNumber = 4,
    kWidthFieldNumber = 1,
    kHeightFieldNumber = 2,
    kTimestampFieldNumber = 3,
  };
  // bytes image_data = 4;
  void clear_image_data();
  const std::string& image_data() const;
  void set_image_data(const std::string& value);
  void set_image_data(std::string&& value);
  void set_image_data(const char* value);
  void set_image_data(const void* value, size_t size);
  std::string* mutable_image_data();
  std::string* release_image_data();
  void set_allocated_image_data(std::string* image_data);
  private:
  const std::string& _internal_image_data() const;
  void _internal_set_image_data(const std::string& value);
  std::string* _internal_mutable_image_data();
  public:

  // int32 width = 1;
  void clear_width();
  ::PROTOBUF_NAMESPACE_ID::int32 width() const;
  void set_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_width() const;
  void _internal_set_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 height = 2;
  void clear_height();
  ::PROTOBUF_NAMESPACE_ID::int32 height() const;
  void set_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_height() const;
  void _internal_set_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int64 timestamp = 3;
  void clear_timestamp();
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp() const;
  void set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_timestamp() const;
  void _internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // @@protoc_insertion_point(class_scope:datatype.image.pbImage)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr image_data_;
  ::PROTOBUF_NAMESPACE_ID::int32 width_;
  ::PROTOBUF_NAMESPACE_ID::int32 height_;
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_pbImage_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// pbImage

// int32 width = 1;
inline void pbImage::clear_width() {
  width_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 pbImage::_internal_width() const {
  return width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 pbImage::width() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbImage.width)
  return _internal_width();
}
inline void pbImage::_internal_set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  width_ = value;
}
inline void pbImage::set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_width(value);
  // @@protoc_insertion_point(field_set:datatype.image.pbImage.width)
}

// int32 height = 2;
inline void pbImage::clear_height() {
  height_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 pbImage::_internal_height() const {
  return height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 pbImage::height() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbImage.height)
  return _internal_height();
}
inline void pbImage::_internal_set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  height_ = value;
}
inline void pbImage::set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_height(value);
  // @@protoc_insertion_point(field_set:datatype.image.pbImage.height)
}

// int64 timestamp = 3;
inline void pbImage::clear_timestamp() {
  timestamp_ = PROTOBUF_LONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbImage::_internal_timestamp() const {
  return timestamp_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbImage::timestamp() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbImage.timestamp)
  return _internal_timestamp();
}
inline void pbImage::_internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  timestamp_ = value;
}
inline void pbImage::set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_timestamp(value);
  // @@protoc_insertion_point(field_set:datatype.image.pbImage.timestamp)
}

// bytes image_data = 4;
inline void pbImage::clear_image_data() {
  image_data_.ClearToEmpty();
}
inline const std::string& pbImage::image_data() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbImage.image_data)
  return _internal_image_data();
}
inline void pbImage::set_image_data(const std::string& value) {
  _internal_set_image_data(value);
  // @@protoc_insertion_point(field_set:datatype.image.pbImage.image_data)
}
inline std::string* pbImage::mutable_image_data() {
  // @@protoc_insertion_point(field_mutable:datatype.image.pbImage.image_data)
  return _internal_mutable_image_data();
}
inline const std::string& pbImage::_internal_image_data() const {
  return image_data_.Get();
}
inline void pbImage::_internal_set_image_data(const std::string& value) {
  
  image_data_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void pbImage::set_image_data(std::string&& value) {
  
  image_data_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:datatype.image.pbImage.image_data)
}
inline void pbImage::set_image_data(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  image_data_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:datatype.image.pbImage.image_data)
}
inline void pbImage::set_image_data(const void* value,
    size_t size) {
  
  image_data_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:datatype.image.pbImage.image_data)
}
inline std::string* pbImage::_internal_mutable_image_data() {
  
  return image_data_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* pbImage::release_image_data() {
  // @@protoc_insertion_point(field_release:datatype.image.pbImage.image_data)
  return image_data_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void pbImage::set_allocated_image_data(std::string* image_data) {
  if (image_data != nullptr) {
    
  } else {
    
  }
  image_data_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), image_data,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:datatype.image.pbImage.image_data)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace image
}  // namespace datatype

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::datatype::image::pbImage_ImageType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::datatype::image::pbImage_ImageType>() {
  return ::datatype::image::pbImage_ImageType_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_pbImage_2eproto
