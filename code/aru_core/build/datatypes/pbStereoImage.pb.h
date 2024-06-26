// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbStereoImage.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_pbStereoImage_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_pbStereoImage_2eproto

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
#include <google/protobuf/unknown_field_set.h>
#include "pbImage.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_pbStereoImage_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_pbStereoImage_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbStereoImage_2eproto;
namespace datatype {
namespace image {
class pbStereoImage;
class pbStereoImageDefaultTypeInternal;
extern pbStereoImageDefaultTypeInternal _pbStereoImage_default_instance_;
}  // namespace image
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> ::datatype::image::pbStereoImage* Arena::CreateMaybeMessage<::datatype::image::pbStereoImage>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace datatype {
namespace image {

// ===================================================================

class pbStereoImage PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:datatype.image.pbStereoImage) */ {
 public:
  inline pbStereoImage() : pbStereoImage(nullptr) {}
  virtual ~pbStereoImage();

  pbStereoImage(const pbStereoImage& from);
  pbStereoImage(pbStereoImage&& from) noexcept
    : pbStereoImage() {
    *this = ::std::move(from);
  }

  inline pbStereoImage& operator=(const pbStereoImage& from) {
    CopyFrom(from);
    return *this;
  }
  inline pbStereoImage& operator=(pbStereoImage&& from) noexcept {
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
  static const pbStereoImage& default_instance();

  static inline const pbStereoImage* internal_default_instance() {
    return reinterpret_cast<const pbStereoImage*>(
               &_pbStereoImage_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(pbStereoImage& a, pbStereoImage& b) {
    a.Swap(&b);
  }
  inline void Swap(pbStereoImage* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(pbStereoImage* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline pbStereoImage* New() const final {
    return CreateMaybeMessage<pbStereoImage>(nullptr);
  }

  pbStereoImage* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<pbStereoImage>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const pbStereoImage& from);
  void MergeFrom(const pbStereoImage& from);
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
  void InternalSwap(pbStereoImage* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "datatype.image.pbStereoImage";
  }
  protected:
  explicit pbStereoImage(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_pbStereoImage_2eproto);
    return ::descriptor_table_pbStereoImage_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kImageLeftFieldNumber = 1,
    kImageRightFieldNumber = 2,
    kTimestampFieldNumber = 3,
  };
  // .datatype.image.pbImage image_left = 1;
  bool has_image_left() const;
  private:
  bool _internal_has_image_left() const;
  public:
  void clear_image_left();
  const ::datatype::image::pbImage& image_left() const;
  ::datatype::image::pbImage* release_image_left();
  ::datatype::image::pbImage* mutable_image_left();
  void set_allocated_image_left(::datatype::image::pbImage* image_left);
  private:
  const ::datatype::image::pbImage& _internal_image_left() const;
  ::datatype::image::pbImage* _internal_mutable_image_left();
  public:
  void unsafe_arena_set_allocated_image_left(
      ::datatype::image::pbImage* image_left);
  ::datatype::image::pbImage* unsafe_arena_release_image_left();

  // .datatype.image.pbImage image_right = 2;
  bool has_image_right() const;
  private:
  bool _internal_has_image_right() const;
  public:
  void clear_image_right();
  const ::datatype::image::pbImage& image_right() const;
  ::datatype::image::pbImage* release_image_right();
  ::datatype::image::pbImage* mutable_image_right();
  void set_allocated_image_right(::datatype::image::pbImage* image_right);
  private:
  const ::datatype::image::pbImage& _internal_image_right() const;
  ::datatype::image::pbImage* _internal_mutable_image_right();
  public:
  void unsafe_arena_set_allocated_image_right(
      ::datatype::image::pbImage* image_right);
  ::datatype::image::pbImage* unsafe_arena_release_image_right();

  // int64 timestamp = 3;
  void clear_timestamp();
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp() const;
  void set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_timestamp() const;
  void _internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // @@protoc_insertion_point(class_scope:datatype.image.pbStereoImage)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::datatype::image::pbImage* image_left_;
  ::datatype::image::pbImage* image_right_;
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_pbStereoImage_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// pbStereoImage

// .datatype.image.pbImage image_left = 1;
inline bool pbStereoImage::_internal_has_image_left() const {
  return this != internal_default_instance() && image_left_ != nullptr;
}
inline bool pbStereoImage::has_image_left() const {
  return _internal_has_image_left();
}
inline const ::datatype::image::pbImage& pbStereoImage::_internal_image_left() const {
  const ::datatype::image::pbImage* p = image_left_;
  return p != nullptr ? *p : reinterpret_cast<const ::datatype::image::pbImage&>(
      ::datatype::image::_pbImage_default_instance_);
}
inline const ::datatype::image::pbImage& pbStereoImage::image_left() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbStereoImage.image_left)
  return _internal_image_left();
}
inline void pbStereoImage::unsafe_arena_set_allocated_image_left(
    ::datatype::image::pbImage* image_left) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_left_);
  }
  image_left_ = image_left;
  if (image_left) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:datatype.image.pbStereoImage.image_left)
}
inline ::datatype::image::pbImage* pbStereoImage::release_image_left() {
  
  ::datatype::image::pbImage* temp = image_left_;
  image_left_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::datatype::image::pbImage* pbStereoImage::unsafe_arena_release_image_left() {
  // @@protoc_insertion_point(field_release:datatype.image.pbStereoImage.image_left)
  
  ::datatype::image::pbImage* temp = image_left_;
  image_left_ = nullptr;
  return temp;
}
inline ::datatype::image::pbImage* pbStereoImage::_internal_mutable_image_left() {
  
  if (image_left_ == nullptr) {
    auto* p = CreateMaybeMessage<::datatype::image::pbImage>(GetArena());
    image_left_ = p;
  }
  return image_left_;
}
inline ::datatype::image::pbImage* pbStereoImage::mutable_image_left() {
  // @@protoc_insertion_point(field_mutable:datatype.image.pbStereoImage.image_left)
  return _internal_mutable_image_left();
}
inline void pbStereoImage::set_allocated_image_left(::datatype::image::pbImage* image_left) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_left_);
  }
  if (image_left) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_left)->GetArena();
    if (message_arena != submessage_arena) {
      image_left = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, image_left, submessage_arena);
    }
    
  } else {
    
  }
  image_left_ = image_left;
  // @@protoc_insertion_point(field_set_allocated:datatype.image.pbStereoImage.image_left)
}

// .datatype.image.pbImage image_right = 2;
inline bool pbStereoImage::_internal_has_image_right() const {
  return this != internal_default_instance() && image_right_ != nullptr;
}
inline bool pbStereoImage::has_image_right() const {
  return _internal_has_image_right();
}
inline const ::datatype::image::pbImage& pbStereoImage::_internal_image_right() const {
  const ::datatype::image::pbImage* p = image_right_;
  return p != nullptr ? *p : reinterpret_cast<const ::datatype::image::pbImage&>(
      ::datatype::image::_pbImage_default_instance_);
}
inline const ::datatype::image::pbImage& pbStereoImage::image_right() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbStereoImage.image_right)
  return _internal_image_right();
}
inline void pbStereoImage::unsafe_arena_set_allocated_image_right(
    ::datatype::image::pbImage* image_right) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_right_);
  }
  image_right_ = image_right;
  if (image_right) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:datatype.image.pbStereoImage.image_right)
}
inline ::datatype::image::pbImage* pbStereoImage::release_image_right() {
  
  ::datatype::image::pbImage* temp = image_right_;
  image_right_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::datatype::image::pbImage* pbStereoImage::unsafe_arena_release_image_right() {
  // @@protoc_insertion_point(field_release:datatype.image.pbStereoImage.image_right)
  
  ::datatype::image::pbImage* temp = image_right_;
  image_right_ = nullptr;
  return temp;
}
inline ::datatype::image::pbImage* pbStereoImage::_internal_mutable_image_right() {
  
  if (image_right_ == nullptr) {
    auto* p = CreateMaybeMessage<::datatype::image::pbImage>(GetArena());
    image_right_ = p;
  }
  return image_right_;
}
inline ::datatype::image::pbImage* pbStereoImage::mutable_image_right() {
  // @@protoc_insertion_point(field_mutable:datatype.image.pbStereoImage.image_right)
  return _internal_mutable_image_right();
}
inline void pbStereoImage::set_allocated_image_right(::datatype::image::pbImage* image_right) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_right_);
  }
  if (image_right) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(image_right)->GetArena();
    if (message_arena != submessage_arena) {
      image_right = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, image_right, submessage_arena);
    }
    
  } else {
    
  }
  image_right_ = image_right;
  // @@protoc_insertion_point(field_set_allocated:datatype.image.pbStereoImage.image_right)
}

// int64 timestamp = 3;
inline void pbStereoImage::clear_timestamp() {
  timestamp_ = PROTOBUF_LONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbStereoImage::_internal_timestamp() const {
  return timestamp_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbStereoImage::timestamp() const {
  // @@protoc_insertion_point(field_get:datatype.image.pbStereoImage.timestamp)
  return _internal_timestamp();
}
inline void pbStereoImage::_internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  timestamp_ = value;
}
inline void pbStereoImage::set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_timestamp(value);
  // @@protoc_insertion_point(field_set:datatype.image.pbStereoImage.timestamp)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace image
}  // namespace datatype

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_pbStereoImage_2eproto
