// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbIndex.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_pbIndex_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_pbIndex_2eproto

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
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_pbIndex_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_pbIndex_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbIndex_2eproto;
namespace datatype {
namespace logging {
class pbIndex;
class pbIndexDefaultTypeInternal;
extern pbIndexDefaultTypeInternal _pbIndex_default_instance_;
}  // namespace logging
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> ::datatype::logging::pbIndex* Arena::CreateMaybeMessage<::datatype::logging::pbIndex>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace datatype {
namespace logging {

// ===================================================================

class pbIndex PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:datatype.logging.pbIndex) */ {
 public:
  inline pbIndex() : pbIndex(nullptr) {}
  virtual ~pbIndex();

  pbIndex(const pbIndex& from);
  pbIndex(pbIndex&& from) noexcept
    : pbIndex() {
    *this = ::std::move(from);
  }

  inline pbIndex& operator=(const pbIndex& from) {
    CopyFrom(from);
    return *this;
  }
  inline pbIndex& operator=(pbIndex&& from) noexcept {
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
  static const pbIndex& default_instance();

  static inline const pbIndex* internal_default_instance() {
    return reinterpret_cast<const pbIndex*>(
               &_pbIndex_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(pbIndex& a, pbIndex& b) {
    a.Swap(&b);
  }
  inline void Swap(pbIndex* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(pbIndex* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline pbIndex* New() const final {
    return CreateMaybeMessage<pbIndex>(nullptr);
  }

  pbIndex* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<pbIndex>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const pbIndex& from);
  void MergeFrom(const pbIndex& from);
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
  void InternalSwap(pbIndex* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "datatype.logging.pbIndex";
  }
  protected:
  explicit pbIndex(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_pbIndex_2eproto);
    return ::descriptor_table_pbIndex_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kTimestampFieldNumber = 1,
    kOffsetFieldNumber = 2,
  };
  // int64 timestamp = 1;
  void clear_timestamp();
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp() const;
  void set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_timestamp() const;
  void _internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 offset = 2;
  void clear_offset();
  ::PROTOBUF_NAMESPACE_ID::int64 offset() const;
  void set_offset(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_offset() const;
  void _internal_set_offset(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // @@protoc_insertion_point(class_scope:datatype.logging.pbIndex)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp_;
  ::PROTOBUF_NAMESPACE_ID::int64 offset_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_pbIndex_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// pbIndex

// int64 timestamp = 1;
inline void pbIndex::clear_timestamp() {
  timestamp_ = PROTOBUF_LONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbIndex::_internal_timestamp() const {
  return timestamp_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbIndex::timestamp() const {
  // @@protoc_insertion_point(field_get:datatype.logging.pbIndex.timestamp)
  return _internal_timestamp();
}
inline void pbIndex::_internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  timestamp_ = value;
}
inline void pbIndex::set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_timestamp(value);
  // @@protoc_insertion_point(field_set:datatype.logging.pbIndex.timestamp)
}

// int64 offset = 2;
inline void pbIndex::clear_offset() {
  offset_ = PROTOBUF_LONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbIndex::_internal_offset() const {
  return offset_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbIndex::offset() const {
  // @@protoc_insertion_point(field_get:datatype.logging.pbIndex.offset)
  return _internal_offset();
}
inline void pbIndex::_internal_set_offset(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  offset_ = value;
}
inline void pbIndex::set_offset(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_offset(value);
  // @@protoc_insertion_point(field_set:datatype.logging.pbIndex.offset)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace logging
}  // namespace datatype

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_pbIndex_2eproto
