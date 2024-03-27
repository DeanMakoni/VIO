// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbMatrix.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_pbMatrix_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_pbMatrix_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_pbMatrix_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_pbMatrix_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbMatrix_2eproto;
namespace datatype {
namespace transform {
class pbMatrix;
class pbMatrixDefaultTypeInternal;
extern pbMatrixDefaultTypeInternal _pbMatrix_default_instance_;
}  // namespace transform
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> ::datatype::transform::pbMatrix* Arena::CreateMaybeMessage<::datatype::transform::pbMatrix>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace datatype {
namespace transform {

// ===================================================================

class pbMatrix PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:datatype.transform.pbMatrix) */ {
 public:
  inline pbMatrix() : pbMatrix(nullptr) {}
  virtual ~pbMatrix();

  pbMatrix(const pbMatrix& from);
  pbMatrix(pbMatrix&& from) noexcept
    : pbMatrix() {
    *this = ::std::move(from);
  }

  inline pbMatrix& operator=(const pbMatrix& from) {
    CopyFrom(from);
    return *this;
  }
  inline pbMatrix& operator=(pbMatrix&& from) noexcept {
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
  static const pbMatrix& default_instance();

  static inline const pbMatrix* internal_default_instance() {
    return reinterpret_cast<const pbMatrix*>(
               &_pbMatrix_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(pbMatrix& a, pbMatrix& b) {
    a.Swap(&b);
  }
  inline void Swap(pbMatrix* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(pbMatrix* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline pbMatrix* New() const final {
    return CreateMaybeMessage<pbMatrix>(nullptr);
  }

  pbMatrix* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<pbMatrix>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const pbMatrix& from);
  void MergeFrom(const pbMatrix& from);
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
  void InternalSwap(pbMatrix* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "datatype.transform.pbMatrix";
  }
  protected:
  explicit pbMatrix(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_pbMatrix_2eproto);
    return ::descriptor_table_pbMatrix_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kDataFieldNumber = 3,
    kRowsFieldNumber = 1,
    kColsFieldNumber = 2,
  };
  // repeated float data = 3;
  int data_size() const;
  private:
  int _internal_data_size() const;
  public:
  void clear_data();
  private:
  float _internal_data(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_data() const;
  void _internal_add_data(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_data();
  public:
  float data(int index) const;
  void set_data(int index, float value);
  void add_data(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      data() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_data();

  // uint32 rows = 1;
  void clear_rows();
  ::PROTOBUF_NAMESPACE_ID::uint32 rows() const;
  void set_rows(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_rows() const;
  void _internal_set_rows(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // uint32 cols = 2;
  void clear_cols();
  ::PROTOBUF_NAMESPACE_ID::uint32 cols() const;
  void set_cols(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_cols() const;
  void _internal_set_cols(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // @@protoc_insertion_point(class_scope:datatype.transform.pbMatrix)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > data_;
  mutable std::atomic<int> _data_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::uint32 rows_;
  ::PROTOBUF_NAMESPACE_ID::uint32 cols_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_pbMatrix_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// pbMatrix

// uint32 rows = 1;
inline void pbMatrix::clear_rows() {
  rows_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 pbMatrix::_internal_rows() const {
  return rows_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 pbMatrix::rows() const {
  // @@protoc_insertion_point(field_get:datatype.transform.pbMatrix.rows)
  return _internal_rows();
}
inline void pbMatrix::_internal_set_rows(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  rows_ = value;
}
inline void pbMatrix::set_rows(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_rows(value);
  // @@protoc_insertion_point(field_set:datatype.transform.pbMatrix.rows)
}

// uint32 cols = 2;
inline void pbMatrix::clear_cols() {
  cols_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 pbMatrix::_internal_cols() const {
  return cols_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 pbMatrix::cols() const {
  // @@protoc_insertion_point(field_get:datatype.transform.pbMatrix.cols)
  return _internal_cols();
}
inline void pbMatrix::_internal_set_cols(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  cols_ = value;
}
inline void pbMatrix::set_cols(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_cols(value);
  // @@protoc_insertion_point(field_set:datatype.transform.pbMatrix.cols)
}

// repeated float data = 3;
inline int pbMatrix::_internal_data_size() const {
  return data_.size();
}
inline int pbMatrix::data_size() const {
  return _internal_data_size();
}
inline void pbMatrix::clear_data() {
  data_.Clear();
}
inline float pbMatrix::_internal_data(int index) const {
  return data_.Get(index);
}
inline float pbMatrix::data(int index) const {
  // @@protoc_insertion_point(field_get:datatype.transform.pbMatrix.data)
  return _internal_data(index);
}
inline void pbMatrix::set_data(int index, float value) {
  data_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.transform.pbMatrix.data)
}
inline void pbMatrix::_internal_add_data(float value) {
  data_.Add(value);
}
inline void pbMatrix::add_data(float value) {
  _internal_add_data(value);
  // @@protoc_insertion_point(field_add:datatype.transform.pbMatrix.data)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbMatrix::_internal_data() const {
  return data_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbMatrix::data() const {
  // @@protoc_insertion_point(field_list:datatype.transform.pbMatrix.data)
  return _internal_data();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbMatrix::_internal_mutable_data() {
  return &data_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbMatrix::mutable_data() {
  // @@protoc_insertion_point(field_mutable_list:datatype.transform.pbMatrix.data)
  return _internal_mutable_data();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace transform
}  // namespace datatype

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_pbMatrix_2eproto