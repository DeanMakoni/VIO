// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbLaser.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_pbLaser_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_pbLaser_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_pbLaser_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_pbLaser_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbLaser_2eproto;
namespace datatype {
namespace laser {
class pbLaser;
class pbLaserDefaultTypeInternal;
extern pbLaserDefaultTypeInternal _pbLaser_default_instance_;
}  // namespace laser
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> ::datatype::laser::pbLaser* Arena::CreateMaybeMessage<::datatype::laser::pbLaser>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace datatype {
namespace laser {

// ===================================================================

class pbLaser PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:datatype.laser.pbLaser) */ {
 public:
  inline pbLaser() : pbLaser(nullptr) {}
  virtual ~pbLaser();

  pbLaser(const pbLaser& from);
  pbLaser(pbLaser&& from) noexcept
    : pbLaser() {
    *this = ::std::move(from);
  }

  inline pbLaser& operator=(const pbLaser& from) {
    CopyFrom(from);
    return *this;
  }
  inline pbLaser& operator=(pbLaser&& from) noexcept {
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
  static const pbLaser& default_instance();

  static inline const pbLaser* internal_default_instance() {
    return reinterpret_cast<const pbLaser*>(
               &_pbLaser_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(pbLaser& a, pbLaser& b) {
    a.Swap(&b);
  }
  inline void Swap(pbLaser* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(pbLaser* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline pbLaser* New() const final {
    return CreateMaybeMessage<pbLaser>(nullptr);
  }

  pbLaser* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<pbLaser>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const pbLaser& from);
  void MergeFrom(const pbLaser& from);
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
  void InternalSwap(pbLaser* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "datatype.laser.pbLaser";
  }
  protected:
  explicit pbLaser(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_pbLaser_2eproto);
    return ::descriptor_table_pbLaser_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kXFieldNumber = 2,
    kYFieldNumber = 3,
    kZFieldNumber = 4,
    kIntensityFieldNumber = 5,
    kReflectanceFieldNumber = 6,
    kTimestampFieldNumber = 1,
  };
  // repeated float x = 2;
  int x_size() const;
  private:
  int _internal_x_size() const;
  public:
  void clear_x();
  private:
  float _internal_x(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_x() const;
  void _internal_add_x(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_x();
  public:
  float x(int index) const;
  void set_x(int index, float value);
  void add_x(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      x() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_x();

  // repeated float y = 3;
  int y_size() const;
  private:
  int _internal_y_size() const;
  public:
  void clear_y();
  private:
  float _internal_y(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_y() const;
  void _internal_add_y(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_y();
  public:
  float y(int index) const;
  void set_y(int index, float value);
  void add_y(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      y() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_y();

  // repeated float z = 4;
  int z_size() const;
  private:
  int _internal_z_size() const;
  public:
  void clear_z();
  private:
  float _internal_z(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_z() const;
  void _internal_add_z(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_z();
  public:
  float z(int index) const;
  void set_z(int index, float value);
  void add_z(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      z() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_z();

  // repeated float intensity = 5;
  int intensity_size() const;
  private:
  int _internal_intensity_size() const;
  public:
  void clear_intensity();
  private:
  float _internal_intensity(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_intensity() const;
  void _internal_add_intensity(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_intensity();
  public:
  float intensity(int index) const;
  void set_intensity(int index, float value);
  void add_intensity(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      intensity() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_intensity();

  // repeated float reflectance = 6;
  int reflectance_size() const;
  private:
  int _internal_reflectance_size() const;
  public:
  void clear_reflectance();
  private:
  float _internal_reflectance(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_reflectance() const;
  void _internal_add_reflectance(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_reflectance();
  public:
  float reflectance(int index) const;
  void set_reflectance(int index, float value);
  void add_reflectance(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      reflectance() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_reflectance();

  // int64 timestamp = 1;
  void clear_timestamp();
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp() const;
  void set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_timestamp() const;
  void _internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // @@protoc_insertion_point(class_scope:datatype.laser.pbLaser)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > x_;
  mutable std::atomic<int> _x_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > y_;
  mutable std::atomic<int> _y_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > z_;
  mutable std::atomic<int> _z_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > intensity_;
  mutable std::atomic<int> _intensity_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > reflectance_;
  mutable std::atomic<int> _reflectance_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::int64 timestamp_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_pbLaser_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// pbLaser

// int64 timestamp = 1;
inline void pbLaser::clear_timestamp() {
  timestamp_ = PROTOBUF_LONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbLaser::_internal_timestamp() const {
  return timestamp_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 pbLaser::timestamp() const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.timestamp)
  return _internal_timestamp();
}
inline void pbLaser::_internal_set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  timestamp_ = value;
}
inline void pbLaser::set_timestamp(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_timestamp(value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.timestamp)
}

// repeated float x = 2;
inline int pbLaser::_internal_x_size() const {
  return x_.size();
}
inline int pbLaser::x_size() const {
  return _internal_x_size();
}
inline void pbLaser::clear_x() {
  x_.Clear();
}
inline float pbLaser::_internal_x(int index) const {
  return x_.Get(index);
}
inline float pbLaser::x(int index) const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.x)
  return _internal_x(index);
}
inline void pbLaser::set_x(int index, float value) {
  x_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.x)
}
inline void pbLaser::_internal_add_x(float value) {
  x_.Add(value);
}
inline void pbLaser::add_x(float value) {
  _internal_add_x(value);
  // @@protoc_insertion_point(field_add:datatype.laser.pbLaser.x)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::_internal_x() const {
  return x_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::x() const {
  // @@protoc_insertion_point(field_list:datatype.laser.pbLaser.x)
  return _internal_x();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::_internal_mutable_x() {
  return &x_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::mutable_x() {
  // @@protoc_insertion_point(field_mutable_list:datatype.laser.pbLaser.x)
  return _internal_mutable_x();
}

// repeated float y = 3;
inline int pbLaser::_internal_y_size() const {
  return y_.size();
}
inline int pbLaser::y_size() const {
  return _internal_y_size();
}
inline void pbLaser::clear_y() {
  y_.Clear();
}
inline float pbLaser::_internal_y(int index) const {
  return y_.Get(index);
}
inline float pbLaser::y(int index) const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.y)
  return _internal_y(index);
}
inline void pbLaser::set_y(int index, float value) {
  y_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.y)
}
inline void pbLaser::_internal_add_y(float value) {
  y_.Add(value);
}
inline void pbLaser::add_y(float value) {
  _internal_add_y(value);
  // @@protoc_insertion_point(field_add:datatype.laser.pbLaser.y)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::_internal_y() const {
  return y_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::y() const {
  // @@protoc_insertion_point(field_list:datatype.laser.pbLaser.y)
  return _internal_y();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::_internal_mutable_y() {
  return &y_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::mutable_y() {
  // @@protoc_insertion_point(field_mutable_list:datatype.laser.pbLaser.y)
  return _internal_mutable_y();
}

// repeated float z = 4;
inline int pbLaser::_internal_z_size() const {
  return z_.size();
}
inline int pbLaser::z_size() const {
  return _internal_z_size();
}
inline void pbLaser::clear_z() {
  z_.Clear();
}
inline float pbLaser::_internal_z(int index) const {
  return z_.Get(index);
}
inline float pbLaser::z(int index) const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.z)
  return _internal_z(index);
}
inline void pbLaser::set_z(int index, float value) {
  z_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.z)
}
inline void pbLaser::_internal_add_z(float value) {
  z_.Add(value);
}
inline void pbLaser::add_z(float value) {
  _internal_add_z(value);
  // @@protoc_insertion_point(field_add:datatype.laser.pbLaser.z)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::_internal_z() const {
  return z_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::z() const {
  // @@protoc_insertion_point(field_list:datatype.laser.pbLaser.z)
  return _internal_z();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::_internal_mutable_z() {
  return &z_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::mutable_z() {
  // @@protoc_insertion_point(field_mutable_list:datatype.laser.pbLaser.z)
  return _internal_mutable_z();
}

// repeated float intensity = 5;
inline int pbLaser::_internal_intensity_size() const {
  return intensity_.size();
}
inline int pbLaser::intensity_size() const {
  return _internal_intensity_size();
}
inline void pbLaser::clear_intensity() {
  intensity_.Clear();
}
inline float pbLaser::_internal_intensity(int index) const {
  return intensity_.Get(index);
}
inline float pbLaser::intensity(int index) const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.intensity)
  return _internal_intensity(index);
}
inline void pbLaser::set_intensity(int index, float value) {
  intensity_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.intensity)
}
inline void pbLaser::_internal_add_intensity(float value) {
  intensity_.Add(value);
}
inline void pbLaser::add_intensity(float value) {
  _internal_add_intensity(value);
  // @@protoc_insertion_point(field_add:datatype.laser.pbLaser.intensity)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::_internal_intensity() const {
  return intensity_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::intensity() const {
  // @@protoc_insertion_point(field_list:datatype.laser.pbLaser.intensity)
  return _internal_intensity();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::_internal_mutable_intensity() {
  return &intensity_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::mutable_intensity() {
  // @@protoc_insertion_point(field_mutable_list:datatype.laser.pbLaser.intensity)
  return _internal_mutable_intensity();
}

// repeated float reflectance = 6;
inline int pbLaser::_internal_reflectance_size() const {
  return reflectance_.size();
}
inline int pbLaser::reflectance_size() const {
  return _internal_reflectance_size();
}
inline void pbLaser::clear_reflectance() {
  reflectance_.Clear();
}
inline float pbLaser::_internal_reflectance(int index) const {
  return reflectance_.Get(index);
}
inline float pbLaser::reflectance(int index) const {
  // @@protoc_insertion_point(field_get:datatype.laser.pbLaser.reflectance)
  return _internal_reflectance(index);
}
inline void pbLaser::set_reflectance(int index, float value) {
  reflectance_.Set(index, value);
  // @@protoc_insertion_point(field_set:datatype.laser.pbLaser.reflectance)
}
inline void pbLaser::_internal_add_reflectance(float value) {
  reflectance_.Add(value);
}
inline void pbLaser::add_reflectance(float value) {
  _internal_add_reflectance(value);
  // @@protoc_insertion_point(field_add:datatype.laser.pbLaser.reflectance)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::_internal_reflectance() const {
  return reflectance_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
pbLaser::reflectance() const {
  // @@protoc_insertion_point(field_list:datatype.laser.pbLaser.reflectance)
  return _internal_reflectance();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::_internal_mutable_reflectance() {
  return &reflectance_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
pbLaser::mutable_reflectance() {
  // @@protoc_insertion_point(field_mutable_list:datatype.laser.pbLaser.reflectance)
  return _internal_mutable_reflectance();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace laser
}  // namespace datatype

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_pbLaser_2eproto
