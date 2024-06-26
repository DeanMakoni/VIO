// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbExperience.proto

#include "pbExperience.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_pbImage_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_pbImage_pbImage_2eproto;
extern PROTOBUF_INTERNAL_EXPORT_pbMatrix_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_pbMatrix_pbMatrix_2eproto;
namespace datatype {
namespace navigation {
class pbExperienceDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<pbExperience> _instance;
} _pbExperience_default_instance_;
}  // namespace navigation
}  // namespace datatype
static void InitDefaultsscc_info_pbExperience_pbExperience_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::datatype::navigation::_pbExperience_default_instance_;
    new (ptr) ::datatype::navigation::pbExperience();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<2> scc_info_pbExperience_pbExperience_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 2, 0, InitDefaultsscc_info_pbExperience_pbExperience_2eproto}, {
      &scc_info_pbImage_pbImage_2eproto.base,
      &scc_info_pbMatrix_pbMatrix_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_pbExperience_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_pbExperience_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_pbExperience_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_pbExperience_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, timestamp_),
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, image_left_),
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, keypoints_),
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, descriptors_),
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, landmarks_),
  PROTOBUF_FIELD_OFFSET(::datatype::navigation::pbExperience, bow_desc_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::datatype::navigation::pbExperience)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::datatype::navigation::_pbExperience_default_instance_),
};

const char descriptor_table_protodef_pbExperience_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\022pbExperience.proto\022\023datatype.navigatio"
  "n\032\rpbImage.proto\032\016pbMatrix.proto\"\216\002\n\014pbE"
  "xperience\022\021\n\ttimestamp\030\001 \001(\003\022+\n\nimage_le"
  "ft\030\002 \001(\0132\027.datatype.image.pbImage\022/\n\tkey"
  "points\030\003 \001(\0132\034.datatype.transform.pbMatr"
  "ix\022,\n\013descriptors\030\004 \001(\0132\027.datatype.image"
  ".pbImage\022/\n\tlandmarks\030\005 \001(\0132\034.datatype.t"
  "ransform.pbMatrix\022.\n\010bow_desc\030\006 \001(\0132\034.da"
  "tatype.transform.pbMatrixb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_pbExperience_2eproto_deps[2] = {
  &::descriptor_table_pbImage_2eproto,
  &::descriptor_table_pbMatrix_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_pbExperience_2eproto_sccs[1] = {
  &scc_info_pbExperience_pbExperience_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_pbExperience_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbExperience_2eproto = {
  false, false, descriptor_table_protodef_pbExperience_2eproto, "pbExperience.proto", 353,
  &descriptor_table_pbExperience_2eproto_once, descriptor_table_pbExperience_2eproto_sccs, descriptor_table_pbExperience_2eproto_deps, 1, 2,
  schemas, file_default_instances, TableStruct_pbExperience_2eproto::offsets,
  file_level_metadata_pbExperience_2eproto, 1, file_level_enum_descriptors_pbExperience_2eproto, file_level_service_descriptors_pbExperience_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_pbExperience_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_pbExperience_2eproto)), true);
namespace datatype {
namespace navigation {

// ===================================================================

class pbExperience::_Internal {
 public:
  static const ::datatype::image::pbImage& image_left(const pbExperience* msg);
  static const ::datatype::transform::pbMatrix& keypoints(const pbExperience* msg);
  static const ::datatype::image::pbImage& descriptors(const pbExperience* msg);
  static const ::datatype::transform::pbMatrix& landmarks(const pbExperience* msg);
  static const ::datatype::transform::pbMatrix& bow_desc(const pbExperience* msg);
};

const ::datatype::image::pbImage&
pbExperience::_Internal::image_left(const pbExperience* msg) {
  return *msg->image_left_;
}
const ::datatype::transform::pbMatrix&
pbExperience::_Internal::keypoints(const pbExperience* msg) {
  return *msg->keypoints_;
}
const ::datatype::image::pbImage&
pbExperience::_Internal::descriptors(const pbExperience* msg) {
  return *msg->descriptors_;
}
const ::datatype::transform::pbMatrix&
pbExperience::_Internal::landmarks(const pbExperience* msg) {
  return *msg->landmarks_;
}
const ::datatype::transform::pbMatrix&
pbExperience::_Internal::bow_desc(const pbExperience* msg) {
  return *msg->bow_desc_;
}
void pbExperience::clear_image_left() {
  if (GetArena() == nullptr && image_left_ != nullptr) {
    delete image_left_;
  }
  image_left_ = nullptr;
}
void pbExperience::clear_keypoints() {
  if (GetArena() == nullptr && keypoints_ != nullptr) {
    delete keypoints_;
  }
  keypoints_ = nullptr;
}
void pbExperience::clear_descriptors() {
  if (GetArena() == nullptr && descriptors_ != nullptr) {
    delete descriptors_;
  }
  descriptors_ = nullptr;
}
void pbExperience::clear_landmarks() {
  if (GetArena() == nullptr && landmarks_ != nullptr) {
    delete landmarks_;
  }
  landmarks_ = nullptr;
}
void pbExperience::clear_bow_desc() {
  if (GetArena() == nullptr && bow_desc_ != nullptr) {
    delete bow_desc_;
  }
  bow_desc_ = nullptr;
}
pbExperience::pbExperience(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:datatype.navigation.pbExperience)
}
pbExperience::pbExperience(const pbExperience& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_image_left()) {
    image_left_ = new ::datatype::image::pbImage(*from.image_left_);
  } else {
    image_left_ = nullptr;
  }
  if (from._internal_has_keypoints()) {
    keypoints_ = new ::datatype::transform::pbMatrix(*from.keypoints_);
  } else {
    keypoints_ = nullptr;
  }
  if (from._internal_has_descriptors()) {
    descriptors_ = new ::datatype::image::pbImage(*from.descriptors_);
  } else {
    descriptors_ = nullptr;
  }
  if (from._internal_has_landmarks()) {
    landmarks_ = new ::datatype::transform::pbMatrix(*from.landmarks_);
  } else {
    landmarks_ = nullptr;
  }
  if (from._internal_has_bow_desc()) {
    bow_desc_ = new ::datatype::transform::pbMatrix(*from.bow_desc_);
  } else {
    bow_desc_ = nullptr;
  }
  timestamp_ = from.timestamp_;
  // @@protoc_insertion_point(copy_constructor:datatype.navigation.pbExperience)
}

void pbExperience::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_pbExperience_pbExperience_2eproto.base);
  ::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
      reinterpret_cast<char*>(&image_left_) - reinterpret_cast<char*>(this)),
      0, static_cast<size_t>(reinterpret_cast<char*>(&timestamp_) -
      reinterpret_cast<char*>(&image_left_)) + sizeof(timestamp_));
}

pbExperience::~pbExperience() {
  // @@protoc_insertion_point(destructor:datatype.navigation.pbExperience)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void pbExperience::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  if (this != internal_default_instance()) delete image_left_;
  if (this != internal_default_instance()) delete keypoints_;
  if (this != internal_default_instance()) delete descriptors_;
  if (this != internal_default_instance()) delete landmarks_;
  if (this != internal_default_instance()) delete bow_desc_;
}

void pbExperience::ArenaDtor(void* object) {
  pbExperience* _this = reinterpret_cast< pbExperience* >(object);
  (void)_this;
}
void pbExperience::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void pbExperience::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const pbExperience& pbExperience::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_pbExperience_pbExperience_2eproto.base);
  return *internal_default_instance();
}


void pbExperience::Clear() {
// @@protoc_insertion_point(message_clear_start:datatype.navigation.pbExperience)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArena() == nullptr && image_left_ != nullptr) {
    delete image_left_;
  }
  image_left_ = nullptr;
  if (GetArena() == nullptr && keypoints_ != nullptr) {
    delete keypoints_;
  }
  keypoints_ = nullptr;
  if (GetArena() == nullptr && descriptors_ != nullptr) {
    delete descriptors_;
  }
  descriptors_ = nullptr;
  if (GetArena() == nullptr && landmarks_ != nullptr) {
    delete landmarks_;
  }
  landmarks_ = nullptr;
  if (GetArena() == nullptr && bow_desc_ != nullptr) {
    delete bow_desc_;
  }
  bow_desc_ = nullptr;
  timestamp_ = PROTOBUF_LONGLONG(0);
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* pbExperience::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // int64 timestamp = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          timestamp_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .datatype.image.pbImage image_left = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_image_left(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .datatype.transform.pbMatrix keypoints = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_keypoints(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .datatype.image.pbImage descriptors = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ctx->ParseMessage(_internal_mutable_descriptors(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .datatype.transform.pbMatrix landmarks = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          ptr = ctx->ParseMessage(_internal_mutable_landmarks(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .datatype.transform.pbMatrix bow_desc = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 50)) {
          ptr = ctx->ParseMessage(_internal_mutable_bow_desc(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* pbExperience::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:datatype.navigation.pbExperience)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 timestamp = 1;
  if (this->timestamp() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(1, this->_internal_timestamp(), target);
  }

  // .datatype.image.pbImage image_left = 2;
  if (this->has_image_left()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        2, _Internal::image_left(this), target, stream);
  }

  // .datatype.transform.pbMatrix keypoints = 3;
  if (this->has_keypoints()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        3, _Internal::keypoints(this), target, stream);
  }

  // .datatype.image.pbImage descriptors = 4;
  if (this->has_descriptors()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        4, _Internal::descriptors(this), target, stream);
  }

  // .datatype.transform.pbMatrix landmarks = 5;
  if (this->has_landmarks()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        5, _Internal::landmarks(this), target, stream);
  }

  // .datatype.transform.pbMatrix bow_desc = 6;
  if (this->has_bow_desc()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        6, _Internal::bow_desc(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:datatype.navigation.pbExperience)
  return target;
}

size_t pbExperience::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:datatype.navigation.pbExperience)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .datatype.image.pbImage image_left = 2;
  if (this->has_image_left()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *image_left_);
  }

  // .datatype.transform.pbMatrix keypoints = 3;
  if (this->has_keypoints()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *keypoints_);
  }

  // .datatype.image.pbImage descriptors = 4;
  if (this->has_descriptors()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *descriptors_);
  }

  // .datatype.transform.pbMatrix landmarks = 5;
  if (this->has_landmarks()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *landmarks_);
  }

  // .datatype.transform.pbMatrix bow_desc = 6;
  if (this->has_bow_desc()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *bow_desc_);
  }

  // int64 timestamp = 1;
  if (this->timestamp() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
        this->_internal_timestamp());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void pbExperience::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:datatype.navigation.pbExperience)
  GOOGLE_DCHECK_NE(&from, this);
  const pbExperience* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<pbExperience>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:datatype.navigation.pbExperience)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:datatype.navigation.pbExperience)
    MergeFrom(*source);
  }
}

void pbExperience::MergeFrom(const pbExperience& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:datatype.navigation.pbExperience)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_image_left()) {
    _internal_mutable_image_left()->::datatype::image::pbImage::MergeFrom(from._internal_image_left());
  }
  if (from.has_keypoints()) {
    _internal_mutable_keypoints()->::datatype::transform::pbMatrix::MergeFrom(from._internal_keypoints());
  }
  if (from.has_descriptors()) {
    _internal_mutable_descriptors()->::datatype::image::pbImage::MergeFrom(from._internal_descriptors());
  }
  if (from.has_landmarks()) {
    _internal_mutable_landmarks()->::datatype::transform::pbMatrix::MergeFrom(from._internal_landmarks());
  }
  if (from.has_bow_desc()) {
    _internal_mutable_bow_desc()->::datatype::transform::pbMatrix::MergeFrom(from._internal_bow_desc());
  }
  if (from.timestamp() != 0) {
    _internal_set_timestamp(from._internal_timestamp());
  }
}

void pbExperience::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:datatype.navigation.pbExperience)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void pbExperience::CopyFrom(const pbExperience& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:datatype.navigation.pbExperience)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool pbExperience::IsInitialized() const {
  return true;
}

void pbExperience::InternalSwap(pbExperience* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(pbExperience, timestamp_)
      + sizeof(pbExperience::timestamp_)
      - PROTOBUF_FIELD_OFFSET(pbExperience, image_left_)>(
          reinterpret_cast<char*>(&image_left_),
          reinterpret_cast<char*>(&other->image_left_));
}

::PROTOBUF_NAMESPACE_ID::Metadata pbExperience::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace navigation
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::datatype::navigation::pbExperience* Arena::CreateMaybeMessage< ::datatype::navigation::pbExperience >(Arena* arena) {
  return Arena::CreateMessageInternal< ::datatype::navigation::pbExperience >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
