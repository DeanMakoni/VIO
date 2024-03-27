// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: pbVoxelBlock.proto

#include "pbVoxelBlock.pb.h"

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
namespace datatype {
namespace mapping {
class pbVoxelBlockDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<pbVoxelBlock> _instance;
} _pbVoxelBlock_default_instance_;
}  // namespace mapping
}  // namespace datatype
static void InitDefaultsscc_info_pbVoxelBlock_pbVoxelBlock_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::datatype::mapping::_pbVoxelBlock_default_instance_;
    new (ptr) ::datatype::mapping::pbVoxelBlock();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_pbVoxelBlock_pbVoxelBlock_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_pbVoxelBlock_pbVoxelBlock_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_pbVoxelBlock_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_pbVoxelBlock_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_pbVoxelBlock_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_pbVoxelBlock_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, voxels_per_side_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, voxel_size_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, origin_x_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, origin_y_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, origin_z_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, has_data_),
  PROTOBUF_FIELD_OFFSET(::datatype::mapping::pbVoxelBlock, voxel_data_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::datatype::mapping::pbVoxelBlock)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::datatype::mapping::_pbVoxelBlock_default_instance_),
};

const char descriptor_table_protodef_pbVoxelBlock_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\022pbVoxelBlock.proto\022\020datatype.mapping\"\227"
  "\001\n\014pbVoxelBlock\022\027\n\017voxels_per_side\030\001 \001(\005"
  "\022\022\n\nvoxel_size\030\002 \001(\001\022\020\n\010origin_x\030\003 \001(\001\022\020"
  "\n\010origin_y\030\004 \001(\001\022\020\n\010origin_z\030\005 \001(\001\022\020\n\010ha"
  "s_data\030\006 \001(\010\022\022\n\nvoxel_data\030\007 \003(\rb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_pbVoxelBlock_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_pbVoxelBlock_2eproto_sccs[1] = {
  &scc_info_pbVoxelBlock_pbVoxelBlock_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_pbVoxelBlock_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_pbVoxelBlock_2eproto = {
  false, false, descriptor_table_protodef_pbVoxelBlock_2eproto, "pbVoxelBlock.proto", 200,
  &descriptor_table_pbVoxelBlock_2eproto_once, descriptor_table_pbVoxelBlock_2eproto_sccs, descriptor_table_pbVoxelBlock_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_pbVoxelBlock_2eproto::offsets,
  file_level_metadata_pbVoxelBlock_2eproto, 1, file_level_enum_descriptors_pbVoxelBlock_2eproto, file_level_service_descriptors_pbVoxelBlock_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_pbVoxelBlock_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_pbVoxelBlock_2eproto)), true);
namespace datatype {
namespace mapping {

// ===================================================================

class pbVoxelBlock::_Internal {
 public:
};

pbVoxelBlock::pbVoxelBlock(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  voxel_data_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:datatype.mapping.pbVoxelBlock)
}
pbVoxelBlock::pbVoxelBlock(const pbVoxelBlock& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      voxel_data_(from.voxel_data_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&voxel_size_, &from.voxel_size_,
    static_cast<size_t>(reinterpret_cast<char*>(&origin_z_) -
    reinterpret_cast<char*>(&voxel_size_)) + sizeof(origin_z_));
  // @@protoc_insertion_point(copy_constructor:datatype.mapping.pbVoxelBlock)
}

void pbVoxelBlock::SharedCtor() {
  ::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
      reinterpret_cast<char*>(&voxel_size_) - reinterpret_cast<char*>(this)),
      0, static_cast<size_t>(reinterpret_cast<char*>(&origin_z_) -
      reinterpret_cast<char*>(&voxel_size_)) + sizeof(origin_z_));
}

pbVoxelBlock::~pbVoxelBlock() {
  // @@protoc_insertion_point(destructor:datatype.mapping.pbVoxelBlock)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void pbVoxelBlock::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void pbVoxelBlock::ArenaDtor(void* object) {
  pbVoxelBlock* _this = reinterpret_cast< pbVoxelBlock* >(object);
  (void)_this;
}
void pbVoxelBlock::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void pbVoxelBlock::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const pbVoxelBlock& pbVoxelBlock::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_pbVoxelBlock_pbVoxelBlock_2eproto.base);
  return *internal_default_instance();
}


void pbVoxelBlock::Clear() {
// @@protoc_insertion_point(message_clear_start:datatype.mapping.pbVoxelBlock)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  voxel_data_.Clear();
  ::memset(&voxel_size_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&origin_z_) -
      reinterpret_cast<char*>(&voxel_size_)) + sizeof(origin_z_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* pbVoxelBlock::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // int32 voxels_per_side = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          voxels_per_side_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // double voxel_size = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          voxel_size_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // double origin_x = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 25)) {
          origin_x_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // double origin_y = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 33)) {
          origin_y_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // double origin_z = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 41)) {
          origin_z_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // bool has_data = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          has_data_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated uint32 voxel_data = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 58)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedUInt32Parser(_internal_mutable_voxel_data(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 56) {
          _internal_add_voxel_data(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
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

::PROTOBUF_NAMESPACE_ID::uint8* pbVoxelBlock::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:datatype.mapping.pbVoxelBlock)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 voxels_per_side = 1;
  if (this->voxels_per_side() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_voxels_per_side(), target);
  }

  // double voxel_size = 2;
  if (!(this->voxel_size() <= 0 && this->voxel_size() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_voxel_size(), target);
  }

  // double origin_x = 3;
  if (!(this->origin_x() <= 0 && this->origin_x() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(3, this->_internal_origin_x(), target);
  }

  // double origin_y = 4;
  if (!(this->origin_y() <= 0 && this->origin_y() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(4, this->_internal_origin_y(), target);
  }

  // double origin_z = 5;
  if (!(this->origin_z() <= 0 && this->origin_z() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(5, this->_internal_origin_z(), target);
  }

  // bool has_data = 6;
  if (this->has_data() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(6, this->_internal_has_data(), target);
  }

  // repeated uint32 voxel_data = 7;
  {
    int byte_size = _voxel_data_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteUInt32Packed(
          7, _internal_voxel_data(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:datatype.mapping.pbVoxelBlock)
  return target;
}

size_t pbVoxelBlock::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:datatype.mapping.pbVoxelBlock)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated uint32 voxel_data = 7;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      UInt32Size(this->voxel_data_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _voxel_data_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // double voxel_size = 2;
  if (!(this->voxel_size() <= 0 && this->voxel_size() >= 0)) {
    total_size += 1 + 8;
  }

  // double origin_x = 3;
  if (!(this->origin_x() <= 0 && this->origin_x() >= 0)) {
    total_size += 1 + 8;
  }

  // int32 voxels_per_side = 1;
  if (this->voxels_per_side() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_voxels_per_side());
  }

  // bool has_data = 6;
  if (this->has_data() != 0) {
    total_size += 1 + 1;
  }

  // double origin_y = 4;
  if (!(this->origin_y() <= 0 && this->origin_y() >= 0)) {
    total_size += 1 + 8;
  }

  // double origin_z = 5;
  if (!(this->origin_z() <= 0 && this->origin_z() >= 0)) {
    total_size += 1 + 8;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void pbVoxelBlock::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:datatype.mapping.pbVoxelBlock)
  GOOGLE_DCHECK_NE(&from, this);
  const pbVoxelBlock* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<pbVoxelBlock>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:datatype.mapping.pbVoxelBlock)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:datatype.mapping.pbVoxelBlock)
    MergeFrom(*source);
  }
}

void pbVoxelBlock::MergeFrom(const pbVoxelBlock& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:datatype.mapping.pbVoxelBlock)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  voxel_data_.MergeFrom(from.voxel_data_);
  if (!(from.voxel_size() <= 0 && from.voxel_size() >= 0)) {
    _internal_set_voxel_size(from._internal_voxel_size());
  }
  if (!(from.origin_x() <= 0 && from.origin_x() >= 0)) {
    _internal_set_origin_x(from._internal_origin_x());
  }
  if (from.voxels_per_side() != 0) {
    _internal_set_voxels_per_side(from._internal_voxels_per_side());
  }
  if (from.has_data() != 0) {
    _internal_set_has_data(from._internal_has_data());
  }
  if (!(from.origin_y() <= 0 && from.origin_y() >= 0)) {
    _internal_set_origin_y(from._internal_origin_y());
  }
  if (!(from.origin_z() <= 0 && from.origin_z() >= 0)) {
    _internal_set_origin_z(from._internal_origin_z());
  }
}

void pbVoxelBlock::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:datatype.mapping.pbVoxelBlock)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void pbVoxelBlock::CopyFrom(const pbVoxelBlock& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:datatype.mapping.pbVoxelBlock)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool pbVoxelBlock::IsInitialized() const {
  return true;
}

void pbVoxelBlock::InternalSwap(pbVoxelBlock* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  voxel_data_.InternalSwap(&other->voxel_data_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(pbVoxelBlock, origin_z_)
      + sizeof(pbVoxelBlock::origin_z_)
      - PROTOBUF_FIELD_OFFSET(pbVoxelBlock, voxel_size_)>(
          reinterpret_cast<char*>(&voxel_size_),
          reinterpret_cast<char*>(&other->voxel_size_));
}

::PROTOBUF_NAMESPACE_ID::Metadata pbVoxelBlock::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace mapping
}  // namespace datatype
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::datatype::mapping::pbVoxelBlock* Arena::CreateMaybeMessage< ::datatype::mapping::pbVoxelBlock >(Arena* arena) {
  return Arena::CreateMessageInternal< ::datatype::mapping::pbVoxelBlock >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
