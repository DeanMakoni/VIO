# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pbVoxelLayer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='pbVoxelLayer.proto',
  package='datatype.mapping',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12pbVoxelLayer.proto\x12\x10\x64\x61tatype.mapping\"I\n\x0cpbVoxelLayer\x12\x12\n\nvoxel_size\x18\x01 \x01(\x01\x12\x17\n\x0fvoxels_per_side\x18\x02 \x01(\r\x12\x0c\n\x04type\x18\x03 \x01(\tb\x06proto3'
)




_PBVOXELLAYER = _descriptor.Descriptor(
  name='pbVoxelLayer',
  full_name='datatype.mapping.pbVoxelLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='voxel_size', full_name='datatype.mapping.pbVoxelLayer.voxel_size', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='voxels_per_side', full_name='datatype.mapping.pbVoxelLayer.voxels_per_side', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='datatype.mapping.pbVoxelLayer.type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=113,
)

DESCRIPTOR.message_types_by_name['pbVoxelLayer'] = _PBVOXELLAYER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

pbVoxelLayer = _reflection.GeneratedProtocolMessageType('pbVoxelLayer', (_message.Message,), {
  'DESCRIPTOR' : _PBVOXELLAYER,
  '__module__' : 'pbVoxelLayer_pb2'
  # @@protoc_insertion_point(class_scope:datatype.mapping.pbVoxelLayer)
  })
_sym_db.RegisterMessage(pbVoxelLayer)


# @@protoc_insertion_point(module_scope)