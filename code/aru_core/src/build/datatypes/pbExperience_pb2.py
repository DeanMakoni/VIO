# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pbExperience.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import pbImage_pb2 as pbImage__pb2
import pbMatrix_pb2 as pbMatrix__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='pbExperience.proto',
  package='datatype.navigation',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12pbExperience.proto\x12\x13\x64\x61tatype.navigation\x1a\rpbImage.proto\x1a\x0epbMatrix.proto\"\x8e\x02\n\x0cpbExperience\x12\x11\n\ttimestamp\x18\x01 \x01(\x03\x12+\n\nimage_left\x18\x02 \x01(\x0b\x32\x17.datatype.image.pbImage\x12/\n\tkeypoints\x18\x03 \x01(\x0b\x32\x1c.datatype.transform.pbMatrix\x12,\n\x0b\x64\x65scriptors\x18\x04 \x01(\x0b\x32\x17.datatype.image.pbImage\x12/\n\tlandmarks\x18\x05 \x01(\x0b\x32\x1c.datatype.transform.pbMatrix\x12.\n\x08\x62ow_desc\x18\x06 \x01(\x0b\x32\x1c.datatype.transform.pbMatrixb\x06proto3'
  ,
  dependencies=[pbImage__pb2.DESCRIPTOR,pbMatrix__pb2.DESCRIPTOR,])




_PBEXPERIENCE = _descriptor.Descriptor(
  name='pbExperience',
  full_name='datatype.navigation.pbExperience',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='datatype.navigation.pbExperience.timestamp', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_left', full_name='datatype.navigation.pbExperience.image_left', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='keypoints', full_name='datatype.navigation.pbExperience.keypoints', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='descriptors', full_name='datatype.navigation.pbExperience.descriptors', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='landmarks', full_name='datatype.navigation.pbExperience.landmarks', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bow_desc', full_name='datatype.navigation.pbExperience.bow_desc', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=75,
  serialized_end=345,
)

_PBEXPERIENCE.fields_by_name['image_left'].message_type = pbImage__pb2._PBIMAGE
_PBEXPERIENCE.fields_by_name['keypoints'].message_type = pbMatrix__pb2._PBMATRIX
_PBEXPERIENCE.fields_by_name['descriptors'].message_type = pbImage__pb2._PBIMAGE
_PBEXPERIENCE.fields_by_name['landmarks'].message_type = pbMatrix__pb2._PBMATRIX
_PBEXPERIENCE.fields_by_name['bow_desc'].message_type = pbMatrix__pb2._PBMATRIX
DESCRIPTOR.message_types_by_name['pbExperience'] = _PBEXPERIENCE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

pbExperience = _reflection.GeneratedProtocolMessageType('pbExperience', (_message.Message,), {
  'DESCRIPTOR' : _PBEXPERIENCE,
  '__module__' : 'pbExperience_pb2'
  # @@protoc_insertion_point(class_scope:datatype.navigation.pbExperience)
  })
_sym_db.RegisterMessage(pbExperience)


# @@protoc_insertion_point(module_scope)
