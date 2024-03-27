# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pbImage.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='pbImage.proto',
  package='datatype.image',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rpbImage.proto\x12\x0e\x64\x61tatype.image\"\xab\x01\n\x07pbImage\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x11\n\ttimestamp\x18\x03 \x01(\x03\x12\x12\n\nimage_data\x18\x04 \x01(\x0c\"Z\n\tImageType\x12\r\n\tRGB_UINT8\x10\x00\x12\x0e\n\nGREY_UINT8\x10\x01\x12\r\n\tRGB_FLOAT\x10\x02\x12\x0e\n\nGREY_FLOAT\x10\x03\x12\x0f\n\x0b\x44\x65pth_FLOAT\x10\x04\x62\x06proto3'
)



_PBIMAGE_IMAGETYPE = _descriptor.EnumDescriptor(
  name='ImageType',
  full_name='datatype.image.pbImage.ImageType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RGB_UINT8', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GREY_UINT8', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RGB_FLOAT', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GREY_FLOAT', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='Depth_FLOAT', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=115,
  serialized_end=205,
)
_sym_db.RegisterEnumDescriptor(_PBIMAGE_IMAGETYPE)


_PBIMAGE = _descriptor.Descriptor(
  name='pbImage',
  full_name='datatype.image.pbImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='datatype.image.pbImage.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='datatype.image.pbImage.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='datatype.image.pbImage.timestamp', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_data', full_name='datatype.image.pbImage.image_data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PBIMAGE_IMAGETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=205,
)

_PBIMAGE_IMAGETYPE.containing_type = _PBIMAGE
DESCRIPTOR.message_types_by_name['pbImage'] = _PBIMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

pbImage = _reflection.GeneratedProtocolMessageType('pbImage', (_message.Message,), {
  'DESCRIPTOR' : _PBIMAGE,
  '__module__' : 'pbImage_pb2'
  # @@protoc_insertion_point(class_scope:datatype.image.pbImage)
  })
_sym_db.RegisterMessage(pbImage)


# @@protoc_insertion_point(module_scope)