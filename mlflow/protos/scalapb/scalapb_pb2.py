# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: scalapb/scalapb.proto
# Protobuf Python Version: 5.27.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    1,
    '',
    'scalapb/scalapb.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15scalapb/scalapb.proto\x12\x07scalapb\x1a google/protobuf/descriptor.proto\"L\n\x0eScalaPbOptions\x12\x14\n\x0cpackage_name\x18\x01 \x01(\t\x12\x14\n\x0c\x66lat_package\x18\x02 \x01(\x08\x12\x0e\n\x06import\x18\x03 \x03(\t\"!\n\x0eMessageOptions\x12\x0f\n\x07\x65xtends\x18\x01 \x03(\t\"\x1c\n\x0c\x46ieldOptions\x12\x0c\n\x04type\x18\x01 \x01(\t:G\n\x07options\x12\x1c.google.protobuf.FileOptions\x18\xfc\x07 \x01(\x0b\x32\x17.scalapb.ScalaPbOptions:J\n\x07message\x12\x1f.google.protobuf.MessageOptions\x18\xfc\x07 \x01(\x0b\x32\x17.scalapb.MessageOptions:D\n\x05\x66ield\x12\x1d.google.protobuf.FieldOptions\x18\xfc\x07 \x01(\x0b\x32\x15.scalapb.FieldOptionsB\x1e\n\x1corg.mlflow.scalapb_interface')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'scalapb.scalapb_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\034org.mlflow.scalapb_interface'
  _globals['_SCALAPBOPTIONS']._serialized_start=68
  _globals['_SCALAPBOPTIONS']._serialized_end=144
  _globals['_MESSAGEOPTIONS']._serialized_start=146
  _globals['_MESSAGEOPTIONS']._serialized_end=179
  _globals['_FIELDOPTIONS']._serialized_start=181
  _globals['_FIELDOPTIONS']._serialized_end=209
# @@protoc_insertion_point(module_scope)
