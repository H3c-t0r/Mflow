# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: databricks_managed_catalog_messages.proto
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
    'databricks_managed_catalog_messages.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2
from . import databricks_pb2 as databricks__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)databricks_managed_catalog_messages.proto\x12\x15mlflow.managedcatalog\x1a\x15scalapb/scalapb.proto\x1a\x10\x64\x61tabricks.proto\"0\n\tTableInfo\x12\x11\n\tfull_name\x18\x0f \x01(\t\x12\x10\n\x08table_id\x18\x16 \x01(\t\"\xc2\x01\n\x08GetTable\x12\x15\n\rfull_name_arg\x18\x01 \x01(\t\x12\x14\n\x0comit_columns\x18\x05 \x01(\x08\x12\x17\n\x0fomit_properties\x18\x06 \x01(\x08\x12\x18\n\x10omit_constraints\x18\x07 \x01(\x08\x12\x19\n\x11omit_dependencies\x18\x08 \x01(\x08\x12\x15\n\romit_username\x18\x0b \x01(\x08\x12$\n\x1comit_storage_credential_name\x18\x0c \x01(\x08\"7\n\x10GetTableResponse\x12\x11\n\tfull_name\x18\x0f \x01(\t\x12\x10\n\x08table_id\x18\x16 \x01(\tB1\n\'com.databricks.api.proto.managedcatalog\xa0\x01\x01\xe2?\x02\x10\x01')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'databricks_managed_catalog_messages_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\'com.databricks.api.proto.managedcatalog\240\001\001\342?\002\020\001'
  _globals['_TABLEINFO']._serialized_start=109
  _globals['_TABLEINFO']._serialized_end=157
  _globals['_GETTABLE']._serialized_start=160
  _globals['_GETTABLE']._serialized_end=354
  _globals['_GETTABLERESPONSE']._serialized_start=356
  _globals['_GETTABLERESPONSE']._serialized_end=411
# @@protoc_insertion_point(module_scope)
