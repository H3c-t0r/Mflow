# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlflow/protos/internal.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlflow.protos.scalapb import scalapb_pb2 as mlflow_dot_protos_dot_scalapb_dot_scalapb__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cmlflow/protos/internal.proto\x12\x0fmlflow.internal\x1a#mlflow/protos/scalapb/scalapb.proto*\'\n\x0fInputVertexType\x12\x07\n\x03RUN\x10\x01\x12\x0b\n\x07\x44\x41TASET\x10\x02\x42\x1e\n\x19org.mlflow.internal.proto\x90\x01\x01')

_INPUTVERTEXTYPE = DESCRIPTOR.enum_types_by_name['InputVertexType']
InputVertexType = enum_type_wrapper.EnumTypeWrapper(_INPUTVERTEXTYPE)
RUN = 1
DATASET = 2


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\031org.mlflow.internal.proto\220\001\001'
  _INPUTVERTEXTYPE._serialized_start=86
  _INPUTVERTEXTYPE._serialized_end=125
# @@protoc_insertion_point(module_scope)
