# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: databricks.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2
from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10\x64\x61tabricks.proto\x12\x06mlflow\x1a google/protobuf/descriptor.proto\x1a\x15scalapb/scalapb.proto\"\xcd\x01\n\x14\x44\x61tabricksRpcOptions\x12\'\n\tendpoints\x18\x01 \x03(\x0b\x32\x14.mlflow.HttpEndpoint\x12&\n\nvisibility\x18\x02 \x01(\x0e\x32\x12.mlflow.Visibility\x12&\n\x0b\x65rror_codes\x18\x03 \x03(\x0e\x32\x11.mlflow.ErrorCode\x12%\n\nrate_limit\x18\x04 \x01(\x0b\x32\x11.mlflow.RateLimit\x12\x15\n\rrpc_doc_title\x18\x05 \x01(\t\"U\n\x0cHttpEndpoint\x12\x14\n\x06method\x18\x01 \x01(\t:\x04POST\x12\x0c\n\x04path\x18\x02 \x01(\t\x12!\n\x05since\x18\x03 \x01(\x0b\x32\x12.mlflow.ApiVersion\"*\n\nApiVersion\x12\r\n\x05major\x18\x01 \x01(\x05\x12\r\n\x05minor\x18\x02 \x01(\x05\"@\n\tRateLimit\x12\x11\n\tmax_burst\x18\x01 \x01(\x03\x12 \n\x18max_sustained_per_second\x18\x02 \x01(\x03\"\x93\x01\n\x15\x44ocumentationMetadata\x12\x11\n\tdocstring\x18\x01 \x01(\t\x12\x10\n\x08lead_doc\x18\x02 \x01(\t\x12&\n\nvisibility\x18\x03 \x01(\x0e\x32\x12.mlflow.Visibility\x12\x1b\n\x13original_proto_path\x18\x04 \x03(\t\x12\x10\n\x08position\x18\x05 \x01(\x05\"n\n\x1f\x44\x61tabricksServiceExceptionProto\x12%\n\nerror_code\x18\x01 \x01(\x0e\x32\x11.mlflow.ErrorCode\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x13\n\x0bstack_trace\x18\x03 \x01(\t*?\n\nVisibility\x12\n\n\x06PUBLIC\x10\x01\x12\x0c\n\x08INTERNAL\x10\x02\x12\x17\n\x13PUBLIC_UNDOCUMENTED\x10\x03*\xf6\x04\n\tErrorCode\x12\x12\n\x0eINTERNAL_ERROR\x10\x01\x12\x1b\n\x17TEMPORARILY_UNAVAILABLE\x10\x02\x12\x0c\n\x08IO_ERROR\x10\x03\x12\x0f\n\x0b\x42\x41\x44_REQUEST\x10\x04\x12\x1c\n\x17INVALID_PARAMETER_VALUE\x10\xe8\x07\x12\x17\n\x12\x45NDPOINT_NOT_FOUND\x10\xe9\x07\x12\x16\n\x11MALFORMED_REQUEST\x10\xea\x07\x12\x12\n\rINVALID_STATE\x10\xeb\x07\x12\x16\n\x11PERMISSION_DENIED\x10\xec\x07\x12\x15\n\x10\x46\x45\x41TURE_DISABLED\x10\xed\x07\x12\x1a\n\x15\x43USTOMER_UNAUTHORIZED\x10\xee\x07\x12\x1b\n\x16REQUEST_LIMIT_EXCEEDED\x10\xef\x07\x12\x1d\n\x18INVALID_STATE_TRANSITION\x10\xd1\x0f\x12\x1b\n\x16\x43OULD_NOT_ACQUIRE_LOCK\x10\xd2\x0f\x12\x1c\n\x17RESOURCE_ALREADY_EXISTS\x10\xb9\x17\x12\x1c\n\x17RESOURCE_DOES_NOT_EXIST\x10\xba\x17\x12\x13\n\x0eQUOTA_EXCEEDED\x10\xa1\x1f\x12\x1c\n\x17MAX_BLOCK_SIZE_EXCEEDED\x10\xa2\x1f\x12\x1b\n\x16MAX_READ_SIZE_EXCEEDED\x10\xa3\x1f\x12\x13\n\x0e\x44RY_RUN_FAILED\x10\x89\'\x12\x1c\n\x17RESOURCE_LIMIT_EXCEEDED\x10\x8a\'\x12\x18\n\x13\x44IRECTORY_NOT_EMPTY\x10\xf1.\x12\x18\n\x13\x44IRECTORY_PROTECTED\x10\xf2.\x12\x1f\n\x1aMAX_NOTEBOOK_SIZE_EXCEEDED\x10\xf3.:G\n\nvisibility\x12\x1d.google.protobuf.FieldOptions\x18\xee\x90\x03 \x01(\x0e\x32\x12.mlflow.Visibility::\n\x11validate_required\x12\x1d.google.protobuf.FieldOptions\x18\xef\x90\x03 \x01(\x08:4\n\x0bjson_inline\x12\x1d.google.protobuf.FieldOptions\x18\xf0\x90\x03 \x01(\x08:1\n\x08json_map\x12\x1d.google.protobuf.FieldOptions\x18\xf1\x90\x03 \x01(\x08:Q\n\tfield_doc\x12\x1d.google.protobuf.FieldOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadata:K\n\x03rpc\x12\x1e.google.protobuf.MethodOptions\x18\xee\x90\x03 \x01(\x0b\x32\x1c.mlflow.DatabricksRpcOptions:S\n\nmethod_doc\x12\x1e.google.protobuf.MethodOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadata:U\n\x0bmessage_doc\x12\x1f.google.protobuf.MessageOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadata:U\n\x0bservice_doc\x12\x1f.google.protobuf.ServiceOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadata:O\n\x08\x65num_doc\x12\x1c.google.protobuf.EnumOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadata:V\n\x15\x65num_value_visibility\x12!.google.protobuf.EnumValueOptions\x18\xee\x90\x03 \x01(\x0e\x32\x12.mlflow.Visibility:Z\n\x0e\x65num_value_doc\x12!.google.protobuf.EnumValueOptions\x18\xf2\x90\x03 \x03(\x0b\x32\x1d.mlflow.DocumentationMetadataB*\n#com.databricks.api.proto.databricks\xe2?\x02\x10\x01')

_VISIBILITY = DESCRIPTOR.enum_types_by_name['Visibility']
Visibility = enum_type_wrapper.EnumTypeWrapper(_VISIBILITY)
_ERRORCODE = DESCRIPTOR.enum_types_by_name['ErrorCode']
ErrorCode = enum_type_wrapper.EnumTypeWrapper(_ERRORCODE)
PUBLIC = 1
INTERNAL = 2
PUBLIC_UNDOCUMENTED = 3
INTERNAL_ERROR = 1
TEMPORARILY_UNAVAILABLE = 2
IO_ERROR = 3
BAD_REQUEST = 4
INVALID_PARAMETER_VALUE = 1000
ENDPOINT_NOT_FOUND = 1001
MALFORMED_REQUEST = 1002
INVALID_STATE = 1003
PERMISSION_DENIED = 1004
FEATURE_DISABLED = 1005
CUSTOMER_UNAUTHORIZED = 1006
REQUEST_LIMIT_EXCEEDED = 1007
INVALID_STATE_TRANSITION = 2001
COULD_NOT_ACQUIRE_LOCK = 2002
RESOURCE_ALREADY_EXISTS = 3001
RESOURCE_DOES_NOT_EXIST = 3002
QUOTA_EXCEEDED = 4001
MAX_BLOCK_SIZE_EXCEEDED = 4002
MAX_READ_SIZE_EXCEEDED = 4003
DRY_RUN_FAILED = 5001
RESOURCE_LIMIT_EXCEEDED = 5002
DIRECTORY_NOT_EMPTY = 6001
DIRECTORY_PROTECTED = 6002
MAX_NOTEBOOK_SIZE_EXCEEDED = 6003

VISIBILITY_FIELD_NUMBER = 51310
visibility = DESCRIPTOR.extensions_by_name['visibility']
VALIDATE_REQUIRED_FIELD_NUMBER = 51311
validate_required = DESCRIPTOR.extensions_by_name['validate_required']
JSON_INLINE_FIELD_NUMBER = 51312
json_inline = DESCRIPTOR.extensions_by_name['json_inline']
JSON_MAP_FIELD_NUMBER = 51313
json_map = DESCRIPTOR.extensions_by_name['json_map']
FIELD_DOC_FIELD_NUMBER = 51314
field_doc = DESCRIPTOR.extensions_by_name['field_doc']
RPC_FIELD_NUMBER = 51310
rpc = DESCRIPTOR.extensions_by_name['rpc']
METHOD_DOC_FIELD_NUMBER = 51314
method_doc = DESCRIPTOR.extensions_by_name['method_doc']
MESSAGE_DOC_FIELD_NUMBER = 51314
message_doc = DESCRIPTOR.extensions_by_name['message_doc']
SERVICE_DOC_FIELD_NUMBER = 51314
service_doc = DESCRIPTOR.extensions_by_name['service_doc']
ENUM_DOC_FIELD_NUMBER = 51314
enum_doc = DESCRIPTOR.extensions_by_name['enum_doc']
ENUM_VALUE_VISIBILITY_FIELD_NUMBER = 51310
enum_value_visibility = DESCRIPTOR.extensions_by_name['enum_value_visibility']
ENUM_VALUE_DOC_FIELD_NUMBER = 51314
enum_value_doc = DESCRIPTOR.extensions_by_name['enum_value_doc']

_DATABRICKSRPCOPTIONS = DESCRIPTOR.message_types_by_name['DatabricksRpcOptions']
_HTTPENDPOINT = DESCRIPTOR.message_types_by_name['HttpEndpoint']
_APIVERSION = DESCRIPTOR.message_types_by_name['ApiVersion']
_RATELIMIT = DESCRIPTOR.message_types_by_name['RateLimit']
_DOCUMENTATIONMETADATA = DESCRIPTOR.message_types_by_name['DocumentationMetadata']
_DATABRICKSSERVICEEXCEPTIONPROTO = DESCRIPTOR.message_types_by_name['DatabricksServiceExceptionProto']
DatabricksRpcOptions = _reflection.GeneratedProtocolMessageType('DatabricksRpcOptions', (_message.Message,), {
  'DESCRIPTOR' : _DATABRICKSRPCOPTIONS,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.DatabricksRpcOptions)
  })
_sym_db.RegisterMessage(DatabricksRpcOptions)

HttpEndpoint = _reflection.GeneratedProtocolMessageType('HttpEndpoint', (_message.Message,), {
  'DESCRIPTOR' : _HTTPENDPOINT,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.HttpEndpoint)
  })
_sym_db.RegisterMessage(HttpEndpoint)

ApiVersion = _reflection.GeneratedProtocolMessageType('ApiVersion', (_message.Message,), {
  'DESCRIPTOR' : _APIVERSION,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.ApiVersion)
  })
_sym_db.RegisterMessage(ApiVersion)

RateLimit = _reflection.GeneratedProtocolMessageType('RateLimit', (_message.Message,), {
  'DESCRIPTOR' : _RATELIMIT,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.RateLimit)
  })
_sym_db.RegisterMessage(RateLimit)

DocumentationMetadata = _reflection.GeneratedProtocolMessageType('DocumentationMetadata', (_message.Message,), {
  'DESCRIPTOR' : _DOCUMENTATIONMETADATA,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.DocumentationMetadata)
  })
_sym_db.RegisterMessage(DocumentationMetadata)

DatabricksServiceExceptionProto = _reflection.GeneratedProtocolMessageType('DatabricksServiceExceptionProto', (_message.Message,), {
  'DESCRIPTOR' : _DATABRICKSSERVICEEXCEPTIONPROTO,
  '__module__' : 'databricks_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.DatabricksServiceExceptionProto)
  })
_sym_db.RegisterMessage(DatabricksServiceExceptionProto)

if _descriptor._USE_C_DESCRIPTORS == False:
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(visibility)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(validate_required)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(json_inline)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(json_map)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(field_doc)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(rpc)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(method_doc)
  google_dot_protobuf_dot_descriptor__pb2.MessageOptions.RegisterExtension(message_doc)
  google_dot_protobuf_dot_descriptor__pb2.ServiceOptions.RegisterExtension(service_doc)
  google_dot_protobuf_dot_descriptor__pb2.EnumOptions.RegisterExtension(enum_doc)
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(enum_value_visibility)
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(enum_value_doc)

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n#com.databricks.api.proto.databricks\342?\002\020\001'
  _VISIBILITY._serialized_start=752
  _VISIBILITY._serialized_end=815
  _ERRORCODE._serialized_start=818
  _ERRORCODE._serialized_end=1448
  _DATABRICKSRPCOPTIONS._serialized_start=86
  _DATABRICKSRPCOPTIONS._serialized_end=291
  _HTTPENDPOINT._serialized_start=293
  _HTTPENDPOINT._serialized_end=378
  _APIVERSION._serialized_start=380
  _APIVERSION._serialized_end=422
  _RATELIMIT._serialized_start=424
  _RATELIMIT._serialized_end=488
  _DOCUMENTATIONMETADATA._serialized_start=491
  _DOCUMENTATIONMETADATA._serialized_end=638
  _DATABRICKSSERVICEEXCEPTIONPROTO._serialized_start=640
  _DATABRICKSSERVICEEXCEPTIONPROTO._serialized_end=750
# @@protoc_insertion_point(module_scope)
