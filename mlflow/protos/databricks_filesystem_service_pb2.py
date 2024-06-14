# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: databricks_filesystem_service.proto
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
    'databricks_filesystem_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import databricks_pb2 as databricks__pb2
from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n#databricks_filesystem_service.proto\x12\x11mlflow.filesystem\x1a\x10\x64\x61tabricks.proto\x1a\x15scalapb/scalapb.proto\")\n\nHttpHeader\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"`\n\x18\x43reateDownloadUrlRequest\x12\x0c\n\x04path\x18\x01 \x01(\t:6\xe2?3\n1com.databricks.rpc.RPC[CreateDownloadUrlResponse]\"\x82\x01\n\x19\x43reateDownloadUrlResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12.\n\x07headers\x18\x02 \x03(\x0b\x32\x1d.mlflow.filesystem.HttpHeader:(\xe2?%\n#com.databricks.rpc.DoNotLogContents\"\\\n\x16\x43reateUploadUrlRequest\x12\x0c\n\x04path\x18\x01 \x01(\t:4\xe2?1\n/com.databricks.rpc.RPC[CreateUploadUrlResponse]\"\x80\x01\n\x17\x43reateUploadUrlResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12.\n\x07headers\x18\x02 \x03(\x0b\x32\x1d.mlflow.filesystem.HttpHeader:(\xe2?%\n#com.databricks.rpc.DoNotLogContents\"\x96\x01\n\x0e\x44irectoryEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x14\n\x0cis_directory\x18\x02 \x01(\x08\x12\x11\n\tfile_size\x18\x03 \x01(\x03\x12\x15\n\rlast_modified\x18\x04 \x01(\x03\x12\x0c\n\x04name\x18\x05 \x01(\t:(\xe2?%\n#com.databricks.rpc.DoNotLogContents\"\x8f\x01\n\x15ListDirectoryResponse\x12\x33\n\x08\x63ontents\x18\x01 \x03(\x0b\x32!.mlflow.filesystem.DirectoryEntry\x12\x17\n\x0fnext_page_token\x18\x02 \x01(\t:(\xe2?%\n#com.databricks.rpc.DoNotLogContents2\xcb\x02\n\x11\x46ilesystemService\x12\x9d\x01\n\x11\x43reateDownloadUrl\x12+.mlflow.filesystem.CreateDownloadUrlRequest\x1a,.mlflow.filesystem.CreateDownloadUrlResponse\"-\xf2\x86\x19)\n%\n\x04POST\x12\x17/fs/create-download-url\x1a\x04\x08\x02\x10\x00\x10\x03\x12\x95\x01\n\x0f\x43reateUploadUrl\x12).mlflow.filesystem.CreateUploadUrlRequest\x1a*.mlflow.filesystem.CreateUploadUrlResponse\"+\xf2\x86\x19\'\n#\n\x04POST\x12\x15/fs/create-upload-url\x1a\x04\x08\x02\x10\x00\x10\x03\x42\x30\n#com.databricks.api.proto.filesystem\x90\x01\x01\xa0\x01\x01\xe2?\x02\x10\x01')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'databricks_filesystem_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n#com.databricks.api.proto.filesystem\220\001\001\240\001\001\342?\002\020\001'
  _globals['_CREATEDOWNLOADURLREQUEST']._loaded_options = None
  _globals['_CREATEDOWNLOADURLREQUEST']._serialized_options = b'\342?3\n1com.databricks.rpc.RPC[CreateDownloadUrlResponse]'
  _globals['_CREATEDOWNLOADURLRESPONSE']._loaded_options = None
  _globals['_CREATEDOWNLOADURLRESPONSE']._serialized_options = b'\342?%\n#com.databricks.rpc.DoNotLogContents'
  _globals['_CREATEUPLOADURLREQUEST']._loaded_options = None
  _globals['_CREATEUPLOADURLREQUEST']._serialized_options = b'\342?1\n/com.databricks.rpc.RPC[CreateUploadUrlResponse]'
  _globals['_CREATEUPLOADURLRESPONSE']._loaded_options = None
  _globals['_CREATEUPLOADURLRESPONSE']._serialized_options = b'\342?%\n#com.databricks.rpc.DoNotLogContents'
  _globals['_DIRECTORYENTRY']._loaded_options = None
  _globals['_DIRECTORYENTRY']._serialized_options = b'\342?%\n#com.databricks.rpc.DoNotLogContents'
  _globals['_LISTDIRECTORYRESPONSE']._loaded_options = None
  _globals['_LISTDIRECTORYRESPONSE']._serialized_options = b'\342?%\n#com.databricks.rpc.DoNotLogContents'
  _globals['_FILESYSTEMSERVICE'].methods_by_name['CreateDownloadUrl']._loaded_options = None
  _globals['_FILESYSTEMSERVICE'].methods_by_name['CreateDownloadUrl']._serialized_options = b'\362\206\031)\n%\n\004POST\022\027/fs/create-download-url\032\004\010\002\020\000\020\003'
  _globals['_FILESYSTEMSERVICE'].methods_by_name['CreateUploadUrl']._loaded_options = None
  _globals['_FILESYSTEMSERVICE'].methods_by_name['CreateUploadUrl']._serialized_options = b'\362\206\031\'\n#\n\004POST\022\025/fs/create-upload-url\032\004\010\002\020\000\020\003'
  _globals['_HTTPHEADER']._serialized_start=99
  _globals['_HTTPHEADER']._serialized_end=140
  _globals['_CREATEDOWNLOADURLREQUEST']._serialized_start=142
  _globals['_CREATEDOWNLOADURLREQUEST']._serialized_end=238
  _globals['_CREATEDOWNLOADURLRESPONSE']._serialized_start=241
  _globals['_CREATEDOWNLOADURLRESPONSE']._serialized_end=371
  _globals['_CREATEUPLOADURLREQUEST']._serialized_start=373
  _globals['_CREATEUPLOADURLREQUEST']._serialized_end=465
  _globals['_CREATEUPLOADURLRESPONSE']._serialized_start=468
  _globals['_CREATEUPLOADURLRESPONSE']._serialized_end=596
  _globals['_DIRECTORYENTRY']._serialized_start=599
  _globals['_DIRECTORYENTRY']._serialized_end=749
  _globals['_LISTDIRECTORYRESPONSE']._serialized_start=752
  _globals['_LISTDIRECTORYRESPONSE']._serialized_end=895
  _globals['_FILESYSTEMSERVICE']._serialized_start=898
  _globals['_FILESYSTEMSERVICE']._serialized_end=1229
_builder.BuildServices(DESCRIPTOR, 'databricks_filesystem_service_pb2', _globals)
# @@protoc_insertion_point(module_scope)
