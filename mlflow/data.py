from __future__ import print_function

import os
import re

from six.moves import urllib

from mlflow.utils import process

DBFS_PREFIX = "dbfs:/"
S3_PREFIX = "s3://"
GS_PREFIX = "gs://"
ACTION_REGEX = re.compile("^-{1,2}[^\d\W]\w*\Z")
DBFS_REGEX = re.compile("^%s" % re.escape(DBFS_PREFIX))
S3_REGEX = re.compile("^%s" % re.escape(S3_PREFIX))
GS_REGEX = re.compile("^%s" % re.escape(GS_PREFIX))


class DownloadException(Exception):
    pass


def _fetch_dbfs(uri, local_path):
    print("=== Downloading DBFS file %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    process.exec_cmd(cmd=["databricks", "fs", "cp", "-r", uri, local_path])


def _fetch_s3(uri, local_path):
    import boto3
    print("=== Downloading S3 object %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    (bucket, s3_path) = parse_s3_uri(uri)
    boto3.client('s3').download_file(bucket, s3_path, local_path)


def _fetch_gs(uri, local_path):
    from google.cloud import storage
    print("=== Downloading GCS file %s to local path %s ===" % (uri, os.path.abspath(local_path)))
    (bucket, gs_path) = parse_gs_uri(uri)
    storage.Client().bucket(bucket).blob(gs_path).download_to_filename(local_path)


def parse_s3_uri(uri):
    """Parse an S3 URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3":
        raise Exception("Not an S3 URI: %s" % uri)
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return parsed.netloc, path


def parse_gs_uri(uri):
    """Parse an GCS URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs":
        raise Exception("Not a GCS URI: %s" % uri)
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return parsed.netloc, path


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0


def is_action(action_string):
    if len(action_string.strip()) > 0:
        return ACTION_REGEX.match(action_string)
    return True


def download_uri(uri, output_path):
    if DBFS_REGEX.match(uri):
        _fetch_dbfs(uri, output_path)
    elif S3_REGEX.match(uri):
        _fetch_s3(uri, output_path)
    elif GS_REGEX.match(uri):
        _fetch_gs(uri, output_path)
    else:
        raise DownloadException("`uri` must be a DBFS (%s), S3 (%s), or GCS (%s) URI, got "
                                "%s" % (DBFS_PREFIX, S3_PREFIX, GS_PREFIX, uri))
