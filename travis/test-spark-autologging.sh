#!/usr/bin/env bash
# Test Spark autologging against the Spark 3.0 preview. This script is temporary and should be
# removed once Spark 3.0 is released in favor of simply updating all tests to run against Spark 3.0
# (i.e. updating the pyspark dependency version in travis/large-requirements.txt)
set -ex

# Build Java package
pushd mlflow/java/spark
mvn package -DskipTests
popd

# Install PySpark 3.0 preview & run tests. For faster local iteration, you can also simply download
# the .tgz used below (http://mirror.cogentco.com/pub/apache/spark/spark-3.0.0-preview/spark-3.0.0-preview-bin-hadoop2.7.tgz),
# extract it, and set SPARK_HOME to the path of the extracted folder while invoking pytest as
# shown below
TEMPDIR=$(mktemp -d)
pushd $TEMPDIR
wget -qO- http://mirror.cogentco.com/pub/apache/spark/spark-3.0.0-preview/spark-3.0.0-preview-bin-hadoop2.7.tgz | tar xvz -
pip install -e spark-3.0.0-preview-bin-hadoop2.7/python
popd

SPARK_HOME=$TEMPDIR/spark-3.0.0-preview-bin-hadoop2.7 pytest tests/spark_autologging/test_spark_datasource_autologging.py
rm -rf $TEMPDIR

# Reinstall old dependencies
pip install -r travis/large-requirements.txt
