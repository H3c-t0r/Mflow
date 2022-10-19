#!/usr/bin/env bash
set -x

export MLFLOW_HOME=$(pwd)

# TODO: Run tests for h2o, shap, and paddle in the cross-version-tests workflow
pytest \
  tests/tracking/fluent/test_fluent_autolog.py \
  tests/autologging \
  tests/test_mlflow_lazily_imports_ml_packages.py
