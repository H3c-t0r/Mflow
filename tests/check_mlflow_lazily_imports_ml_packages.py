"""
Tests that mlflow doesn't import ml packages when it's imported
"""

import sys
import importlib
import mlflow  # pylint: disable=unused-import


def main():
    ml_packages = {
        "catboost",
        "fastai",
        "mxnet",
        "h2o",
        "keras",
        "lightgbm",
        "mleap",
        "onnx",
        "pytorch_lightning",
        "pyspark",
        "shap",
        "sklearn",
        "spacy",
        "statsmodels",
        "tensorflow",
        "torch",
        "xgboost",
        "pmdarima",
        "diviner",
    }
    imported = ml_packages.intersection(set(sys.modules))
    assert imported == set(), f"mlflow imports {imported} when it's imported but it should not"

    # Ensure that the ML packages are importable
    failed_to_import = []
    for package in ml_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            failed_to_import.append(package)

    message = (
        f"Failed to import {failed_to_import}. Please install packages that provide these modules."
    )
    assert failed_to_import == [], message


if __name__ == "__main__":
    main()
