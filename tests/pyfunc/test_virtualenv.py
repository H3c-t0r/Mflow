import os
from unittest import mock
from collections import namedtuple

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

import mlflow
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON_SPLIT_ORIENTED
from mlflow.pyfunc.backend import _MLFLOW_ENV_ROOT_ENV_VAR

from tests.helper_functions import pyfunc_serve_and_score_model


@pytest.fixture(scope="module")
def sklearn_model():
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = LogisticRegression().fit(X, y)
    X_pred = X.sample(frac=0.1, random_state=0)
    y_pred = model.predict(X_pred)
    return namedtuple("Model", ["model", "X_pred", "y_pred"])(model, X_pred, y_pred)


def serve_and_score(model_uri, data, extra_args=None):
    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=data,
        content_type=CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--env-manager=virtualenv"] + (extra_args or []),
    )
    return pd.read_json(resp.content, orient="records").values.squeeze()


@pytest.fixture
def temp_env_root(tmp_path, monkeypatch):
    env_root = tmp_path / "envs"
    monkeypatch.setenv(_MLFLOW_ENV_ROOT_ENV_VAR, str(env_root))
    return env_root


def test_virtualenv_serving(temp_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    # This call should reuse the environment created in the previous run
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    assert len(os.listdir(temp_env_root)) == 1


@pytest.mark.usefixtures(temp_env_root.__name__)
def test_virtualenv_serving_when_python_env_does_not_exist(sklearn_model):
    with mlflow.start_run():
        with mock.patch("mlflow.utils.environment.PythonEnv.to_yaml") as mock_to_yaml:
            model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")
            mock_to_yaml.assert_called_once()

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


def test_virtualenv_serving_pip_install_fails(temp_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            # Enforce pip install to fail using a non-existing package version
            pip_requirements=["mlflow==999.999.999"],
        )
    try:
        serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    except Exception:
        pass
    assert len(list(temp_env_root.iterdir())) == 0
