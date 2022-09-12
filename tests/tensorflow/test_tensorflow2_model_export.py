# pep8: disable=E501

import collections
import os
from pathlib import Path
import shutil
import pickle
import pytest
import copy
import json

import numpy as np
import pandas as pd
import pandas.testing
import tensorflow as tf
from tensorflow import estimator as tf_estimator
import iris_data_utils

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.tensorflow import _TF2Wrapper
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.pyfunc.backend import _execute_in_conda_env
from tests.pipelines.helper_functions import chdir

from tests.helper_functions import pyfunc_serve_and_score_model

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    [
        "path",
        "meta_graph_tags",
        "signature_def_key",
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
    ],
)


def save_tf_estimator_model(
    tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, path, mlflow_model=None
):
    """
    A helper method to save tf estimator model as a mlflow model, it is used for testing loading
    tf estimator model saved by previous mlflow version.
    """
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()

    flavor_conf = dict(
        saved_model_dir="tfmodel",
        meta_graph_tags=tf_meta_graph_tags,
        signature_def_key=tf_signature_def_key,
    )
    mlflow_model.add_flavor("tensorflow", code=None, **flavor_conf)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.tensorflow",
        env="conda.yaml",
        code=None,
    )
    mlflow_model.save(os.path.join(path, "MLmodel"))
    with open(os.path.join(path, "conda.yaml"), "w") as f:
        f.write(
            """
channels:
- conda-forge
dependencies:
- python=3.8.12
- pip<=21.2.4
- pip:
  - mlflow
  - bcrypt==3.2.0
  - boto3==1.20.46
  - defusedxml==0.7.1
  - fsspec==2022.1.0
  - keras==2.7.0
  - pandas==1.4.0
  - pillow==9.0.0
  - pyopenssl==22.0.0
  - scipy==1.7.3
  - tensorflow==2.7.0
"""
        )
    with open(os.path.join(path, "python_env.yaml"), "w") as f:
        f.write(
            """
python: 3.8.12
build_dependencies:
- pip==21.2.4
- setuptools==61.2.0
- wheel==0.37.1
dependencies:
- -r requirements.txt
"""
        )
    with open(os.path.join(path, "requirements.txt"), "w") as f:
        f.write(
            """
mlflow
bcrypt==3.2.0
boto3==1.20.46
defusedxml==0.7.1
fsspec==2022.1.0
keras==2.7.0
pandas==1.4.0
pillow==9.0.0
pyopenssl==22.0.0
scipy==1.7.3
tensorflow==2.7.0                                                                                             
"""
        )
    shutil.copytree(tf_saved_model_dir, os.path.join(path, "tfmodel"))


def log_tf_estimator_model(
    tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, artifact_path
):
    """
    A helper method to log tf estimator model as a mlflow model artifact,
    it is used for testing loading tf estimator model saved by previous mlflow version.
    """
    with TempDir() as tmp:
        local_path = tmp.path("model")
        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        save_tf_estimator_model(
            path=local_path,
            tf_saved_model_dir=tf_saved_model_dir,
            tf_meta_graph_tags=tf_meta_graph_tags,
            tf_signature_def_key=tf_signature_def_key,
            mlflow_model=mlflow_model,
        )
        mlflow.tracking.fluent.log_artifacts(local_path, artifact_path)
    return mlflow_model.get_model_info()


@pytest.fixture
def tf_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "pytest"])
    return conda_env


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


tracking_uri = f"file:{os.getcwd()}/mlruns"

ModelDataInfo = collections.namedtuple(
    "ModelDataInfo",
    [
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
    ]
)


def save_tf_model_by_mlflow128(model_type, model_path):
    conda_env = get_or_create_conda_env("tests/tensorflow/mlflow-128-tf-23-env.yaml")
    with TempDir() as tmpdir:
        output_data_file_path = tmpdir.path("output_data.pkl")
        with chdir("tests/tensorflow"):
            # change cwd to avoid it imports current repo mlflow.
            _execute_in_conda_env(
                conda_env,
                f"python save_tf_estimator_model.py {model_type} save_model {model_path} {output_data_file_path}",
                install_mlflow=False,
                command_env={"MLFLOW_TRACKING_URI": tracking_uri}
            )
        with open(output_data_file_path, "rb") as f:
            return ModelDataInfo(*pickle.load(f))


def test_load_model_from_remote_uri_succeeds(model_path, mock_s3_bucket):
    model_data_info = save_tf_model_by_mlflow128("iris", model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    infer = mlflow.tensorflow.load_model(model_uri=model_uri)
    feed_dict = {
        df_column_name: tf.constant(model_data_info.inference_df[df_column_name])
        for df_column_name in list(model_data_info.inference_df)
    }
    raw_preds = infer(**feed_dict)
    pred_dict = {column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()}
    for col in pred_dict:
        np.testing.assert_allclose(
            np.array(pred_dict[col], dtype=np.float),
            np.array(model_data_info.raw_results[col], dtype=np.float),
        )


def test_iris_model_can_be_loaded_and_evaluated_successfully(model_path):
    save_tf_estimator_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    def load_and_evaluate():

        infer = mlflow.tensorflow.load_model(model_uri=model_path)
        feed_dict = {
            df_column_name: tf.constant(saved_tf_iris_model.inference_df[df_column_name])
            for df_column_name in list(saved_tf_iris_model.inference_df)
        }
        raw_preds = infer(**feed_dict)
        pred_dict = {
            column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()
        }
        for col in pred_dict:
            np.testing.assert_array_equal(pred_dict[col], saved_tf_iris_model.raw_results[col])

    load_and_evaluate()

    with tf.device("/CPU:0"):
        load_and_evaluate()


def test_load_model_loads_artifacts_from_specified_model_directory(saved_tf_iris_model, model_path):
    save_tf_estimator_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    # Verify that the MLflow model can be loaded even after deleting the TensorFlow `SavedModel`
    # directory that was used to create it, implying that the artifacts were copied to and are
    # loaded from the specified MLflow model path
    shutil.rmtree(saved_tf_iris_model.path)

    mlflow.tensorflow.load_model(model_uri=model_path)


def test_log_and_load_model_persists_and_restores_model_successfully(saved_tf_iris_model):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = log_tf_estimator_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        assert model_info.model_uri == model_uri

    mlflow.tensorflow.load_model(model_uri=model_uri)


def test_iris_data_model_can_be_loaded_and_evaluated_as_pyfunc(saved_tf_iris_model, model_path):
    save_tf_estimator_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(saved_tf_iris_model.inference_df)
    assert isinstance(results_df, pd.DataFrame)
    for key in results_df.keys():
        np.testing.assert_array_equal(results_df[key], saved_tf_iris_model.raw_df[key])

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(saved_tf_iris_model.inference_df):
        inp_dict[df_col_name] = saved_tf_iris_model.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    for key in results.keys():
        np.testing.assert_array_equal(results[key], saved_tf_iris_model.raw_df[key].tolist())

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(saved_tf_iris_model.inference_df):
        inp_list.append(saved_tf_iris_model.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_categorical_model_can_be_loaded_and_evaluated_as_pyfunc(
    saved_tf_categorical_model, model_path
):
    save_tf_estimator_model(
        tf_saved_model_dir=saved_tf_categorical_model.path,
        tf_meta_graph_tags=saved_tf_categorical_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_categorical_model.signature_def_key,
        path=model_path,
    )

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(saved_tf_categorical_model.inference_df)
    # Precision is less accurate for the categorical model when we load back the saved model.
    pandas.testing.assert_frame_equal(
        results_df, saved_tf_categorical_model.expected_results_df, check_less_precise=3
    )

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(saved_tf_categorical_model.inference_df):
        inp_dict[df_col_name] = saved_tf_categorical_model.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_dict(data=results),
        saved_tf_categorical_model.expected_results_df,
        check_less_precise=3,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(saved_tf_categorical_model.inference_df):
        inp_list.append(saved_tf_categorical_model.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_pyfunc_serve_and_score(saved_tf_iris_model):
    artifact_path = "model"

    with mlflow.start_run():
        log_tf_estimator_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=saved_tf_iris_model.inference_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--env-manager", "local"],
    )
    actual = pd.DataFrame(json.loads(resp.content))["class_ids"].values
    expected = (
        saved_tf_iris_model.expected_results_df["predictions"]
        .map(iris_data_utils.SPECIES.index)
        .values
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_tf_saved_model_model_with_tf_keras_api(tmpdir):
    tf.random.set_seed(1337)

    mlflow_model_path = os.path.join(str(tmpdir), "mlflow_model")
    tf_model_path = os.path.join(str(tmpdir), "tf_model")

    # Build TensorFlow model.
    inputs = tf.keras.layers.Input(shape=1, name="feature1", dtype=tf.float32)
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs])

    # Save model in TensorFlow SavedModel format.
    tf.saved_model.save(model, tf_model_path)

    # Save TensorFlow SavedModel as MLflow model.
    save_tf_estimator_model(
        tf_saved_model_dir=tf_model_path,
        tf_meta_graph_tags=["serve"],
        tf_signature_def_key="serving_default",
        path=mlflow_model_path,
    )

    def load_and_predict():
        model_uri = mlflow_model_path
        mlflow_model = mlflow.pyfunc.load_model(model_uri)
        feed_dict = {"feature1": tf.constant([[2.0]])}
        predictions = mlflow_model.predict(feed_dict)
        np.testing.assert_allclose(predictions["dense"], model.predict(feed_dict).squeeze())

    load_and_predict()


def test_saved_model_support_array_type_input():
    def infer(features):
        res = np.expand_dims(features.numpy().sum(axis=1), axis=1)
        return {"prediction": tf.constant(res)}

    model = _TF2Wrapper(None, infer)
    infer_df = pd.DataFrame({"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]})

    result = model.predict(infer_df)

    np.testing.assert_allclose(result["prediction"], infer_df.applymap(sum).values[:, 0])


def test_virtualenv_subfield_points_to_correct_path(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()
