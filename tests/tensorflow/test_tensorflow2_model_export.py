# pep8: disable=E501

import collections
import os
import shutil
import sys
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


@pytest.fixture
def saved_tf_iris_model(tmpdir):
    # Following code from
    # https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
    train_x, train_y = iris_data_utils.load_data()[0]

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    estimator = tf_estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
    )

    # Train the Model.
    batch_size = 100
    train_steps = 1000
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps,
    )

    # Generate predictions from the model
    predict_x = {
        "SepalLength": [5.1, 5.9, 6.9],
        "SepalWidth": [3.3, 3.0, 3.1],
        "PetalLength": [1.7, 4.2, 5.4],
        "PetalWidth": [0.5, 1.5, 2.1],
    }

    estimator_preds = estimator.predict(
        lambda: iris_data_utils.eval_input_fn(predict_x, None, batch_size)
    )

    # Building a dictionary of the predictions by the estimator.
    if sys.version_info < (3, 0):
        estimator_preds_dict = estimator_preds.next()
    else:
        estimator_preds_dict = next(estimator_preds)
    for row in estimator_preds:
        for key in row.keys():
            estimator_preds_dict[key] = np.vstack((estimator_preds_dict[key], row[key]))

    # Building a pandas DataFrame out of the prediction dictionary.
    estimator_preds_df = copy.deepcopy(estimator_preds_dict)
    for col in estimator_preds_df.keys():
        if all(len(element) == 1 for element in estimator_preds_df[col]):
            estimator_preds_df[col] = estimator_preds_df[col].ravel()
        else:
            estimator_preds_df[col] = estimator_preds_df[col].tolist()

    # Building a DataFrame that contains the names of the flowers predicted.
    estimator_preds_df = pandas.DataFrame.from_dict(data=estimator_preds_df)
    estimator_preds_results = [
        iris_data_utils.SPECIES[id[0]] for id in estimator_preds_dict["class_ids"]
    ]
    estimator_preds_results_df = pd.DataFrame({"predictions": estimator_preds_results})

    # Define a function for estimator inference
    feature_spec = {}
    for name in my_feature_columns:
        feature_spec[name.key] = tf.Variable([], dtype=tf.float64, name=name.key)

    receiver_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(
            data=predict_x, columns=[name.key for name in my_feature_columns]
        ),
        expected_results_df=estimator_preds_results_df,
        raw_results=estimator_preds_dict,
        raw_df=estimator_preds_df,
    )


@pytest.fixture
def saved_tf_categorical_model(tmpdir):
    path = os.path.abspath("tests/data/uci-autos-imports-85.data")
    # Order is important for the csv-readers, so we use an OrderedDict here
    defaults = collections.OrderedDict(
        [("body-style", [""]), ("curb-weight", [0.0]), ("highway-mpg", [0.0]), ("price", [0.0])]
    )
    types = collections.OrderedDict((key, type(value[0])) for key, value in defaults.items())
    df = pd.read_csv(path, names=list(types.keys()), dtype=types, na_values="?")
    df = df.dropna()

    # Extract the label from the features dataframe
    y_train = df.pop("price")

    # Create the required input training function
    trainingFeatures = {}
    for i in df:
        trainingFeatures[i] = df[i].values

    # Create the feature columns required for the DNNRegressor
    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        key="body-style", vocabulary_list=body_style_vocab
    )
    feature_columns = [
        tf.feature_column.numeric_column(key="curb-weight"),
        tf.feature_column.numeric_column(key="highway-mpg"),
        # Since this is a DNN model, convert categorical columns from sparse to dense.
        # Then, wrap them in an `indicator_column` to create a one-hot vector from the input
        tf.feature_column.indicator_column(body_style),
    ]

    # Build a DNNRegressor, with 20x20-unit hidden layers, with the feature columns
    # defined above as input
    estimator = tf_estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)

    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(trainingFeatures, y_train, 1), steps=10
    )
    estimator_preds = np.array(
        [
            s["predictions"]
            for s in estimator.predict(
                lambda: iris_data_utils.eval_input_fn(trainingFeatures, None, 1)
            )
        ]
    ).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {
        "body-style": tf.Variable([], dtype=tf.string, name="body-style"),
        "curb-weight": tf.Variable([], dtype=tf.float64, name="curb-weight"),
        "highway-mpg": tf.Variable([], dtype=tf.float64, name="highway-mpg"),
    }
    receiver_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(trainingFeatures),
        expected_results_df=estimator_preds_df,
        raw_results=None,
        raw_df=None,
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


def test_load_model_from_remote_uri_succeeds(saved_tf_iris_model, model_path, mock_s3_bucket):
    save_tf_estimator_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    infer = mlflow.tensorflow.load_model(model_uri=model_uri)
    feed_dict = {
        df_column_name: tf.constant(saved_tf_iris_model.inference_df[df_column_name])
        for df_column_name in list(saved_tf_iris_model.inference_df)
    }
    raw_preds = infer(**feed_dict)
    pred_dict = {column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()}
    for col in pred_dict:
        np.testing.assert_allclose(
            np.array(pred_dict[col], dtype=np.float),
            np.array(saved_tf_iris_model.raw_results[col], dtype=np.float),
        )


def test_iris_model_can_be_loaded_and_evaluated_successfully(saved_tf_iris_model, model_path):
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
