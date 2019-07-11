# pep8: disable=E501

from __future__ import print_function

import collections
import shutil
import pytest
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers

import mlflow
import mlflow.tensorflow
import mlflow.keras

SavedModelInfo = collections.namedtuple(
        "SavedModelInfo",
        ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"])

client = mlflow.tracking.MlflowClient()


@pytest.fixture
def random_train_data():
    return np.random.random((1000, 32))


@pytest.fixture
def tf_keras_random_data_run(random_train_data):
    mlflow.tensorflow.autolog(metrics_every_n_steps=5)

    def random_one_hot_labels(shape):
        n, n_class = shape
        classes = np.random.randint(0, n_class, n)
        labels = np.zeros((n, n_class))
        labels[np.arange(n), classes] = 1
        return labels

    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels((1000, 10))

        model = tf.keras.Sequential()

        model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=10)

    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_keras_autolog_logs_expected_data(tf_keras_random_data_run):
    data = tf_keras_random_data_run.data

    assert 'epoch_acc' in data.metrics
    assert 'epoch_loss' in data.metrics
    assert 'optimizer_name' in data.params
    assert data.params['optimizer_name'] == 'AdamOptimizer'
    assert 'summary' in tf_keras_random_data_run.data.tags
    assert 'Total params: 6,922' in tf_keras_random_data_run.data.tags['summary']
    all_epoch_acc = client.get_metric_history(tf_keras_random_data_run.info.run_id, 'epoch_acc')
    assert all((x.step - 1) % 5 == 0 for x in all_epoch_acc)


@pytest.mark.large
def test_tf_keras_autolog_model_can_load_from_artifact(tf_keras_random_data_run, random_train_data):
    artifacts = client.list_artifacts(tf_keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    assert 'tensorboard_logs' in artifacts
    model = mlflow.keras.load_model("runs:/" + tf_keras_random_data_run.info.run_id +
                                    "/model")
    model.predict(random_train_data)


@pytest.fixture
def tf_core_random_tensors():
    mlflow.tensorflow.autolog(metrics_every_n_steps=4)
    with mlflow.start_run() as run:
        sess = tf.Session()
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)
        total = a + b
        tf.summary.scalar('a', a)
        tf.summary.scalar('b', b)
        merged = tf.summary.merge_all()
        dir = tempfile.mkdtemp()
        writer = tf.summary.FileWriter(dir, sess.graph)
        for i in range(40):
            with sess.as_default():
                summary, _ = sess.run([merged, total])
            writer.add_summary(summary, global_step=i)
        shutil.rmtree(dir)
        writer.close()
        sess.close()

    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_core_autolog_logs_scalars(tf_core_random_tensors):
    assert 'a' in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics['a'] == 3.0
    assert 'b' in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics['b'] == 4.0
    all_a = client.get_metric_history(tf_core_random_tensors.info.run_id, 'a')
    assert all((x.step - 1) % 4 == 0 for x in all_a)
    assert mlflow.active_run() is None


@pytest.fixture
def tf_estimator_random_data_run():
    mlflow.tensorflow.autolog()
    with mlflow.start_run() as run:
        dir = tempfile.mkdtemp()
        CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        SPECIES = ['Setosa', 'Versicolor', 'Virginica']

        train_path = tf.keras.utils.get_file(
            "iris_training.csv", "https://storage.googleapis.com/download"
                                 ".tensorflow.org/data/iris_training.csv")
        test_path = tf.keras.utils.get_file(
            "iris_test.csv", "https://storage.googleapis.com/download"
                             ".tensorflow.org/data/iris_test.csv")

        train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
        test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

        train_y = train.pop('Species')
        test_y = test.pop('Species')

        def input_fn(features, labels, training=True, batch_size=256):
            """An input function for training or evaluating"""
            # Convert the inputs to a Dataset.
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

            # Shuffle and repeat if you are in training mode.
            if training:
                dataset = dataset.shuffle(1000).repeat()

            return dataset.batch(batch_size)

        my_feature_columns = []
        for key in train.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        feature_spec = {}
        for feature in CSV_COLUMN_NAMES:
            feature_spec[feature] = tf.placeholder(dtype="float", name=feature, shape=[150])

        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[30, 10],
            # The model must choose between 3 classes.
            n_classes=3,
            model_dir=dir)

        classifier.train(
            input_fn=lambda: input_fn(train, train_y, training=True),
            steps=500)
        classifier.export_saved_model(dir, receiver_fn)

    shutil.rmtree(dir)
    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_estimator_autolog_logs_metrics(tf_estimator_random_data_run):
    assert 'loss' in tf_estimator_random_data_run.data.metrics
    metrics = client.get_metric_history(tf_estimator_random_data_run.info.run_id, 'loss')
    assert all((x.step-1) % 100 == 0 for x in metrics)


@pytest.mark.large
def test_tf_keras_autolog_model_can_load_from_artifact(tf_estimator_random_data_run):
    artifacts = client.list_artifacts(tf_estimator_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    session = tf.Session()
    model = mlflow.tensorflow.load_model("runs:/" + tf_estimator_random_data_run.info.run_id +
                                         "/model", session)


@pytest.fixture
def duplicate_autolog_tf_estimator_run():
    mlflow.tensorflow.autolog(metrics_every_n_steps=23)  # 23 is prime; no false positives in test
    run = tf_estimator_random_data_run()
    return run  # should be autologged every 4 steps


@pytest.mark.large
def test_duplicate_autolog_second_overrides(duplicate_autolog_tf_estimator_run):
    metrics = client.get_metric_history(duplicate_autolog_tf_estimator_run.info.run_id, 'loss')
    assert all((x.step - 1) % 4 == 0 for x in metrics)
