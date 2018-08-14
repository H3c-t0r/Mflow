# pep8: disable=E501

from __future__ import print_function

import os
import tempfile
import pytest
from keras.models import Sequential
from keras.layers import Dense
import sklearn.datasets as datasets
import pandas as pd
import numpy as np

import mlflow.keras
import mlflow
from mlflow import pyfunc


@pytest.fixture(scope='module')
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    y = data['target']
    x = data.drop('target', axis=1)
    return (x, y)


@pytest.fixture(scope='module')
def model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    model.fit(x, y)
    return model


@pytest.fixture(scope='module')
def predicted(model, data):
    return model.predict(data[0])


def test_model_save_load(model, data, predicted):
    x, y = data
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "model")
    mlflow.keras.save_model(model, path)

    # Loading Keras model
    model_loaded = mlflow.keras.load_model(path)
    assert all(model_loaded.predict(x) == predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(path)
    assert all(pyfunc_loaded.predict(x).values == predicted)


def test_model_log(model, data, predicted):
    x, y = data
    old_uri = mlflow.get_tracking_uri()
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        tmp = tempfile.mkdtemp()
        try:
            print("SAVING TO %s" % tmp)
            mlflow.set_tracking_uri(tmp)
            if should_start_run:
                mlflow.start_run()
            mlflow.keras.log_model(model, artifact_path="keras_model")

            # Load model
            model_loaded = mlflow.keras.load_model(
                "keras_model",
                run_id=mlflow.active_run().info.run_uuid)
            assert all(model_loaded.predict(x) == predicted)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(
                "keras_model",
                run_id=mlflow.active_run().info.run_uuid)
            assert all(pyfunc_loaded.predict(x).values == predicted)
        finally:
            mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)
