import random
import warnings

import mxnet.ndarray as nd
import numpy as np
import pytest
from mxnet.gluon import Trainer
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense
from mxnet.metric import Accuracy

import mlflow
import mlflow.gluon

client = mlflow.tracking.MlflowClient()


class LogsDataset(Dataset):
    def __init__(self):
        self.len = 1000

    def __getitem__(self, idx):
        return nd.array(np.random.rand(1, 32)), nd.full(1, random.randint(0, 10), dtype="float32")

    def __len__(self):
        return self.len


@pytest.fixture
def gluon_random_data_run():
    mlflow.gluon.autolog()

    with mlflow.start_run() as run:
        data = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")

        model = HybridSequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10))
        model.initialize()
        model.hybridize()
        trainer = Trainer(model.collect_params(), "adam",
                          optimizer_params={"learning_rate": .001, "epsilon": 1e-07})
        est = estimator.Estimator(net=model, loss=SoftmaxCrossEntropyLoss(),
                                  metrics=Accuracy(), trainer=trainer)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(data, epochs=5)

    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_gluon_autolog_logs_expected_data(gluon_random_data_run):
    data = gluon_random_data_run.data
    assert "train accuracy" in data.metrics
    assert "train softmaxcrossentropyloss" in data.metrics
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"
    assert "epsilon" in data.params
    assert data.params["epsilon"] == "1e-07"


@pytest.mark.large
def test_gluon_autolog_model_can_load_from_artifact(gluon_random_data_run):
    artifacts = client.list_artifacts(gluon_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    model = mlflow.gluon.load_model("runs:/" + gluon_random_data_run.info.run_id + "/model")
    model(nd.array(np.random.rand(1000, 1, 32)))
