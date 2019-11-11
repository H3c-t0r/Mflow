import os

import gorilla
from mxnet import gluon
from mxnet import sym
from mxnet.gluon.contrib.estimator import Estimator, TrainEnd, EpochEnd
from mxnet.gluon.nn import HybridSequential

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import experimental
from mlflow.utils.autologging_utils import try_mlflow_log

FLAVOR_NAME = "gluon"
_MODEL_SAVE_PATH = "net"


@experimental
def load_model(model_uri, ctx):
    """
    Load a Gluon model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param ctx: Either CPU or GPU

    :return: A Gluon model instance.

    >>> # Load persisted model as a Gluon model, make inferences against an NDArray
    >>> model = mlflow.gluon.load_model("runs:/" + gluon_random_data_run.info.run_id + "/model")
    >>> model(nd.array(np.random.rand(1000, 1, 32)))
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)

    model_arch_path = os.path.join(local_model_path, "data", _MODEL_SAVE_PATH) + "-symbol.json"
    model_params_path = os.path.join(local_model_path, "data", _MODEL_SAVE_PATH) + "-0000.params"
    symbol = sym.load(model_arch_path)
    inputs = sym.var('data', dtype='float32')
    net = gluon.SymbolBlock(symbol, inputs)
    net.collect_params().load(model_params_path, ctx)
    return net


@experimental
def save_model(gluon_model, path, mlflow_model=Model()):
    """
    Save a Gluon model to a path on the local file system.

    :param gluon_model: Gluon model to be saved. Must be already hybridized.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.

    >>> from mxnet.gluon import Trainer
    >>> from mxnet.gluon.contrib import estimator
    >>> from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
    >>> from mxnet.gluon.nn import HybridSequential
    >>> from mxnet.metric import Accuracy
    >>> import mlflow
    >>> # Build, compile, and train your model
    >>> gluon_model_path = ...
    >>> net = HybridSequential()
    >>> with net.name_scope():
    >>> ...
    >>> net.hybridize()
    >>> net.collect_params().initialize()
    >>> softmax_loss = SoftmaxCrossEntropyLoss()
    >>> trainer = Trainer(net.collect_params())
    >>> est = estimator.Estimator(net=net, loss=softmax_loss, metrics=Accuracy(), trainer=trainer)
    >>> est.fit(train_data=train_data, epochs=100, val_data=validation_data)
    ... # Save the model as an MLflow Model
    >>> mlflow.gluon.save_model(net, gluon_model_path)
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)
    # The epoch argument of the export method does not play any role in selecting
    # a specific epoch's paramaters, and is there only for display purposes.
    gluon_model.export(os.path.join(data_path, _MODEL_SAVE_PATH))
    with open(os.path.join(path, "architecture.txt"), "w") as fp:
        fp.write(str(gluon_model))
    mlflow_model.save(os.path.join(path, "MLmodel"))


@experimental
def log_model(gluon_model, artifact_path):
    """
    Log a Gluon model as an MLflow artifact for the current run.

    :param gluon_model: Gluon model to be saved. Must be already hybridized.
    :param artifact_path: Run-relative artifact path.

    >>> from mxnet.gluon import Trainer
    >>> from mxnet.gluon.contrib import estimator
    >>> from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
    >>> from mxnet.gluon.nn import HybridSequential
    >>> from mxnet.metric import Accuracy
    >>> import mlflow
    >>> # Build, compile, and train your model
    >>> net = HybridSequential()
    >>> with net.name_scope():
    >>> ...
    >>> net.hybridize()
    >>> net.collect_params().initialize()
    >>> softmax_loss = SoftmaxCrossEntropyLoss()
    >>> trainer = Trainer(net.collect_params())
    >>> est = estimator.Estimator(net=net, loss=softmax_loss, metrics=Accuracy(), trainer=trainer)
    >>> # Log metrics and log the model
    >>> with mlflow.start_run() as run:
    >>>   est.fit(train_data=train_data, epochs=100, val_data=validation_data)
    >>>   mlflow.gluon.log_model(net, "model")
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.gluon, gluon_model=gluon_model)


@experimental
def autolog():
    """
    Enable automatic logging from Gluon to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.
    """

    class __MLflowGluonCallback(TrainEnd, EpochEnd):
        def __init__(self):
            self.current_epoch = 0

        def epoch_end(self, estimator, *args, **kwargs):
            logs = {}
            for metric in estimator.train_metrics:
                metric_name, metric_val = metric.get()
                logs[metric_name] = metric_val
            for metric in estimator.val_metrics:
                metric_name, metric_val = metric.get()
                logs[metric_name] = metric_val
            try_mlflow_log(mlflow.log_metrics, logs, step=self.current_epoch)
            self.current_epoch += 1

        def train_end(self, estimator, *args, **kwargs):
            try_mlflow_log(mlflow.log_param, "num_layers", len(estimator.net))
            if estimator.max_epoch is not None:
                try_mlflow_log(mlflow.log_param, "epochs", estimator.max_epoch)
            if estimator.max_batch is not None:
                try_mlflow_log(mlflow.log_param, "batches", estimator.max_batch)
            try_mlflow_log(mlflow.log_param, "optimizer_name",
                           type(estimator.trainer.optimizer).__name__)
            if hasattr(estimator.trainer.optimizer, "lr"):
                try_mlflow_log(mlflow.log_param, "learning_rate",
                               estimator.trainer.optimizer.lr)
            if hasattr(estimator.trainer.optimizer, "epsilon"):
                try_mlflow_log(mlflow.log_param, "epsilon",
                               estimator.trainer.optimizer.epsilon)
            if isinstance(estimator.net, HybridSequential):
                try_mlflow_log(log_model, estimator.net, artifact_path="model")

    @gorilla.patch(Estimator)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(Estimator, "fit")
        if len(args) >= 4:
            l = list(args)
            l[3] += [__MLflowGluonCallback()]
            args = tuple(l)
        elif "event_handlers" in kwargs:
            kwargs["event_handlers"] += [__MLflowGluonCallback()]
        else:
            kwargs["event_handlers"] = [__MLflowGluonCallback()]
        return original(self, *args, **kwargs)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(Estimator, "fit", fit, settings=settings))
