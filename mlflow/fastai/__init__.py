"""
The ``mlflow.fastai`` module provides an API for logging and loading fast.ai models. This module
exports fast.ai models with the following flavors:

fastai (native) format
    This is the main flavor that can be loaded back into fastai.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _fastai.Learner:
    https://docs.fast.ai/basic_train.html#Learner
.. _fastai.Learner.export:
    https://docs.fast.ai/basic_train.html#Learner.export
"""
import os
import yaml
import tempfile
import shutil
import pandas as pd
import numpy as np

from mlflow import pyfunc
from mlflow.models import Model, ModelSignature, ModelInputExample
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _save_example
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    try_mlflow_log,
    log_fn_args_as_params,
    safe_patch,
    batch_metrics_logger,
    autologging_integration,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "fastai"


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.


    .. code-block:: python
        :caption: Example

        import mlflow.fastai

        # Start MLflow session and log the fastai learner model
        with mlflow.start_run():
           model.fit(epochs, learning_rate)
           mlflow.fastai.log_model(model, "model")

        # Fetch the default conda environment
        env = mlflow.fastai.get_default_conda_env()
        print("conda environment: {}".format(env))

    .. code-block:: text
        :caption: Output

        conda environment: {'name': 'mlflow-env',
                            'channels': ['defaults', 'conda-forge', 'fastai',
                                         'pytorch'],
                            'dependencies': ['python=3.7.5', 'fastai=2.2.7',
                                             'pytorch=1.8.0, 'torchvision=0.9.0',
                                             'pip', {'pip': ['mlflow']}]}
    """

    import fastai
    import torch
    import torchvision

    pip_deps = None
    if include_cloudpickle:
        import cloudpickle

        pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _mlflow_conda_env(
        additional_conda_deps=[
            "fastai={}".format(fastai.__version__),
            "pytorch={}".format(torch.__version__),
            "torchvision={}".format(torchvision.__version__),
        ],
        additional_pip_deps=pip_deps,
        additional_conda_channels=["fastai", "pytorch"],
    )


def save_model(
    fastai_learner,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    **kwargs,
):
    """
    Save a fastai Learner to a path on the local file system.

    :param fastai_learner: fastai Learner to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the
                      dependencies contained in :func:`get_default_conda_env()`. If
                      ``None``, the default :func:`get_default_conda_env()` environment is
                      added to the model. The following is an *example* dictionary
                      representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'fastai=2.2.7',
                                'pytorch=1.8.0',
                                'torchvision=0.9.0'
                            ]
                        }
    :param mlflow_model: MLflow model config this flavor is being added to.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param kwargs: kwargs to pass to ``Learner.save`` method.

    .. code-block:: python
        :caption: Example

        import os

        import mlflow.fastai

        # Create a fastai Learner model
        model = ...

        # Start MLflow session and save model to current working directory
        with mlflow.start_run():
            model.fit(epochs, learning_rate)
            mlflow.fastai.save_model(model, 'model')

        # Load saved model for inference
        model_uri = "{}/{}".format(os.getcwd(), 'model')
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    import fastai
    from fastai.callback.all import ParamScheduler
    from pathlib import Path

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.fastai"
    model_data_path = os.path.join(path, model_data_subpath)
    model_data_path = Path(model_data_path)
    os.makedirs(path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # ParamScheduler currently is not pickable
    # hence it is been removed before export and added again after export
    def pop_not_pickle_cbs():
        cbs = []
        i = 0
        while i < len(fastai_learner.cbs):
            cb = fastai_learner.cbs[i]
            if isinstance(cb, ParamScheduler):
                cbs.append(cb)
            i = i + 1
        fastai_learner.remove_cbs(cbs)
        return cbs

    cbs = pop_not_pickle_cbs()

    # Save an Learner
    fastai_learner.export(model_data_path, **kwargs)

    fastai_learner.add_cbs(cbs)

    conda_env_subpath = "conda.yaml"

    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.fastai",
        data=model_data_subpath,
        env=conda_env_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME, fastai_version=fastai.__version__, data=model_data_subpath
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    fastai_learner,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    **kwargs,
):
    """
    Log a fastai model as an MLflow artifact for the current run.

    :param fastai_learner: Fastai model (an instance of `fastai.Learner`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'fastai=1.0.60',
                            ]
                        }
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param kwargs: kwargs to pass to `fastai.Learner.export`_ method.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    .. code-block:: python
        :caption: Example

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow.tracking import MlflowClient

        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)

           # Prepare, transform, and normalize the data
           data = vis.ImageDataBunch.from_folder(path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64)
           data.normalize(vis.imagenet_stats)

           # Create the CNN Learner model
           model = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

           # Start MLflow session and log model
           with mlflow.start_run() as run:
                model.fit(epochs, learning_rate)
                mlflow.fastai.log_model(model, 'model')

           # fetch the logged model artifacts
           artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id, 'model')]
           print("artifacts: {}".format(artifacts))

        main()

    .. code-block:: text
        :caption: Output

        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.fastai']
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.fastai,
        registered_model_name=registered_model_name,
        fastai_learner=fastai_learner,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        **kwargs,
    )


def _load_model(path):
    from fastai.learner import load_learner

    abspath = os.path.abspath(path)
    return load_learner(abspath)


class _FastaiModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, dataframe):
        dl = self.learner.dls.test_dl(dataframe)
        preds, _ = self.learner.get_preds(dl=dl)
        preds = pd.DataFrame(map(np.array, preds.numpy()), columns=["predictions"])
        return preds


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``fastai`` flavor.
    """
    return _FastaiModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load a fastai model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A fastai model (an instance of `fastai.Learner`_).

    .. code-block:: python
        :caption: Example

        import mlflow.fastai

        # Define the Learner model
        model = ...

        # log the fastai Leaner model
        with mlflow.start_run() as run:
            model.fit(epochs, learning_rate)
            mlflow.fastai.log_model(model, "model")

        # Load the model for scoring
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    model_file_path = os.path.join(
        local_model_path, flavor_conf.get("data", "model.fastai")
    )
    return _load_model(path=model_file_path)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enable automatic logging from Fastai to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    MLflow will also log the parameters of the
    `EarlyStoppingCallback <https://docs.fast.ai/callback.tracker.html#EarlyStoppingCallback>`_
    and `OneCycleScheduler <https://docs.fast.ai/callback.schedule.html#ParamScheduler>`_ callbacks

    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the Fastai autologging integration. If ``False``,
                    enables the Fastai autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      fastai that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Fastai
                   autologging. If ``False``, show all events and warnings during Fastai
                   autologging.

    .. code-block:: python
        :caption: Example

        # This is a modified example from
        # https://github.com/mlflow/mlflow/tree/master/examples/fastai
        # demonstrating autolog capabilites.

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow.tracking import MlflowClient

        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))

        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)

            # Prepare, transform, and normalize the data
            data = vis.ImageDataBunch.from_folder(path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64)
            data.normalize(vis.imagenet_stats)

            # Create CNN the Learner model
            model = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

            # Enable auto logging
            mlflow.fastai.autolog()

            # Start MLflow session
            with mlflow.start_run() as run:
                model.fit(epochs, learning_rate)

            # fetch the auto logged parameters, metrics, and artifacts
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        main()

    .. code-block:: text
        :caption: output

        run_id: 5a23dcbcaa334637814dbce7a00b2f6a
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.fastai']
        params: {'wd': 'None',
                 'bn_wd': 'True',
                 'opt_func': 'Adam',
                 'epochs': '5', '
                 train_bn': 'True',
                 'num_layers': '60',
                 'lr': '0.01',
                 'true_wd': 'True'}
        metrics: {'train_loss': 0.024,
                  'accuracy': 0.99214,
                  'valid_loss': 0.021}
        # Tags model summary omitted too long
        tags: {...}

    .. figure:: ../_static/images/fastai_autolog.png

        Fastai autologged MLflow entities
    """
    from fastai.learner import Learner
    from fastai.callback.hook import module_summary, layer_info, find_bs, _print_shapes
    from fastai.callback.all import EarlyStoppingCallback, TrackerCallback

    def getFastaiCallback(metrics_logger, is_fine_tune=False):
        from mlflow.fastai.callback import __MLflowFastaiCallback

        return __MLflowFastaiCallback(
            metrics_logger=metrics_logger,
            log_models=log_models,
            is_fine_tune=is_fine_tune,
        )

    def _find_callback_of_type(callback_type, callbacks):
        for callback in callbacks:
            if isinstance(callback, callback_type):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {
                    "early_stop_monitor": callback.monitor,
                    "early_stop_min_delta": callback.min_delta,
                    "early_stop_patience": callback.patience,
                    "early_stop_comp": callback.comp.__name__,
                }
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _log_model_info(learner):
        # The process excuted here, are incompatible with TrackerCallback
        # Hence it is removed and add again after the execution
        remove_cbs = [cb for cb in learner.cbs if isinstance(cb, TrackerCallback)]
        if len(remove_cbs):
            learner.remove_cbs(remove_cbs)

        xb = learner.dls.train.one_batch()[: learner.dls.train.n_inp]
        infos = layer_info(learner, *xb)
        bs = find_bs(xb)
        inp_sz = _print_shapes(map(lambda x: x.shape, xb), bs)
        try_mlflow_log(mlflow.log_param, "input_size", inp_sz)
        try_mlflow_log(mlflow.log_param, "num_layers", len(infos))

        summary = module_summary(learner, *xb)

        # Add again TrackerCallback
        if len(remove_cbs):
            learner.add_cbs(remove_cbs)

        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, "module_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)
            try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def _run_and_log_function(
        self, original, args, kwargs, unlogged_params, is_fine_tune=False
    ):

        # Check if is trying to fit while fine tuning or not
        mlflow_cbs = [cb for cb in self.cbs if cb.name == "__m_lflow_fastai"]
        fit_in_fine_tune = (
            original.__name__ == "fit"
            and len(mlflow_cbs) > 0
            and mlflow_cbs[0].is_fine_tune
        )

        if not fit_in_fine_tune:
            log_fn_args_as_params(original, list(args), kwargs, unlogged_params)

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:

            if not fit_in_fine_tune:
                early_stop_callback = _find_callback_of_type(
                    EarlyStoppingCallback, self.cbs
                )
                _log_early_stop_callback_params(early_stop_callback)

                # First try to remove if any already registered callback
                self.remove_cbs(mlflow_cbs)

                # Log information regarding model and data without bar and print-out
                with self.no_bar(), self.no_logging():
                    try_mlflow_log(_log_model_info, learner=self)

                mlflowFastaiCallback = getFastaiCallback(
                    metrics_logger=metrics_logger, is_fine_tune=is_fine_tune
                )

                # Add the new callback
                self.add_cb(mlflowFastaiCallback)

            result = original(self, *args, **kwargs)

        return result

    def fit(original, self, *args, **kwargs):
        unlogged_params = ["self", "cbs", "learner", "lr", "lr_max", "wd"]
        return _run_and_log_function(
            self, original, args, kwargs, unlogged_params, is_fine_tune=False
        )

    safe_patch(FLAVOR_NAME, Learner, "fit", fit, manage_run=True)

    def fine_tune(original, self, *args, **kwargs):
        unlogged_params = ["self", "cbs", "learner", "lr", "lr_max", "wd"]
        return _run_and_log_function(
            self, original, args, kwargs, unlogged_params, is_fine_tune=True
        )

    safe_patch(FLAVOR_NAME, Learner, "fine_tune", fine_tune, manage_run=True)
