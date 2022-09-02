"""
The ``mlflow.tensorflow`` module provides an API for logging and loading TensorFlow models.
This module exports TensorFlow models with the following flavors:

TensorFlow (native) format
    This is the main flavor that can be loaded back into TensorFlow.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import os
import shutil
import logging
import concurrent.futures
import warnings
import atexit
import tempfile
from collections import namedtuple
import pandas
from packaging.version import Version
from threading import RLock
import numpy as np

import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import is_iterator
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _add_code_from_conf_to_system_path,
)
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    resolve_input_example_and_signature,
    picklable_exception_safe_function,
    PatchFunction,
    log_fn_args_as_params,
    batch_metrics_logger,
    get_autologging_config,
    AUTOLOGGING_CONF_KEY_IS_GLOBALLY_CONFIGURED,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.models import infer_signature
from mlflow.tensorflow import keras as mlflow_keras

FLAVOR_NAME = "tensorflow"

_logger = logging.getLogger(__name__)

_MAX_METRIC_QUEUE_SIZE = 500

_LOG_EVERY_N_STEPS = 1

_metric_queue_lock = RLock()
_metric_queue = []

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# For tracking if the run was started by autologging.
_AUTOLOG_RUN_ID = None


def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("tensorflow"), _get_pinned_requirement("keras")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path,
    custom_objects=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a Keras model.

    :param model: The Keras model to be saved.
    :param artifact_path: The run-relative path to which to log model artifacts.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.tensorflow.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``model.save`` method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    return mlflow_keras._log_keras_model(
        artifact_path=artifact_path,
        keras_model=model,
        custom_objects=custom_objects,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path,
    custom_objects=None,
    mlflow_model=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Save a Keras model to a path on the local file system.

    :param model: The Keras model to be saved.
    :param path: Local path where the MLflow model is to be saved.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.tensorflow.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param mlflow_model: MLflow model configuration to which to add the ``tensorflow`` flavor.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``model.save`` method.
    """
    mlflow_keras._save_keras_model(
        keras_model=model,
        path=path,
        custom_objects=custom_objects,
        mlflow_model=mlflow_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load an MLflow model that contains the TensorFlow flavor from the specified path.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :param kwargs: kwargs to pass to ``keras.models.load_model`` method. Only available
                   when you are loading a Keras model.

    :return: A callable graph (tf.function) that takes inputs and returns inferences.

    .. code-block:: python
        :caption: Example

        import mlflow
        import tensorflow as tf
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            signature_definition = mlflow.tensorflow.load_model(model_uri="model_uri",
                                    tf_sess=tf_sess)
            input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                                for _, input_signature in signature_definition.inputs.items()]
            output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                                for _, output_signature in signature_definition.outputs.items()]
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)

    model_configuration_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)
    model_conf = Model.load(model_configuration_path)

    if "keras" in model_conf.flavors or "keras_module" in flavor_conf:
        return mlflow_keras._load_keras_model(local_model_path, flavor_conf, **kwargs)

    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    (
        tf_saved_model_dir,
        tf_meta_graph_tags,
        tf_signature_def_key,
    ) = _parse_flavor_configuration(flavor_conf, local_model_path)
    return _load_tensorflow_saved_model(
        tf_saved_model_dir=tf_saved_model_dir,
        tf_meta_graph_tags=tf_meta_graph_tags,
        tf_signature_def_key=tf_signature_def_key,
    )


def _load_tensorflow_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Load a specified TensorFlow model consisting of a TensorFlow metagraph and signature definition
    from a serialized TensorFlow ``SavedModel`` collection.

    :param tf_saved_model_dir: The local filesystem path or run-relative artifact path to the model.
    :param tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the `tf.saved_model.builder.SavedModelBuilder
                               method <https://www.tensorflow.org/api_docs/python/tf/saved_model/
                               builder/SavedModelBuilder#add_meta_graph>`_.
    :param tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``SavedModel``'s
                                 signature definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.SavedModelBuilder`` method.
    :return: A callable graph (tensorflow.function) that takes inputs and returns inferences.
    """
    import tensorflow

    loaded = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
        tags=tf_meta_graph_tags, export_dir=tf_saved_model_dir
    )
    loaded_sig = loaded.signatures
    if tf_signature_def_key not in loaded_sig:
        raise MlflowException(
            "Could not find signature def key %s. Available keys are: %s"
            % (tf_signature_def_key, list(loaded_sig.keys()))
        )
    return loaded_sig[tf_signature_def_key]


def _parse_flavor_configuration(flavor_conf, model_path):
    """
    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    :return: A triple containing the following elements:

             - ``tf_saved_model_dir``: The local filesystem path to the underlying TensorFlow
                                       SavedModel directory.
             - ``tf_meta_graph_tags``: A list of tags identifying the TensorFlow model's metagraph
                                       within the serialized ``SavedModel`` object.
             - ``tf_signature_def_key``: A string identifying the input/output signature associated
                                         with the model. This is a key within the serialized
                                         ``SavedModel``'s signature definition mapping.
    """
    tf_saved_model_dir = os.path.join(model_path, flavor_conf["saved_model_dir"])
    tf_meta_graph_tags = flavor_conf["meta_graph_tags"]
    tf_signature_def_key = flavor_conf["signature_def_key"]
    return tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``. This function loads an MLflow
    model with the TensorFlow flavor into a new TensorFlow graph and exposes it behind the
    ``pyfunc.predict`` interface.

    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    """
    import tensorflow

    if os.path.exists(os.path.join(path, mlflow_keras._KERAS_MODULE_SPEC_PATH)):
        return mlflow_keras._load_pyfunc(path)

    flavor_conf = _get_flavor_configuration(path, FLAVOR_NAME)
    (
        tf_saved_model_dir,
        tf_meta_graph_tags,
        tf_signature_def_key,
    ) = _parse_flavor_configuration(flavor_conf, path)

    loaded_model = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
        export_dir=tf_saved_model_dir, tags=tf_meta_graph_tags
    )
    return _TF2Wrapper(model=loaded_model, infer=loaded_model.signatures[tf_signature_def_key])


class _TF2Wrapper:
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``. For TensorFlow versions >= 2.0.0.
    """

    def __init__(self, model, infer):
        """
        :param model: A Tensorflow SavedModel.
        :param infer: Tensorflow function returned by a saved model that is used for inference.
        """
        # Note: we need to retain the model reference in TF2Wrapper object, because the infer
        #  function in tensorflow will be `ConcreteFunction` which only retains WeakRefs to the
        #  variables they close over.
        #  See https://www.tensorflow.org/guide/function#deleting_tfvariables_between_function_calls
        self.model = model
        self.infer = infer

    def predict(self, data):
        import tensorflow

        feed_dict = {}
        if isinstance(data, dict):
            feed_dict = {k: tensorflow.constant(v) for k, v in data.items()}
        elif isinstance(data, pandas.DataFrame):
            for df_col_name in list(data):
                # If there are multiple columns with the same name, selecting the shared name
                # from the DataFrame will result in another DataFrame containing the columns
                # with the shared name. TensorFlow cannot make eager tensors out of pandas
                # DataFrames, so we convert the DataFrame to a numpy array here.
                val = data[df_col_name]
                if isinstance(val, pandas.DataFrame):
                    val = val.values
                else:
                    val = np.array(val.to_list())
                feed_dict[df_col_name] = tensorflow.constant(val)
        else:
            raise TypeError("Only dict and DataFrame input types are supported")

        raw_preds = self.infer(**feed_dict)
        pred_dict = {col_name: raw_preds[col_name].numpy() for col_name in raw_preds.keys()}
        for col in pred_dict.keys():
            if all(len(element) == 1 for element in pred_dict[col]):
                pred_dict[col] = pred_dict[col].ravel()
            else:
                pred_dict[col] = pred_dict[col].tolist()

        if isinstance(data, dict):
            return pred_dict
        else:
            return pandas.DataFrame.from_dict(data=pred_dict)


def _assoc_list_to_map(lst):
    """
    Convert an association list to a dictionary.
    """
    d = {}
    for run_id, metric in lst:
        d[run_id] = d[run_id] + [metric] if run_id in d else [metric]
    return d


def _flush_queue():
    """
    Flush the metric queue and log contents in batches to MLflow.
    Queue is divided into batches according to run id.
    """
    try:
        # Multiple queue flushes may be scheduled simultaneously on different threads
        # (e.g., if the queue is at its flush threshold and several more items
        # are added before a flush occurs). For correctness and efficiency, only one such
        # flush operation should proceed; all others are redundant and should be dropped
        acquired_lock = _metric_queue_lock.acquire(blocking=False)
        if acquired_lock:
            client = MlflowClient()
            # For thread safety and to avoid modifying a list while iterating over it, we record a
            # separate list of the items being flushed and remove each one from the metric queue,
            # rather than clearing the metric queue or reassigning it (clearing / reassigning is
            # dangerous because we don't block threads from adding to the queue while a flush is
            # in progress)
            snapshot = _metric_queue[:]
            for item in snapshot:
                _metric_queue.remove(item)

            metrics_by_run = _assoc_list_to_map(snapshot)
            for run_id, metrics in metrics_by_run.items():
                client.log_batch(run_id, metrics=metrics, params=[], tags=[])
    finally:
        if acquired_lock:
            _metric_queue_lock.release()


@picklable_exception_safe_function
def _get_tensorboard_callback(lst):
    import tensorflow

    for x in lst:
        if isinstance(x, tensorflow.keras.callbacks.TensorBoard):
            return x
    return None


# A representation of a TensorBoard event logging directory with two attributes:
# :location - string: The filesystem location of the logging directory
# :is_temp - boolean: `True` if the logging directory was created for temporary use by MLflow,
#                     `False` otherwise
_TensorBoardLogDir = namedtuple("_TensorBoardLogDir", ["location", "is_temp"])


def _setup_callbacks(lst, metrics_logger):
    """
    Adds TensorBoard and MlfLowTfKeras callbacks to the
    input list, and returns the new list and appropriate log directory.
    """
    # pylint: disable=no-name-in-module
    from mlflow.tensorflow._autolog import _TensorBoard, __MLflowTfKeras2Callback

    tb = _get_tensorboard_callback(lst)
    if tb is None:
        log_dir = _TensorBoardLogDir(location=tempfile.mkdtemp(), is_temp=True)

        out_list = lst + [_TensorBoard(log_dir.location)]
    else:
        log_dir = _TensorBoardLogDir(location=tb.log_dir, is_temp=False)
        out_list = lst
    out_list += [__MLflowTfKeras2Callback(metrics_logger, _LOG_EVERY_N_STEPS)]
    return out_list, log_dir


@autologging_integration(FLAVOR_NAME)
def autolog(
    every_n_iter=1,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    log_input_examples=False,
    log_model_signatures=False,
):  # pylint: disable=unused-argument
    # pylint: disable=E0611
    """
    Enables automatic logging from TensorFlow to MLflow.
    Note that autologging for ``tf.keras`` and ``keras`` are also
    handled by :py:func:`mlflow.tensorflow.autolog`.
    As an example, try running the
    `TensorFlow examples <https://github.com/mlflow/mlflow/tree/master/examples/tensorflow>`_.

    For each TensorFlow module, autologging captures the following information:

    **tf.keras**
     - **Metrics** and **Parameters**

      - Training loss; validation loss; user-specified metrics
      - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon

     - **Artifacts**

      - Model summary on training start
      - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model)
      - TensorBoard logs on training end

    **tf.keras.callbacks.EarlyStopping**
     - **Metrics** and **Parameters**

      - Metrics from the ``EarlyStopping`` callbacks: ``stopped_epoch``, ``restored_epoch``,
        ``restore_best_weight``, etc
      - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``:
        ``min_delta``, ``patience``, ``baseline``, ``restore_best_weights``, etc

    **tf.estimator**
     - **Metrics** and **Parameters**

      - TensorBoard metrics: ``average_loss``, ``loss``, etc
      - Parameters ``steps`` and ``max_steps``

     - **Artifacts**

      - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (TF saved model) on call
        to ``tf.estimator.export_saved_model``

    **TensorFlow Core**
     - **Metrics**

      - All ``tf.summary.scalar`` calls

    Refer to the autologging tracking documentation for more
    information on `TensorFlow workflows
    <https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras-experimental>`_.

    :param every_n_iter: The frequency with which metrics should be logged. For example, a value of
                         100 will log metrics at step 0, 100, 200, etc.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the TensorFlow autologging integration. If ``False``,
                    enables the TensorFlow integration autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      tensorflow that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during TensorFlow
                   autologging. If ``False``, show all events and warnings during TensorFlow
                   autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with tf/keras model artifacts during training. If
                               ``False``, input examples are not logged.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with tf/keras model artifacts during training. If ``False``,
                                 signatures are not logged. ``False`` by default because
                                 logging TensorFlow models with signatures changes their pyfunc
                                 inference behavior when Pandas DataFrames are passed to
                                 ``predict()``: when a signature is present, an ``np.ndarray``
                                 (for single-output models) or a mapping from
                                 ``str`` -> ``np.ndarray`` (for multi-output models) is returned;
                                 when a signature is not present, a Pandas DataFrame is returned.
    """
    import tensorflow

    global _LOG_EVERY_N_STEPS
    _LOG_EVERY_N_STEPS = every_n_iter

    atexit.register(_flush_queue)

    if Version(tensorflow.__version__) < Version("2.6"):
        warnings.warn("Could not log to MLflow. TensorFlow versions below 1.12 are not supported.")
        return

    def _should_log_model_signatures():
        return (
            log_model_signatures
            and
            # `log_model_signatures` is `False` by default for
            # `mlflow.tensorflow.autolog()` in order to to preserve
            # backwards-compatible inference behavior with older versions of MLflow
            # that did not support signature autologging for TensorFlow (
            # unfortunately, adding a signature to a TensorFlow model has the
            # unintended consequence of changing the output type produced by
            # inference with pyfunc `predict()` for Pandas DataFrame inputs).
            # However, `log_model_signatures` is `True` by default for
            # `mlflow.autolog()`. To ensure that we maintain backwards compatibility
            # when TensorFlow autologging is enabled via `mlflow.autolog()`,
            # we only enable signature logging if `mlflow.tensorflow.autolog()` is
            # called explicitly with `log_model_signatures=True`
            not get_autologging_config(
                FLAVOR_NAME, AUTOLOGGING_CONF_KEY_IS_GLOBALLY_CONFIGURED, False
            )
        )

    @picklable_exception_safe_function
    def _get_early_stop_callback(callbacks):
        for callback in callbacks:
            if isinstance(callback, tensorflow.keras.callbacks.EarlyStopping):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {
                    "monitor": callback.monitor,
                    "min_delta": callback.min_delta,
                    "patience": callback.patience,
                    "baseline": callback.baseline,
                    "restore_best_weights": callback.restore_best_weights,
                }
                mlflow.log_params(earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history, metrics_logger):
        if callback is None or not callback.model.stop_training:
            return

        callback_attrs = _get_early_stop_callback_attrs(callback)
        if callback_attrs is None:
            return

        stopped_epoch, restore_best_weights, _ = callback_attrs
        metrics_logger.record_metrics({"stopped_epoch": stopped_epoch})

        if not restore_best_weights or callback.best_weights is None:
            return

        monitored_metric = history.history.get(callback.monitor)
        if not monitored_metric:
            return

        initial_epoch = history.epoch[0]
        # If `monitored_metric` contains multiple best values (e.g. [0.1, 0.1, 0.2] where 0.1 is
        # the minimum loss), the epoch corresponding to the first occurrence of the best value is
        # the best epoch. In keras > 2.6.0, the best epoch can be obtained via the `best_epoch`
        # attribute of an `EarlyStopping` instance: https://github.com/keras-team/keras/pull/15197
        restored_epoch = initial_epoch + monitored_metric.index(callback.best)
        metrics_logger.record_metrics({"restored_epoch": restored_epoch})
        restored_index = history.epoch.index(restored_epoch)
        restored_metrics = {
            key: metrics[restored_index] for key, metrics in history.history.items()
        }
        # Checking that a metric history exists
        metric_key = next(iter(history.history), None)
        if metric_key is not None:
            metrics_logger.record_metrics(restored_metrics, stopped_epoch + 1)

    def _log_keras_model(history, args):
        def _infer_model_signature(input_data_slice):
            # In certain TensorFlow versions, calling `predict()` on model  may modify
            # the `stop_training` attribute, so we save and restore it accordingly
            original_stop_training = history.model.stop_training
            model_output = history.model.predict(input_data_slice)
            history.model.stop_training = original_stop_training
            return infer_signature(input_data_slice, model_output)

        from mlflow.tensorflow._autolog import extract_tf_keras_input_example

        def _get_tf_keras_input_example_slice():
            from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

            input_training_data = args[0]
            keras_input_example_slice = extract_tf_keras_input_example(input_training_data)
            if keras_input_example_slice is None:
                raise MlflowException(
                    "Cannot log input example or model signature for input with type"
                    f" {type(input_training_data)}. TensorFlow Keras autologging can"
                    " only log input examples and model signatures for the following"
                    " input types: numpy.ndarray, dict[string -> numpy.ndarray],"
                    " tensorflow.keras.utils.Sequence, and"
                    " tensorflow.data.Dataset (TensorFlow >= 2.1.0 required)",
                    INVALID_PARAMETER_VALUE,
                )
            return keras_input_example_slice

        input_example, signature = resolve_input_example_and_signature(
            _get_tf_keras_input_example_slice,
            _infer_model_signature,
            log_input_examples,
            _should_log_model_signatures(),
            _logger,
        )

        log_model(
            model=history.model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            registered_model_name=get_autologging_config(
                FLAVOR_NAME, "registered_model_name", None
            ),
        )

    class FitPatch(PatchFunction):
        def __init__(self):
            self.log_dir = None

        def _patch_implementation(
            self, original, inst, *args, **kwargs
        ):  # pylint: disable=arguments-differ
            unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]

            batch_size = None
            try:
                training_data = kwargs["x"] if "x" in kwargs else args[0]
                if isinstance(training_data, tensorflow.data.Dataset) and hasattr(
                    training_data, "_batch_size"
                ):
                    batch_size = training_data._batch_size.numpy()
                elif isinstance(training_data, tensorflow.keras.utils.Sequence):
                    first_batch_inputs, _ = training_data[0]
                    batch_size = len(first_batch_inputs)
                elif is_iterator(training_data):
                    peek = next(training_data)
                    batch_size = len(peek[0])

                    def __restore_generator(prev_generator):
                        yield peek
                        yield from prev_generator

                    restored_generator = __restore_generator(training_data)
                    if "x" in kwargs:
                        kwargs["x"] = restored_generator
                    else:
                        args = (restored_generator,) + args[1:]
            except Exception as e:
                _logger.warning(
                    "Encountered unexpected error while inferring batch size from training"
                    " dataset: %s",
                    e,
                )

            if batch_size is not None:
                mlflow.log_param("batch_size", batch_size)
                unlogged_params.append("batch_size")

            log_fn_args_as_params(original, args, kwargs, unlogged_params)

            run_id = mlflow.active_run().info.run_id
            with batch_metrics_logger(run_id) as metrics_logger:
                # Check if the 'callback' argument of fit() is set positionally
                if len(args) >= 6:
                    # Convert the positional training function arguments to a list in order to
                    # mutate the contents
                    args = list(args)
                    # Make a shallow copy of the preexisting callbacks to avoid permanently
                    # modifying their contents for future training invocations. Introduce
                    # TensorBoard & tf.keras callbacks if necessary
                    callbacks = list(args[5])
                    callbacks, self.log_dir = _setup_callbacks(callbacks, metrics_logger)
                    # Replace the callbacks positional entry in the copied arguments and convert
                    # the arguments back to tuple form for usage in the training function
                    args[5] = callbacks
                    args = tuple(args)
                else:
                    # Make a shallow copy of the preexisting callbacks and introduce TensorBoard
                    # & tf.keras callbacks if necessary
                    callbacks = list(kwargs.get("callbacks") or [])
                    kwargs["callbacks"], self.log_dir = _setup_callbacks(callbacks, metrics_logger)

                early_stop_callback = _get_early_stop_callback(callbacks)
                _log_early_stop_callback_params(early_stop_callback)

                history = original(inst, *args, **kwargs)

                if log_models:
                    _log_keras_model(history, args)

                _log_early_stop_callback_metrics(
                    callback=early_stop_callback,
                    history=history,
                    metrics_logger=metrics_logger,
                )

                _flush_queue()
                mlflow.log_artifacts(
                    local_dir=self.log_dir.location,
                    artifact_path="tensorboard_logs",
                )
            if self.log_dir.is_temp:
                shutil.rmtree(self.log_dir.location)
            return history

        def _on_exception(self, exception):
            if (
                self.log_dir is not None
                and self.log_dir.is_temp
                and os.path.exists(self.log_dir.location)
            ):
                shutil.rmtree(self.log_dir.location)

    managed = [
        (tensorflow.keras.Model, "fit", FitPatch),
    ]

    for p in managed:
        safe_patch(FLAVOR_NAME, *p, manage_run=True)
