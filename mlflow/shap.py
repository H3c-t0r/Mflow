from contextlib import contextmanager
import os
import tempfile

import numpy as np

import mlflow
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import append_to_uri_path


_BACKGROUND_DATA_SIZE = 100
_DEFAULT_ARTIFACT_PATH = "model_explanations_shap"
_SUMMARY_BAR_PLOT_FILE_NAME = "summary_bar_plot.png"
_BASE_VALUES_FILE_NAME = "base_values.npy"
_SHAP_VALUES_FILE_NAME = "shap_values.npy"


@contextmanager
def _log_artifact_contextmanager(out_file, artifact_path=None):
    """
    A context manager to make it easier to log an artifact.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, out_file)
        yield tmp_path
        mlflow.log_artifact(tmp_path, artifact_path)


def _log_numpy(numpy_obj, out_file, artifact_path=None):
    """
    Log a numpy object.
    """
    with _log_artifact_contextmanager(out_file, artifact_path) as tmp_path:
        np.save(tmp_path, numpy_obj)


def _log_matplotlib_figure(fig, out_file, artifact_path=None):
    """
    Log a matplotlib figure.
    """
    with _log_artifact_contextmanager(out_file, artifact_path) as tmp_path:
        fig.savefig(tmp_path)


@experimental
def log_explanation(predict_function, features, artifact_path=None):
    """
    Given a ``predict_function`` capable of computing ML model output on the provided ``features``,
    computes and logs explanations of an ML model's output. Explanations are logged as a directory
    of artifacts containing the following items generated by SHAP:

        - Base values
        - SHAP values (computed using `shap.KerneExplainer`_)
        - Summary bar plot (shows the average impact of each feature on model output)

    .. _shap.KerneExplainer: https://github.com/slundberg/shap#kernelexplainer

    :param predict_function:
        A function to compute the output of a model (e.g. ``predict`` method of
        scikit-learn regressors). Must have the following signature:

        .. code-block:: python

            def predict_function(X) -> pred:
                ...

        - ``X``: An array-like object whose shape should be (# samples,
          # features).
        - ``pred``: An array-like object whose shape should be (# samples) for
          a regressor or (# classes, # samples) for a classifier.

        Acceptable array-like object types:

            - ``numpy.array``
            - ``pandas.DataFrame``
            - ``shap.common.DenseData``
            - ``scipy.sparse matrix``

    :param features:
        A matrix of features to compute SHAP values with. The provided features should
        have shape (# samples, # features), and can be either of the array-like object
        types listed above.

    :param artifact_path:
        The run-relative artifact path to which the explanation is saved.
        If unspecified, defaults to "model_explanations_shap".

    :return: Artifact URI of the logged explanations

    .. code-block:: python
        :caption: Example

        import pandas as pd
        from sklearn.datasets import load_boston
        from sklearn.linear_model import LinearRegression

        import mlflow

        # prepare training data
        dataset = load_boston()
        X = pd.DataFrame(dataset.data[:, :4], columns=dataset.feature_names[:4])
        y = dataset.target

        # train a model
        model = LinearRegression()
        model.fit(X, y)

        # log an explanation
        with mlflow.start_run() as run:
            mlflow.shap.log_explanation(model.predict, X)

        # print out logged items
        client = mlflow.tracking.MlflowClient()
        artifact_path = "model_explanations_shap"
        artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
        print(artifacts)

    .. code-block:: text
        :caption: Output

        ['shap/base_values.npy', 'shap/shap_values.npy', 'shap/summary_bar_plot.png']

    """
    import matplotlib.pyplot as plt
    import shap

    artifact_path = _DEFAULT_ARTIFACT_PATH if artifact_path is None else artifact_path
    background_data = shap.kmeans(features, min(_BACKGROUND_DATA_SIZE, len(features)))
    explainer = shap.KernelExplainer(predict_function, background_data)
    shap_values = explainer.shap_values(features)

    _log_numpy(explainer.expected_value, _BASE_VALUES_FILE_NAME, artifact_path)
    _log_numpy(shap_values, _SHAP_VALUES_FILE_NAME, artifact_path)

    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    fig = plt.gcf()
    _log_matplotlib_figure(fig, _SUMMARY_BAR_PLOT_FILE_NAME, artifact_path)
    plt.close(fig)

    return append_to_uri_path(mlflow.active_run().info.artifact_uri, artifact_path)
