import mlflow

from mlflow.evaluation import evaluate, EvaluationDataset
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pytest

from tests.sklearn.test_sklearn_autolog import get_iris, get_run_data, load_json_artifact
from sklearn.metrics import mean_absolute_error, mean_squared_error


@pytest.fixture(scope="module")
def regressor_model():
    X, y = get_iris()
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(X, y)
    return reg


@pytest.fixture(scope="module")
def classifier_model():
    X, y = get_iris()
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="module")
def evaluation_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    return EvaluationDataset(data=eval_X, labels=eval_y, name='eval_data_1')


def test_reg_evaluate(regressor_model, evaluation_dataset):
    y_true = evaluation_dataset.labels
    y_pred = regressor_model.predict(evaluation_dataset.data)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_metrics = {
        'mean_absolute_error': expected_mae,
        'mean_squared_error': expected_mse,
    }

    expected_artifact = expected_metrics

    with mlflow.start_run() as run:
        eval_result = evaluate(
            regressor_model, 'regressor', evaluation_dataset,
            run_id=None, evaluators='dummy_regressor_evaluator',
            evaluator_config={
                'can_evaluate': True,
                'metrics_to_calc': ['mean_absolute_error', 'mean_squared_error']
            }
        )
        saved_artifact_uri = mlflow.get_artifact_uri('metrics_artifact.json')
        saved_artifact = load_json_artifact('metrics_artifact.json')
        assert saved_artifact == expected_artifact

    _, saved_metrics, _, _ = get_run_data(run.info.run_id)
    assert saved_metrics == expected_metrics

    assert eval_result.metrics == expected_metrics
    assert eval_result.artifacts.content == expected_artifact
    assert eval_result.artifacts.location == saved_artifact_uri
