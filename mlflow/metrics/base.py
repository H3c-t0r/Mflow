import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from mlflow.models import make_metric

_logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """
    The value of a metric.
    :param scores: The value of the metric per row
    :param justifications: The justification (if applicable) for the respective score
    :param aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    scores: List[float] = None
    justifications: List[float] = None
    aggregate_results: Dict[str, float] = None


def _validate_text_predictions(predictions, metric_name):
    if len(predictions) == 0:
        return False

    if any(not isinstance(prediction, str) for prediction in predictions):
        _logger.warning(
            f"Cannot calculate {metric_name} for non-string inputs, skipping metric logging."
        )
        return False

    return True


def _toxicity_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "toxicity"):
        return

    try:
        _logger.info("Loading toxicity metric:")
        import evaluate

        toxicity = evaluate.load("toxicity", module_type="measurement")
    except Exception as e:
        _logger.warning(
            f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging."
        )
        return

    _logger.info("Computing toxicity metric:")
    scores = toxicity.compute(predictions=predictions)["toxicity"]
    toxicity_ratio = toxicity.compute(predictions=predictions, aggregation="ratio")[
        "toxicity_ratio"
    ]
    return MetricValue(scores=scores, aggregate_results={"ratio": toxicity_ratio})


def _perplexity_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "perplexity"):
        return

    try:
        _logger.info("Loading perplexity metric:")
        import evaluate

        perplexity = evaluate.load("perplexity", module_type="metric")
    except Exception as e:
        _logger.warning(
            f"Failed to load 'perplexity' metric (error: {e!r}), skipping metric logging."
        )
        return

    _logger.info("Computing perplexity metric:")
    results = perplexity.compute(predictions=predictions, model_id="gpt2")
    return MetricValue(
        scores=results["perplexities"], aggregate_results={"mean": results["mean_perplexity"]}
    )


def _flesch_kincaid_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "flesch_kincaid"):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning("Failed to load flesch kincaid metric, skipping metric logging.")
        return

    _logger.info("Computing flesch kincaid metric:")
    scores = [textstat.flesch_kincaid_grade(prediction) for prediction in predictions]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def _ari_eval_fn(eval_df, metrics):
    y_pred = eval_df["prediction"]
    predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred

    if not _validate_text_predictions(predictions, "ari"):
        return

    try:
        import textstat
    except ImportError:
        _logger.warning(
            "Failed to load automated readability index metric, skipping metric logging."
        )
        return

    _logger.info("Computing automated readability index metric:")
    scores = [textstat.automated_readability_index(prediction) for prediction in predictions]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def _accuracy_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_true=eval_df["target"], y_pred=eval_df["prediction"])
        return MetricValue(aggregate_results={"": acc})


def _rouge1_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1"],
            use_aggregator=False,
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1"],
            use_aggregator=True,
        )
        return MetricValue(scores=scores["rouge1"], aggregate_results={"": aggregates["rouge1"]})


def _rouge2_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge2"],
            use_aggregator=False,
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge2"],
            use_aggregator=True,
        )
        return MetricValue(scores=scores["rouge2"], aggregate_results={"": aggregates["rouge2"]})


def _rougeL_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
            use_aggregator=False,
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
            use_aggregator=True,
        )
        return MetricValue(scores=scores["rougeL"], aggregate_results={"": aggregates["rougeL"]})


def _rougeLsum_eval_fn(eval_df, metrics):
    if "target" in eval_df:
        try:
            import evaluate

            rouge = evaluate.load("rouge")
        except Exception as e:
            _logger.warning(
                f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging."
            )
            return

        y_pred = eval_df["prediction"]
        predictions = y_pred.squeeze() if isinstance(y_pred, pd.DataFrame) else y_pred
        references = eval_df["target"]

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeLsum"],
            use_aggregator=False,
        )
        aggregates = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeLsum"],
            use_aggregator=True,
        )
        return MetricValue(
            scores=scores["rougeLsum"], aggregate_results={"": aggregates["rougeLsum"]}
        )


# general text metrics
toxicity = make_metric(
    eval_fn=_toxicity_eval_fn,
    greater_is_better=False,
    name="toxicity",
    long_name="toxicity/roberta-hate-speech-dynabench-r4",
    version="v1",
)

perplexity = make_metric(
    eval_fn=_perplexity_eval_fn,
    greater_is_better=False,
    name="perplexity",
    long_name="perplexity/gpt2",
    version="v1",
)

flesch_kincaid_grade_level = make_metric(
    eval_fn=_flesch_kincaid_eval_fn,
    greater_is_better=False,
    name="flesch_kincaid_grade_level",
    version="v1",
)

ari_grade_level = make_metric(
    eval_fn=_ari_eval_fn,
    greater_is_better=False,
    name="ari_grade_level",
    long_name="automated_readability_index_grade_level",
    version="v1",
)

# question answering metrics
accuracy = make_metric(
    eval_fn=_accuracy_eval_fn, greater_is_better=True, name="exact_match", version="v1"
)

# text summarization metrics
rouge1 = make_metric(
    eval_fn=_rouge1_eval_fn,
    greater_is_better=True,
    name="rouge1",
    version="v1",
)

rouge2 = make_metric(
    eval_fn=_rouge2_eval_fn,
    greater_is_better=True,
    name="rouge2",
    version="v1",
)

rougeL = make_metric(
    eval_fn=_rougeL_eval_fn,
    greater_is_better=True,
    name="rougeL",
    version="v1",
)

rougeLsum = make_metric(
    eval_fn=_rougeLsum_eval_fn,
    greater_is_better=True,
    name="rougeLsum",
    version="v1",
)
