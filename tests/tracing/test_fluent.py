import time
from datetime import datetime
from unittest import mock

import pytest

import mlflow
from mlflow.entities import SpanStatusCode, SpanType, Trace, TraceData, TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import TraceMetadataKey


def test_trace(mock_client):
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1}
        )
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    trace = mlflow.get_traces()[0]
    trace_info = trace.info
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == "0"  # default experiment
    assert trace_info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace_info.status == SpanStatusCode.OK
    assert trace_info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace_info.request_metadata[TraceMetadataKey.OUTPUTS] == "64"

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response == "64"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }

    child_span_1 = span_name_to_span["add_one_with_custom_name"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "square",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 8},
        "mlflow.spanOutputs": 64,
    }


def test_trace_handle_exception_during_prediction(mock_client):
    # This test is to make sure that the exception raised by the main prediction
    # logic is raised properly and the trace is still logged.
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return self.some_operation_raise_error(x, y)

        @mlflow.trace()
        def some_operation_raise_error(self, x, y):
            raise ValueError("Some error")

    model = TestModel()

    with pytest.raises(ValueError, match=r"Some error"):
        model.predict(2, 5)

    # Trace should be logged even if the function fails, with status code ERROR
    trace = mlflow.get_traces()[0]
    assert trace.info.request_id is not None
    assert trace.info.status == SpanStatusCode.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == ""

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response is None
    assert len(trace.data.spans) == 2


def test_trace_ignore_exception_from_tracing_logic(mock_client):
    # This test is to make sure that the main prediction logic is not affected
    # by the exception raised by the tracing logic.
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # Exception during span creation: no-op span wrapper created and no trace is logged
    with mock.patch("mlflow.tracing.fluent.get_tracer", side_effect=ValueError("Some error")):
        output = model.predict(2, 5)

    assert output == 7
    assert mlflow.get_traces() == []

    # Exception during inspecting inputs: trace is logged without inputs field
    with mock.patch("mlflow.tracing.utils.inspect.signature", side_effect=ValueError("Some error")):
        output = model.predict(2, 5)

    assert output == 7
    trace = mlflow.get_traces()[0]
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == "{}"
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "7"


def test_start_span_context_manager(mock_client):
    datetime_now = datetime.now()

    class TestModel:
        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                with mlflow.start_span(name="child_span", span_type=SpanType.LLM) as child_span:
                    child_span.set_inputs(z)
                    z = z + 2
                    child_span.set_outputs(z)
                    child_span.set_attributes({"delta": 2, "time": datetime_now})

                res = self.square(z)
                root_span.set_outputs(res)
            return res

        def square(self, t):
            with mlflow.start_span(name="child_span") as span:
                span.set_inputs({"t": t})
                res = t**2
                time.sleep(0.1)
                span.set_outputs(res)
                return res

    model = TestModel()
    model.predict(1, 2)

    trace = mlflow.get_traces()[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == SpanStatusCode.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "25"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "25"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace.info.execution_time_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 25,
    }

    # Span with duplicate name should be renamed to have an index number like "_1", "_2", ...
    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "time": str(datetime_now),
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": 25,
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


def test_start_span_context_manager_with_imperative_apis(mock_client):
    # This test is to make sure that the spans created with fluent APIs and imperative APIs
    # (via MLflow client) are correctly linked together. This usage is not recommended but
    # should be supported for the advanced use cases like using LangChain callbacks as a
    # part of broader tracing.
    class TestModel:
        def __init__(self):
            self._mlflow_client = mlflow.tracking.MlflowClient()

        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                child_span = self._mlflow_client.start_span(
                    name="child_span_1",
                    span_type=SpanType.LLM,
                    request_id=root_span.request_id,
                    parent_id=root_span.span_id,
                )
                child_span.set_inputs(z)

                z = z + 2
                time.sleep(0.1)

                child_span.set_outputs(z)
                child_span.set_attributes({"delta": 2})
                child_span.end()

                root_span.set_outputs(z)
            return z

    model = TestModel()
    model.predict(1, 2)

    trace = mlflow.get_traces()[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == SpanStatusCode.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "5"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "5"
    assert len(trace.data.spans) == 2

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace.info.execution_time_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 5,
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }


def test_search_traces_yields_expected_dataframe_contents(monkeypatch, create_trace):
    traces_to_return = [create_trace("a"), create_trace("b"), create_trace("c")]

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return traces_to_return

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df.columns.tolist() == [
        "request_id",
        "timestamp_ms",
        "status",
        "execution_time_ms",
        "request",
        "response",
        "request_metadata",
        "spans",
        "tags",
    ]
    for idx, trace in enumerate(traces_to_return):
        assert df.iloc[idx].request_id == trace.info.request_id
        assert df.iloc[idx].timestamp_ms == trace.info.timestamp_ms
        assert df.iloc[idx].status == trace.info.status
        assert df.iloc[idx].execution_time_ms == trace.info.execution_time_ms
        assert df.iloc[idx].request == trace.data.request
        assert df.iloc[idx].response == trace.data.response
        assert df.iloc[idx].request_metadata == trace.info.request_metadata
        assert df.iloc[idx].spans == trace.data.spans
        assert df.iloc[idx].tags == trace.info.tags


def test_search_traces_handles_missing_response_tags_and_metadata(monkeypatch, create_trace):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return [
                Trace(
                    info=TraceInfo(
                        request_id=5,
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=2,
                        status=TraceStatus.OK,
                    ),
                    data=TraceData(
                        spans=[],
                        request="request",
                        # Response is missing
                    ),
                )
            ]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df["response"].isnull().all()
    assert df["tags"].tolist() == [{}]
    assert df["request_metadata"].tolist() == [{}]


def test_search_traces_extracts_fields_as_expected(monkeypatch):
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1}
        )
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return mlflow.get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["predict.inputs.x", "predict.outputs", "add_one_with_custom_name.inputs.z"]
    )
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]
