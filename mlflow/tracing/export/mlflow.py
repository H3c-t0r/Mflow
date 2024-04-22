import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.fluent import TRACE_BUFFER, TRACE_BUFFER_LOCK
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


class MlflowSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow.

    MLflow backend (will) only support logging the complete trace, not incremental updates
    for spans, so this exporter is designed to aggregate the spans into traces in memory.
    Therefore, this only works within a single process application and not intended to work
    in a distributed environment. For the same reason, this exporter should only be used with
    SimpleSpanProcessor.

    If we want to support distributed tracing, we should first implement an incremental trace
    logging in MLflow backend, then we can get rid of the in-memory trace aggregation.
    """

    def __init__(self, client=MlflowClient(), display_handler=get_display_handler()):
        self._client = client
        self._display_handler = display_handler
        self._trace_manager = InMemoryTraceManager.get_instance()

    def export(self, root_spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
                Only root spans for each trace are passed to this method.
        """
        for span in root_spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"TraceInfo for span {span} not found. Skipping export.")
                continue

            # Add the trace to the in-memory buffer
            with TRACE_BUFFER_LOCK:
                TRACE_BUFFER.append(trace)

            # Display the trace in the UI
            self._display_handler.display_traces([trace])

            # Log the trace to MLflow
            self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        # TODO: Make this async
        try:
            self._client._upload_trace_data(trace.info, trace.data)
            self._client._upload_ended_trace_info(
                request_id=trace.info.request_id,
                timestamp_ms=trace.info.timestamp_ms + trace.info.execution_time_ms,
                status=trace.info.status,
                request_metadata=trace.info.request_metadata,
                tags=trace.info.tags,
            )
        except Exception as e:
            _logger.debug(f"Failed to log trace to MLflow backend: {e}", exc_info=True)
