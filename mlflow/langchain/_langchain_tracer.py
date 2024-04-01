import logging
from typing import Any, Dict, List, Optional, Sequence, Union, cast
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from tenacity import RetryCallState
from typing_extensions import override

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanEvent, SpanStatus, SpanType, TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.tracing.types.wrapper import MlflowSpanWrapper
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass

_logger = logging.getLogger(__name__)


class MlflowLangchainTracer(BaseCallbackHandler, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging traces.
    We need to inherit ExceptionSafeAbstractClass to avoid invalid new
    input arguments added to original function call.
    """

    def __init__(self):
        super().__init__()
        self._mlflow_client = MlflowClient()
        self._run_span_mapping: Dict[str, MlflowSpanWrapper] = {}

    def _get_span_by_run_id(self, run_id: UUID) -> Optional[MlflowSpanWrapper]:
        if span := self._run_span_mapping.get(str(run_id)):
            return span
        raise MlflowException(f"Span for run_id {run_id!s} not found.")

    def _start_span(
        self,
        span_name: str,
        parent_run_id: Optional[UUID],
        span_type: str,
        run_id: UUID,
        inputs: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> MlflowSpanWrapper:
        """Start MLflow Span (or Trace if it is root component)"""
        parent = self._get_span_by_run_id(parent_run_id) if parent_run_id else None
        if parent:
            span = self._mlflow_client.start_span(
                name=span_name,
                request_id=parent.request_id,
                parent_span_id=parent.span_id,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
            )
        else:
            # When parent_run_id is None, this is root component so start trace
            span = self._mlflow_client.start_trace(
                name=span_name, inputs=inputs, attributes=attributes
            )
        self._run_span_mapping[str(run_id)] = span
        return span

    def _end_span(
        self,
        span: MlflowSpanWrapper,
        outputs=None,
        attributes=None,
        status=SpanStatus(TraceStatus.OK),
    ):
        """Close MLflow Span (or Trace if it is root component)"""
        self._mlflow_client.end_span(
            request_id=span.request_id,
            span_id=span.span_id,
            outputs=outputs,
            attributes=attributes,
            status=status,
        )

    def _reset(self):
        self._run_span_mapping = {}

    @override
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Run when a chat model starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or "chat model",
            parent_run_id=parent_run_id,
            # we use LLM for chat models as well
            span_type=SpanType.LLM,
            run_id=run_id,
            inputs={"messages": messages},
            attributes=kwargs,
        )

    @override
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM (non-chat models) starts running."""
        inputs = {"prompts": prompts}
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or "llm",
            parent_run_id=parent_run_id,
            span_type=SpanType.LLM,
            run_id=run_id,
            inputs=inputs,
            attributes=kwargs,
        )

    @override
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ):
        """Run on new LLM token. Only available when streaming is enabled."""
        llm_span = self._get_span_by_run_id(run_id)
        event_kwargs = {"token": token}
        if chunk:
            event_kwargs["chunk"] = chunk
        llm_span.add_event(
            SpanEvent(
                name="new_token",
                attributes=event_kwargs,
            )
        )

    @override
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run on a retry event."""
        span = self._get_span_by_run_id(run_id)
        retry_d: Dict[str, Any] = {
            "slept": retry_state.idle_for,
            "attempt": retry_state.attempt_number,
        }
        if retry_state.outcome is None:
            retry_d["outcome"] = "N/A"
        elif retry_state.outcome.failed:
            retry_d["outcome"] = "failed"
            exception = retry_state.outcome.exception()
            retry_d["exception"] = str(exception)
            retry_d["exception_type"] = exception.__class__.__name__
        else:
            retry_d["outcome"] = "success"
            retry_d["result"] = str(retry_state.outcome.result())
        span.add_event(
            SpanEvent(
                name="retry",
                attributes=retry_d,
            )
        )

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any):
        """End the span for an LLM run."""
        llm_span = self._get_span_by_run_id(run_id)
        outputs = response.dict()
        for i, generations in enumerate(response.generations):
            for j, generation in enumerate(generations):
                output_generation = outputs["generations"][i][j]
                if "message" in output_generation:
                    output_generation["message"] = cast(ChatGeneration, generation).message
        self._end_span(llm_span, outputs=outputs)

    @override
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Handle an error for an LLM run."""
        llm_span = self._get_span_by_run_id(run_id)
        llm_span.add_event(SpanEvent.from_exception(error))
        self._end_span(llm_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

    def _get_chain_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        return inputs if isinstance(inputs, dict) else {"input": inputs}

    @override
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Union[Dict[str, Any], Any],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Start span for a chain run."""
        if metadata:
            kwargs.update({"metadata": metadata})
        # not considering streaming events for now
        self._start_span(
            span_name=name or "chain",
            parent_run_id=parent_run_id,
            span_type=SpanType.CHAIN,
            run_id=run_id,
            inputs=self._get_chain_inputs(inputs),
            attributes=kwargs,
        )

    @override
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ):
        """Run when chain ends running."""
        chain_span = self._get_span_by_run_id(run_id)
        if inputs:
            chain_span.set_inputs(self._get_chain_inputs(inputs))
        self._end_span(chain_span, outputs=outputs)

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: Optional[Union[Dict[str, Any], Any]] = None,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when chain errors."""
        chain_span = self._get_span_by_run_id(run_id)
        if inputs:
            chain_span.set_inputs(self._get_chain_inputs(inputs))
        chain_span.add_event(SpanEvent.from_exception(error))
        self._end_span(chain_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

    @override
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Start span for a tool run."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or "tool",
            parent_run_id=parent_run_id,
            span_type=SpanType.TOOL,
            run_id=run_id,
            inputs={"input_str": input_str},
            attributes=kwargs,
        )

    @override
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any):
        """Run when tool ends running."""
        tool_span = self._get_span_by_run_id(run_id)
        self._end_span(tool_span, outputs=str(output))

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when tool errors."""
        tool_span = self._get_span_by_run_id(run_id)
        tool_span.add_event(SpanEvent.from_exception(error))
        self._end_span(tool_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

    @override
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Run when Retriever starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or "retriever",
            parent_run_id=parent_run_id,
            span_type=SpanType.RETRIEVER,
            run_id=run_id,
            inputs={"query": query},
            attributes=kwargs,
        )

    @override
    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any):
        """Run when Retriever ends running."""
        retriever_span = self._get_span_by_run_id(run_id)
        self._end_span(retriever_span, outputs={"documents": documents})

    @override
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when Retriever errors."""
        retriever_span = self._get_span_by_run_id(run_id)
        retriever_span.add_event(SpanEvent.from_exception(error))
        self._end_span(retriever_span, status=SpanStatus(TraceStatus.ERROR, str(error)))

    @override
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        kwargs.update({"log": action.log})
        self._start_span(
            span_name=action.tool,
            parent_run_id=parent_run_id,
            span_type=SpanType.AGENT,
            run_id=run_id,
            inputs={"tool_input": action.tool_input},
            attributes=kwargs,
        )

    @override
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        agent_span = self._get_span_by_run_id(run_id)
        kwargs.update({"log": finish.log})
        self._end_span(agent_span, outputs=finish.return_values, attributes=kwargs)

    @override
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary text."""
        try:
            span = self._get_span_by_run_id(run_id)
        except MlflowException:
            _logger.warning("Span not found for text event. Skipping text event logging.")
        else:
            span.add_event(
                SpanEvent(
                    "text",
                    attributes={"text": text},
                )
            )

    def flush_tracker(self):
        mlflow.get_traces()
        self._reset()
