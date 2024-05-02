import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

# A thread local variable to store the context of the current prediction request.
# This is particularly used to associate logs/traces with a specific prediction request in the
# caller side. The context variable is intended to be set by the called before invoking the
# predict method, using the set_prediction_context context manager.
_PREDICTION_REQUEST_CTX = ContextVar("mlflow_prediction_request_context", default=None)


@dataclass
class Context:
    # A unique identifier for the current prediction request.
    request_id: str
    # Whether the current prediction request is as a part of MLflow model evaluation.
    is_evaluate: bool = False


@contextlib.contextmanager
def set_prediction_context(context: Context):
    """
    Set the context for the current prediction request. The context will be set as a thread-local
    variable and will be accessible globally within the same thread.

    Args:
        context: The context for the current prediction request.
    """
    if not isinstance(context, Context):
        raise TypeError(f"Expected context to be an instance of Context, but got: {context}")

    token = _PREDICTION_REQUEST_CTX.set(context)
    try:
        yield
    finally:
        _PREDICTION_REQUEST_CTX.reset(token)


def get_prediction_context() -> Optional[Context]:
    """
    Get the context for the current prediction request. The context is thread-local and is set
    using the set_prediction_context context manager.

    Returns:
        The context for the current prediction request, or None if no context is set.
    """
    return _PREDICTION_REQUEST_CTX.get()
