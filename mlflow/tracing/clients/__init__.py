from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.ipython_wrapper import IPythonTraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient

__all__ = ["IPythonTraceClient", "InMemoryTraceClient", "TraceClient", "get_trace_client"]


def get_trace_client() -> TraceClient:
    try:
        import IPython

        if IPython.get_ipython() is not None:
            return IPythonTraceClient.get_instance()
    except ImportError:
        pass
    return InMemoryTraceClient.get_instance()
