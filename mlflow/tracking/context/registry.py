import entrypoints
import warnings

from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext


class RunContextProviderRegistry(object):
    """Registry for run context provider implementations

    This class allows the registration of a run context provider which can be used to infer meta
    information about the context of an MLflow experiment run. Implementations declared though the
    entrypoints `mlflow.run_context_provider` group can be automatically registered through the
    `register_entrypoints` method.

    Registered run context providers can return tags that override those implemented in the core
    library, however the order in which plugins are resolved is undefined.
    """

    def __init__(self):
        self._registry = []

    def register(self, run_context_provider_cls):
        self._registry.append(run_context_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all("mlflow.run_context_provider"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register context provider "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2
                )

    def _run(self, provider, func, default):
        try:
            return func(provider)
        except NotImplementedError:
            return default

    def run(self, func, default=None):
        return (
            list(
                map(
                    lambda x: self._run(x, func, default),
                    filter(
                        lambda x: x.in_context(),
                        self
                    )
                )
            )
        )

    def __iter__(self):
        return iter(self._registry)


_run_context_provider_registry = RunContextProviderRegistry()
_run_context_provider_registry.register(DefaultRunContext)
_run_context_provider_registry.register(GitRunContext)
_run_context_provider_registry.register(DatabricksNotebookRunContext)
_run_context_provider_registry.register(DatabricksJobRunContext)
_run_context_provider_registry.register(DatabricksClusterRunContext)

_run_context_provider_registry.register_entrypoints()


def resolve_tags(tags=None):
    """Generate a set of tags for the current run context. Tags are resolved in the order,
    contexts are registered. Argument tags are applied last.

    This function iterates through all run context providers in the registry. Additional context
    providers can be registered as described in
    :py:class:`mlflow.tracking.context.RunContextProvider`.

    :param tags: A dictionary of tags to override. If specified, tags passed in this argument will
                 override those inferred from the context.
    :return: A dicitonary of resolved tags.
    """

    all_tags = {}

    _run_context_provider_registry.run(
        lambda x: all_tags.update(x.tags()), {}
    )

    if tags is not None:
        all_tags.update(tags)

    return all_tags


def execute_start_run_actions(run):
    """
    Execute context-specific for all the registered contexts when a MLflow run is started

    :param run: An instance of :py:class:`mlflow.entities.Run` of the run started
    run that started
    :return: None
    """
    _run_context_provider_registry.run(
        lambda x: x.execute_start_run_actions(run)
    )


def execute_end_run_actions(run, status):
    """
    Execute context-specific for all the registered contexts when a MLflow run is finished

    :param run: An instance of :py:class:`mlflow.entities.Run` of the run finished
    :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
    :return: None
    """
    _run_context_provider_registry.run(
        lambda x: x.execute_end_run_actions(run)
    )


def execute_create_experiment_actions(experiment_id):
    """
    Execute context-specific actions for all the registered contexts
    when a MLflow experiment is created

    :param experiment_id: Experiment ID of the created experiments.
    :return: None
    """
    _run_context_provider_registry.run(
        lambda x: x.execute_create_experiment_actions(experiment_id)
    )


def execute_delete_experiment_actions(experiment_id):
    """
    Execute context-specific actions for all the registered contexts
    when a MLflow experiment is deleted

    :param experiment_id: Experiment ID of the deletd experiments.
    :return: None
    """
    _run_context_provider_registry.run(
        lambda x: x.execute_delete_experiment_actions(experiment_id)
    )
