from abc import ABC, abstractmethod


class ProjectBackend(ABC):
    """
    Wrapper around an MLflow project remote backend (e.g. databricks, azure)
    for methods
    """

    def __init__(self, project, active_run, backend_config):
        self.active_run = active_run
        self.backend_config = backend_config

    @abstractmethod
    def validate(self, backend_config):
        """
        Validates that the configuration is good. Also checks if supported
        """
        pass

    @abstractmethod
    def configure(self):
        """
        Set up things like logging
        """
        pass

    @abstractmethod
    def submit_run(self):
        """
        returns a SubmittedRun, like DatabricksSubmittedRun
        """
        pass

    @property
    @abstractmethod
    def backend_type(self):
        pass
