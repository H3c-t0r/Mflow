"""Class for monitoring CPU stats."""

import psutil

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class CPUMonitor(BaseMetricsMonitor):
    """Class for monitoring CPU stats."""

    def __init__(self, name="cpu"):
        super().__init__(name)

    def collect_metrics(self):
        # Get CPU metrics.
        cpu_percent = psutil.cpu_percent()
        self._metrics["cpu_percentage"].append(cpu_percent)

        system_memory = psutil.virtual_memory()._asdict()
        self._metrics["cpu_memory_used"].append(int(system_memory["used"] / 1e6))

    def aggregate_metrics(self):
        for name, values in self._metrics.items():
            self._metrics[name] = sum(values) / len(values)
