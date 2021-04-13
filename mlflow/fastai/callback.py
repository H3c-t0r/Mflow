import numpy as np
import os
import shutil
import tempfile
from functools import partial
import matplotlib.pyplot as plt

import mlflow.tracking
from mlflow.utils.autologging_utils import try_mlflow_log
from mlflow.fastai import log_model

from fastai.callback.core import Callback


# Move outside, because it cannot be pickled. Besides, ExceptionSafeClass was giving some issues
class __MLflowFastaiCallback(Callback):
    from fastai.learner import Recorder
    from fastai.callback.all import TrackerCallback

    """
    Callback for auto-logging metrics and parameters.
    Records model structural information as params when training begins
    """
    remove_on_fetch, run_before, run_after = True, TrackerCallback, Recorder

    def __init__(self, metrics_logger, log_models, is_fine_tune=False):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.log_models = log_models

        self.is_fine_tune = is_fine_tune
        self.freeze_prefix = ""

    def after_epoch(self):
        """
        Log loss and other metrics values after each epoch
        """

        # Do not record in case of predicting
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return

        metrics = self.recorder.log
        metrics = dict(zip(self.recorder.metric_names, metrics))

        keys = list(metrics.keys())
        i = 0
        while i < len(metrics):
            key = keys[i]
            try:
                float(metrics[key])
                i += 1
            except (ValueError, TypeError):
                del metrics[key]
                del keys[i]

        self.metrics_logger.record_metrics(metrics, step=metrics["epoch"])

    def before_fit(self):
        from fastai.callback.all import ParamScheduler

        # Do not record in case of predicting or lr_finder
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return

        if self.is_fine_tune and len(self.opt.param_lists) == 1:
            print(
                """ WARNING: Using `fine_tune` with model which cannot be freeze.
                Current model have only one param group which makes imposible to freeze.
                Because of this it will record some fitting params twice (overriding o throwing exception) """
            )

        frozen = self.opt.frozen_idx != 0
        if frozen and self.is_fine_tune:
            self.freeze_prefix = "freeze_"
        else:
            self.freeze_prefix = ""

        # Extract function name when `opt_func` is partial function
        if isinstance(self.opt_func, partial):
            try_mlflow_log(
                mlflow.log_param,
                self.freeze_prefix + "opt_func",
                self.opt_func.keywords["opt"].__name__,
            )
        else:
            try_mlflow_log(
                mlflow.log_param, self.freeze_prefix + "opt_func", self.opt_func.__name__
            )

        params_not_to_log = []
        for cb in self.cbs:
            if isinstance(cb, ParamScheduler):
                params_not_to_log = list(cb.scheds.keys())
                for param, f in cb.scheds.items():
                    values = []
                    for step in np.linspace(0, 1, num=100, endpoint=False):
                        values.append(f(step))
                    values = np.array(values)

                    # Log params main values from scheduling
                    try_mlflow_log(
                        mlflow.log_param, self.freeze_prefix + param + "_min", np.min(values, 0)
                    )
                    try_mlflow_log(
                        mlflow.log_param, self.freeze_prefix + param + "_max", np.max(values, 0)
                    )
                    try_mlflow_log(
                        mlflow.log_param, self.freeze_prefix + param + "_init", values[0]
                    )
                    try_mlflow_log(
                        mlflow.log_param, self.freeze_prefix + param + "_final", values[-1]
                    )

                    # Plot and save image of scheduling
                    fig = plt.figure()
                    plt.plot(values)
                    plt.ylabel(param)

                    tempdir = tempfile.mkdtemp()
                    try:
                        scheds_file = os.path.join(tempdir, self.freeze_prefix + param + ".png")
                        plt.savefig(scheds_file)
                        plt.close(fig)
                        try_mlflow_log(mlflow.log_artifact, local_path=scheds_file)
                    finally:
                        shutil.rmtree(tempdir)
                break

        for param in self.opt.hypers[0]:
            if param not in params_not_to_log:
                try_mlflow_log(
                    mlflow.log_param,
                    self.freeze_prefix + param,
                    [h[param] for h in self.opt.hypers],
                )

        if hasattr(self.opt, "true_wd"):
            try_mlflow_log(mlflow.log_param, self.freeze_prefix + "true_wd", self.opt.true_wd)

        if hasattr(self.opt, "bn_wd"):
            try_mlflow_log(mlflow.log_param, self.freeze_prefix + "bn_wd", self.opt.bn_wd)

        if hasattr(self.opt, "train_bn"):
            try_mlflow_log(mlflow.log_param, self.freeze_prefix + "train_bn", self.opt.train_bn)

    def after_fit(self):
        from fastai.callback.all import SaveModelCallback

        # Do not log model in case of predicting
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return

        # Workaround to log model from SaveModelCallback
        # Use this till able to set order between SaveModelCallback and EarlyStoppingCallback
        for cb in self.cbs:
            if isinstance(cb, SaveModelCallback):
                cb("after_fit")

        if self.log_models:
            try_mlflow_log(log_model, self.learn, artifact_path="model")
