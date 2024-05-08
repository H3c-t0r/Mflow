from mlflow.models import ModelConfig, set_model
from mlflow.pyfunc import PythonModel

base_config = ModelConfig(development_config="tests/pyfunc/pyfunc_sample_config.yml")


class MyModel(PythonModel):
    def predict(self, context=None, model_input=None):
        timeout = base_config.get("timeout")
        return f"Predict called with context {context} and input {model_input}, timeout {timeout}"


set_model(MyModel())