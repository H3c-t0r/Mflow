import numpy as np
import pandas as pd
import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import (
    PydanticResponse,
    Response,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import NodeWithScore
from pyspark.sql import SparkSession

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.llama_index import _LlamaIndexModelWrapper

from tests.llama_index._llama_index_test_fixtures import (
    embed_model,  # noqa: F401
    llm,  # noqa: F401
    multi_index,  # noqa: F401
    settings,  # noqa: F401
    single_graph,  # noqa: F401
    single_index,  # noqa: F401
)


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


@pytest.mark.parametrize(
    "index_fixture",
    [
        "single_index",
        "multi_index",
        "single_graph",
    ],
)
def test_llama_index_native_save_and_load_model(request, index_fixture, tmp_path):
    index = request.getfixturevalue(index_fixture)
    mlflow.llama_index.save_model(index, tmp_path, engine_type="query")

    loaded_model = mlflow.llama_index.load_model(tmp_path)

    assert type(loaded_model) == type(index)
    assert loaded_model.as_chat_engine().query("Spell llamaindex").response.lower() != ""


@pytest.mark.parametrize(
    "index_fixture",
    [
        "single_index",
        "multi_index",
        "single_graph",
    ],
)
def test_llama_index_native_log_and_load_model(request, index_fixture):
    index = request.getfixturevalue(index_fixture)
    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(index, "model", engine_type="query")

    loaded_model = mlflow.llama_index.load_model(logged_model.model_uri)

    assert "llama_index" in logged_model.flavors
    assert type(loaded_model) == type(index)
    assert loaded_model.as_chat_engine().query("Spell llamaindex").response.lower() != ""


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_query_and_retriever_correct(single_index, engine_type):
    wrapped_model = _LlamaIndexModelWrapper(single_index, engine_type)
    format_func = wrapped_model._format_predict_input_query_and_retriever

    assert isinstance(format_func(pd.DataFrame({"query_str": ["hi"]})), QueryBundle)
    assert isinstance(format_func(np.array(["hi"])), str)
    assert isinstance(format_func({"query_str": ["hi"]}), QueryBundle)
    assert isinstance(format_func({"query_str": "hi"}), QueryBundle)
    assert isinstance(format_func(["hi"]), str)
    assert isinstance(format_func("hi"), str)
    assert format_func([]) is None


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_query_and_retriever_too_many_elements(single_index, engine_type):
    wrapped_model = _LlamaIndexModelWrapper(single_index, engine_type)
    format_func = wrapped_model._format_predict_input_query_and_retriever

    with pytest.raises(MlflowException, match="only take one message"):
        format_func(pd.DataFrame({"query_str": ["hi", "bye"]}))
    with pytest.raises(MlflowException, match="only take one message"):
        format_func(np.array(["hi", "bye"]))
    with pytest.raises(MlflowException, match="only take one message"):
        format_func(["hi", "there"])


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_query_and_retriever_incorrect_schema(single_index, engine_type):
    wrapped_model = _LlamaIndexModelWrapper(single_index, engine_type)
    format_func = wrapped_model._format_predict_input_query_and_retriever

    with pytest.raises(MlflowException, match="incorrect schema"):
        format_func(pd.DataFrame({"incorrect": ["hi"]}))
    with pytest.raises(MlflowException, match="incorrect schema"):
        format_func({"incorrect": ["hi"]})


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_query_and_retriever_correct_schema_complex(single_index, engine_type):
    wrapped_model = _LlamaIndexModelWrapper(single_index, engine_type)
    format_func = wrapped_model._format_predict_input_query_and_retriever

    payload = {
        "query_str": ["hi"],
        "custom_embedding_strs": ["a"],
        "embedding": [[1.0]],
    }
    assert isinstance(format_func(pd.DataFrame(payload)), QueryBundle)
    assert isinstance(format_func(payload), QueryBundle)


def test_format_predict_output_as_str(single_index):
    wrapped_model = _LlamaIndexModelWrapper(single_index, "query")
    format_func = wrapped_model._format_predict_output_as_str

    assert isinstance(format_func(Response(response="asdf")), str)
    assert isinstance(format_func(PydanticResponse(response=BaseModel())), str)


def test_format_predict_output_as_list(single_index, document):
    wrapped_model = _LlamaIndexModelWrapper(single_index, "query")
    format_func = wrapped_model._format_predict_output_as_list

    nodes = [NodeWithScore(node=document, score=1.0), NodeWithScore(node=document, score=1.0)]
    assert isinstance(format_func(nodes), list)
    assert isinstance(format_func(nodes)[0], dict)


@pytest.mark.parametrize(
    "engine_type",
    [
        "query",
        # "chat",
        "retriever",
    ],
)
def test_pyfunc_predict_with_index_valid_schema(single_index, engine_type):
    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(single_index, "model", engine_type=engine_type)

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert loaded_model.predict(
        pd.DataFrame({"query_str": ["hi"], "custom_embedding_strs": ["a"], "embedding": [[1.0]]})
    )
    assert loaded_model.predict("hi")
