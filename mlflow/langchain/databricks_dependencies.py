from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

_DATABRICKS_DEPENDENCY_KEY = "databricks_dependency"
_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY = "databricks_vector_search_index_name"
_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY = "databricks_vector_search_endpoint_name"
_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY = "databricks_embeddings_endpoint_name"
_DATABRICKS_LLM_ENDPOINT_NAME_KEY = "databricks_llm_endpoint_name"
_DATABRICKS_CHAT_ENDPOINT_NAME_KEY = "databricks_chat_endpoint_name"


def _assign_value_or_append_to_list(d, key, value):
    d[key].append(value)


def _extract_databricks_dependencies_from_retriever(
    retriever, dependency_dict: DefaultDict[str, List[Any]]
):
    import langchain

    if hasattr(retriever, "vectorstore") and hasattr(retriever.vectorstore, "embeddings"):
        vectorstore = retriever.vectorstore
        embeddings = vectorstore.embeddings

        if isinstance(vectorstore, langchain.vectorstores.DatabricksVectorSearch):
            index = vectorstore.index
            _assign_value_or_append_to_list(
                dependency_dict, _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY, index.name
            )
            _assign_value_or_append_to_list(
                dependency_dict, _DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY, index.endpoint_name
            )

        if isinstance(embeddings, langchain.embeddings.DatabricksEmbeddings):
            _assign_value_or_append_to_list(
                dependency_dict, _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY, embeddings.endpoint
            )
        elif (
            hasattr(vectorstore, "_is_databricks_managed_embeddings")
            and callable(getattr(vectorstore, "_is_databricks_managed_embeddings"))
            and vectorstore._is_databricks_managed_embeddings()
        ):
            _assign_value_or_append_to_list(
                dependency_dict,
                _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY,
                "_is_databricks_managed_embeddings",
            )


def _extract_databricks_dependencies_from_llm(llm, dependency_dict: DefaultDict[str, List[Any]]):
    import langchain

    if isinstance(llm, langchain.llms.Databricks):
        _assign_value_or_append_to_list(
            dependency_dict, _DATABRICKS_LLM_ENDPOINT_NAME_KEY, llm.endpoint_name
        )


def _extract_databricks_dependencies_from_chat_model(
    chat_model, dependency_dict: DefaultDict[str, List[Any]]
):
    import langchain

    if isinstance(chat_model, langchain.chat_models.ChatDatabricks):
        _assign_value_or_append_to_list(
            dependency_dict, _DATABRICKS_CHAT_ENDPOINT_NAME_KEY, chat_model.endpoint
        )


def _extract_dependency_dict_from_lc_model(lc_model, dependency_dict: DefaultDict[str, List[Any]]):
    """
    This function contains the logic to examine a non-Runnable component of a langchain model.
    The logic here does not cover all legacy chains. If you need to support a custom chain,
    you need to monkey patch this function.
    """
    if lc_model is None:
        return

    # leaf node
    _extract_databricks_dependencies_from_chat_model(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_retriever(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_llm(lc_model, dependency_dict)

    # recursively inspect legacy chain
    if hasattr(lc_model, "retriever"):
        _extract_dependency_dict_from_lc_model(lc_model.retriever, dependency_dict)
    if hasattr(
        lc_model, "llm_chain"
    ):  # StuffDocumentsChain, MapRerankDocumentsChain, MapReduceDocumentsChain
        _extract_dependency_dict_from_lc_model(lc_model.llm_chain.llm, dependency_dict)
    if hasattr(lc_model, "question_generator"):  # BaseConversationalRetrievalChain
        _extract_dependency_dict_from_lc_model(lc_model.question_generator.llm, dependency_dict)
    if hasattr(lc_model, "initial_llm_chain") and hasattr(
        lc_model, "refine_llm_chain"
    ):  # RefineDocumentsChain
        _extract_dependency_dict_from_lc_model(lc_model.initial_llm_chain.llm, dependency_dict)
        _extract_dependency_dict_from_lc_model(lc_model.refine_llm_chain.llm, dependency_dict)

    if hasattr(lc_model, "combine_documents_chain"):  # RetrievalQA, ReduceDocumentsChain
        _extract_dependency_dict_from_lc_model(lc_model.combine_documents_chain, dependency_dict)
    if hasattr(lc_model, "combine_docs_chain"):  # BaseConversationalRetrievalChain
        _extract_dependency_dict_from_lc_model(lc_model.combine_docs_chain, dependency_dict)
    if (
        hasattr(lc_model, "collapse_documents_chain")
        and lc_model.collapse_documents_chain is not None
    ):  # ReduceDocumentsChain
        _extract_dependency_dict_from_lc_model(lc_model.collapse_documents_chain, dependency_dict)


def _traverse_runnable(lc_model, dependency_dict: DefaultDict[str, List[Any]], visited: Set[str]):
    """
    This function contains the logic to traverse a langchain_core.runnables.RunnableSerializable
    object. It first inspects the current object using _extract_dependency_dict_from_lc_model
    and then, if the current object is a Runnable, it recursively inspects its children returned
    by lc_model.get_graph().nodes.values().
    This function supports arbitrary LCEL chain.
    """
    import langchain_core

    current_object_id = id(lc_model)
    if current_object_id in visited:
        return

    # Visit the current object
    visited.add(current_object_id)
    _extract_dependency_dict_from_lc_model(lc_model, dependency_dict)

    if isinstance(lc_model, langchain_core.runnables.RunnableSerializable):
        # Visit the returned graph
        for node in lc_model.get_graph().nodes.values():
            _traverse_runnable(node.data, dependency_dict, visited)
    else:
        # This is a leaf node
        pass
    return


def _detect_databricks_dependencies(lc_model) -> Dict[str, List[Any]]:
    """
    Detects the databricks dependencies of a langchain model and returns a dictionary of
    detected endpoint names and index names.

    lc_model can be an arbirary [chain that is built with LCEL](https://python.langchain.com
    /docs/modules/chains#lcel-chains), which is a langchain_core.runnables.RunnableSerializable.
    [Legacy chains](https://python.langchain.com/docs/modules/chains#legacy-chains) have limited
    support. Only RetrievalQA, StuffDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain,
    MapRerankDocumentsChain, MapReduceDocumentsChain, BaseConversationalRetrievalChain are
    supported. If you need to support a custom chain, you need to monkey patch
    the function mlflow.langchain.databricks_dependencies._extract_dependency_dict_from_lc_model().

    For an LCEL chain, all the langchain_core.runnables.RunnableSerializable nodes will be
    traversed.

    If a retriever is found, it will be used to extract the databricks vector search and embeddings
    dependencies.
    If an llm is found, it will be used to extract the databricks llm dependencies.
    If a chat_model is found, it will be used to extract the databricks chat dependencies.
    """
    dependency_dict = defaultdict(list)
    _traverse_runnable(lc_model, dependency_dict, set())
    return dict(dependency_dict)
