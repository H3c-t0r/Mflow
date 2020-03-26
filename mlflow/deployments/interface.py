from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.utils import experimental


listType = list
plugin_store = DeploymentPlugins(auto_register=True)


@experimental
def create(target, model_uri, flavor=None, **kwargs):
    """
    Create the deployment using target function's create method

    :param target:  The deployment target name The deployment target name
    :param model_uri: The location, in URI format, of the MLflow model
    :param flavor: The name of the flavor of the model to use for deployment. if this is
                   ``None``, the plugin need to choose the flavor. In any case, it's better
                   to validate the flavor by calling `_validate_deployment_flavor` method
    :param kwargs: The keyword arguments either parsed from the CLI options or passed by the
                   user specifically using the python API
    :return: dict, A python dictionary with keys ``deployment_id`` and ``flavor``
    """
    deployment = plugin_store[target].create(model_uri, flavor, **kwargs)
    if not isinstance(deployment, dict) or \
            not all([k in ('deployment_id', 'flavor') for k in deployment]):
        raise TypeError("Deployment creation must return a dictionary with values for "
                        "``deployment_id`` and ``flavor``")
    return deployment


@experimental
def delete(target, deployment_id, **kwargs):
    """
    Delete the deployment associated with the deployment ID

    :param target:  The deployment target name
    :param deployment_id: The ID generated by the plugin while creating the deployment
    :param kwargs: The keyword arguments either parsed from the CLI options or passed by the
                   user specifically using the python API
    :return: None
    """
    plugin_store[target].delete(deployment_id, **kwargs)


@experimental
def update(target, deployment_id, model_uri=None, flavor=None, **kwargs):
    """
    Update the existing deployment with a new model

    :param target:  The deployment target name
    :param deployment_id: The ID generated by the plugin while creating the deployment
    :param model_uri: The location, in URI format, of the MLflow model
    :param flavor: The name of the flavor of the model to use for deployment. if this is
                   ``None``, the plugin need to choose the flavor. In any case, it's better
                   to validate the flavor by calling `_validate_deployment_flavor` method
    :param kwargs: The keyword arguments either parsed from the CLI options or passed by the
                   user specifically using the python API
    :return: None
    """
    if flavor and not model_uri:
        raise RuntimeError("``update`` has got ``flavor`` but not ``model_uri``")
    if not any([flavor, model_uri, kwargs]):
        raise RuntimeError("``update`` did not get any arguments")
    plugin_store[target].update(deployment_id, model_uri, flavor, **kwargs)


@experimental
def list(target, **kwargs):  # pylint: disable=W0622
    """
    List the deployment IDs of the deployments. These IDs can be used to do other operations
    such as delete, update or get more description about the deployments

    :param target:  The deployment target name
    :param kwargs: The keyword arguments either parsed from the CLI options or passed by the
                   user specifically using the python API
    :return: list, A list of deployment IDs
    """
    ids = plugin_store[target].list(**kwargs)
    if not isinstance(ids, listType):
        raise TypeError("IDs must be returned as a ``list``")
    return ids


@experimental
def describe(target, deployment_id, **kwargs):
    """
    Get more description about the deployments. MLFlow does not control what information
    should be part of the return value here and it is solely depends on the plugin

    :param target:  The deployment target name
    :param deployment_id: The ID generated by the plugin while creating the deployment
    :param kwargs: The keyword arguments either parsed from the CLI options or passed by the
                   user specifically using the python API
    :return: dict, A dictionary with all the important descriptions
    """
    desc = plugin_store[target].describe(deployment_id, **kwargs)
    if not isinstance(desc, dict):
        raise TypeError("Description must be returned as a dictionary")
    return desc
