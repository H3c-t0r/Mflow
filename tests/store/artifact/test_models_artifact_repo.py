import os
import shutil
from unittest import mock
from unittest.mock import Mock

import pytest

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)

from tests.store.artifact.constants import (
    UC_MODELS_ARTIFACT_REPOSITORY,
    WORKSPACE_MODELS_ARTIFACT_REPOSITORY,
)


@pytest.mark.parametrize(
    "uri_with_profile",
    [
        "models://profile@databricks/MyModel/12",
        "models://profile@databricks/MyModel/Staging",
        "models://profile@databricks/MyModel/Production",
    ],
)
def test_models_artifact_repo_init_with_uri_containing_profile(uri_with_profile):
    with mock.patch(WORKSPACE_MODELS_ARTIFACT_REPOSITORY, autospec=True) as mock_repo:
        mock_repo.return_value.model_name = "MyModel"
        mock_repo.return_value.model_version = "12"
        models_repo = ModelsArtifactRepository(uri_with_profile)
        assert models_repo.artifact_uri == uri_with_profile
        assert isinstance(models_repo.repo, DatabricksModelsArtifactRepository)
        mock_repo.assert_called_once_with(uri_with_profile)


@pytest.mark.parametrize(
    "uri_without_profile",
    ["models:/MyModel/12", "models:/MyModel/Staging", "models:/MyModel/Production"],
)
def test_models_artifact_repo_init_with_db_profile_inferred_from_context(uri_without_profile):
    with mock.patch(WORKSPACE_MODELS_ARTIFACT_REPOSITORY, autospec=True) as mock_repo, mock.patch(
        "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
        return_value="databricks://getRegistryUriDefault",
    ):
        mock_repo.return_value.model_name = "MyModel"
        mock_repo.return_value.model_version = "12"
        models_repo = ModelsArtifactRepository(uri_without_profile)
        assert models_repo.artifact_uri == uri_without_profile
        assert isinstance(models_repo.repo, DatabricksModelsArtifactRepository)
        mock_repo.assert_called_once_with(uri_without_profile)


def test_models_artifact_repo_init_with_uc_registry_db_profile_inferred_from_context():
    model_uri = "models:/MyModel/12"
    uc_registry_uri = "databricks-uc://getRegistryUriDefault"
    with mock.patch(UC_MODELS_ARTIFACT_REPOSITORY, autospec=True) as mock_repo, mock.patch(
        "mlflow.get_registry_uri", return_value=uc_registry_uri
    ):
        mock_repo.return_value.model_name = "MyModel"
        mock_repo.return_value.model_version = "12"
        models_repo = ModelsArtifactRepository(model_uri)
        assert models_repo.artifact_uri == model_uri
        assert isinstance(models_repo.repo, UnityCatalogModelsArtifactRepository)
        mock_repo.assert_called_once_with(model_uri, registry_uri=uc_registry_uri)


def test_models_artifact_repo_init_with_version_uri_and_not_using_databricks_registry():
    non_databricks_uri = "non_databricks_uri"
    artifact_location = "s3://blah_bucket/"
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch(
        "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
        return_value=non_databricks_uri,
    ), mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=None,
    ) as get_repo_mock:
        model_uri = "models:/MyModel/12"
        ModelsArtifactRepository(model_uri)
        get_repo_mock.assert_called_once_with(artifact_location)


def test_models_artifact_repo_init_with_stage_uri_and_not_using_databricks_registry():
    model_uri = "models:/MyModel/Staging"
    artifact_location = "s3://blah_bucket/"
    model_version_detailed = ModelVersion(
        "MyModel",
        "10",
        "2345671890",
        "234567890",
        "some description",
        "UserID",
        "Production",
        "source",
        "run12345",
    )
    get_latest_versions_patch = mock.patch.object(
        MlflowClient, "get_latest_versions", return_value=[model_version_detailed]
    )
    get_model_version_download_uri_patch = mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    )
    with get_latest_versions_patch, get_model_version_download_uri_patch, mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=None,
    ) as get_repo_mock:
        ModelsArtifactRepository(model_uri)
        get_repo_mock.assert_called_once_with(artifact_location)


def test_models_artifact_repo_uses_repo_download_artifacts():
    """
    ``ModelsArtifactRepository`` should delegate `download_artifacts` to its
    ``self.repo.download_artifacts`` function.
    """
    artifact_location = "s3://blah_bucket/"
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch.object(ModelsArtifactRepository, "_add_registered_model_meta_file"):
        model_uri = "models:/MyModel/12"
        models_repo = ModelsArtifactRepository(model_uri)
        models_repo.repo = Mock()
        models_repo.download_artifacts("artifact_path", "dst_path")
        models_repo.repo.download_artifacts.assert_called_once()


def test_models_artifact_repo_add_registered_model_meta_file():
    from mlflow.store.artifact.models_artifact_repo import REGISTERED_MODEL_META_FILE_NAME

    artifact_path = "artifact_path"
    dst_path = "dst_path"
    artifact_location = f"s3://blah_bucket/{artifact_path}"
    artifact_dst_path = f"{dst_path}/{artifact_path}/"
    model_name = "MyModel"
    model_version = "12"

    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("os.path.isdir", return_value=True), mock.patch(
        "mlflow.store.artifact.models_artifact_repo.write_yaml"
    ) as write_yaml_mock:
        models_repo = ModelsArtifactRepository(f"models:/{model_name}/{model_version}")
        models_repo.repo = Mock(**{"download_artifacts.return_value": artifact_dst_path})

        models_repo.download_artifacts(artifact_path, dst_path)

        write_yaml_mock.assert_called_with(
            artifact_dst_path,
            REGISTERED_MODEL_META_FILE_NAME,
            {
                "model_name": model_name,
                "model_version": model_version,
            },
            overwrite=True,
            ensure_yaml_extension=False,
        )
    # Calling the download_artifacts method on local FileStore will create an ./mlruns directory
    # which is a test side effect. Clean this up.
    mlruns_dir = "./mlruns"
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)


def test_models_artifact_repo_add_registered_model_meta_file_to_file_dest():
    artifact_path = "artifact_path"
    dst_path = "dst_path"
    artifact_location = f"s3://blah_bucket/{artifact_path}"
    artifact_dst_file_path = f"{dst_path}/{artifact_path}/MLModel.yaml"

    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch(
        "mlflow.store.artifact.models_artifact_repo.write_yaml"
    ) as write_yaml_mock, mock.patch(
        "os.path.isdir", return_value=False
    ), mock.patch(
        "mlflow.store.artifact.models_artifact_repo.ModelsArtifactRepository"
    ) as mock_repo:
        mock_repo.repo = Mock(**{"download_artifacts.return_value": artifact_dst_file_path})

        mock_repo.download_artifacts(artifact_path, dst_path)

        # Assert that write_yaml is not called since the destination is a file
        write_yaml_mock.assert_not_called()


def test_split_models_uri():
    assert ModelsArtifactRepository.split_models_uri("models:/model/1") == ("models:/model/1", "")
    assert ModelsArtifactRepository.split_models_uri("models:/model/1/path") == (
        "models:/model/1",
        "path",
    )
    assert ModelsArtifactRepository.split_models_uri("models:/model/1/path/to/artifact") == (
        "models:/model/1",
        "path/to/artifact",
    )
