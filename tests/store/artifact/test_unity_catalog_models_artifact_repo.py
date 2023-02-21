import json
from unittest import mock
from unittest.mock import Mock

import pytest
from requests import Response

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.unity_catalog_models_artifact_repo import UnityCatalogModelsArtifactRepository
from mlflow import MlflowClient
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"


@pytest.mark.parametrize(
    "uri_with_profile",
    [
        "models://profile@databricks-uc/MyModel/12"
    ],
)
def test_uc_models_artifact_repo_init_with_uri_containing_profile(uri_with_profile):
    models_repo = UnityCatalogModelsArtifactRepository(uri_with_profile, _DATABRICKS_UNITY_CATALOG_SCHEME)
    assert models_repo.artifact_uri == uri_with_profile
    assert models_repo.client._registry_uri == f"{_DATABRICKS_UNITY_CATALOG_SCHEME}://profile"


def test_uc_models_artifact_repo_init_with_db_profile_inferred_from_context():
    uri_without_profile = "models:/MyModel/12"
    profile_in_registry_uri = "some_profile"
    registry_uri = f"databricks-uc://{profile_in_registry_uri}"
    models_repo = UnityCatalogModelsArtifactRepository(artifact_uri=uri_without_profile, registry_uri=registry_uri)
    assert models_repo.artifact_uri == uri_without_profile
    assert models_repo.client._registry_uri == registry_uri


def test_uc_models_artifact_repo_init_not_using_databricks_registry_raises():
    non_databricks_uri = "non_databricks_uri"
    model_uri = "models:/MyModel/12"
    with pytest.raises(MlflowException,
                       match="Attempted to instantiate an artifact repo to access models in the Unity Catalog with non-Unity Catalog registry URI"):
        UnityCatalogModelsArtifactRepository(model_uri, non_databricks_uri)


def test_uc_models_artifact_repo_with_stage_uri_raises():
    model_uri = "models:/MyModel/Staging"
    with pytest.raises(MlflowException):
        UnityCatalogModelsArtifactRepository(model_uri, _DATABRICKS_UNITY_CATALOG_SCHEME)


def test_uc_models_artifact_repo_uses_repo_download_artifacts():
    """
    ``ModelsArtifactRepository`` should delegate `download_artifacts` to its
    ``self.repo.download_artifacts`` function.
    """
    # TODO: each time we do a download_artifacts(), which is the only supported operation
    # for this read-only models:// artifact store, we re-fetch the scoped token and blob storage
    # location for the current store object.
    # in contrast, the ModelsArtifactRepo just delegates to its underlying artifact repo
    # based on the URI
    # We could mock the result of getting download URI and scoped tokens, and make sure the
    # cloud-appropriate artifact repo is used to download artifacts. That's prob the best
    # thing to test here, the real request flow for download
    artifact_location = "s3://blah_bucket/"
    mock_response = mock.MagicMock(autospec=Response)
    mock_response.status_code = 200
    fake_key_id = "fake_key_id"
    fake_secret_access_key = "fake_secret_access_key"
    fake_session_token = "fake_session_token"
    temporary_creds = {
        "aws_temp_credentials": {
            "access_key_id": fake_key_id,
            "secret_access_key": fake_secret_access_key,
            "session_token": fake_session_token
        }
    }
    mock_response.text = json.dumps({"credentials": temporary_creds})
    with mock.patch.object(
            MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch(
        "mlflow.utils.rest_utils.http_request"
    ) as request_mock, mock.patch(
        "mlflow.store.artifact.s3_artifact_repo.S3ArtifactRepository") as s3_artifact_repo_class_mock:
        mock_s3_repo = mock.MagicMock(autospec=S3ArtifactRepository)
        s3_artifact_repo_class_mock.return_value = mock_s3_repo
        request_mock.return_value = mock_response
        model_uri = "models:/MyModel/12"
        models_repo = UnityCatalogModelsArtifactRepository(model_uri, _DATABRICKS_UNITY_CATALOG_SCHEME)
        models_repo.download_artifacts("artifact_path", "dst_path")
        s3_artifact_repo_class_mock.assert_called_once_with(artifact_uri=artifact_location, access_key_id=fake_key_id, secret_access_key=fake_secret_access_key, session_token=fake_session_token)
        mock_s3_repo.download_artifacts.assert_called_once()
