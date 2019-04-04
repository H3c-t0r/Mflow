from click.testing import CliRunner
from mock import mock

from mlflow.cli import server, run


def test_server_uri_validation():
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--backend-store-uri", "sqlite://"])
        assert result.output.startswith("Option 'default-artifact-root' is required")
        run_server_mock.assert_not_called()

    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # SQLAlchemy expects postgresql:// not postgres://
        result = CliRunner().invoke(server,
                                    ["--backend-store-uri", "postgres://user:pwd@host:5432/mydb",
                                     "--default-artifact-root", "./mlruns"])
        assert result.output.startswith("Error related to option backend-store-uri")
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server,
                                    ["--backend-store-uri", "sqlite://invalid-sqlite-url",
                                     "--default-artifact-root", "./mlruns"])
        assert result.output.startswith("Error related to option backend-store-uri")
        run_server_mock.assert_not_called()

    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--default-artifact-root", "bad-scheme://afdf/dfd"])
        assert result.output.startswith("Error related to option default-artifact-root")
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--default-artifact-root", "s3://unauthorized-bucket"])
        assert result.output.startswith("Error related to option default-artifact-root")
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server,
                                    ["--default-artifact-root", "gs://some-bucket"],
                                    env={'GOOGLE_APPLICATION_CREDENTIALS': '/etc/not-exists.json'})
        assert result.output.startswith("Error related to option default-artifact-root")
        run_server_mock.assert_not_called()


def test_server_static_prefix_validation():
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        CliRunner().invoke(server)
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--static-prefix", "/mlflow"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "mlflow/"])
        assert "--static-prefix must begin with a '/'." in result.output
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "/mlflow/"])
        assert "--static-prefix should not end with a '/'." in result.output
        run_server_mock.assert_not_called()


def test_mlflow_run():
    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(run)
        mock_projects.run.assert_not_called()
        assert 'Missing argument "URI"' in result.output

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["--experiment-id", "5", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["--experiment-name", "random name", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(run, ["--experiment-id", "51",
                                          "--experiment-name", "name blah", "uri"])
        mock_projects.run.assert_not_called()
        assert "Specify only one of 'experiment-name' or 'experiment-id' options." in result.output
