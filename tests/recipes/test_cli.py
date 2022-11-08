import pytest

import mlflow.recipes.cli as recipes_cli

from click.testing import CliRunner
from mlflow.exceptions import MlflowException
from mlflow.recipes.utils import _RECIPE_PROFILE_ENV_VAR

# pylint: disable=unused-import
from tests.recipes.helper_functions import (
    enter_recipe_example_directory,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate", "register"]


@pytest.fixture
def clean_up_recipe():
    try:
        yield
    finally:
        CliRunner().invoke(recipes_cli.clean, env={_RECIPE_PROFILE_ENV_VAR: "local"})


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
@pytest.mark.parametrize("step", _STEP_NAMES)
def test_recipes_cli_step_works(step):
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert (
            CliRunner()
            .invoke(command, args=f"--step {step}", env={_RECIPE_PROFILE_ENV_VAR: "local"})
            .exit_code
            == 0
        )
        assert CliRunner().invoke(command, args=f"--step {step} --profile=local").exit_code == 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_flow_completes_successfully():
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert CliRunner().invoke(command, env={_RECIPE_PROFILE_ENV_VAR: "local"}).exit_code == 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_fails_without_profile():
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert CliRunner().invoke(command, env={_RECIPE_PROFILE_ENV_VAR: ""}).exit_code != 0
        assert CliRunner().invoke(command).exit_code != 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_fails_with_illegal_profile():
    result = CliRunner().invoke(recipes_cli.clean, env={_RECIPE_PROFILE_ENV_VAR: "illegal_profile"})
    assert result.exit_code != 0
    assert isinstance(result.exception, MlflowException)


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_works_with_non_default_profile():
    result = CliRunner().invoke(recipes_cli.clean, env={_RECIPE_PROFILE_ENV_VAR: "databricks"})
    assert "with profile: 'databricks'" in result.output
    assert result.exit_code == 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_works():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        args="--artifact model",
        env={_RECIPE_PROFILE_ENV_VAR: "local"},
    )
    assert result.exit_code == 0
    assert result.output is not None


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_with_bad_artifact_name_fails():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        args="--artifact foo",
        env={_RECIPE_PROFILE_ENV_VAR: "local"},
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, MlflowException)


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_with_no_artifact_name_fails():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        env={_RECIPE_PROFILE_ENV_VAR: "local"},
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
