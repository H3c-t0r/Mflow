"""Tests verifying that the SQLAlchemyStore generates the expected database schema"""
import os

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from alembic.migration import MigrationContext  # pylint: disable=import-error
from alembic.autogenerate import compare_metadata
import sqlalchemy
from sqlalchemy.schema import MetaData

import mlflow.db
from mlflow.exceptions import MlflowException
from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.dbmodels.models import Base
from mlflow.store.sqlalchemy_store import SqlAlchemyStore
from tests.resources.db.initial_models import Base as InitialBase
from tests.store.dump_schema import dump_sqlalchemy_store_schema, dump_db_schema
from tests.integration.utils import invoke_cli_runner


def _assert_schema_files_equal(generated_schema_file, expected_schema_file):
    """
    Assert equivalence of two SQL schema dump files consisting of CREATE TABLE statements delimited
    by double-newlines, allowing for the reordering of individual lines within each CREATE TABLE
    statement to account for differences in schema-dumping across platforms & Python versions.
    """
    # Extract "CREATE TABLE" statement chunks from both files, assuming tables are listed in the
    # same order across files
    with open(generated_schema_file, "r") as generated_schema_handle:
        generated_schema_table_chunks = generated_schema_handle.read().split("\n\n")
    with open(expected_schema_file, "r") as expected_schema_handle:
        expected_schema_table_chunks = expected_schema_handle.read().split("\n\n")
    # Compare the two files table-by-table. We assume each CREATE TABLE statement is valid and
    # so sort the lines within the statements before comparing them.
    for generated_schema_table, expected_schema_table \
            in zip(generated_schema_table_chunks, expected_schema_table_chunks):
        generated_lines = sorted(generated_schema_table.split("\n"))
        expected_lines = sorted(expected_schema_table.split("\n"))
        assert generated_lines == expected_lines,\
            "Generated schema did not match expected schema. Generated schema had table " \
            "definition:\n{generated_table}\nExpected schema had table definition:" \
            "\n{expected_table}\nIf you intended to make schema changes, run " \
            "'python tests/store/dump_schema.py {expected_file} from your checkout of MLflow to " \
            "update the schema snapshot.".format(
                generated_table=generated_schema_table, expected_table=expected_schema_table,
                expected_file=expected_schema_file)


@pytest.fixture()
def expected_schema_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yield os.path.normpath(
        os.path.join(current_dir, os.pardir, "resources", "db", "latest_schema.sql"))


@pytest.fixture()
def db_url(tmpdir):
    return "sqlite:///%s" % tmpdir.join("db_file").strpath


def test_sqlalchemystore_generates_up_to_date_schema(tmpdir, expected_schema_file):
    generated_schema_file = tmpdir.join("generated-schema.sql").strpath
    dump_sqlalchemy_store_schema(dst_file=generated_schema_file)
    _assert_schema_files_equal(generated_schema_file, expected_schema_file)


def test_running_migrations_generates_expected_schema(tmpdir, expected_schema_file, db_url):
    """Test that migrating an existing database generates the desired schema."""
    engine = sqlalchemy.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    invoke_cli_runner(mlflow.db.commands, ['upgrade', db_url])
    engine = sqlalchemy.create_engine(db_url)
    created_tables_metadata = MetaData(bind=engine)
    created_tables_metadata.reflect()
    generated_schema_file = tmpdir.join("generated-schema.sql").strpath
    dump_db_schema(created_tables_metadata, generated_schema_file)
    _assert_schema_files_equal(generated_schema_file, expected_schema_file)


def test_sqlalchemy_store_detects_schema_mismatch(
        tmpdir, db_url):  # pylint: disable=unused-argument
    def _assert_invalid_schema(engine):
        with pytest.raises(MlflowException) as ex:
            SqlAlchemyStore._verify_schema(engine)
            assert ex.message.contains("Detected out-of-date database schema.")

    # Initialize an empty database & verify that we detect a schema mismatch
    engine = sqlalchemy.create_engine(db_url)
    _assert_invalid_schema(engine)
    # Create legacy tables, verify schema is still out of date
    InitialBase.metadata.create_all(engine)
    _assert_invalid_schema(engine)
    # Run each migration. Until the last one, schema should be out of date
    config = _get_alembic_config(db_url)
    script = ScriptDirectory.from_config(config)
    revisions = list(script.walk_revisions())
    revisions.reverse()
    for rev in revisions[:-1]:
        command.upgrade(config, rev.revision)
        _assert_invalid_schema(engine)
    # Run migrations, schema verification should now pass
    invoke_cli_runner(mlflow.db.commands, ['upgrade', db_url])
    SqlAlchemyStore._verify_schema(engine)


def test_store_generated_schema_matches_base(tmpdir, db_url):
    # Create a SQLAlchemyStore against tmpfile, directly verify that tmpfile contains a
    # database with a valid schema
    SqlAlchemyStore(db_url, tmpdir.join("ARTIFACTS").strpath)
    engine = sqlalchemy.create_engine(db_url)
    mc = MigrationContext.configure(engine.connect())
    diff = compare_metadata(mc, Base.metadata)
    assert len(diff) == 0
