from pathlib import Path

from sqlalchemy.engine.base import Engine
from alembic.command import upgrade
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.migration import MigrationContext


def _get_alembic_dir() -> str:
    return Path(__file__).parent / "migrations"


def _get_alembic_config() -> Config:
    alembic_dir = _get_alembic_dir()
    alembic_ini_path = alembic_dir / "alembic.ini"
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    return alembic_cfg


def migrate(engine: Engine) -> None:
    alembic_cfg = _get_alembic_config()
    with engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        upgrade(alembic_cfg, "head")


def migrate_if_needed(engine: Engine) -> None:
    alembic_cfg = _get_alembic_config()
    script_dir = ScriptDirectory.from_config(alembic_cfg)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        if context.get_current_revision() != script_dir.get_current_head():
            upgrade(alembic_cfg, "head")
