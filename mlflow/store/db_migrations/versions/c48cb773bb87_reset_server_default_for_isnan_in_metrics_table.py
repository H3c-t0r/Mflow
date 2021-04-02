"""reset_server_default_for_isnan_in_metrics_table

Revision ID: c48cb773bb87
Revises: 39d1c3be5f05
Create Date: 2021-04-02 15:43:28.466043

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c48cb773bb87"
down_revision = "39d1c3be5f05"
branch_labels = None
depends_on = None


def upgrade():
    # In 39d1c3be5f05_add_is_nan_constraint_for_metrics_tables_if_necessary.py
    # (added in MLflow 1.15.0), `alter_column` is called on the `is_nan` column in the `metrics`
    # table without specifying `existing_server_default`. This alters the column default value to
    # NULL in MySQL (see the doc below).
    #
    # https://alembic.sqlalchemy.org/en/latest/ops.html#alembic.operations.Operations.alter_column
    #
    # This migration reverts this change by setting the default column value to "0".
    with op.batch_alter_table("metrics") as batch_op:
        batch_op.alter_column(
            "is_nan",
            type_=sa.types.Boolean(create_constraint=True),
            nullable=False,
            server_default="0",
        )


def downgrade():
    pass
