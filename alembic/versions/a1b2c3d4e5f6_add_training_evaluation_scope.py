"""Allow frozen checkpoint evaluation on the effective training range."""
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "f0a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("evaluation_data_scope", "evaluations", type_="check")
    op.create_check_constraint("evaluation_data_scope", "evaluations",
        "data_scope IN ('run_training', 'run_validation', 'extended_out_of_sample', 'custom_range')")


def downgrade():
    # Existing TRAIN rows must be explicitly removed before downgrading.
    op.drop_constraint("evaluation_data_scope", "evaluations", type_="check")
    op.create_check_constraint("evaluation_data_scope", "evaluations",
        "data_scope IN ('run_validation', 'extended_out_of_sample', 'custom_range')")
