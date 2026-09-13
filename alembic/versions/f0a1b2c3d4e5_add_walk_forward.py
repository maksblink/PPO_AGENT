"""Add walk-forward protocol/cycles and explicit run windows, without deleting history."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "f0a1b2c3d4e5"
down_revision = "97d8e21241ba"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "walk_forward_studies",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("protocol_sha256", sa.String(64), nullable=False),
        sa.Column("protocol", JSONB(), nullable=False),
        sa.Column("plan", JSONB(), nullable=False),
        sa.Column("git_commit", sa.String(40), nullable=False),
        sa.Column("git_branch", sa.String(200), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_walk_forward_studies_name"),
    )
    op.create_table(
        "walk_forward_cycles",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("study_id", sa.BigInteger(), sa.ForeignKey("walk_forward_studies.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("plan", JSONB(), nullable=False),
        sa.Column("source_checkpoint_id", sa.BigInteger(), sa.ForeignKey("checkpoints.id")),
        sa.Column("selected_checkpoint_id", sa.BigInteger(), sa.ForeignKey("checkpoints.id")),
        sa.Column("refit_checkpoint_id", sa.BigInteger(), sa.ForeignKey("checkpoints.id")),
        sa.Column("test_evaluation_id", sa.BigInteger(), sa.ForeignKey("evaluations.id")),
        sa.Column("reference_evaluation_id", sa.BigInteger(), sa.ForeignKey("evaluations.id")),
        sa.Column("selection", JSONB()),
        sa.Column("error", sa.Text()),
        sa.UniqueConstraint("study_id", "number", name="uq_walk_forward_cycles_study_id"),
        sa.CheckConstraint("number >= 1", name=op.f("ck_walk_forward_cycles_positive_number")),
    )
    op.add_column("runs", sa.Column("window_metadata", JSONB()))
    op.add_column("runs", sa.Column("cycle_id", sa.BigInteger()))
    op.add_column("runs", sa.Column("stage_role", sa.String(32)))
    op.add_column("runs", sa.Column("candidate_id", sa.String(64)))
    op.add_column("runs", sa.Column("stage_summary", JSONB()))
    op.create_foreign_key("fk_runs_cycle_id_walk_forward_cycles", "runs", "walk_forward_cycles", ["cycle_id"], ["id"])
    op.drop_constraint(op.f("ck_runs_validation_rows_positive"), "runs", type_="check")
    op.create_check_constraint(op.f("ck_runs_validation_rows_positive"), "runs", "validation_rows >= 0")


def downgrade():
    # Never discard protocol history or make no-validation runs invalid silently.
    bind = op.get_bind()
    if bind.execute(sa.text("SELECT EXISTS(SELECT 1 FROM walk_forward_studies) OR EXISTS(SELECT 1 FROM runs WHERE validation_rows=0 OR window_metadata IS NOT NULL)")).scalar():
        raise RuntimeError("Downgrade refused: walk-forward history exists. Archive it and use an explicit migration.")
    op.drop_constraint(op.f("ck_runs_validation_rows_positive"), "runs", type_="check")
    op.create_check_constraint(op.f("ck_runs_validation_rows_positive"), "runs", "validation_rows >= 1")
    op.drop_constraint("fk_runs_cycle_id_walk_forward_cycles", "runs", type_="foreignkey")
    for column in ("stage_summary", "candidate_id", "stage_role", "cycle_id", "window_metadata"):
        op.drop_column("runs", column)
    op.drop_table("walk_forward_cycles")
    op.drop_table("walk_forward_studies")
