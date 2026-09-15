from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile
import sys

from sqlalchemy import select, text
import yaml

from train_and_eval.database.models import (
    Checkpoint, CheckpointSaveReason, Evaluation, EvaluationDataScope,
    EvaluationStatus, Run, RunStatus, WalkForwardCycle, WalkForwardStudy,
)
from train_and_eval.evaluation.service import evaluate_run_validation_checkpoint
from train_and_eval.market_data.load_market_data import load_market_data
from train_and_eval.reproducibility import require_clean_git
from train_and_eval.run_config import RunConfig, normalize_config, validate_resume_compatibility
from train_and_eval.training.service import train_ppo_run
from train_and_eval.training.progress import LiveTrainingProgress
from train_and_eval.walk_forward.config import build_plan, digest, load_protocol, stage_config


@contextmanager
def study_lock(session_factory, name: str):
    """A PostgreSQL session lock prevents two processes advancing the same study."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)
    with session_factory() as session:
        engine = session.get_bind()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        if not connection.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar():
            raise RuntimeError(f"Study {name!r} is already running")
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})


def select_candidate(records: list[dict]) -> dict:
    if len(records) != 1:
        raise ValueError("Stage two requires exactly one candidate")
    if not math.isfinite(records[0]["balanced_score"]):
        raise ValueError("The candidate must have a completed finite validation score")
    return records[0]


def prepare(config_path, root, factory=None):
    protocol = load_protocol(config_path)
    if factory is None:
        raise ValueError("Stage-two planning requires a database connection to resolve the checkpoint")
    with factory() as session:
        checkpoint = session.get(Checkpoint, protocol.source_checkpoint_id)
        if checkpoint is None:
            raise ValueError("Stage-one checkpoint does not exist")
        run = session.get(Run, checkpoint.run_id)
        template = dict(run.normalized_config_json)
        template["continuation"] = {"mode": "fresh"}
        inherited = RunConfig.model_validate(template)
        if (not inherited.environment.force_close_on_done
                or inherited.evaluation.policy_mode != "deterministic_argmax"
                or inherited.ppo.n_steps % inherited.ppo.batch_size):
            raise ValueError("Stage two requires force_close_on_done, deterministic_argmax and batch-aligned n_steps")
        protocol = protocol.model_copy(update={"run": inherited})
    data_dir = root / "data"
    manifest_path = root / "train_and_eval/market_data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = load_market_data(protocol.run.data.path, data_directory=data_dir, manifest_path=manifest_path)
    source = _stage_one_source(factory, protocol)
    plan = build_plan(protocol, frame, manifest, source)
    plan["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return protocol, plan


def _stage_one_source(factory, protocol):
    """Freeze source identity and reject lineage that has already trained ahead."""
    with factory() as session:
        checkpoint = session.get(Checkpoint, protocol.source_checkpoint_id)
        if checkpoint is None:
            raise ValueError("Stage-one checkpoint does not exist")
        selected_run = session.get(Run, checkpoint.run_id)
        source_config = RunConfig.model_validate(selected_run.normalized_config_json)
        train = source_config.data.train_range
        validation = source_config.data.validation_range
        if train is None or validation is None or source_config.data.alignment != "trim_start":
            raise ValueError("Stage one requires explicit train/validation ranges and trim_start alignment")
        if source_config.run.seed != protocol.seed or source_config.environment != protocol.run.environment:
            raise ValueError("Stage-two seed and environment must match the stage-one source")
        # The source may be an ordinary resume, but every ancestor must use the same
        # bounded development period. A trading/refit branch cannot masquerade as a base.
        run = selected_run
        ancestors, seen = [], set()
        while run is not None:
            if run.id in seen:
                raise ValueError("Cyclic checkpoint ancestry")
            seen.add(run.id)
            cfg = RunConfig.model_validate(run.normalized_config_json)
            if (run.status != RunStatus.COMPLETED or run.cycle_id is not None or run.stage_role is not None
                    or cfg.data.train_range != train or cfg.data.validation_range != validation
                    or run.data_sha256 != selected_run.data_sha256 or run.data_path != protocol.run.data.path):
                raise ValueError("Stage-one ancestry must contain completed ordinary runs on the same explicit ranges and data")
            if run.window_metadata is None:
                raise ValueError("Stage-one source lacks persisted temporal window metadata")
            ancestors.append({"run_id": run.id, "config_sha256": run.normalized_config_sha256,
                              "source_checkpoint_id": run.source_checkpoint_id})
            if run.source_checkpoint_id is None:
                break
            parent = session.get(Checkpoint, run.source_checkpoint_id)
            if parent is None:
                raise ValueError("Missing ancestor checkpoint")
            run = session.get(Run, parent.run_id)
            if run is None:
                raise ValueError("Missing ancestor run")
        validated = session.scalar(select(Evaluation.id).where(
            Evaluation.checkpoint_id == checkpoint.id,
            Evaluation.data_scope == EvaluationDataScope.RUN_VALIDATION,
            Evaluation.status == EvaluationStatus.COMPLETED).limit(1))
        if validated is None:
            raise ValueError("Choose a stage-one checkpoint with a completed validation")
        # Reuse the existing architecture/observation compatibility check before any run writes.
        probe = protocol.run.model_dump(mode="json")
        probe["run"]["name"] = protocol.name + "_source_check"
        probe["continuation"] = {"mode": "resume", "source_run": selected_run.name, "checkpoint": str(checkpoint.id)}
        validate_resume_compatibility(RunConfig.model_validate(probe), source_config)
        return {"checkpoint_id": checkpoint.id, "checkpoint_sha256": checkpoint.sha256,
                "run_id": selected_run.id, "run_name": selected_run.name,
                "validation_evaluation_id": validated,
                "git_commit": selected_run.git_commit, "data_sha256": selected_run.data_sha256,
                "train_range": train.model_dump(), "validation_range": validation.model_dump(),
                "ancestors": ancestors}


def _set_cycle(factory, cycle_id, **changes):
    with factory() as session:
        row = session.get(WalkForwardCycle, cycle_id)
        for field, value in changes.items():
            setattr(row, field, value)
        session.commit()


def _source(factory, checkpoint_id):
    with factory() as session:
        checkpoint = session.get(Checkpoint, checkpoint_id)
        run = session.get(Run, checkpoint.run_id)
        if run.stage_role != "candidate" or run.status != RunStatus.COMPLETED:
            raise RuntimeError("Base lineage must come from a completed pre-refit candidate")
        return run.name, checkpoint.id


def _run_stage(factory, protocol, cycle, role, source, root, live):
    if source is None:
        raise ValueError("Every stage-two training run must resume a checkpoint")
    config = stage_config(protocol, cycle.plan, role=role, source=source)
    with factory() as session:
        existing = session.scalar(select(Run).where(Run.name == config.run.name))
        if existing is not None:
            if (existing.normalized_config_json != json.loads(normalize_config(config)) or
                    existing.cycle_id != cycle.id or existing.stage_role != role or
                    existing.source_checkpoint_id != (source[1] if source else None)):
                raise RuntimeError("Existing stage identity does not match the frozen plan")
            if existing.status != RunStatus.COMPLETED:
                raise RuntimeError(f"Run #{existing.id} is {existing.status.value}; refusing automatic retraining. "
                                   "Preserve this study and use a new study name for a restart.")
            checkpoints = list(session.scalars(select(Checkpoint).where(
                Checkpoint.run_id == existing.id, Checkpoint.save_reason == CheckpointSaveReason.FINAL)))
            if len(checkpoints) != 1:
                raise RuntimeError("Completed stage must have one final checkpoint")
            return existing.id, checkpoints[0].id
    print(f"Cycle {cycle.number}: {role} | LR={config.ppo.learning_rate} | {config.data.train_range.start} -> {config.data.train_range.end}", flush=True)
    window = cycle.plan["refit_window" if role == "refit" else "candidate_window"]
    print(f"  available scored steps={window['train']['rows']}, context={window['history_rows']}, "
          f"trimmed={window['trimmed_steps']}, prepended={window['prepended_steps']}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ppo-walk-forward-") as temporary:
        path = Path(temporary) / "stage.yml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
        result = train_ppo_run(factory, config_path=path, project_root=root,
                               data_directory=root/"data", manifest_path=root/"train_and_eval/market_data/manifest.json",
                               cycle_id=cycle.id, stage_role=role, candidate_id="0" if role == "candidate" else None,
                               progress_reporter=LiveTrainingProgress(stream=sys.stdout) if live else None)
    return result.run.run_id, result.checkpoint.checkpoint_id


def _evaluation(factory, checkpoint_id, bounds, scope, root):
    # Reuse a completed exact evaluation after interruption, never overwrite its result.
    with factory() as session:
        checkpoint = session.get(Checkpoint, checkpoint_id)
        seed = session.get(Run, checkpoint.run_id).seed
        rows = list(session.scalars(select(Evaluation).where(
            Evaluation.checkpoint_id == checkpoint_id, Evaluation.data_scope == scope,
            Evaluation.policy_mode == "deterministic_argmax", Evaluation.seed == seed,
            Evaluation.threshold_action.is_(None), Evaluation.probability_threshold.is_(None),
            Evaluation.evaluation_start_index == bounds["start_index"],
            Evaluation.evaluation_end_index == bounds["end_index"],
            Evaluation.status == EvaluationStatus.COMPLETED,
        )))
        if len(rows) > 1:
            raise RuntimeError("Ambiguous completed evaluation; refusing to choose using test results")
        if rows:
            return rows[0].id
    result = evaluate_run_validation_checkpoint(
        factory, checkpoint_id=checkpoint_id, trigger="final", policy_mode="deterministic_argmax",
        evaluation_range=(bounds["start_index"], bounds["end_index"]), data_scope=scope,
        project_root=root, data_directory=root/"data", manifest_path=root/"train_and_eval/market_data/manifest.json",
        persist_trajectory=True,
    )
    return result.evaluation_id


def execute(factory, config_path, *, project_root, max_cycles=None, live=True):
    root = Path(project_root).resolve()
    git = require_clean_git(root)
    protocol, plan = prepare(config_path, root, factory)
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    definition = protocol.model_dump(mode="json")
    definition["run"] = protocol.run.model_dump(mode="json")
    protocol_hash = digest({"protocol": definition, "plan": plan})
    with study_lock(factory, protocol.name):
        with factory() as session:
            study = session.scalar(select(WalkForwardStudy).where(WalkForwardStudy.name == protocol.name))
            if study is None:
                study = WalkForwardStudy(name=protocol.name, protocol=definition, protocol_sha256=protocol_hash,
                                         plan=plan, git_commit=git.commit, git_branch=git.branch, status="running")
                session.add(study)
                session.flush()
                for item in plan["cycles"]:
                    session.add(WalkForwardCycle(study_id=study.id, number=item["number"], plan=item, status="pending"))
                session.commit()
            elif study.protocol_sha256 != protocol_hash or study.git_commit != git.commit:
                raise RuntimeError("Study protocol, dataset, plan or code changed; use a new study name")
            study_id = study.id
            study.status = "running"
            session.commit()
        current_id = None
        try:
            with factory() as session:
                cycles = list(session.scalars(select(WalkForwardCycle).where(WalkForwardCycle.study_id == study_id).order_by(WalkForwardCycle.number)))
            previous = protocol.source_checkpoint_id
            for cycle in cycles:
                if max_cycles is not None and cycle.number > max_cycles:
                    break
                if cycle.status == "completed":
                    if cycle.source_checkpoint_id != previous or not cycle.selected_checkpoint_id or not cycle.test_evaluation_id:
                        raise RuntimeError("Completed cycle lineage is inconsistent")
                    previous = cycle.selected_checkpoint_id
                    continue
                current_id = cycle.id
                source = ((plan["stage_one_source"]["run_name"], previous)
                          if cycle.number == 1 else _source(factory, previous))
                _set_cycle(factory, cycle.id, source_checkpoint_id=previous, status="running", error=None)
                if cycle.selection is None:
                    records = []
                    run_id, checkpoint_id = _run_stage(factory, protocol, cycle, "candidate", source, root, live)
                    with factory() as session:
                        evaluations = list(session.scalars(select(Evaluation).where(
                            Evaluation.checkpoint_id == checkpoint_id,
                            Evaluation.data_scope == EvaluationDataScope.RUN_VALIDATION,
                            Evaluation.status == EvaluationStatus.COMPLETED)))
                        if len(evaluations) != 1:
                            raise RuntimeError("Candidate must have exactly one final validation")
                        evaluation = evaluations[0]
                        run = session.get(Run, run_id)
                        records.append({"candidate_order": 0, "learning_rate": protocol.learning_rate, "seed": protocol.seed,
                                        "run_id": run_id, "checkpoint_id": checkpoint_id,
                                        "evaluation_id": evaluation.id, "balanced_score": float(evaluation.balanced_score),
                                        "initial_policy_sha256": run.stage_summary["initial_policy_sha256"],
                                        "initial_checkpoint_id": run.stage_summary["initial_checkpoint_id"]})
                    winner = select_candidate(records)
                    selection = {"rule": protocol.selection_rule, "candidates": records, "winner": winner,
                                 "reject_updates": False, "checkpoint_policy": "final_only"}
                    _set_cycle(factory, cycle.id, selection=selection, selected_checkpoint_id=winner["checkpoint_id"])
                else:
                    selection = cycle.selection
                    winner = select_candidate(selection["candidates"])
                    if winner != selection["winner"] or winner["checkpoint_id"] != cycle.selected_checkpoint_id:
                        raise RuntimeError("Persisted selection is inconsistent")
                # Reference is diagnostic only and is never eligible for selection.
                if cycle.reference_evaluation_id is None:
                    reference = _evaluation(factory, previous, cycle.plan["candidate_window"]["validation"], EvaluationDataScope.CUSTOM_RANGE, root)
                    _set_cycle(factory, cycle.id, reference_evaluation_id=reference)
                selected = winner["checkpoint_id"]
                selected_source = _source(factory, selected)
                _, refit = _run_stage(factory, protocol, cycle, "refit", selected_source, root, live)
                _set_cycle(factory, cycle.id, refit_checkpoint_id=refit)
                # The test is first read by the evaluator only after selection and refit are persisted.
                test_id = _evaluation(factory, refit, cycle.plan["test_rows"], EvaluationDataScope.EXTENDED_OUT_OF_SAMPLE, root)
                _set_cycle(factory, cycle.id, test_evaluation_id=test_id, status="completed")
                previous = selected  # Deliberately never refit.
                print(f"Cycle {cycle.number} completed: base={selected}, trading={refit}, test={test_id}", flush=True)
                current_id = None
            with factory() as session:
                remaining = session.scalar(select(WalkForwardCycle.id).where(WalkForwardCycle.study_id == study_id, WalkForwardCycle.status != "completed").limit(1))
                session.get(WalkForwardStudy, study_id).status = "paused" if remaining is not None else "completed"
                session.commit()
        except BaseException as error:
            if current_id is not None:
                _set_cycle(factory, current_id, status="failed", error=f"{type(error).__name__}: {error}")
            with factory() as session:
                session.get(WalkForwardStudy, study_id).status = "failed"
                session.commit()
            raise
    return study_id
