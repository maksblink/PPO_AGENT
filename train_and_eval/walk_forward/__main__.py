from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Frozen-protocol, two-branch PPO walk-forward training")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
        if name == "run":
            sub.add_argument("--max-cycles", type=int, help="Execute through this cycle number; rerun without it to continue")
            sub.add_argument("--plain-output", action="store_true", help="Print study snapshots at start, cycle completion and exit instead of refreshing a live panel")
        else:
            sub.add_argument("--output", type=Path, help="Optional JSON plan destination, e.g. /tmp/ppo_plan.json")
    report = commands.add_parser("report")
    report.add_argument("--study-id", required=True, type=int)
    report.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    from train_and_eval.walk_forward.service import prepare, execute
    if args.command == "plan":
        from train_and_eval.database.session import create_database_engine, create_session_factory
        engine = create_database_engine()
        try:
            protocol, plan = prepare(args.config, args.project_root.resolve(), create_session_factory(engine))
        finally:
            engine.dispose()
        print(f"Study: {protocol.name}; seed={protocol.seed}; full tests={len(plan['cycles'])}")
        print(f"Initial A: {plan['initial_train']}")
        if "stage_one_source" in plan:
            print(f"Stage-one source: {plan['stage_one_source']}")
        for cycle in plan["cycles"][:3]:
            window = cycle["candidate_window"]
            print(f"Cycle {cycle['number']}: train={cycle['update']}; validation={cycle['validation']}; test={cycle['test']}")
            print(f"  context={window['history_rows']}, trimmed={window['trimmed_steps']}, prepended={window['prepended_steps']}, train steps={window['train']['rows']}, test steps={cycle['test_rows']['rows']}")
        print(f"Unused tail: [{plan['unused_tail_start']}, {plan['unused_tail_end']})")
        if args.output:
            args.output.write_text(json.dumps(plan, indent=2))
            print(args.output.resolve())
        return
    from train_and_eval.database.session import create_database_engine, create_session_factory
    from train_and_eval.walk_forward.reporting import generate_report
    engine = create_database_engine()
    try:
        factory = create_session_factory(engine)
        study_id = args.study_id if args.command == "report" else execute(
            factory, args.config, project_root=args.project_root, max_cycles=args.max_cycles, live=not args.plain_output)
        print(f"Study ID: {study_id}")
        print(generate_report(factory, study_id, project_root=args.project_root))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
