from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Experiment queue
#
# Order here == execution order.
# 00_base_seed1.yml is intentionally NOT included.
# ---------------------------------------------------------------------------

CONFIGS = [
    "configs/experiments/nq1h_search_v1/01_nepochs2_lr2p25e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/02_nepochs2_lr2p5e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/03_nepochs2_lr2p75e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/04_nepochs2_lr3e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/05_nepochs2_lr4e4_seed1.yml",

    "configs/experiments/nq1h_search_v1/06_nepochs3_lr2p25e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/07_nepochs3_lr2p5e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/08_nepochs3_lr2p75e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/09_nepochs3_lr3e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/10_nepochs3_lr4e4_seed1.yml",
]


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


RUN_RE = re.compile(
    r"PPO RUN #(?P<run_id>\d+)\s+(?P<run_name>\S+)"
)

VALIDATION_RE = re.compile(
    rf"balanced_score\s+(?P<balanced_score>{NUMBER})"
    rf".*?agent_return\s+(?P<agent_return>{NUMBER})"
    rf".*?max_drawdown\s+(?P<max_drawdown>{NUMBER})"
    rf".*?profit_factor\s+(?P<profit_factor>{NUMBER})"
    rf".*?win_rate\s+(?P<win_rate>{NUMBER})"
    rf".*?exposure\s+(?P<exposure>{NUMBER})"
    rf".*?round_trips\s+(?P<round_trips>[\d,]+)"
    rf".*?always_long\s+(?P<always_long>{NUMBER})"
)

BEST_RE = re.compile(
    rf"Best balanced_score\s+(?P<best_score>{NUMBER})"
    r"\s+\|\s+checkpoint\s+#(?P<checkpoint>\d+)"
    r"\s+@\s+run step\s+(?P<step>[\d,]+)"
)

TRAIN_LINE_1_RE = re.compile(
    rf"ep_reward\s+(?P<ep_reward>{NUMBER})"
    rf".*?approx_kl\s+(?P<approx_kl>{NUMBER})"
    rf".*?clip_fraction\s+(?P<clip_fraction>{NUMBER})"
)

TRAIN_LINE_2_RE = re.compile(
    rf"entropy_loss\s+(?P<entropy_loss>{NUMBER})"
    rf".*?explained_variance\s+(?P<explained_variance>{NUMBER})"
    rf".*?learning_rate\s+(?P<learning_rate>{NUMBER})"
)


@dataclass
class ConfigMeta:
    path: Path
    name: str
    seed: int
    n_epochs: int
    learning_rate: float


@dataclass
class RunResult:
    queue_index: int
    config: ConfigMeta

    status: str = "PENDING"

    run_id: int | None = None
    run_name: str | None = None

    balanced_score: float | None = None
    agent_return: float | None = None
    max_drawdown: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    exposure: float | None = None
    round_trips: int | None = None
    always_long: float | None = None

    best_score: float | None = None
    best_checkpoint: int | None = None
    best_step: int | None = None

    approx_kl: float | None = None
    clip_fraction: float | None = None
    entropy_loss: float | None = None
    explained_variance: float | None = None

    wall_seconds: float | None = None


def find_single(text: str, pattern: str, field: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)

    if len(matches) != 1:
        raise RuntimeError(
            f"{field}: expected exactly one match, found {len(matches)}"
        )

    return matches[0]


def read_config_meta(relative_path: str) -> ConfigMeta:
    path = ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")

    text = path.read_text()

    name = find_single(
        text,
        r"^\s*name:\s*(\S+)\s*$",
        "name",
    )

    seed = int(
        find_single(
            text,
            r"^\s*seed:\s*(\d+)\s*$",
            "seed",
        )
    )

    n_epochs = int(
        find_single(
            text,
            r"^\s*n_epochs:\s*(\d+)\s*$",
            "n_epochs",
        )
    )

    learning_rate = float(
        find_single(
            text,
            rf"^\s*learning_rate:\s*({NUMBER})\s*$",
            "learning_rate",
        )
    )

    return ConfigMeta(
        path=path,
        name=name,
        seed=seed,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )


def assert_git_clean() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Could not check git status:\n"
            + completed.stderr
        )

    dirty = completed.stdout.strip()

    if dirty:
        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or discard changes before training.\n\n"
            + dirty
        )


def print_queue(configs: list[ConfigMeta], start_at: int) -> None:
    print()
    print("=" * 92)
    print("PPO SEARCH QUEUE")
    print("=" * 92)
    print(
        f"{'#':>3}  "
        f"{'n_epochs':>8}  "
        f"{'learning_rate':>13}  "
        f"{'seed':>4}  "
        f"name"
    )
    print("-" * 92)

    for i, cfg in enumerate(configs, start=1):
        marker = "->" if i == start_at else "  "

        print(
            f"{marker} "
            f"{i:>2}  "
            f"{cfg.n_epochs:>8}  "
            f"{cfg.learning_rate:>13.8f}  "
            f"{cfg.seed:>4}  "
            f"{cfg.name}"
        )

    print("=" * 92)
    print()


def parse_output_line(result: RunResult, line: str) -> None:
    match = RUN_RE.search(line)

    if match:
        result.run_id = int(match.group("run_id"))
        result.run_name = match.group("run_name")

    match = VALIDATION_RE.search(line)

    if match:
        # Every validation updates these fields.
        # Therefore after the process finishes they contain the FINAL
        # validation because it is the last validation printed.
        result.balanced_score = float(
            match.group("balanced_score")
        )
        result.agent_return = float(
            match.group("agent_return")
        )
        result.max_drawdown = float(
            match.group("max_drawdown")
        )
        result.profit_factor = float(
            match.group("profit_factor")
        )
        result.win_rate = float(
            match.group("win_rate")
        )
        result.exposure = float(
            match.group("exposure")
        )
        result.round_trips = int(
            match.group("round_trips").replace(",", "")
        )
        result.always_long = float(
            match.group("always_long")
        )

    match = BEST_RE.search(line)

    if match:
        result.best_score = float(
            match.group("best_score")
        )
        result.best_checkpoint = int(
            match.group("checkpoint")
        )
        result.best_step = int(
            match.group("step").replace(",", "")
        )

    match = TRAIN_LINE_1_RE.search(line)

    if match:
        result.approx_kl = float(
            match.group("approx_kl")
        )
        result.clip_fraction = float(
            match.group("clip_fraction")
        )

    match = TRAIN_LINE_2_RE.search(line)

    if match:
        result.entropy_loss = float(
            match.group("entropy_loss")
        )
        result.explained_variance = float(
            match.group("explained_variance")
        )


def run_config(
    queue_index: int,
    config: ConfigMeta,
) -> RunResult:
    result = RunResult(
        queue_index=queue_index,
        config=config,
        status="RUNNING",
    )

    command = [
        sys.executable,
        "-m",
        "train_and_eval.training",
        "--config",
        str(config.path.relative_to(ROOT)),
        "--traceback",
    ]

    print()
    print("#" * 100)
    print(
        f"QUEUE {queue_index}/{len(CONFIGS)}"
        f" | n_epochs={config.n_epochs}"
        f" | lr={config.learning_rate}"
        f" | seed={config.seed}"
    )
    print(f"CONFIG: {config.path.relative_to(ROOT)}")
    print("#" * 100)
    print()

    started = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            parse_output_line(result, line)

        return_code = process.wait()

    except KeyboardInterrupt:
        print()
        print("Interrupted by user. Terminating current training...")
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        result.status = "INTERRUPTED"
        result.wall_seconds = time.perf_counter() - started
        raise

    result.wall_seconds = time.perf_counter() - started

    if return_code == 0:
        result.status = "COMPLETED"
    else:
        result.status = f"FAILED({return_code})"

    return result


def format_float(
    value: float | None,
    digits: int = 4,
) -> str:
    if value is None:
        return "-"

    return f"{value:.{digits}f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value * 100:+.2f}%"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"

    minutes = int(value // 60)
    seconds = int(value % 60)

    return f"{minutes:02d}:{seconds:02d}"


def print_summary(results: list[RunResult]) -> None:
    if not results:
        print()
        print("No runs were executed.")
        return

    print()
    print()
    print("=" * 154)
    print("FINAL SEARCH SUMMARY")
    print("=" * 154)

    header = (
        f"{'#':>3} "
        f"{'run':>4} "
        f"{'ep':>3} "
        f"{'LR':>9} "
        f"{'score':>10} "
        f"{'return':>9} "
        f"{'maxDD':>9} "
        f"{'PF':>8} "
        f"{'win':>8} "
        f"{'exposure':>9} "
        f"{'trips':>7} "
        f"{'best':>10} "
        f"{'KL':>10} "
        f"{'clip':>8} "
        f"{'time':>6} "
        f"{'status':>12}"
    )

    print(header)
    print("-" * 154)

    for result in results:
        print(
            f"{result.queue_index:>3} "
            f"{str(result.run_id or '-'):>4} "
            f"{result.config.n_epochs:>3} "
            f"{result.config.learning_rate:>9.6f} "
            f"{format_float(result.balanced_score, 5):>10} "
            f"{format_percent(result.agent_return):>9} "
            f"{format_percent(result.max_drawdown):>9} "
            f"{format_float(result.profit_factor, 3):>8} "
            f"{format_percent(result.win_rate):>8} "
            f"{format_percent(result.exposure):>9} "
            f"{str(result.round_trips if result.round_trips is not None else '-'):>7} "
            f"{format_float(result.best_score, 5):>10} "
            f"{format_float(result.approx_kl, 3):>10} "
            f"{format_float(result.clip_fraction, 4):>8} "
            f"{format_seconds(result.wall_seconds):>6} "
            f"{result.status:>12}"
        )

    print("=" * 154)

    completed = [
        result
        for result in results
        if (
            result.status == "COMPLETED"
            and result.balanced_score is not None
        )
    ]

    if completed:
        ranked = sorted(
            completed,
            key=lambda result: result.balanced_score,
            reverse=True,
        )

        print()
        print("RANKING BY FINAL balanced_score")
        print("-" * 92)

        for rank, result in enumerate(ranked, start=1):
            print(
                f"{rank:>2}. "
                f"score={result.balanced_score:+.6f} | "
                f"return={format_percent(result.agent_return)} | "
                f"maxDD={format_percent(result.max_drawdown)} | "
                f"PF={format_float(result.profit_factor, 3)} | "
                f"epochs={result.config.n_epochs} | "
                f"lr={result.config.learning_rate:.8f} | "
                f"run=#{result.run_id} | "
                f"{result.config.name}"
            )

        winner = ranked[0]

        print()
        print("WINNER")
        print("-" * 92)
        print(f"Run:             #{winner.run_id}")
        print(f"Name:            {winner.config.name}")
        print(f"n_epochs:        {winner.config.n_epochs}")
        print(f"learning_rate:   {winner.config.learning_rate}")
        print(f"seed:            {winner.config.seed}")
        print(
            f"balanced_score:  "
            f"{winner.balanced_score:+.8f}"
        )
        print(
            f"agent_return:    "
            f"{format_percent(winner.agent_return)}"
        )
        print(
            f"max_drawdown:    "
            f"{format_percent(winner.max_drawdown)}"
        )
        print(
            f"profit_factor:   "
            f"{format_float(winner.profit_factor, 5)}"
        )
        print(
            f"exposure:        "
            f"{format_percent(winner.exposure)}"
        )
        print(
            f"round_trips:     "
            f"{winner.round_trips}"
        )
        print(
            f"best_score:      "
            f"{format_float(winner.best_score, 8)}"
        )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PPO experiment configs sequentially "
            "and print a final comparison."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue without starting training.",
    )

    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help=(
            "Start from this 1-based queue position. "
            "Useful after an interrupted run."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configs = [
        read_config_meta(path)
        for path in CONFIGS
    ]

    if args.start_at < 1 or args.start_at > len(configs):
        raise SystemExit(
            f"--start-at must be between 1 and {len(configs)}"
        )

    print_queue(configs, args.start_at)

    if args.dry_run:
        print("DRY RUN: no training was started.")
        return 0

    assert_git_clean()

    selected = configs[args.start_at - 1:]

    results: list[RunResult] = []

    try:
        for offset, config in enumerate(
            selected,
            start=args.start_at,
        ):
            result = run_config(offset, config)
            results.append(result)

            if result.status != "COMPLETED":
                print()
                print(
                    f"Stopping queue because experiment "
                    f"{offset} ended with status "
                    f"{result.status}."
                )
                break

    except KeyboardInterrupt:
        print()
        print("Queue interrupted.")

    finally:
        print_summary(results)

    failed = any(
        result.status != "COMPLETED"
        for result in results
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
