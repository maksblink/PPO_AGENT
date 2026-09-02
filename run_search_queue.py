from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


QUEUE_DIRECTORY = (
    ROOT / "configs" / "search_queues"
)
QUEUE_SCHEMA_VERSION = 1
DEFAULT_QUEUE = "nq1h_search_v1"


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
    gamma: float
    gae_lambda: float
    ent_coef: float
    exposure_penalty: float


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


def discover_queue_manifests() -> dict[str, Path]:
    if not QUEUE_DIRECTORY.is_dir():
        raise RuntimeError(
            f"Queue directory does not exist: {QUEUE_DIRECTORY}"
        )

    manifests: dict[str, Path] = {}

    paths = sorted(
        [
            *QUEUE_DIRECTORY.glob("*.yml"),
            *QUEUE_DIRECTORY.glob("*.yaml"),
        ]
    )

    for path in paths:
        queue_name = path.stem

        if queue_name in manifests:
            raise RuntimeError(
                f"Duplicate queue manifest name: {queue_name}"
            )

        manifests[queue_name] = path

    if not manifests:
        raise RuntimeError(
            f"No queue manifests found in {QUEUE_DIRECTORY}"
        )

    if DEFAULT_QUEUE not in manifests:
        raise RuntimeError(
            f"Default queue manifest is missing: {DEFAULT_QUEUE}"
        )

    return manifests


def load_queue_config_paths(queue_name: str) -> list[str]:
    manifests = discover_queue_manifests()

    try:
        path = manifests[queue_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown experiment queue: {queue_name}"
        ) from exc

    try:
        payload = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"Invalid queue YAML: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Queue manifest must contain a mapping: {path}"
        )

    if payload.get("queue_schema_version") != QUEUE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported queue_schema_version in {path}; "
            f"expected {QUEUE_SCHEMA_VERSION}."
        )

    manifest_name = payload.get("name")

    if manifest_name != queue_name:
        raise RuntimeError(
            f"Queue name in {path} must be {queue_name!r}; "
            f"found {manifest_name!r}."
        )

    raw_configs = payload.get("configs")

    if not isinstance(raw_configs, list) or not raw_configs:
        raise RuntimeError(
            f"Queue configs must be a non-empty list: {path}"
        )

    config_paths: list[str] = []
    seen: set[str] = set()
    root = ROOT.resolve()

    for index, raw_config in enumerate(raw_configs, start=1):
        if not isinstance(raw_config, str) or not raw_config.strip():
            raise RuntimeError(
                f"Queue config #{index} in {path} "
                "must be a non-empty string."
            )

        relative = Path(raw_config)

        if relative.is_absolute():
            raise RuntimeError(
                f"Queue config #{index} in {path} "
                "must be relative to the repository root."
            )

        resolved = (ROOT / relative).resolve()

        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Queue config #{index} in {path} "
                "escapes the repository root."
            ) from exc

        normalized = relative.as_posix()

        if normalized in seen:
            raise RuntimeError(
                f"Duplicate queue config in {path}: {normalized}"
            )

        seen.add(normalized)
        config_paths.append(normalized)

    return config_paths


def find_single(text: str, pattern: str, field: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)

    if len(matches) != 1:
        raise RuntimeError(
            f"{field}: expected exactly one match, found {len(matches)}"
        )

    return matches[0]


def parse_queue_selection(value: str) -> tuple[int, ...]:
    positions: set[int] = set()

    for raw_token in value.split(","):
        token = raw_token.strip()

        if not token:
            raise argparse.ArgumentTypeError(
                "Queue selection contains an empty item."
            )

        if "-" in token:
            parts = token.split("-")

            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid queue range: {token!r}."
                )

            try:
                first = int(parts[0].strip())
                last = int(parts[1].strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid queue range: {token!r}."
                ) from exc

            if first < 1 or last < 1:
                raise argparse.ArgumentTypeError(
                    "Queue positions must be positive integers."
                )

            if first > last:
                raise argparse.ArgumentTypeError(
                    f"Queue range start exceeds its end: {token!r}."
                )

            positions.update(range(first, last + 1))
            continue

        try:
            position = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid queue position: {token!r}."
            ) from exc

        if position < 1:
            raise argparse.ArgumentTypeError(
                "Queue positions must be positive integers."
            )

        positions.add(position)

    if not positions:
        raise argparse.ArgumentTypeError(
            "At least one queue position is required."
        )

    return tuple(sorted(positions))


def resolve_queue_selection(
    configs: list[ConfigMeta],
    *,
    start_at: int | None,
    selected_positions: tuple[int, ...] | None,
) -> list[tuple[int, ConfigMeta]]:
    if start_at is not None and selected_positions is not None:
        raise ValueError(
            "--start-at and --select cannot be combined."
        )

    queue_total = len(configs)

    if start_at is not None:
        if start_at < 1 or start_at > queue_total:
            raise ValueError(
                f"--start-at must be between 1 and {queue_total}."
            )

        positions = tuple(range(start_at, queue_total + 1))
    elif selected_positions is not None:
        invalid = [
            position
            for position in selected_positions
            if position > queue_total
        ]

        if invalid:
            invalid_text = ", ".join(
                str(position)
                for position in invalid
            )
            raise ValueError(
                f"Queue positions exceed queue size {queue_total}: "
                f"{invalid_text}."
            )

        positions = selected_positions
    else:
        positions = tuple(range(1, queue_total + 1))

    return [
        (position, configs[position - 1])
        for position in positions
    ]


def find_existing_selected_runs(
    selected: list[tuple[int, ConfigMeta]],
    database_results: list[RunResult],
) -> list[tuple[int, ConfigMeta, RunResult]]:
    existing_by_name = {
        result.config.name: result
        for result in database_results
        if result.status != "NOT_FOUND"
    }

    return [
        (position, config, existing_by_name[config.name])
        for position, config in selected
        if config.name in existing_by_name
    ]


def print_existing_selected_runs(
    existing: list[tuple[int, ConfigMeta, RunResult]],
) -> None:
    print()
    print("Selected configs already registered in PostgreSQL:")

    for position, config, result in existing:
        run_text = (
            "-"
            if result.run_id is None
            else f"#{result.run_id}"
        )
        print(
            f"  queue_position={position}"
            f" | run={run_text}"
            f" | status={result.status}"
            f" | {config.name}"
        )


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

    gamma = float(
        find_single(
            text,
            rf"^\s*gamma:\s*({NUMBER})\s*$",
            "gamma",
        )
    )

    gae_lambda = float(
        find_single(
            text,
            rf"^\s*gae_lambda:\s*({NUMBER})\s*$",
            "gae_lambda",
        )
    )

    ent_coef = float(
        find_single(
            text,
            rf"^\s*ent_coef:\s*({NUMBER})\s*$",
            "ent_coef",
        )
    )

    exposure_penalty = float(
        find_single(
            text,
            rf"^\s*exposure_penalty:\s*({NUMBER})\s*$",
            "exposure_penalty",
        )
    )

    return ConfigMeta(
        path=path,
        name=name,
        seed=seed,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        exposure_penalty=exposure_penalty,
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


def print_queue(
    queue_name: str,
    configs: list[ConfigMeta],
    selected_positions: set[int],
) -> None:
    print()
    print("=" * 100)
    print(f"PPO SEARCH QUEUE | {queue_name}")
    print("=" * 100)
    print(
        f"{'run':>3}  "
        f"{'#':>3}  "
        f"{'n_epochs':>8}  "
        f"{'learning_rate':>13}  "
        f"{'seed':>4}  "
        f"name"
    )
    print("-" * 100)

    for position, config in enumerate(configs, start=1):
        marker = "[x]" if position in selected_positions else "[ ]"

        print(
            f"{marker:>3}  "
            f"{position:>3}  "
            f"{config.n_epochs:>8}  "
            f"{config.learning_rate:>13.8f}  "
            f"{config.seed:>4}  "
            f"{config.name}"
        )

    print("=" * 100)
    print(
        f"Selected: {len(selected_positions)}/{len(configs)}"
    )
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
    *,
    queue_name: str,
    queue_position: int,
    queue_total: int,
    selected_index: int,
    selected_total: int,
    config: ConfigMeta,
) -> RunResult:
    result = RunResult(
        queue_index=queue_position,
        config=config,
        status="RUNNING",
    )

    command = [
        sys.executable,
        "-u",
        "-m",
        "train_and_eval.training",
        "--config",
        str(config.path.relative_to(ROOT)),
        "--traceback",
    ]

    print()
    print("=" * 100)
    print(
        f"PPO QUEUE PROGRESS "
        f"[{selected_index}/{selected_total}]"
    )
    print(f"Queue: {queue_name}")
    print(
        f"Queue position: "
        f"{queue_position}/{queue_total}"
    )
    print(
        f"Remaining after this run: "
        f"{selected_total - selected_index}"
    )
    print(
        f"Config: {config.path.relative_to(ROOT)}"
    )
    print("=" * 100)
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
            parse_output_line(result, line)
            print(line, end="", flush=True)

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


def format_scientific(
    value: float | None,
    digits: int = 3,
) -> str:
    if value is None:
        return "-"

    return f"{value:.{digits}e}"


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



DB_NULL = "__NULL__"


def _db_float(value: str) -> float | None:
    if value == DB_NULL:
        return None
    return float(value)


def _db_int(value: str) -> int | None:
    if value == DB_NULL:
        return None
    return int(value)


def load_results_from_database(
    configs: list[ConfigMeta],
) -> list[RunResult]:
    """
    Load search results from PostgreSQL.

    PostgreSQL is the source of truth for the final summary.
    Terminal output is presentation only.
    """

    sql = """
SELECT
    r.id,
    r.name,
    r.status,
    EXTRACT(EPOCH FROM (r.finished_at - r.started_at)),

    final_eval.balanced_score,
    final_eval.agent_return,
    final_eval.agent_max_drawdown,
    final_eval.profit_factor,
    final_eval.win_rate,
    final_eval.market_exposure,
    final_eval.round_trips,
    final_eval.always_long_return,

    best_eval.best_score,

    latest_tm.approx_kl,
    latest_tm.clip_fraction,
    latest_tm.entropy_loss,
    latest_tm.explained_variance

FROM runs AS r

LEFT JOIN LATERAL (
    SELECT
        e.balanced_score,
        e.agent_return,
        e.agent_max_drawdown,
        e.profit_factor,
        e.win_rate,
        e.market_exposure,
        e.round_trips,
        e.always_long_return
    FROM evaluations AS e
    JOIN checkpoints AS c
        ON c.id = e.checkpoint_id
    WHERE
        c.run_id = r.id
        AND e.status = 'completed'
        AND e.trigger = 'final'
    ORDER BY e.id DESC
    LIMIT 1
) AS final_eval ON TRUE

LEFT JOIN LATERAL (
    SELECT
        MAX(e.balanced_score) AS best_score
    FROM evaluations AS e
    JOIN checkpoints AS c
        ON c.id = e.checkpoint_id
    WHERE
        c.run_id = r.id
        AND e.status = 'completed'
) AS best_eval ON TRUE

LEFT JOIN LATERAL (
    SELECT
        tm.approx_kl,
        tm.clip_fraction,
        tm.entropy_loss,
        tm.explained_variance
    FROM training_metrics AS tm
    WHERE tm.run_id = r.id
    ORDER BY
        tm.run_step DESC,
        tm.id DESC
    LIMIT 1
) AS latest_tm ON TRUE

ORDER BY r.id;
"""

    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "ppo_agent",
        "-d",
        "ppo_agent",
        "-A",
        "-t",
        "-F",
        "\t",
        "-P",
        f"null={DB_NULL}",
        "-c",
        sql,
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Could not load search results from PostgreSQL:\n"
            + completed.stderr
        )

    config_by_name = {
        config.name: (index, config)
        for index, config in enumerate(configs, start=1)
    }

    result_by_name: dict[str, RunResult] = {}

    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue

        fields = raw_line.split("\t")

        if len(fields) != 17:
            raise RuntimeError(
                "Unexpected PostgreSQL result shape: "
                f"expected 17 fields, got {len(fields)}\n"
                f"{raw_line}"
            )

        (
            run_id,
            run_name,
            status,
            wall_seconds,
            balanced_score,
            agent_return,
            max_drawdown,
            profit_factor,
            win_rate,
            exposure,
            round_trips,
            always_long,
            best_score,
            approx_kl,
            clip_fraction,
            entropy_loss,
            explained_variance,
        ) = fields

        config_entry = config_by_name.get(run_name)

        # Ignore database runs that are not part of this queue.
        if config_entry is None:
            continue

        queue_index, config = config_entry

        result_by_name[run_name] = RunResult(
            queue_index=queue_index,
            config=config,
            status=status.upper(),
            run_id=int(run_id),
            run_name=run_name,
            balanced_score=_db_float(balanced_score),
            agent_return=_db_float(agent_return),
            max_drawdown=_db_float(max_drawdown),
            profit_factor=_db_float(profit_factor),
            win_rate=_db_float(win_rate),
            exposure=_db_float(exposure),
            round_trips=_db_int(round_trips),
            always_long=_db_float(always_long),
            best_score=_db_float(best_score),
            approx_kl=_db_float(approx_kl),
            clip_fraction=_db_float(clip_fraction),
            entropy_loss=_db_float(entropy_loss),
            explained_variance=_db_float(explained_variance),
            wall_seconds=_db_float(wall_seconds),
        )

    results: list[RunResult] = []

    for queue_index, config in enumerate(configs, start=1):
        result = result_by_name.get(config.name)

        if result is None:
            result = RunResult(
                queue_index=queue_index,
                config=config,
                status="NOT_FOUND",
            )

        results.append(result)

    return results


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
        f"{'gamma':>6} "
        f"{'GAE':>6} "
        f"{'ent':>9} "
        f"{'exp_pen':>9} "
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
            f"{result.config.gamma:>6.3f} "
            f"{result.config.gae_lambda:>6.3f} "
            f"{result.config.ent_coef:>9.2e} "
            f"{result.config.exposure_penalty:>9.2e} "
            f"{format_float(result.balanced_score, 5):>10} "
            f"{format_percent(result.agent_return):>9} "
            f"{format_percent(result.max_drawdown):>9} "
            f"{format_float(result.profit_factor, 3):>8} "
            f"{format_percent(result.win_rate):>8} "
            f"{format_percent(result.exposure):>9} "
            f"{str(result.round_trips if result.round_trips is not None else '-'):>7} "
            f"{format_float(result.best_score, 5):>10} "
            f"{format_scientific(result.approx_kl, 3):>10} "
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
                f"gamma={result.config.gamma:.3f} | "
                f"gae={result.config.gae_lambda:.3f} | "
                f"ent={result.config.ent_coef:.2e} | "
                f"exp_pen={result.config.exposure_penalty:.2e} | "
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
        print(f"gamma:           {winner.config.gamma}")
        print(f"gae_lambda:      {winner.config.gae_lambda}")
        print(f"ent_coef:        {winner.config.ent_coef}")
        print(
            f"exposure_penalty:{winner.config.exposure_penalty:>12.6g}"
        )
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


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    queue_names = tuple(
        discover_queue_manifests()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run PPO experiment configs sequentially "
            "and print a final comparison."
        )
    )

    parser.add_argument(
        "--queue",
        choices=queue_names,
        default=DEFAULT_QUEUE,
        help=(
            "Experiment queue to run. "
            f"Default: {DEFAULT_QUEUE}"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue without starting training.",
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--start-at",
        type=int,
        default=None,
        help=(
            "Run this 1-based queue position and every "
            "position after it. Useful after interruption."
        ),
    )

    selection.add_argument(
        "--select",
        type=parse_queue_selection,
        metavar="POSITIONS",
        help=(
            "Run selected 1-based queue positions in manifest "
            "order, for example: 1,3,5-8."
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip selected configs whose run names are already "
            "registered in PostgreSQL."
        ),
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Read existing queue results from PostgreSQL and "
            "print the final summary without starting training."
        ),
    )

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    config_paths = load_queue_config_paths(args.queue)
    configs = [
        read_config_meta(path)
        for path in config_paths
    ]

    try:
        selected = resolve_queue_selection(
            configs,
            start_at=args.start_at,
            selected_positions=args.select,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selected_positions = {
        position
        for position, _ in selected
    }

    if args.summary_only:
        database_results = load_results_from_database(configs)
        print_summary(database_results)
        return 0

    if args.dry_run:
        print_queue(
            args.queue,
            configs,
            selected_positions,
        )
        print(
            f"DRY RUN: selected {len(selected)} "
            f"of {len(configs)} configs; "
            "no training was started."
        )
        return 0

    assert_git_clean()

    preflight_results = load_results_from_database(configs)
    existing = find_existing_selected_runs(
        selected,
        preflight_results,
    )

    if existing and not args.skip_existing:
        print_queue(
            args.queue,
            configs,
            selected_positions,
        )
        print_existing_selected_runs(existing)
        print()
        print(
            "QUEUE PREFLIGHT FAILED: no training was started."
        )
        print(
            "Choose configs that are not registered, or pass "
            "--skip-existing to run only missing selections."
        )
        return 2

    if existing:
        print_existing_selected_runs(existing)
        print()
        print(
            f"Skipping {len(existing)} existing selected "
            f"config(s)."
        )

        existing_names = {
            config.name
            for _, config, _ in existing
        }
        selected = [
            (position, config)
            for position, config in selected
            if config.name not in existing_names
        ]
        selected_positions = {
            position
            for position, _ in selected
        }

    print_queue(
        args.queue,
        configs,
        selected_positions,
    )

    if not selected:
        print(
            "No selected configs remain after skipping existing "
            "runs. No training was started."
        )
        print_summary(preflight_results)
        return 0

    results: list[RunResult] = []

    try:
        for selected_index, (
            queue_position,
            config,
        ) in enumerate(selected, start=1):
            result = run_config(
                queue_name=args.queue,
                queue_position=queue_position,
                queue_total=len(configs),
                selected_index=selected_index,
                selected_total=len(selected),
                config=config,
            )
            results.append(result)

            if result.status != "COMPLETED":
                print()
                print(
                    f"Stopping queue because experiment "
                    f"{queue_position} ended with status "
                    f"{result.status}."
                )
                break

    except KeyboardInterrupt:
        print()
        print("Queue interrupted.")

    finally:
        try:
            database_results = load_results_from_database(configs)
        except Exception as exc:
            print()
            print(
                "WARNING: could not build summary from PostgreSQL:"
            )
            print(exc)
            print()
            print("Falling back to in-process parsed results.")
            print_summary(results)
        else:
            print_summary(database_results)

    failed = any(
        result.status != "COMPLETED"
        for result in results
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
