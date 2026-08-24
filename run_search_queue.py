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

NQ1H_SEARCH_V1_CONFIGS = [
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

    # Winner confirmation on additional seeds
    "configs/experiments/nq1h_search_v1/11_nepochs2_lr2p5e4_seed2.yml",
    "configs/experiments/nq1h_search_v1/12_nepochs2_lr2p5e4_seed3.yml",
    "configs/experiments/nq1h_search_v1/13_nepochs3_lr2p25e4_seed2.yml",
    "configs/experiments/nq1h_search_v1/14_nepochs3_lr2p25e4_seed3.yml",

    # Gamma screening — n_epochs=3, LR=2.25e-4, seed=1
    "configs/experiments/nq1h_search_v1/15_nepochs3_lr2p25e4_gamma085_seed1.yml",
    "configs/experiments/nq1h_search_v1/16_nepochs3_lr2p25e4_gamma095_seed1.yml",
    "configs/experiments/nq1h_search_v1/17_nepochs3_lr2p25e4_gamma097_seed1.yml",
    "configs/experiments/nq1h_search_v1/18_nepochs3_lr2p25e4_gamma099_seed1.yml",

    # GAE lambda screening — n_epochs=3, LR=2.25e-4, gamma=.90, seed=1
    "configs/experiments/nq1h_search_v1/19_nepochs3_lr2p25e4_gamma090_gae085_seed1.yml",
    "configs/experiments/nq1h_search_v1/20_nepochs3_lr2p25e4_gamma090_gae090_seed1.yml",
    "configs/experiments/nq1h_search_v1/21_nepochs3_lr2p25e4_gamma090_gae098_seed1.yml",
    "configs/experiments/nq1h_search_v1/22_nepochs3_lr2p25e4_gamma090_gae100_seed1.yml",

    # GAE=.85 winner confirmation on additional seeds
    "configs/experiments/nq1h_search_v1/23_nepochs3_lr2p25e4_gamma090_gae085_seed2.yml",
    "configs/experiments/nq1h_search_v1/24_nepochs3_lr2p25e4_gamma090_gae085_seed3.yml",

    # Entropy coefficient screening
    # n_epochs=3, LR=2.25e-4, gamma=.90, GAE=.85, seed=1
    "configs/experiments/nq1h_search_v1/25_nepochs3_lr2p25e4_gamma090_gae085_ent0_seed1.yml",
    "configs/experiments/nq1h_search_v1/26_nepochs3_lr2p25e4_gamma090_gae085_ent1e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/27_nepochs3_lr2p25e4_gamma090_gae085_ent5e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/28_nepochs3_lr2p25e4_gamma090_gae085_ent1e3_seed1.yml",

    # Exposure penalty screening
    # n_epochs=3, LR=2.25e-4, gamma=.90, GAE=.85, ent=2e-4, seed=1
    "configs/experiments/nq1h_search_v1/29_nepochs3_lr2p25e4_gamma090_gae085_exp2p5em6_seed1.yml",
    "configs/experiments/nq1h_search_v1/30_nepochs3_lr2p25e4_gamma090_gae085_exp5em6_seed1.yml",
    "configs/experiments/nq1h_search_v1/31_nepochs3_lr2p25e4_gamma090_gae085_exp1em5_seed1.yml",
    "configs/experiments/nq1h_search_v1/32_nepochs3_lr2p25e4_gamma090_gae085_exp2em5_seed1.yml",

    # Exposure penalty refinement for active-policy regime
    "configs/experiments/nq1h_search_v1/33_nepochs3_lr2p25e4_gamma090_gae085_exp1p25em5_seed1.yml",
    "configs/experiments/nq1h_search_v1/34_nepochs3_lr2p25e4_gamma090_gae085_exp1p5em5_seed1.yml",
    "configs/experiments/nq1h_search_v1/35_nepochs3_lr2p25e4_gamma090_gae085_exp1p75em5_seed1.yml",

    # Active-policy exposure penalty confirmation
    "configs/experiments/nq1h_search_v1/36_nepochs3_lr2p25e4_gamma090_gae085_exp1p5em5_seed2.yml",
    "configs/experiments/nq1h_search_v1/37_nepochs3_lr2p25e4_gamma090_gae085_exp1p5em5_seed3.yml",

    # Exposure-penalty x seed behavior map
    "configs/experiments/nq1h_search_v1/38_nepochs3_lr2p25e4_gamma090_gae085_exp1p25em5_seed2.yml",
    "configs/experiments/nq1h_search_v1/39_nepochs3_lr2p25e4_gamma090_gae085_exp1p25em5_seed3.yml",
    "configs/experiments/nq1h_search_v1/40_nepochs3_lr2p25e4_gamma090_gae085_exp1p75em5_seed2.yml",
    "configs/experiments/nq1h_search_v1/41_nepochs3_lr2p25e4_gamma090_gae085_exp1p75em5_seed3.yml",

    # Exposure-free seed ablation
    "configs/experiments/nq1h_search_v1/38_nepochs3_lr2p25e4_gamma090_gae085_exp0_seed1.yml",
    "configs/experiments/nq1h_search_v1/39_nepochs3_lr2p25e4_gamma090_gae085_exp0_seed2.yml",
    "configs/experiments/nq1h_search_v1/40_nepochs3_lr2p25e4_gamma090_gae085_exp0_seed3.yml",

    # Turnover-penalty seed ablation
    "configs/experiments/nq1h_search_v1/42_nepochs3_lr2p25e4_gamma090_gae085_exp0_turn5em5_seed1.yml",
    "configs/experiments/nq1h_search_v1/43_nepochs3_lr2p25e4_gamma090_gae085_exp0_turn5em5_seed2.yml",
    "configs/experiments/nq1h_search_v1/44_nepochs3_lr2p25e4_gamma090_gae085_exp0_turn5em5_seed3.yml",

    # LONG initialization prior seed ablation
    "configs/experiments/nq1h_search_v1/45_nepochs3_lr2p25e4_gamma090_gae085_exp0_initlong55_seed1.yml",
    "configs/experiments/nq1h_search_v1/46_nepochs3_lr2p25e4_gamma090_gae085_exp0_initlong55_seed2.yml",
    "configs/experiments/nq1h_search_v1/47_nepochs3_lr2p25e4_gamma090_gae085_exp0_initlong55_seed3.yml",

    # Resume phase: checkpoint #192 learning-rate screening
    "configs/experiments/nq1h_search_v1/48_resume48_ckpt192_lr1p5e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/49_resume48_ckpt192_lr2p25e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/50_resume48_ckpt192_lr3e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/51_resume48_ckpt192_lr4e4_seed1.yml",

    # Resume phase 2: checkpoint #204 low learning-rate screening
    "configs/experiments/nq1h_search_v1/52_resume51_ckpt204_lr5e5_seed1.yml",
    "configs/experiments/nq1h_search_v1/53_resume51_ckpt204_lr7p5e5_seed1.yml",
    "configs/experiments/nq1h_search_v1/54_resume51_ckpt204_lr1e4_seed1.yml",
    "configs/experiments/nq1h_search_v1/55_resume51_ckpt204_lr1p5e4_seed1.yml",

    # Checkpoint #204 continuation confirmation on additional seeds
    "configs/experiments/nq1h_search_v1/56_resume51_ckpt204_lr1p5e4_seed2.yml",
    "configs/experiments/nq1h_search_v1/57_resume51_ckpt204_lr1p5e4_seed3.yml",
]


NQ1H_LR_DECAY_SPLIT90_V1_CONFIGS = [
    "configs/experiments/nq1h_lr_decay_split90_v1/01_fresh_lr3e4_seed1.yml",
    "configs/experiments/nq1h_lr_decay_split90_v1/02_resume_epoch1_lr2p25e4_seed1.yml",
    "configs/experiments/nq1h_lr_decay_split90_v1/03_resume_epoch2_lr1p5e4_seed1.yml",
    "configs/experiments/nq1h_lr_decay_split90_v1/04_resume_epoch3_lr1e4_seed1.yml",
]


CONFIGS = [
    *NQ1H_SEARCH_V1_CONFIGS,
    *NQ1H_LR_DECAY_SPLIT90_V1_CONFIGS,
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


def should_echo_training_line(line: str) -> bool:
    """Hide raw SB3 logger tables while preserving our normal training UI."""
    stripped = line.strip()

    # Stable-Baselines3 logger rows:
    #
    # | time/              |          |
    # |    fps             | 1500     |
    # | train/             |          |
    # |    approx_kl       | ...      |
    #
    if stripped.startswith("|") and stripped.endswith("|"):
        return False

    # SB3 table separators such as:
    # -----------------------------------------
    if stripped and set(stripped) == {"-"}:
        return False

    return True


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
        f" | gamma={config.gamma}"
        f" | gae={config.gae_lambda}"
        f" | ent={config.ent_coef}"
        f" | exp_pen={config.exposure_penalty}"
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
            # Always parse the complete child-process output so the final
            # summary still has all available metrics.
            parse_output_line(result, line)

            # But keep the terminal clean: show our regular preflight,
            # progress bars, validation output and completion summary while
            # suppressing raw Stable-Baselines3 logger tables.
            if should_echo_training_line(line):
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

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Read existing queue results from PostgreSQL and "
            "print the final summary without starting training."
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

    if args.summary_only:
        database_results = load_results_from_database(configs)
        print_summary(database_results)
        return 0

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
