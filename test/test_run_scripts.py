"""Smoke tests for the top-level multi-task launcher scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "script_name",
    ["run_grpo_all.sh", "run_tgrpo_all.sh", "run_hrpo_all.sh", "run_thrpo_all.sh", "srun.sh"],
)
def test_shell_entrypoints_are_executable(script_name: str):
    assert os.access(REPO_ROOT / script_name, os.X_OK)


@pytest.mark.parametrize(
    ("script_name", "mode", "expects_time_conditioning"),
    [
        ("run_grpo_all.sh", "grpo", False),
        ("run_tgrpo_all.sh", "tgrpo", True),
        ("run_hrpo_all.sh", "hrpo", False),
        ("run_thrpo_all.sh", "thrpo", True),
    ],
)
def test_run_script_dry_run_accepts_smoke_flags(script_name: str, mode: str, expects_time_conditioning: bool):
    script_path = REPO_ROOT / script_name
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--tasks",
            "gsm8k",
            "--skip-eval",
            "--no-wandb",
            "--exp-suffix",
            "pytest-smoke",
            "--max-steps",
            "1",
            "--max-train-samples",
            "8",
            "--predictor-hidden-states",
            "7",
            "--time-predictor-warmup-fraction",
            "0.4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"--mode {mode}" in result.stdout
    assert "--max_steps 1" in result.stdout
    assert "--max_train_samples 8" in result.stdout
    assert "--exp-suffix pytest-smoke" in result.stdout
    if expects_time_conditioning:
        assert "--thinking_time_loss_weight" in result.stdout
        assert "--lr_time_conditioning" in result.stdout
        assert "--thinking_time_predictor_num_hidden_states 7" in result.stdout
        assert "--time_predictor_warmup_fraction 0.4" in result.stdout
        assert "-last7" in result.stdout
    else:
        assert "--time_predictor_warmup_fraction" not in result.stdout
        assert "-last7" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_grpo_all.sh", "run_hrpo_all.sh"])
def test_run_script_rejects_unknown_tasks_before_dispatch(script_name: str):
    script_path = REPO_ROOT / script_name
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--tasks",
            "gsm8k,unknown",
            "--skip-eval",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Unknown task: unknown" in output
    assert "command not found" not in output


@pytest.mark.parametrize(
    ("script_name", "mode", "exp_dir"),
    [
        (
            "run_grpo_all.sh",
            "tgrpo",
            "experiments/Hyphen-Model-gsm8k-tgrpo-group4-lora32-temp0.5-last3-pytest-latest",
        ),
        (
            "run_hrpo_all.sh",
            "thrpo",
            "experiments/Hyphen-Model-gsm8k-thrpo-group4-lora32-rmin0.99-temp0.5-last3-pytest-latest",
        ),
    ],
)
def test_run_script_dry_run_eval_uses_highest_numeric_checkpoint(
    tmp_path: Path,
    script_name: str,
    mode: str,
    exp_dir: str,
):
    script_path = tmp_path / script_name
    shutil.copy2(REPO_ROOT / script_name, script_path)
    checkpoint_root = tmp_path / exp_dir
    (checkpoint_root / "checkpoint-20").mkdir(parents=True)
    (checkpoint_root / "checkpoint-100").mkdir()

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--eval-only",
            "--tasks",
            "gsm8k",
            "--model",
            "Org/Hyphen-Model",
            "--mode",
            mode,
            "--exp-suffix",
            "pytest-latest",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "checkpoint-100" in result.stdout
    assert "eval_gsm8k.py --checkpoint_path" in result.stdout
