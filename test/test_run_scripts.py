from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


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
    else:
        assert "--time_predictor_warmup_fraction" not in result.stdout
