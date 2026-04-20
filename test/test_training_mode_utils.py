import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_conditioning import THINKING_RESIDUAL_MODULE_NAMES, TIME_CONDITIONING_MODULE_NAMES
from utils import (
    ANSWER_START,
    ADAPTER_METADATA_KEY,
    attach_adapter_metadata,
    build_training_exp_name,
    build_adapter_metadata,
    configure_model_for_training_mode,
    create_training_parser,
    get_modules_to_save_for_mode,
    load_adapter_metadata,
    resolve_time_conditioning_predictor_num_hidden_states,
    resolve_resume_from_checkpoint,
)


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class _DummyInnerModel(nn.Module):
    def __init__(self, hidden_size=8, thinking_time_embed_dim=6, num_layers=2):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.layers = nn.ModuleList([_DummyLayer() for _ in range(num_layers)])
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            thinking_time_embed_dim=thinking_time_embed_dim,
            use_time_conditioning=False,
            num_hidden_layers=num_layers,
        )


class _DummyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyInnerModel()


def test_get_modules_to_save_for_mode_matches_expected_components():
    assert get_modules_to_save_for_mode("grpo") is None
    assert set(get_modules_to_save_for_mode("tgrpo")) == set(TIME_CONDITIONING_MODULE_NAMES)
    assert set(get_modules_to_save_for_mode("hrpo")) == set(THINKING_RESIDUAL_MODULE_NAMES)
    assert set(get_modules_to_save_for_mode("thrpo")) == set(
        THINKING_RESIDUAL_MODULE_NAMES + TIME_CONDITIONING_MODULE_NAMES
    )


def test_build_training_exp_name_normalizes_suffix_without_extra_separator():
    no_suffix = build_training_exp_name(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        task="gsm8k",
        mode="tgrpo",
        group_size=4,
        lora_rank=32,
        temperature=0.5,
        exp_suffix="",
    )
    normalized_suffix = build_training_exp_name(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        task="gsm8k",
        mode="thrpo",
        group_size=4,
        lora_rank=32,
        temperature=0.5,
        residual_r_min=0.99,
        exp_suffix="  smoke 2026/04  ",
    )

    assert no_suffix == "./experiments/Qwen2.5-3B-Instruct-gsm8k-tgrpo-group4-lora32-temp0.5"
    assert not no_suffix.endswith("-")
    assert normalized_suffix.endswith("-smoke-2026-04")


def test_configure_model_for_training_mode_enables_time_conditioning_only_for_tgrpo_and_thrpo():
    tgrpo = _DummyWrapper()
    thrpo = _DummyWrapper()
    grpo = _DummyWrapper()

    configure_model_for_training_mode(
        tgrpo,
        "tgrpo",
        max_completion_length=77,
        thinking_time_predictor_num_hidden_states=6,
    )
    configure_model_for_training_mode(thrpo, "thrpo")
    configure_model_for_training_mode(grpo, "grpo")

    assert tgrpo.answer_start == ANSWER_START
    assert getattr(tgrpo, "disable_thinking_residual", False) is True
    assert getattr(tgrpo.model, "use_time_conditioning", False) is True
    assert tgrpo.model.config.max_completion_length == 77
    assert tgrpo.model.config.thinking_time_predictor_num_hidden_states == 6
    assert tgrpo.model.thinking_time_predictor.net[0].in_features == 48
    assert thrpo.answer_start == ANSWER_START
    assert getattr(thrpo, "disable_thinking_residual", False) is False
    assert getattr(thrpo.model, "use_time_conditioning", False) is True
    assert not hasattr(grpo, "answer_start")
    assert getattr(grpo.model, "use_time_conditioning", False) is False


def test_attach_adapter_metadata_wraps_peft_config_to_dict():
    class _DummyPeftConfig:
        def to_dict(self):
            return {"peft_type": "LORA"}

    model = SimpleNamespace(peft_config={"default": _DummyPeftConfig()})
    metadata = {"mode": "thrpo", "temperature": 0.5}

    attach_adapter_metadata(model, metadata)

    saved = model.peft_config["default"].to_dict()
    assert saved["peft_type"] == "LORA"
    assert saved[ADAPTER_METADATA_KEY] == metadata


def test_resolve_resume_from_checkpoint_handles_resume_and_non_resume_paths(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_resume_from_checkpoint(str(tmp_path / "missing-exp"), resume=True)

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "checkpoint-10").mkdir()
    (exp_dir / "checkpoint-25").mkdir()
    checkpoint = resolve_resume_from_checkpoint(str(exp_dir), resume=True)
    assert checkpoint.endswith("checkpoint-25")

    with pytest.raises(ValueError, match="already exists"):
        resolve_resume_from_checkpoint(str(exp_dir), resume=False)


def test_create_training_parser_preserves_common_cli_shape():
    parser = create_training_parser(
        group_size=8,
        per_device_train_batch_size=16,
        max_prompt_length=2048,
        max_completion_length=1024,
        dataset_root_default="../DATA",
    )

    args = parser.parse_args(
        [
            "--model_name",
            "Qwen/Qwen2.5-3B-Instruct",
            "--exp_suffix",
            "smoke",
            "--dataset_root",
            "/tmp/data",
            "--mode",
            "tgrpo",
            "--thinking_time_predictor_num_hidden_states",
            "7",
            "--max_steps",
            "3",
            "--max_train_samples",
            "12",
            "--resume",
        ]
    )

    assert args.group_size == 8
    assert args.per_device_train_batch_size == 16
    assert args.max_prompt_length == 2048
    assert args.max_completion_length == 1024
    assert args.exp_suffix == "smoke"
    assert args.dataset_root == "/tmp/data"
    assert args.mode == "tgrpo"
    assert args.thinking_time_predictor_num_hidden_states == 7
    assert args._thinking_time_predictor_num_hidden_states_explicit is True
    assert args.max_steps == 3
    assert args.max_train_samples == 12
    assert args.resume is True


def test_create_training_parser_accepts_time_predictor_warmup_fraction():
    parser = create_training_parser(
        group_size=4,
        per_device_train_batch_size=8,
        max_prompt_length=1024,
        max_completion_length=1024,
    )

    args = parser.parse_args(
        [
            "--mode",
            "tgrpo",
            "--time-predictor-warmup-fraction",
            "0.35",
        ]
    )

    assert args.time_predictor_warmup_fraction == pytest.approx(0.35)


def test_create_training_parser_defaults_time_predictor_warmup_fraction():
    parser = create_training_parser(
        group_size=4,
        per_device_train_batch_size=8,
        max_prompt_length=1024,
        max_completion_length=1024,
    )

    args = parser.parse_args(["--mode", "thrpo"])

    assert args.time_predictor_warmup_fraction == pytest.approx(0.2)


def test_load_adapter_metadata_reads_standard_rot_metadata(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    metadata = {
        "schema_version": 2,
        "task": "gsm8k",
        "mode": "thrpo",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "temperature": 0.7,
        "thinking_time_predictor_num_hidden_states": 5,
    }
    (adapter_dir / "adapter_config.json").write_text(
        '{"peft_type":"LORA","rot_metadata":{"schema_version":2,"task":"gsm8k","mode":"thrpo","base_model":"Qwen/Qwen2.5-3B-Instruct","temperature":0.7,"thinking_time_predictor_num_hidden_states":5}}'
    )

    loaded = load_adapter_metadata(str(adapter_dir))

    assert loaded == metadata


def test_build_adapter_metadata_includes_time_conditioning_hidden_state_count():
    args = SimpleNamespace(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        temperature=0.5,
        group_size=4,
        lora_rank=32,
        max_prompt_length=1024,
        max_completion_length=1024,
        residual_r_min=0.99,
        residual_r_max=0.999,
        thinking_time_loss_weight=0.1,
        lr=5e-6,
        lr_residual_gate=1e-4,
        lr_residual_Lambda=1e-3,
        lr_time_conditioning=1e-4,
        thinking_time_predictor_num_hidden_states=6,
    )

    metadata = build_adapter_metadata(args, task="gsm8k", mode="tgrpo")

    assert metadata["thinking_time_predictor_num_hidden_states"] == 6


def test_resolve_time_conditioning_predictor_num_hidden_states_uses_checkpoint_value_and_rejects_conflict(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text(
        '{"peft_type":"LORA","rot_metadata":{"schema_version":2,"task":"gsm8k","mode":"tgrpo","base_model":"Qwen/Qwen2.5-3B-Instruct","temperature":0.5,"thinking_time_predictor_num_hidden_states":6}}'
    )

    implicit_args = SimpleNamespace(
        thinking_time_predictor_num_hidden_states=4,
        _thinking_time_predictor_num_hidden_states_explicit=False,
    )
    resolved = resolve_time_conditioning_predictor_num_hidden_states(
        implicit_args,
        mode="tgrpo",
        checkpoint_path=str(checkpoint_dir),
    )
    assert resolved == 6
    assert implicit_args.thinking_time_predictor_num_hidden_states == 6

    explicit_args = SimpleNamespace(
        thinking_time_predictor_num_hidden_states=5,
        _thinking_time_predictor_num_hidden_states_explicit=True,
    )
    with pytest.raises(ValueError, match="thinking_time_predictor_num_hidden_states"):
        resolve_time_conditioning_predictor_num_hidden_states(
            explicit_args,
            mode="tgrpo",
            checkpoint_path=str(checkpoint_dir),
        )
