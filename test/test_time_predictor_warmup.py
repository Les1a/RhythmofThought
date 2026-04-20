import sys
from pathlib import Path

import pytest
from datasets import Dataset
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_predictor_warmup import (
    build_time_predictor_warmup_dataset,
    build_time_predictor_warmup_trainer,
    collect_time_predictor_parameters,
    maybe_run_time_predictor_warmup,
    run_time_predictor_warmup,
    validate_time_predictor_warmup_fraction,
)


def test_validate_time_predictor_warmup_fraction_rejects_out_of_range():
    with pytest.raises(ValueError, match="time_predictor_warmup_fraction"):
        validate_time_predictor_warmup_fraction(-0.01)
    with pytest.raises(ValueError, match="time_predictor_warmup_fraction"):
        validate_time_predictor_warmup_fraction(1.01)


def test_build_time_predictor_warmup_dataset_is_seed_stable_and_uses_ceil():
    dataset = Dataset.from_dict({"prompt": [f"p{i}" for i in range(7)]})

    subset_a, size_a = build_time_predictor_warmup_dataset(dataset, fraction=0.2, seed=123)
    subset_b, size_b = build_time_predictor_warmup_dataset(dataset, fraction=0.2, seed=123)
    subset_c, size_c = build_time_predictor_warmup_dataset(dataset, fraction=0.2, seed=999)

    assert size_a == 2
    assert size_b == 2
    assert size_c == 2
    assert subset_a["prompt"] == subset_b["prompt"]
    assert subset_a["prompt"] != subset_c["prompt"]


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.backbone = nn.Linear(4, 4)
        self.reasoning_time_embedding = nn.Linear(1, 4)
        self.thinking_time_predictor = nn.Linear(4, 1)
        self.use_time_conditioning = True
        self.config = type("_Cfg", (), {"hidden_size": 4})()


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyBase()

    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None):
        batch_size, seq_len = input_ids.shape
        return type(
            "_Out",
            (),
            {"logits": torch.zeros(batch_size, seq_len, 8, device=input_ids.device)},
        )()


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)


class _DummyTrainer:
    instances = []

    def __init__(
        self,
        model=None,
        processing_class=None,
        reward_funcs=None,
        args=None,
        train_dataset=None,
    ):
        self.model = model or _DummyModel()
        self.processing_class = processing_class if processing_class is not None else object()
        self.reward_funcs = reward_funcs if reward_funcs is not None else [lambda **_: [1.0]]
        self.args = args or type(
            "_Args",
            (),
            {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 0.1,
                "optim": "adamw_torch",
                "learning_rate": 5e-6,
                "adam_beta1": 0.9,
                "adam_beta2": 0.99,
                "adam_epsilon": 1e-8,
                "weight_decay": 0.0,
                "output_dir": "/tmp/original-run",
                "run_name": "original-run",
                "report_to": "wandb",
                "save_strategy": "steps",
                "max_steps": 10,
                "num_train_epochs": 3,
                "logging_steps": 1,
            },
        )()
        self.accelerator = _DummyAccelerator()
        self.train_dataset = train_dataset or Dataset.from_dict({"prompt": ["a", "b"]})
        self.data_collator = lambda features: features
        self.added_callbacks = []
        self.train_invocations = 0
        self.trainable_names_seen = None
        self.resume_from_checkpoint_seen = None
        self.optimizer = None
        _DummyTrainer.instances.append(self)

    def add_callback(self, callback):
        self.added_callbacks.append(callback)

    def create_optimizer(self):
        return None

    def train(self, resume_from_checkpoint=None):
        self.train_invocations += 1
        self.resume_from_checkpoint_seen = resume_from_checkpoint
        self.trainable_names_seen = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        for callback in self.added_callbacks:
            callback.on_log(
                args=self.args,
                state=type("_State", (), {"global_step": 1})(),
                control=None,
                logs={
                    "reward": 0.625,
                    "thinking_time_aux_loss": 1.0,
                    "thinking_time_pred_error": 0.2,
                    "thinking_time_rollout_error": 0.5,
                },
            )
        return type("_TrainOutput", (), {"metrics": {}})()


def test_collect_time_predictor_parameters_only_returns_predictor_weights():
    params = collect_time_predictor_parameters(_DummyModel())

    assert {name for name, _ in params} == {
        "model.thinking_time_predictor.weight",
        "model.thinking_time_predictor.bias",
    }


def test_maybe_run_time_predictor_warmup_skips_non_time_conditioning_modes(monkeypatch):
    _DummyTrainer.instances.clear()
    trainer = _DummyTrainer()

    maybe_run_time_predictor_warmup(
        trainer,
        args=type(
            "_Args",
            (),
            {
                "seed": 7,
                "lr_time_conditioning": 1e-4,
                "time_predictor_warmup_fraction": 0.2,
            },
        )(),
        mode="grpo",
        resume_from_checkpoint=None,
        task="gsm8k",
    )

    assert len(_DummyTrainer.instances) == 1


def test_maybe_run_time_predictor_warmup_forwards_standard_trainer_logs():
    _DummyTrainer.instances.clear()
    trainer = _DummyTrainer()
    captured = {"reward": None}

    def _capture_metrics(task, step, metrics):
        captured["reward"] = metrics["warmup/reward"]

    maybe_run_time_predictor_warmup(
        trainer,
        args=type(
            "_Args",
            (),
            {
                "seed": 7,
                "lr_time_conditioning": 1e-4,
                "max_grad_norm": 0.1,
                "time_predictor_warmup_fraction": 1.0,
            },
        )(),
        mode="tgrpo",
        resume_from_checkpoint=None,
        task="gsm8k",
        log_metrics_fn=_capture_metrics,
    )

    assert captured["reward"] == pytest.approx(0.625)


def test_maybe_run_time_predictor_warmup_skips_resume(monkeypatch):
    _DummyTrainer.instances.clear()
    trainer = _DummyTrainer()

    maybe_run_time_predictor_warmup(
        trainer,
        args=type(
            "_Args",
            (),
            {
                "seed": 7,
                "lr_time_conditioning": 1e-4,
                "time_predictor_warmup_fraction": 0.2,
            },
        )(),
        mode="tgrpo",
        resume_from_checkpoint="checkpoint-10",
        task="gsm8k",
    )

    assert len(_DummyTrainer.instances) == 1


def test_build_time_predictor_warmup_dataset_does_not_mutate_original_dataset():
    dataset = Dataset.from_dict({"prompt": ["p0", "p1", "p2", "p3"]})
    original_rows = dataset["prompt"]

    subset, size = build_time_predictor_warmup_dataset(dataset, fraction=0.5, seed=42)

    assert size == 2
    assert dataset["prompt"] == original_rows
    assert len(dataset) == 4
    assert len(subset) == 2


def test_build_time_predictor_warmup_trainer_uses_isolated_training_args():
    _DummyTrainer.instances.clear()
    trainer = _DummyTrainer()
    warmup_dataset = Dataset.from_dict({"prompt": ["w0", "w1"]})

    warmup_trainer = build_time_predictor_warmup_trainer(
        trainer,
        warmup_dataset,
        args=type(
            "_Args",
            (),
            {
                "lr_time_conditioning": 1e-4,
                "lr_residual_gate": 1e-4,
                "lr_residual_Lambda": 1e-3,
            },
        )(),
        mode="thrpo",
        task="gsm8k",
    )

    assert warmup_trainer is not trainer
    assert warmup_trainer.model is trainer.model
    assert warmup_trainer.train_dataset["prompt"] == ["w0", "w1"]
    assert warmup_trainer.args.output_dir.endswith("predictor_warmup")
    assert warmup_trainer.args.run_name.endswith("predictor_warmup")
    assert warmup_trainer.args.report_to == "none"
    assert warmup_trainer.args.save_strategy == "no"
    assert warmup_trainer.args.max_steps == -1
    assert warmup_trainer.args.num_train_epochs == 1
    assert warmup_trainer.thinking_time_loss_weight == 1.0


def test_run_time_predictor_warmup_uses_standard_trainer_and_restores_grad_flags():
    _DummyTrainer.instances.clear()
    trainer = _DummyTrainer()
    original_states = {
        name: parameter.requires_grad for name, parameter in trainer.model.named_parameters()
    }

    run_time_predictor_warmup(
        trainer,
        trainer.train_dataset,
        args=type(
            "_Args",
            (),
            {
                "lr_time_conditioning": 1e-4,
                "lr_residual_gate": 1e-4,
                "lr_residual_Lambda": 1e-3,
            },
        )(),
        mode="thrpo",
        task="gsm8k",
    )

    assert len(_DummyTrainer.instances) == 2
    warmup_trainer = _DummyTrainer.instances[-1]
    assert warmup_trainer.train_invocations == 1
    assert warmup_trainer.trainable_names_seen == {
        "model.thinking_time_predictor.weight",
        "model.thinking_time_predictor.bias",
    }
    restored_states = {
        name: parameter.requires_grad for name, parameter in trainer.model.named_parameters()
    }
    assert restored_states == original_states
