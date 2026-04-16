import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_conditioning import (
    AdaLNProjection,
    ReasoningTimeEmbedding,
    ThinkingTimePredictor,
    build_replay_lagged_predictor_trace,
    compute_thinking_time_aux_loss,
    enable_time_conditioning,
    prepare_adaln_modulation,
    prepare_online_thinking_time_conditioning,
    prepare_trainer_time_conditioning,
    reset_time_conditioning_state,
    select_training_thinking_time_embedding,
    set_thinking_residual_disabled,
    summarize_residual_rollout_metrics,
    update_online_thinking_time_hidden_cache,
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


class _RecordingPredictor(nn.Module):
    def __init__(self, next_thinking_time=None):
        super().__init__()
        self.next_thinking_time = next_thinking_time
        self.last_inputs = None
        self.last_mask = None

    def forward(self, inputs_embeds, thinking_mask=None):
        self.last_inputs = inputs_embeds.detach().clone()
        self.last_mask = None if thinking_mask is None else thinking_mask.detach().clone()
        if self.next_thinking_time is None:
            return torch.zeros(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return self.next_thinking_time.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    def forward_step(self, inputs_embeds, is_thinking=None):
        self.last_inputs = inputs_embeds.detach().clone()
        if is_thinking is None:
            mask = torch.ones(inputs_embeds.shape[0], 1, device=inputs_embeds.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(is_thinking, device=inputs_embeds.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
        self.last_mask = mask.detach().clone()
        if self.next_thinking_time is None:
            values = torch.zeros(inputs_embeds.shape[0], 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        else:
            values = self.next_thinking_time.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return torch.where(mask, values, torch.zeros_like(values))


def test_predictor_masks_finished_rows():
    predictor = ThinkingTimePredictor(hidden_size=4)
    with torch.no_grad():
        predictor.net[0].weight.fill_(1.0)
        predictor.net[0].bias.zero_()
        predictor.net[2].weight.fill_(20.0)
        predictor.net[2].bias.zero_()

    hidden = torch.tensor([[[5.0] * 4], [[-5.0] * 4]])
    used_thinking_time = predictor.forward_step(hidden, is_thinking=[True, False])

    assert used_thinking_time.shape == (2, 1)
    assert used_thinking_time[0, 0] > 0.0
    assert used_thinking_time[1, 0] == 0.0


def test_online_conditioning_uses_cached_hidden_and_freezes_finished_rows():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)
    base = wrapper.model
    old_hidden = torch.randn(2, 1, base.config.hidden_size)
    new_hidden = torch.randn(2, 1, base.config.hidden_size)
    base._cached_last_hidden = old_hidden.clone()
    base.thinking_time_predictor = _RecordingPredictor(torch.tensor([[0.6], [0.95]]))

    thinking_time_emb, thinking_time_mask, used_thinking_time = prepare_online_thinking_time_conditioning(
        base,
        batch_size=2,
        device=base.anchor.device,
        is_thinking=[True, False],
    )
    update_online_thinking_time_hidden_cache(base, new_hidden, is_thinking=[True, False])

    assert thinking_time_emb.shape == (2, 1, base.config.thinking_time_embed_dim)
    assert torch.equal(thinking_time_mask, torch.tensor([[True], [False]], device=base.anchor.device))
    assert used_thinking_time[0, 0].item() == pytest.approx(0.6)
    assert used_thinking_time[1, 0].item() == 0.0
    assert torch.allclose(base._cached_last_hidden[0], new_hidden[0])
    assert torch.allclose(base._cached_last_hidden[1], old_hidden[1])


def test_training_replay_stats_drop_stale_source_metrics():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)
    base = wrapper.model

    _, selected_thinking_time, stats = select_training_thinking_time_embedding(
        base,
        gt_thinking_time=torch.tensor([[0.0, 0.5, 1.0]]),
        rollout_thinking_time=torch.ones(1, 3),
        thinking_mask=torch.tensor([[True, True, True]]),
    )

    assert selected_thinking_time.shape == (1, 3)
    assert "thinking_time_source_gt" not in stats
    assert "thinking_time_source_self" not in stats
    assert set(stats) == {"thinking_time_rollout_error"}


def test_prepare_trainer_time_conditioning_sets_cached_training_tensors():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)
    base = wrapper.model

    gt_time, _ = prepare_trainer_time_conditioning(
        base,
        input_ids=torch.zeros((1, 3), dtype=torch.long),
        thinking_mask=torch.tensor([[False, True, True]]),
        rollout_thinking_time=torch.tensor([[0.0, 0.25, 0.75]]),
    )

    assert gt_time.shape == (1, 3)
    assert base._train_thinking_time_emb.shape == (1, 3, base.config.thinking_time_embed_dim)
    assert torch.equal(base._train_thinking_time_mask, torch.tensor([[False, True, True]]))


def test_replay_lagged_trace_uses_last_prompt_token_then_last_thinking_token():
    replay_hidden = torch.tensor([[[10.0], [11.0], [20.0], [21.0], [30.0]]])
    lagged_hidden, lagged_mask = build_replay_lagged_predictor_trace(
        replay_hidden,
        prompt_mask=torch.tensor([[True, True]]),
        thinking_mask=torch.tensor([[False, False, True, False, True]]),
    )

    assert torch.equal(lagged_mask, torch.tensor([[False, False, True, False, True]]))
    assert torch.equal(lagged_hidden[0, 2], replay_hidden[0, 1])
    assert torch.equal(lagged_hidden[0, 3], replay_hidden[0, 2])
    assert torch.equal(lagged_hidden[0, 4], replay_hidden[0, 2])


def test_aux_loss_metrics_use_current_names():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)
    base = wrapper.model
    base.thinking_time_predictor = _RecordingPredictor()
    base._train_predictor_hidden_states = torch.randn(1, 4, base.config.hidden_size)

    aux_loss, metrics = compute_thinking_time_aux_loss(
        base,
        gt_thinking_time=torch.tensor([[0.0, 0.0, 0.5, 1.0]]),
        thinking_mask=torch.tensor([[False, False, True, True]]),
        prompt_mask=torch.tensor([[True, True]]),
        rewards=torch.tensor([1.0]),
    )

    assert aux_loss is not None
    assert "thinking_time_pred_mse" not in metrics
    assert "thinking_time_aux_total" not in metrics
    assert metrics["thinking_time_aux_loss"] == pytest.approx(aux_loss.item())
    assert "thinking_time_pred_error" in metrics
    assert "thinking_time_embed_std" in metrics
    assert "thinking_time_aux_weight_mean" not in metrics
    assert "thinking_time_pred_mean" not in metrics


def test_residual_rollout_metrics_use_current_names():
    metrics = summarize_residual_rollout_metrics(torch.tensor([[1.0, 0.8, 0.6]]))

    assert "embeds_ratio" not in metrics
    assert "hidden_ratio" not in metrics
    assert metrics["residual_active_fraction"] == pytest.approx(2 / 3)
    assert metrics["residual_embed_ratio"] == pytest.approx(0.7)
    assert metrics["residual_hidden_ratio"] > 0.0


def test_reset_time_conditioning_state_clears_cached_training_state():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)
    base = wrapper.model
    base._train_thinking_time_emb = torch.randn(1, 3, base.config.thinking_time_embed_dim)
    base._train_thinking_time_mask = torch.ones(1, 3, dtype=torch.bool)
    base._train_predictor_hidden_states = torch.randn(1, 3, base.config.hidden_size)
    base._cached_last_hidden = torch.randn(1, 1, base.config.hidden_size)
    base._used_thinking_time = torch.randn(1, 1)

    reset_time_conditioning_state(base)

    assert base._train_thinking_time_emb is None
    assert base._train_thinking_time_mask is None
    assert base._train_predictor_hidden_states is None
    assert base._cached_last_hidden is None
    assert base._used_thinking_time is None


def test_set_thinking_residual_disabled_updates_wrapper_and_base():
    wrapper = _DummyWrapper()
    enable_time_conditioning(wrapper)

    set_thinking_residual_disabled(wrapper, True)

    base = wrapper.model
    assert getattr(wrapper, "disable_thinking_residual", False) is True
    assert getattr(base, "disable_thinking_residual", False) is True


def test_embedding_and_adaln_helpers_preserve_boundaries():
    embedding = ReasoningTimeEmbedding(thinking_time_embed_dim=8, num_frequencies=4)
    actual = embedding(torch.tensor([[0.0], [1.0]]))
    zeros = tuple(torch.zeros(2, 3, 4) for _ in range(6))
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = prepare_adaln_modulation(zeros)
    projection = AdaLNProjection(thinking_time_embed_dim=4, hidden_size=2, gamma_cap=0.3, beta_cap=0.2, alpha_cap=0.1)

    assert actual.shape == (2, 1, 8)
    assert not torch.allclose(actual[0], actual[1])
    assert torch.equal(gamma1, torch.zeros_like(gamma1))
    assert torch.equal(beta1, torch.zeros_like(beta1))
    assert torch.equal(gamma2, torch.zeros_like(gamma2))
    assert torch.equal(beta2, torch.zeros_like(beta2))
    assert torch.equal(alpha1, torch.ones_like(alpha1))
    assert torch.equal(alpha2, torch.ones_like(alpha2))
    with torch.no_grad():
        projection.proj.weight.fill_(10.0)
        projection.proj.bias.fill_(10.0)
    bounded = projection(torch.ones(1, 1, 4))
    assert bounded[0].abs().max().item() <= 0.3001
    assert bounded[1].abs().max().item() <= 0.2001
    assert bounded[2].abs().max().item() <= 0.1001
