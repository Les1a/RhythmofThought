"""Time-conditioning modules and runtime helpers for TGRPO/THRPO.

The stack predicts normalized thinking progress from lagged hidden states,
embeds that scalar into Fourier-style features, and applies per-layer AdaLN
modulation only while a generated sample is still in the thinking span. Rollout
uses cached one-step-lag hidden states; replay rebuilds the same lagged contract
from trainer hidden traces so auxiliary supervision does not peek at future
tokens.
"""

import math

import torch
import torch.nn as nn


TIME_CONDITIONING_MODULE_NAMES = (
    "adaln_proj",
    "thinking_time_predictor",
    "reasoning_time_embedding",
)
DEFAULT_TIME_CONDITIONING_PREDICTOR_NUM_HIDDEN_STATES = 3
THINKING_RESIDUAL_MODULE_NAMES = (
    "thinking_residual_gate_r",
    "thinking_residual_gate_i",
    "thinking_residual_Lambda",
)


def get_time_conditioning_predictor_num_hidden_states(config_or_model):
    """Return how many recent output hidden states to concatenate for the predictor."""
    if hasattr(config_or_model, "config"):
        config_or_model = config_or_model.config
    raw_value = getattr(
        config_or_model,
        "thinking_time_predictor_num_hidden_states",
        DEFAULT_TIME_CONDITIONING_PREDICTOR_NUM_HIDDEN_STATES,
    )
    try:
        num_hidden_states = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "thinking_time_predictor_num_hidden_states must be an integer, "
            f"got {raw_value!r}"
        ) from exc
    if num_hidden_states <= 0:
        raise ValueError(
            "thinking_time_predictor_num_hidden_states must be positive, "
            f"got {num_hidden_states}"
        )
    return num_hidden_states


def _coerce_time_conditioning_predictor_num_hidden_states(num_hidden_states):
    return get_time_conditioning_predictor_num_hidden_states(
        type("_PredictorConfig", (), {"thinking_time_predictor_num_hidden_states": num_hidden_states})()
    )


def get_time_conditioning_predictor_input_size(hidden_size, num_hidden_states=None):
    """Return predictor input width after concatenating the last hidden-state outputs."""
    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if num_hidden_states is None:
        num_hidden_states = DEFAULT_TIME_CONDITIONING_PREDICTOR_NUM_HIDDEN_STATES
    else:
        num_hidden_states = _coerce_time_conditioning_predictor_num_hidden_states(num_hidden_states)
    return num_hidden_states * hidden_size


def append_time_conditioning_hidden_state(hidden_history, hidden_state, num_hidden_states=None):
    """Append one hidden-state tensor while bounding history to the most recent N outputs."""
    if hidden_history is None:
        raise ValueError("hidden_history must be provided")
    if not torch.is_tensor(hidden_state):
        raise ValueError("hidden_state must be a tensor")

    if num_hidden_states is None:
        num_hidden_states = DEFAULT_TIME_CONDITIONING_PREDICTOR_NUM_HIDDEN_STATES
    else:
        num_hidden_states = _coerce_time_conditioning_predictor_num_hidden_states(num_hidden_states)

    hidden_history.append(hidden_state)
    overflow = len(hidden_history) - num_hidden_states
    if overflow > 0:
        del hidden_history[:overflow]
    return hidden_history


def concat_time_conditioning_hidden_states(hidden_states, num_hidden_states=None):
    """Concatenate the most recent output hidden states along the hidden dimension."""
    hidden_states = list(hidden_states)
    if num_hidden_states is None:
        num_hidden_states = DEFAULT_TIME_CONDITIONING_PREDICTOR_NUM_HIDDEN_STATES
    else:
        num_hidden_states = _coerce_time_conditioning_predictor_num_hidden_states(num_hidden_states)
    if len(hidden_states) < num_hidden_states:
        raise ValueError(
            "expected at least "
            f"{num_hidden_states} hidden-state outputs, got {len(hidden_states)}"
        )

    selected_hidden_states = hidden_states[-num_hidden_states:]
    ref_shape = None
    for idx, hidden_state in enumerate(selected_hidden_states):
        if not torch.is_tensor(hidden_state):
            raise ValueError(f"hidden state at index {idx} is not a tensor")
        if hidden_state.ndim != 3:
            raise ValueError(
                "hidden states must be rank-3 (B, L, H), "
                f"got shape {tuple(hidden_state.shape)} at index {idx}"
            )
        if ref_shape is None:
            ref_shape = hidden_state.shape[:2]
        elif hidden_state.shape[:2] != ref_shape:
            raise ValueError(
                "hidden states must share leading dims before concatenation: "
                f"{tuple(hidden_state.shape[:2])} vs {tuple(ref_shape)}"
            )

    return torch.cat(selected_hidden_states, dim=-1)


class ThinkingTimePredictor(nn.Module):
    """Predict normalized thinking progress from concatenated recent hidden states."""

    def __init__(self, hidden_size):
        super().__init__()
        bottleneck = hidden_size // 4
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )

    def _predict_token_thinking_time(self, hidden_states):
        net_dtype = next(self.net.parameters()).dtype
        return torch.sigmoid(self.net(hidden_states.to(net_dtype))).squeeze(-1)

    def forward(self, inputs_embeds, thinking_mask=None):
        thinking_time = self._predict_token_thinking_time(inputs_embeds).clamp(0.0, 1.0)
        mask = None
        if thinking_mask is not None:
            mask = _normalize_sequence_mask(thinking_mask, thinking_time.shape, thinking_time.device)
        if mask is not None:
            thinking_time = torch.where(mask, thinking_time, torch.zeros_like(thinking_time))
        return thinking_time

    def forward_step(self, inputs_embeds, is_thinking=None):
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] != 1:
            raise ValueError(
                "forward_step expects inputs_embeds with shape (B, 1, hidden_size), "
                f"got {tuple(inputs_embeds.shape)}"
            )

        thinking_time = self._predict_token_thinking_time(inputs_embeds).clamp(0.0, 1.0)

        mask = _normalize_step_mask(is_thinking, thinking_time.shape[0], thinking_time.device)
        if mask is None:
            return thinking_time
        return torch.where(mask, thinking_time, torch.zeros_like(thinking_time))


class ReasoningTimeEmbedding(nn.Module):
    """Map scalar thinking-time values to Fourier features with learnable bandwidth."""

    def __init__(
        self,
        thinking_time_embed_dim=256,
        num_frequencies=32,
        min_frequency=1.0,
        max_frequency=6.0,
        thinking_time_std_init=0.05,
    ):
        super().__init__()
        self.thinking_time_embed_dim = int(thinking_time_embed_dim)
        self.num_frequencies = int(num_frequencies)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.thinking_time_std_init = float(thinking_time_std_init)
        if self.thinking_time_embed_dim <= 0:
            raise ValueError(
                f"thinking_time_embed_dim must be positive, got {self.thinking_time_embed_dim}"
            )
        if self.num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be positive, got {self.num_frequencies}")
        if self.min_frequency <= 0.0:
            raise ValueError(f"min_frequency must be positive, got {self.min_frequency}")
        if self.max_frequency < self.min_frequency:
            raise ValueError(
                f"max_frequency must be >= min_frequency, got {self.max_frequency} < {self.min_frequency}"
            )
        if self.thinking_time_std_init <= 0.0:
            raise ValueError(
                f"thinking_time_std_init must be positive, got {self.thinking_time_std_init}"
            )

        if self.num_frequencies == 1:
            frequencies = torch.full((1,), self.min_frequency)
        else:
            log_freqs = torch.linspace(
                math.log(self.min_frequency),
                math.log(self.max_frequency),
                steps=self.num_frequencies,
            )
            frequencies = log_freqs.exp()

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.log_thinking_time_std = nn.Parameter(torch.tensor(math.log(self.thinking_time_std_init)))
        input_dim = 3 + 2 * self.num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.thinking_time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(self.thinking_time_embed_dim * 4, self.thinking_time_embed_dim),
        )

    def thinking_time_std(self):
        return self.log_thinking_time_std.exp()

    def _build_features(self, thinking_time):
        thinking_time = thinking_time.float().clamp(0.0, 1.0)
        sigma = self.thinking_time_std().to(device=thinking_time.device, dtype=thinking_time.dtype)
        sigma_field = sigma.expand_as(thinking_time)
        frequencies = self.frequencies.to(device=thinking_time.device, dtype=thinking_time.dtype)
        angles = thinking_time.unsqueeze(-1) * frequencies
        attenuation = torch.exp(-0.5 * sigma.square() * frequencies.square())
        attenuation = attenuation.view(*([1] * thinking_time.ndim), -1)
        spectral = torch.cat(
            [
                attenuation * torch.sin(angles),
                attenuation * torch.cos(angles),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                thinking_time.unsqueeze(-1),
                thinking_time.square().unsqueeze(-1),
                sigma_field.unsqueeze(-1),
                spectral,
            ],
            dim=-1,
        )

    def _project_embedding(self, embedding):
        first_param = next(self.mlp.parameters(), None)
        if first_param is None:
            return self.mlp(embedding)
        return self.mlp(embedding.to(first_param.dtype))

    def forward(self, thinking_time):
        return self._project_embedding(self._build_features(thinking_time))


class AdaLNProjection(nn.Module):
    """Project a time embedding into bounded AdaLN chunks for two layer sites."""

    def __init__(
        self,
        thinking_time_embed_dim,
        hidden_size,
        gamma_cap=0.99,
        beta_cap=0.99,
        alpha_cap=0.99,
    ):
        super().__init__()
        self.proj = nn.Linear(thinking_time_embed_dim, 6 * hidden_size)
        self.set_caps(gamma_cap=gamma_cap, beta_cap=beta_cap, alpha_cap=alpha_cap)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def set_caps(self, gamma_cap=0.99, beta_cap=0.99, alpha_cap=0.99):
        self.gamma_cap = max(float(gamma_cap), 0.0)
        self.beta_cap = max(float(beta_cap), 0.0)
        self.alpha_cap = max(float(alpha_cap), 0.0)

    @staticmethod
    def _bound_chunk(chunk, cap):
        if cap <= 0.0:
            return torch.zeros_like(chunk)
        return cap * torch.tanh(chunk / cap)

    def forward(self, thinking_time_emb):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.proj(thinking_time_emb).chunk(6, dim=-1)
        gamma1 = self._bound_chunk(gamma1, self.gamma_cap)
        beta1 = self._bound_chunk(beta1, self.beta_cap)
        alpha1 = self._bound_chunk(alpha1, self.alpha_cap)
        gamma2 = self._bound_chunk(gamma2, self.gamma_cap)
        beta2 = self._bound_chunk(beta2, self.beta_cap)
        alpha2 = self._bound_chunk(alpha2, self.alpha_cap)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


def prepare_adaln_modulation(raw_chunks, thinking_time_mask=None):
    """Convert raw AdaLN projection chunks into identity-preserving modulation tensors."""
    if len(raw_chunks) != 6:
        raise ValueError(f"expected 6 AdaLN chunks, got {len(raw_chunks)}")
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = raw_chunks
    alpha1 = 1.0 + alpha1
    alpha2 = 1.0 + alpha2

    if thinking_time_mask is not None:
        mask = thinking_time_mask.unsqueeze(-1).to(dtype=gamma1.dtype)
        gamma1, beta1 = gamma1 * mask, beta1 * mask
        alpha1 = 1.0 + (alpha1 - 1.0) * mask
        gamma2, beta2 = gamma2 * mask, beta2 * mask
        alpha2 = 1.0 + (alpha2 - 1.0) * mask

    return gamma1, beta1, alpha1, gamma2, beta2, alpha2


def compute_thinking_time_targets(thinking_mask):
    """Compute a 0->1 ramp over each sample's active thinking span."""
    mask_f = thinking_mask.float()
    lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    cumsum = mask_f.cumsum(dim=1)
    ramp = (cumsum - mask_f) / (lengths - 1).clamp(min=1)
    return ramp.clamp(0.0, 1.0) * mask_f


def dead_zone_mse(pred, target, margin=0.01):
    """Zero loss inside a small error band, quadratic outside it."""
    error = (pred - target).abs()
    return torch.where(error < margin, torch.zeros_like(error), (error - margin) ** 2)


def compute_thinking_time_aux_weights(rewards):
    """Convert rewards into non-negative auxiliary weights using direct reward scale."""
    if rewards is None:
        return None
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contains non-finite values")
    rewards = rewards.float()
    r_max = rewards.max()
    if r_max > 1.0:
        rewards = rewards / r_max
    return rewards.clamp_min(0.0)


def summarize_residual_rollout_metrics(embeds_ratio):
    """Summarize rollout-time residual usage with names that reflect current behavior."""
    if embeds_ratio is None:
        return {}
    if not torch.is_tensor(embeds_ratio):
        embeds_ratio = torch.tensor(embeds_ratio)
    embeds_ratio = embeds_ratio.float()
    active_mask = embeds_ratio < 1.0
    metrics = {"residual_active_fraction": active_mask.float().mean().item()}
    if active_mask.any():
        active_ratio = embeds_ratio[active_mask]
        metrics["residual_embed_ratio"] = active_ratio.mean().item()
        metrics["residual_hidden_ratio"] = torch.sqrt((1.0 - active_ratio.square()).clamp_min(0.0)).mean().item()
    else:
        metrics["residual_embed_ratio"] = 1.0
        metrics["residual_hidden_ratio"] = 0.0
    return metrics


TRAINING_REPLAY_NOISE_EPS = 0.01


def _sample_training_replay_noisy_prediction(predicted_thinking_time):
    noise = torch.empty_like(predicted_thinking_time).uniform_(
        -TRAINING_REPLAY_NOISE_EPS,
        TRAINING_REPLAY_NOISE_EPS,
    )
    return (predicted_thinking_time + noise).clamp(0.0, 1.0)


def select_training_thinking_time_embedding(
    base_model,
    gt_thinking_time,
    rollout_thinking_time=None,
    thinking_mask=None,
):
    """Build replay-time conditioning from rollout self-predicted time plus fixed noise."""
    if gt_thinking_time.ndim != 2:
        raise ValueError(
            f"gt_thinking_time must be rank-2 (B, L), got shape {tuple(gt_thinking_time.shape)}"
        )
    if not torch.isfinite(gt_thinking_time).all():
        raise ValueError("gt_thinking_time contains non-finite values")

    if thinking_mask is None:
        thinking_mask = gt_thinking_time > 0
    thinking_mask = _normalize_sequence_mask(thinking_mask, gt_thinking_time.shape, gt_thinking_time.device)

    if rollout_thinking_time is None:
        raise ValueError("rollout_thinking_time must be provided for replay training")
    if rollout_thinking_time.shape != gt_thinking_time.shape:
        raise ValueError(
            "rollout_thinking_time shape does not match gt_thinking_time: "
            f"{tuple(rollout_thinking_time.shape)} vs {tuple(gt_thinking_time.shape)}"
        )
    rollout_thinking_time = rollout_thinking_time.to(
        device=gt_thinking_time.device,
        dtype=gt_thinking_time.dtype,
    ).detach()
    if not torch.isfinite(rollout_thinking_time).all():
        raise ValueError("rollout_thinking_time contains non-finite values")
    rollout_thinking_time = rollout_thinking_time.clamp(0.0, 1.0)

    config = getattr(base_model, "config", None)
    try:
        rollout_margin = max(float(getattr(config, "thinking_time_rollout_trust_margin", 0.05)), 0.0)
    except (TypeError, ValueError):
        rollout_margin = 0.05
    noisy_rollout_thinking_time = _sample_training_replay_noisy_prediction(rollout_thinking_time)
    mask_float = thinking_mask.float()
    rollout_error = (
        dead_zone_mse(rollout_thinking_time, gt_thinking_time, margin=rollout_margin) * mask_float
    ).sum(1) / mask_float.sum(1).clamp(min=1)

    selected_thinking_time = noisy_rollout_thinking_time
    selected_thinking_time = torch.where(
        thinking_mask,
        selected_thinking_time,
        torch.zeros_like(selected_thinking_time),
    ).clamp(0.0, 1.0)
    thinking_time_emb = base_model.reasoning_time_embedding(selected_thinking_time.clone())
    source_stats = {"thinking_time_rollout_error": rollout_error.mean().item()}
    return thinking_time_emb, selected_thinking_time, source_stats


def prepare_training_thinking_time_conditioning(
    base_model,
    thinking_mask,
    rollout_thinking_time,
):
    """Populate cached AdaLN tensors for the inner decoder during replay training."""
    if thinking_mask is None:
        base_model._train_thinking_time_emb = None
        base_model._train_thinking_time_mask = None
        return None, {}

    gt_thinking_time = compute_thinking_time_targets(thinking_mask.clone())
    base_model._train_thinking_time_emb, _, source_stats = select_training_thinking_time_embedding(
        base_model,
        gt_thinking_time,
        rollout_thinking_time,
        thinking_mask=thinking_mask.clone(),
    )
    base_model._train_thinking_time_mask = thinking_mask.clone()
    return gt_thinking_time, source_stats


def prepare_trainer_time_conditioning(
    base_model,
    input_ids,
    thinking_mask,
    rollout_thinking_time,
):
    """Validate position-aligned trainer inputs and populate cached training tensors."""
    if thinking_mask is None:
        base_model._train_thinking_time_emb = None
        base_model._train_thinking_time_mask = None
        return None, {}

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be a rank-2 tensor with shape (B, L), got {type(input_ids)}"
        )

    target_shape = input_ids.shape
    device = input_ids.device
    normalized_thinking_mask = _normalize_sequence_mask(thinking_mask, target_shape, device)

    if rollout_thinking_time is not None:
        if not torch.is_tensor(rollout_thinking_time):
            rollout_thinking_time = torch.tensor(rollout_thinking_time, device=device)
        rollout_thinking_time = rollout_thinking_time.to(device=device)
        if rollout_thinking_time.shape != target_shape:
            raise ValueError(
                "rollout_thinking_time shape does not match input_ids shape: "
                f"{tuple(rollout_thinking_time.shape)} vs {tuple(target_shape)}"
            )

    return prepare_training_thinking_time_conditioning(
        base_model,
        thinking_mask=normalized_thinking_mask,
        rollout_thinking_time=rollout_thinking_time,
    )


def prepare_time_conditioning_training_step(
    model,
    input_ids,
    thinking_mask,
    rollout_thinking_time,
    thinking_time_loss_weight=0.0,
):
    """Normalize trainer masks and prepare cached tensors for one replay forward pass."""
    base_model = None
    gt_thinking_time = None
    source_stats = {}
    use_time_conditioning = False
    if thinking_time_loss_weight > 0 or thinking_mask is not None:
        base_model = get_time_conditioning_base_model(model)
        use_time_conditioning = has_time_conditioning(base_model)
        reset_time_conditioning_state(base_model)
    if thinking_mask is not None and thinking_mask.shape[1] != input_ids.shape[1]:
        if use_time_conditioning or thinking_time_loss_weight > 0:
            raise ValueError(
                f"time_conditioning: thinking_mask shape {thinking_mask.shape} does not match "
                f"input_ids shape {input_ids.shape}. This indicates a data pipeline bug."
            )
        thinking_mask = None
    if use_time_conditioning and base_model is not None:
        gt_thinking_time, source_stats = prepare_trainer_time_conditioning(
            base_model,
            input_ids=input_ids,
            thinking_mask=thinking_mask,
            rollout_thinking_time=rollout_thinking_time,
        )
    return base_model, use_time_conditioning, gt_thinking_time, thinking_mask, source_stats


def clear_training_time_conditioning_forward_cache(base_model):
    """Drop cached training embeddings after the forward pass while keeping replay hidden states for aux loss."""
    if base_model is None:
        return
    base_model._train_thinking_time_emb = None
    base_model._train_thinking_time_mask = None


def build_replay_lagged_predictor_trace(
    replay_hidden,
    prompt_mask,
    thinking_mask,
):
    """Materialize rollout-equivalent lagged predictor inputs from replay hidden states."""
    if replay_hidden is None:
        raise ValueError("replay_hidden must be provided for time_conditioning aux supervision")
    if replay_hidden.ndim != 3:
        raise ValueError(
            "replay_hidden must be rank-3 (B, L, H), "
            f"got shape {tuple(replay_hidden.shape)}"
        )

    batch_size, seq_len, _ = replay_hidden.shape
    lagged_mask = _normalize_sequence_mask(
        thinking_mask,
        (batch_size, seq_len),
        replay_hidden.device,
    )

    if prompt_mask is None:
        raise ValueError("prompt_mask must be provided for replay lagged predictor traces")
    if not torch.is_tensor(prompt_mask):
        prompt_mask = torch.tensor(prompt_mask, device=replay_hidden.device)
    prompt_mask = prompt_mask.to(device=replay_hidden.device, dtype=torch.bool)
    if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
        raise ValueError(
            "prompt_mask must have shape (B, P), "
            f"got shape {tuple(prompt_mask.shape)} for batch size {batch_size}"
        )

    prompt_lengths = prompt_mask.long().sum(dim=1)
    if (prompt_lengths <= 0).any():
        raise ValueError("prompt_mask must include at least one prompt token per sample")
    if (prompt_lengths > seq_len).any():
        raise ValueError(
            "prompt_mask describes more prompt tokens than replay_hidden sequence length: "
            f"{prompt_lengths.tolist()} vs seq_len={seq_len}"
        )

    lagged_hidden = torch.zeros_like(replay_hidden)
    for row_idx in range(batch_size):
        prompt_len = int(prompt_lengths[row_idx].item())
        carry = replay_hidden[row_idx, prompt_len - 1]
        for pos_idx in range(prompt_len, seq_len):
            lagged_hidden[row_idx, pos_idx] = carry
            if lagged_mask[row_idx, pos_idx]:
                carry = replay_hidden[row_idx, pos_idx]

    return lagged_hidden, lagged_mask


def predict_replay_thinking_time_from_hidden_trace(
    predictor,
    replay_hidden,
    *,
    prompt_mask,
    thinking_mask,
):
    """Predict replay thinking time while streaming the lagged hidden carry."""
    if replay_hidden is None:
        raise ValueError("replay_hidden must be provided for replay prediction")
    if replay_hidden.ndim != 3:
        raise ValueError(
            "replay_hidden must be rank-3 (B, L, H), "
            f"got shape {tuple(replay_hidden.shape)}"
        )

    batch_size, seq_len, _ = replay_hidden.shape
    lagged_mask = _normalize_sequence_mask(
        thinking_mask,
        (batch_size, seq_len),
        replay_hidden.device,
    )

    if prompt_mask is None:
        raise ValueError("prompt_mask must be provided for replay prediction")
    if not torch.is_tensor(prompt_mask):
        prompt_mask = torch.tensor(prompt_mask, device=replay_hidden.device)
    prompt_mask = prompt_mask.to(device=replay_hidden.device, dtype=torch.bool)
    if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
        raise ValueError(
            "prompt_mask must have shape (B, P), "
            f"got shape {tuple(prompt_mask.shape)} for batch size {batch_size}"
        )

    prompt_lengths = prompt_mask.long().sum(dim=1)
    if (prompt_lengths <= 0).any():
        raise ValueError("prompt_mask must include at least one prompt token per sample")
    if (prompt_lengths > seq_len).any():
        raise ValueError(
            "prompt_mask describes more prompt tokens than replay_hidden sequence length: "
            f"{prompt_lengths.tolist()} vs seq_len={seq_len}"
        )

    batch_indices = torch.arange(batch_size, device=replay_hidden.device)
    carry = replay_hidden[batch_indices, prompt_lengths - 1].unsqueeze(1)
    preds = []
    zero_step = torch.zeros(batch_size, 1, device=replay_hidden.device, dtype=replay_hidden.dtype)

    for pos_idx in range(seq_len):
        after_prompt_mask = (prompt_lengths <= pos_idx).unsqueeze(1)
        step_mask = lagged_mask[:, pos_idx : pos_idx + 1] & after_prompt_mask
        if step_mask.any():
            step_pred = _predict_thinking_time_from_hidden_trace(
                predictor,
                carry,
                step_mask,
            )
        else:
            step_pred = zero_step
        preds.append(step_pred)

        current_hidden = replay_hidden[:, pos_idx : pos_idx + 1, :]
        carry = torch.where(
            lagged_mask[:, pos_idx : pos_idx + 1].unsqueeze(-1),
            current_hidden,
            carry,
        )

    return torch.cat(preds, dim=1).clamp(0.0, 1.0), lagged_mask


def _predict_thinking_time_from_hidden_trace(
    predictor,
    predictor_hidden_input,
    predictor_step_mask,
):
    if isinstance(predictor, ThinkingTimePredictor) or not hasattr(predictor, "forward_step"):
        return predictor(
            predictor_hidden_input,
            thinking_mask=predictor_step_mask,
        ).clamp(0.0, 1.0)

    preds = []
    for idx in range(predictor_hidden_input.shape[1]):
        used_thinking_time = predictor.forward_step(
            predictor_hidden_input[:, idx : idx + 1, :],
            is_thinking=predictor_step_mask[:, idx : idx + 1],
        )
        preds.append(used_thinking_time)
    return torch.cat(preds, dim=1).clamp(0.0, 1.0)


def compute_thinking_time_aux_loss(
    base_model,
    gt_thinking_time,
    thinking_mask,
    prompt_mask,
    rewards=None,
):
    """Compute predictor supervision from replay hidden states using the rollout lag contract."""
    if gt_thinking_time is None or thinking_mask is None:
        return None, {}

    replay_hidden = getattr(base_model, "_train_predictor_hidden_states", None)
    if replay_hidden is None:
        raise ValueError("time_conditioning aux requires cached replay hidden states")
    if replay_hidden.shape[:2] != gt_thinking_time.shape:
        raise ValueError(
            "cached replay hidden leading dims do not match gt_thinking_time: "
            f"{tuple(replay_hidden.shape[:2])} vs {tuple(gt_thinking_time.shape)}"
        )

    thinking_time_pred, lagged_mask = predict_replay_thinking_time_from_hidden_trace(
        base_model.thinking_time_predictor,
        replay_hidden,
        prompt_mask=prompt_mask,
        thinking_mask=thinking_mask,
    )

    mask_bool = lagged_mask.bool() & thinking_mask.bool()
    mask_float = mask_bool.float()
    per_token_loss = dead_zone_mse(thinking_time_pred, gt_thinking_time)
    per_sample_loss = (per_token_loss * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)

    aux_weights = compute_thinking_time_aux_weights(rewards)
    if aux_weights is None:
        aux_weights = torch.ones_like(per_sample_loss)
    aux_weights = aux_weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    aux_loss = (per_sample_loss * aux_weights).mean()

    metrics = {
        "thinking_time_aux_loss": aux_loss.detach().item(),
        "thinking_time_pred_error": per_sample_loss.mean().detach().item(),
    }
    embedding_module = getattr(base_model, "reasoning_time_embedding", None)
    if embedding_module is not None and hasattr(embedding_module, "thinking_time_std"):
        metrics["thinking_time_embed_std"] = embedding_module.thinking_time_std().detach().float().item()

    return aux_loss, metrics


def get_time_conditioning_base_model(model):
    """Resolve the inner model that owns time-conditioning state."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model"):
        return model.base_model.model.model
    if hasattr(model, "model"):
        return model.model
    return model


def has_time_conditioning(model):
    """Return whether the resolved model currently owns active time-conditioning modules."""
    return bool(
        model is not None
        and getattr(model, "use_time_conditioning", False)
        and hasattr(model, "thinking_time_predictor")
    )


def set_thinking_residual_disabled(model, disabled):
    """Synchronize the runtime thinking-residual toggle across common wrappers."""
    disabled = bool(disabled)
    targets = [model, getattr(model, "model", None), get_time_conditioning_base_model(model)]
    seen = set()
    for target in targets:
        if target is None or id(target) in seen:
            continue
        setattr(target, "disable_thinking_residual", disabled)
        seen.add(id(target))


def reset_time_conditioning_state(model):
    """Clear cached time-conditioning state to avoid stale rollout leakage."""
    if model is None:
        return
    for attr in (
        "_train_thinking_time_emb",
        "_train_thinking_time_mask",
        "_train_predictor_hidden_states",
        "_cached_last_hidden",
        "_used_thinking_time",
    ):
        setattr(model, attr, None)


def prepare_online_thinking_time_conditioning(base_model, batch_size, device, is_thinking=None):
    """Build one-step-lag rollout conditioning from cached hidden states only."""
    used_thinking_time = torch.zeros(batch_size, 1, device=device)
    thinking_time_emb = None
    thinking_time_mask = _normalize_step_mask(is_thinking, batch_size, device)
    cached_hidden = getattr(base_model, "_cached_last_hidden", None)

    if thinking_time_mask is None and cached_hidden is None:
        base_model._used_thinking_time = used_thinking_time
        return None, None, used_thinking_time

    active_mask = thinking_time_mask
    if active_mask is None:
        active_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)

    if cached_hidden is not None and cached_hidden.shape[:2] == (batch_size, 1):
        with torch.no_grad():
            predicted_thinking_time = base_model.thinking_time_predictor.forward_step(
                cached_hidden,
                is_thinking=is_thinking,
            )
        used_thinking_time = predicted_thinking_time.clamp(0.0, 1.0)
        if active_mask.any():
            thinking_time_emb = base_model.reasoning_time_embedding(used_thinking_time.clone())
            thinking_time_mask = active_mask
        else:
            thinking_time_mask = None
    else:
            thinking_time_mask = None

    base_model._used_thinking_time = used_thinking_time.detach()
    return thinking_time_emb, thinking_time_mask, used_thinking_time


def update_online_thinking_time_hidden_cache(base_model, new_hidden, is_thinking=None):
    """Update the one-step-lag hidden cache while freezing rows that finished thinking."""
    old_hidden = getattr(base_model, "_cached_last_hidden", None)
    mask = _normalize_step_mask(is_thinking, new_hidden.shape[0], new_hidden.device)
    if mask is not None and old_hidden is not None and old_hidden.shape == new_hidden.shape:
        base_model._cached_last_hidden = torch.where(mask.unsqueeze(-1), new_hidden, old_hidden)
    else:
        base_model._cached_last_hidden = new_hidden


def _normalize_sequence_mask(thinking_mask, target_shape, device):
    mask = thinking_mask
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.shape != target_shape:
        raise ValueError(f"thinking_mask shape {tuple(mask.shape)} does not match target shape {tuple(target_shape)}")
    return mask


def _normalize_step_mask(is_thinking, batch_size, device):
    if is_thinking is None:
        return None
    mask = is_thinking
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(1)
    if mask.shape != (batch_size, 1):
        raise ValueError(f"is_thinking shape {tuple(mask.shape)} does not match expected {(batch_size, 1)}")
    return mask


def enable_time_conditioning(model):
    """Attach predictor, time embedding, and AdaLN modules before LoRA wrapping."""
    inner = get_time_conditioning_base_model(model)
    config = inner.config

    def _read_non_negative_float(attr_name, default):
        try:
            return max(float(getattr(config, attr_name, default)), 0.0)
        except (TypeError, ValueError):
            return default

    thinking_time_embed_dim = getattr(config, "thinking_time_embed_dim", 256)
    thinking_time_num_frequencies = getattr(config, "thinking_time_num_frequencies", 32)
    thinking_time_min_frequency = getattr(config, "thinking_time_min_frequency", 1.0)
    thinking_time_max_frequency = getattr(config, "thinking_time_max_frequency", 6.0)
    thinking_time_std_init = getattr(config, "thinking_time_std_init", 0.05)
    thinking_time_rollout_trust_margin = _read_non_negative_float("thinking_time_rollout_trust_margin", 0.05)
    thinking_time_gamma_cap = _read_non_negative_float("thinking_time_gamma_cap", 0.99)
    thinking_time_beta_cap = _read_non_negative_float("thinking_time_beta_cap", 0.99)
    thinking_time_alpha_cap = _read_non_negative_float("thinking_time_alpha_cap", 0.99)
    thinking_time_predictor_num_hidden_states = get_time_conditioning_predictor_num_hidden_states(config)
    device = next(inner.parameters()).device
    dtype = next(inner.parameters()).dtype
    predictor_input_size = get_time_conditioning_predictor_input_size(
        config.hidden_size,
        thinking_time_predictor_num_hidden_states,
    )

    for layer in inner.layers:
        if not hasattr(layer, "adaln_proj"):
            layer.adaln_proj = AdaLNProjection(
                thinking_time_embed_dim=thinking_time_embed_dim,
                hidden_size=config.hidden_size,
                gamma_cap=thinking_time_gamma_cap,
                beta_cap=thinking_time_beta_cap,
                alpha_cap=thinking_time_alpha_cap,
            ).to(device=device, dtype=dtype)
        else:
            layer.adaln_proj.set_caps(
                gamma_cap=thinking_time_gamma_cap,
                beta_cap=thinking_time_beta_cap,
                alpha_cap=thinking_time_alpha_cap,
            )

    predictor_input_features = None
    if hasattr(inner, "thinking_time_predictor") and hasattr(inner.thinking_time_predictor, "net"):
        first_layer = inner.thinking_time_predictor.net[0]
        predictor_input_features = getattr(first_layer, "in_features", None)
    if predictor_input_features != predictor_input_size:
        inner.thinking_time_predictor = ThinkingTimePredictor(predictor_input_size).to(
            device=device, dtype=dtype
        )
    if not hasattr(inner, "reasoning_time_embedding"):
        inner.reasoning_time_embedding = ReasoningTimeEmbedding(
            thinking_time_embed_dim=thinking_time_embed_dim,
            num_frequencies=thinking_time_num_frequencies,
            min_frequency=thinking_time_min_frequency,
            max_frequency=thinking_time_max_frequency,
            thinking_time_std_init=thinking_time_std_init,
        ).to(
            device=device,
            dtype=dtype,
        )

    inner.use_time_conditioning = True
    config.use_time_conditioning = True
    config.thinking_time_embed_dim = thinking_time_embed_dim
    config.thinking_time_num_frequencies = thinking_time_num_frequencies
    config.thinking_time_min_frequency = thinking_time_min_frequency
    config.thinking_time_max_frequency = thinking_time_max_frequency
    config.thinking_time_std_init = thinking_time_std_init
    config.thinking_time_rollout_trust_margin = float(thinking_time_rollout_trust_margin)
    config.thinking_time_gamma_cap = float(thinking_time_gamma_cap)
    config.thinking_time_beta_cap = float(thinking_time_beta_cap)
    config.thinking_time_alpha_cap = float(thinking_time_alpha_cap)
    config.thinking_time_predictor_num_hidden_states = int(thinking_time_predictor_num_hidden_states)
    reset_time_conditioning_state(inner)
