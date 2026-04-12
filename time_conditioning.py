"""
Time-Conditioned LayerNorm Modulation for HRPO.

Provides modules for predicting reasoning progress, encoding it sinusoidally,
and modulating RMSNorm layers via per-layer AdaLN with scale (γ), shift (β),
and gate (α) projections.
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeProgressPredictor(nn.Module):
    """Predicts monotonic per-token reasoning progress t ∈ [0,1]."""

    def __init__(self, hidden_size, delta_eps=1e-6, init_bias=-4.0):
        super().__init__()
        self.delta_eps = float(delta_eps)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
        )
        # Start near zero progress increments so early rollout predictions do not
        # immediately saturate the time embedding.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def _delta_from_hidden(self, hidden_states):
        return F.softplus(self.net(hidden_states)).squeeze(-1) + self.delta_eps

    def forward(self, inputs_embeds, thinking_mask=None):
        # inputs_embeds: (B, L, hidden_size)
        delta = self._delta_from_hidden(inputs_embeds)
        mask = None
        if thinking_mask is not None:
            mask = _normalize_sequence_mask(thinking_mask, delta.shape, delta.device)
            delta = delta * mask.to(delta.dtype)
        cum = torch.cumsum(delta, dim=1)
        time = 1.0 - torch.exp(-cum)
        if mask is not None:
            time = time * mask.to(time.dtype)
        return time  # (B, L)

    def forward_step(self, inputs_embeds, prev_cum=None, is_thinking=None):
        # inputs_embeds: (B, 1, hidden_size)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] != 1:
            raise ValueError(
                "forward_step expects inputs_embeds with shape (B, 1, hidden_size), "
                f"got {tuple(inputs_embeds.shape)}"
            )
        delta = self._delta_from_hidden(inputs_embeds)
        if prev_cum is None:
            prev_cum = torch.zeros_like(delta)
        else:
            if prev_cum.shape != delta.shape:
                raise ValueError(
                    f"prev_cum shape {tuple(prev_cum.shape)} does not match delta shape {tuple(delta.shape)}"
                )
            prev_cum = prev_cum.to(device=delta.device, dtype=delta.dtype)
        mask = _normalize_step_mask(is_thinking, delta.shape[0], delta.device)
        if mask is None:
            mask = torch.ones_like(delta, dtype=torch.bool)
        next_cum = torch.where(mask, prev_cum + delta, prev_cum)
        time = 1.0 - torch.exp(-next_cum)
        time = torch.where(mask, time, torch.zeros_like(time))
        return time, next_cum


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal encoding of scalar t ∈ [0,1], projected through 2-layer MLP."""

    def __init__(self, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        half_dim = time_embed_dim // 2
        # Precompute log-spaced frequencies (not trainable)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("freqs", freqs)
        # 2-layer MLP to refine the sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t):
        # t: (B, L) scalar time values in [0, 1]
        # Expand t for broadcasting: (B, L, 1) * (half_dim,) -> (B, L, half_dim)
        args = t.unsqueeze(-1).float() * self.freqs
        # Sinusoidal encoding: (B, L, time_embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # Cast to MLP weight dtype (handles bf16/fp32 mismatch after PEFT wrapping)
        mlp_dtype = next(self.mlp.parameters()).dtype
        return self.mlp(embedding.to(mlp_dtype))


class AdaLNProjection(nn.Module):
    """Per-layer AdaLN projection: time_emb → (γ1, β1, α1, γ2, β2, α2).

    Produces 3 modulation parameters (scale, shift, gate) for each of the
    2 LayerNorms in a decoder layer. Zero-initialized so the model starts
    as identity (AdaLN-Zero).
    """

    def __init__(self, time_embed_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(time_embed_dim, 6 * hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Gate (α) bias must be 1 for pre-trained model identity:
        # residual + 1*sublayer = normal pre-trained behavior
        # Layout: [γ1(D), β1(D), α1(D), γ2(D), β2(D), α2(D)]
        with torch.no_grad():
            self.proj.bias[2 * hidden_size : 3 * hidden_size] = 1.0
            self.proj.bias[5 * hidden_size : 6 * hidden_size] = 1.0

    def forward(self, time_emb):
        # time_emb: (B, L, time_embed_dim)
        # Returns 6 tensors each of shape (B, L, hidden_size)
        return self.proj(time_emb).chunk(6, dim=-1)


def compute_gt_time(thinking_mask):
    """Compute GT time values for thinking tokens.

    Args:
        thinking_mask: (B, N) boolean, True for thinking tokens.

    Returns:
        (B, N) float tensor. 0 for non-thinking positions,
        linearly ramps from 1 / n to 1.0 across each sample's thinking span.
    """
    mask_f = thinking_mask.float()
    lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    ramp = mask_f.cumsum(dim=1) / lengths
    return ramp.clamp(0, 1) * mask_f  # zero out non-thinking positions


def dead_zone_mse(pred, target, margin=0.01):
    """Dead-zone MSE: zero loss when |pred - target| < margin, quadratic beyond."""
    error = (pred - target).abs()
    return torch.where(error < margin, torch.zeros_like(error), (error - margin) ** 2)


def select_training_time_embedding(
    base_model,
    gt_time_vals,
    rollout_time_pred,
    global_step,
    max_steps,
    thinking_mask=None,
):
    """Build mixed GT / rollout time embeddings for AdaLN during training.

    The rollout mixing coefficient linearly anneals from 0 at step 0 to 1 at
    ``max_steps``. Rollout predictions are detached, clamped to [0, 1], and
    mixed only on thinking positions. Non-thinking positions remain zero.

    Args:
        base_model: Inner model that owns ``sinusoidal_time_embedding``.
        gt_time_vals: (B, L) GT reasoning progress values.
        rollout_time_pred: Optional (B, L) rollout-time predictions.
        global_step: Current completed optimization step.
        max_steps: Total optimization steps. If <= 0, always uses GT time.

    Returns:
        Tuple ``(time_emb, mix_alpha, effective_rollout_alpha)`` where
        ``time_emb`` is ready for AdaLN consumption.
    """
    if gt_time_vals.ndim != 2:
        raise ValueError(f"gt_time_vals must be rank-2 (B, L), got shape {tuple(gt_time_vals.shape)}")
    if not torch.isfinite(gt_time_vals).all():
        raise ValueError("gt_time_vals contains non-finite values")

    mix_alpha = 0.0
    if max_steps is not None and max_steps > 0:
        mix_alpha = float(global_step) / float(max_steps)
        mix_alpha = min(max(mix_alpha, 0.0), 1.0)

    if thinking_mask is None:
        thinking_mask = gt_time_vals > 0
    thinking_mask = _normalize_sequence_mask(thinking_mask, gt_time_vals.shape, gt_time_vals.device)

    if rollout_time_pred is not None:
        if rollout_time_pred.shape != gt_time_vals.shape:
            raise ValueError(
                "rollout_time_pred shape does not match gt_time_vals: "
                f"{tuple(rollout_time_pred.shape)} vs {tuple(gt_time_vals.shape)}"
            )
        rollout_time_pred = rollout_time_pred.to(device=gt_time_vals.device, dtype=gt_time_vals.dtype).detach()
        if not torch.isfinite(rollout_time_pred).all():
            raise ValueError("rollout_time_pred contains non-finite values")
        rollout_time_pred = rollout_time_pred.clamp(0.0, 1.0)

    selected_time = gt_time_vals
    effective_rollout_alpha = 0.0
    if rollout_time_pred is not None and mix_alpha > 0.0:
        selected_time = (1.0 - mix_alpha) * gt_time_vals + mix_alpha * rollout_time_pred
        effective_rollout_alpha = mix_alpha
    selected_time = torch.where(thinking_mask, selected_time, torch.zeros_like(selected_time)).clamp(0.0, 1.0)
    time_emb = base_model.sinusoidal_time_embedding(selected_time.clone())
    return time_emb, mix_alpha, effective_rollout_alpha


def get_time_conditioning_base_model(model):
    """Resolve the inner model that owns time-conditioning state."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model"):
        return model.base_model.model.model
    if hasattr(model, "model"):
        return model.model
    return model


def reset_time_conditioning_state(model, clear_last_time_pred=True):
    """Clear cached time-conditioning state to avoid stale rollout leakage."""
    if model is None:
        return
    attrs = (
        "_gt_time_emb",
        "_gt_time_mask",
        "_cached_last_hidden",
        "_used_time_pred",
        "_time_progress_cum",
    )
    for attr in attrs:
        setattr(model, attr, None)
    if clear_last_time_pred:
        model._last_time_pred = None


def prepare_online_time_conditioning(base_model, batch_size, device, is_thinking=None):
    """Build rollout-time embeddings from cached hidden state and cumulative progress."""
    used_time_pred = torch.zeros(batch_size, 1, device=device)
    time_emb = None
    time_mask = _normalize_step_mask(is_thinking, batch_size, device)
    cached_hidden = getattr(base_model, "_cached_last_hidden", None)
    prev_cum = getattr(base_model, "_time_progress_cum", None)
    if prev_cum is not None and prev_cum.shape != (batch_size, 1):
        prev_cum = None
        base_model._time_progress_cum = None
    if cached_hidden is not None and cached_hidden.shape[:2] == (batch_size, 1):
        with torch.no_grad():
            used_time_pred, next_cum = base_model.time_progress_predictor.forward_step(
                cached_hidden,
                prev_cum=prev_cum,
                is_thinking=is_thinking,
            )
            base_model._time_progress_cum = next_cum.detach()
            time_emb = base_model.sinusoidal_time_embedding(used_time_pred)
    base_model._used_time_pred = used_time_pred
    return time_emb, time_mask, used_time_pred


def update_online_time_conditioning_hidden_cache(base_model, new_hidden, is_thinking=None):
    """Update the one-step-lag hidden cache, freezing stopped samples."""
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
    """Dynamically create and attach time conditioning modules to a loaded model.

    Call after FastLanguageModel.from_pretrained() and before get_peft_model().
    The model.__init__ won't create these modules unless config.use_time_conditioning
    is set, so this function handles post-hoc creation.

    Args:
        model: The top-level CausalLM (e.g. from FastLanguageModel.from_pretrained).
    """
    inner = get_time_conditioning_base_model(model)
    config = inner.config
    _te_dim = getattr(config, 'time_embed_dim', 256)
    device = next(inner.parameters()).device
    dtype = next(inner.parameters()).dtype

    # Per-layer AdaLN
    for layer in inner.layers:
        if not hasattr(layer, 'adaln_proj'):
            layer.adaln_proj = AdaLNProjection(
                time_embed_dim=_te_dim, hidden_size=config.hidden_size
            ).to(device=device, dtype=dtype)

    # Model-level modules
    if not hasattr(inner, 'time_progress_predictor'):
        inner.time_progress_predictor = TimeProgressPredictor(
            config.hidden_size
        ).to(device=device, dtype=dtype)
    if not hasattr(inner, 'sinusoidal_time_embedding'):
        inner.sinusoidal_time_embedding = SinusoidalTimeEmbedding(
            time_embed_dim=_te_dim
        ).to(device=device, dtype=dtype)
    inner.use_time_conditioning = True
    reset_time_conditioning_state(inner)


def detect_time_conditioning(model, adapter_path):
    """Auto-detect and enable time conditioning from a saved adapter checkpoint.

    Reads adapter_config.json to check if 'adaln_proj' is in modules_to_save.
    If so, creates the time conditioning modules (via enable_time_conditioning)
    so that load_adapter() can restore the saved weights.

    Call after from_pretrained() but before load_adapter().

    Args:
        model: The top-level CausalLM.
        adapter_path: Path to the adapter checkpoint directory.
    """
    cfg_path = os.path.join(adapter_path, 'adapter_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            acfg = json.load(f)
        modules_to_save = acfg.get('modules_to_save') or []
        if 'adaln_proj' in modules_to_save:
            enable_time_conditioning(model)
