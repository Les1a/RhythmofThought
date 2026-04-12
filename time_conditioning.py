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
from tqdm.auto import tqdm


class TimeProgressPredictor(nn.Module):
    """Predicts per-token reasoning progress t ∈ [0,1] from input embeddings."""

    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, inputs_embeds):
        # inputs_embeds: (B, L, hidden_size)
        return torch.sigmoid(self.net(inputs_embeds)).squeeze(-1)  # (B, L)


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
        linearly ramps from 0.0 to 1.0 across each sample's thinking span.
        A single thinking token is treated as progress 0.0.
    """
    mask_f = thinking_mask.float()
    lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    cumsum = mask_f.cumsum(dim=1)
    # Shift cumsum so first thinking token is 0 and the last is 1.
    # For a single-token span we intentionally keep t=0.0 to represent the
    # degenerate start/end position without fabricating a terminal ramp.
    ramp = (cumsum - mask_f) / (lengths - 1).clamp(min=1)
    return ramp.clamp(0, 1) * mask_f  # zero out non-thinking positions


def dead_zone_mse(pred, target, margin=0.01):
    """Dead-zone MSE: zero loss when |pred - target| < margin, quadratic beyond."""
    error = (pred - target).abs()
    return torch.where(error < margin, torch.zeros_like(error), (error - margin) ** 2)


def select_training_time_embedding(base_model, gt_time_vals, rollout_time_pred, global_step, max_steps):
    """Choose GT vs. predicted time for AdaLN during training.

    The predicted-time probability linearly anneals from 0 at step 0 to 1 at
    ``max_steps``. If the rollout prediction is unavailable, the helper falls
    back to GT time while reporting that fallback to the caller.

    Args:
        base_model: Inner model that owns ``sinusoidal_time_embedding``.
        gt_time_vals: (B, L) GT reasoning progress values.
        rollout_time_pred: Optional (B, L) rollout-time predictions.
        global_step: Current completed optimization step.
        max_steps: Total optimization steps. If <= 0, always uses GT time.

    Returns:
        Tuple ``(time_emb, pred_prob, used_pred, used_fallback_gt)`` where
        ``time_emb`` is ready for AdaLN consumption.
    """
    if gt_time_vals.ndim != 2:
        raise ValueError(f"gt_time_vals must be rank-2 (B, L), got shape {tuple(gt_time_vals.shape)}")
    if not torch.isfinite(gt_time_vals).all():
        raise ValueError("gt_time_vals contains non-finite values")

    pred_prob = 0.0
    if max_steps is not None and max_steps > 0:
        pred_prob = float(global_step) / float(max_steps)
        pred_prob = min(max(pred_prob, 0.0), 1.0)

    if rollout_time_pred is not None:
        if rollout_time_pred.shape != gt_time_vals.shape:
            raise ValueError(
                "rollout_time_pred shape does not match gt_time_vals: "
                f"{tuple(rollout_time_pred.shape)} vs {tuple(gt_time_vals.shape)}"
            )
        rollout_time_pred = rollout_time_pred.to(device=gt_time_vals.device, dtype=gt_time_vals.dtype)
        if not torch.isfinite(rollout_time_pred).all():
            raise ValueError("rollout_time_pred contains non-finite values")
        rollout_time_pred = rollout_time_pred.clamp(0.0, 1.0)

    use_pred = False
    used_fallback_gt = False
    if rollout_time_pred is not None and pred_prob > 0.0:
        use_pred = bool((torch.rand((), device=gt_time_vals.device) < pred_prob).item())
    elif rollout_time_pred is None and pred_prob > 0.0:
        used_fallback_gt = True

    selected_time = rollout_time_pred if use_pred else gt_time_vals
    time_emb = base_model.sinusoidal_time_embedding(selected_time.clone())
    return time_emb, pred_prob, use_pred, used_fallback_gt


def enable_time_conditioning(model):
    """Dynamically create and attach time conditioning modules to a loaded model.

    Call after FastLanguageModel.from_pretrained() and before get_peft_model().
    The model.__init__ won't create these modules unless config.use_time_conditioning
    is set, so this function handles post-hoc creation.

    Args:
        model: The top-level CausalLM (e.g. from FastLanguageModel.from_pretrained).
    """
    inner = model.model  # CausalLM.model → inner model (LlamaModel / Qwen2Model)
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
        if 'adaln_proj' in acfg.get('modules_to_save', []):
            enable_time_conditioning(model)


def pretrain_time_predictor(model, tokenizer, dataset, num_samples=1024,
                            num_epochs=3, lr=1e-4, temperature=0.5,
                            max_completion_length=1024, batch_size=128):
    """Phase 0: Pre-train TimeProgressPredictor before main GRPO training.

    Generates completions for a subset of training data, builds thinking masks
    from the ``####`` answer marker, then trains the predictor to map final
    hidden states to a linear time ramp [0, 1] across thinking tokens.

    Only the predictor is updated; all other model parameters are frozen.

    Args:
        model: PEFT-wrapped CausalLM with time conditioning enabled.
        tokenizer: Tokenizer / processing class.
        dataset: Training dataset (must have a ``prompt`` column).
        num_samples: Number of prompts to generate completions for.
        num_epochs: Supervised training epochs over the generated data.
        lr: Learning rate for the predictor optimizer.
        temperature: Sampling temperature for completion generation.
        max_completion_length: Maximum new tokens per generation.
        batch_size: Batch size for both generation and training.
    """
    from transformers import GenerationConfig

    device = next(model.parameters()).device

    # Resolve inner model (Qwen2Model / LlamaModel)
    if hasattr(model, 'base_model'):
        inner = model.base_model.model.model
    else:
        inner = model.model

    n = min(num_samples, len(dataset))
    print(f"[Phase 0] Generating {n} completions for time predictor pre-training...")

    # ---- Generation phase ----
    gen_config = GenerationConfig(
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
    )

    model.eval()
    # Reset any stale state from prior .generate() calls so the first
    # batch's forward takes the fresh predictor-free path.
    inner._gt_time_emb = None
    inner._gt_time_mask = None
    inner._cached_last_hidden = None
    inner._last_time_pred = None

    all_full_ids = []
    all_thinking_masks = []
    all_attention_masks = []

    with tqdm(total=n, desc="[Phase 0 inference]", unit="completions") as pbar:
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_prompts = dataset[i:end]["prompt"]
            prompts_text = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in batch_prompts
            ]
            inputs = tokenizer(
                prompts_text, return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_out = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=gen_config,
                    processing_class=tokenizer,
                    return_thinking_embeds=True,
                )
                # gen_out is a 5-tuple: (input_ids, thinking_embeds, thinking_mask,
                # embeds_ratio, time_preds). Only grab what we actually use; drop
                # the rest via `del gen_out` to free (B, L, D) tensors early.
                full_ids = gen_out[0]
                thinking_mask = gen_out[2]
                del gen_out

            pbar.update(end - i)

            # Skip batches with zero thinking tokens
            if thinking_mask.sum() == 0:
                del full_ids, thinking_mask, inputs
                torch.cuda.empty_cache()
                continue

            attn_mask = (full_ids != tokenizer.pad_token_id).long()
            all_full_ids.append(full_ids.cpu())
            all_thinking_masks.append(thinking_mask.cpu())
            all_attention_masks.append(attn_mask.cpu())

            # Free per-batch GPU tensors before the next iteration.
            del full_ids, thinking_mask, attn_mask, inputs
            torch.cuda.empty_cache()

            print(f"  [Phase 0 inference] Generated {end}/{n} completions")

    if not all_full_ids:
        print("[Phase 0] No thinking tokens found in any completion — skipping pre-training.")
        model.train()
        return

    # Free generation-phase GPU buffers before training phase
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Training phase ----
    print(f"[Phase 0] Training time predictor for {num_epochs} epochs "
          f"on {len(all_full_ids)} batches...")

    # Freeze everything except TimeProgressPredictor
    saved_grad = {}
    predictor_params = []
    for name, param in model.named_parameters():
        saved_grad[name] = param.requires_grad
        if "time_progress_predictor" in name:
            param.requires_grad_(True)
            predictor_params.append(param)
        else:
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(predictor_params, lr=lr)
    # Keep the model in evaluation mode for the whole training phase:
    # - disables gradient checkpointing (unsloth/llama.py:838)
    # - keeps attention_mask 4D-prepared for variable-length padded batches
    #   (unsloth/llama.py:764-783 would zero it out under self.training=True)
    # - the predictor has no dropout/batchnorm, so evaluation mode doesn't
    #   change its output
    model.train(False)
    predictor = inner.time_progress_predictor

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        indices = torch.randperm(len(all_full_ids))

        for idx in indices:
            full_ids = all_full_ids[idx].to(device, non_blocking=True)
            t_mask = all_thinking_masks[idx].to(device, non_blocking=True)
            attn_mask = all_attention_masks[idx].to(device, non_blocking=True)

            if t_mask.sum() == 0:
                continue

            # Force the backbone into its unmodulated predictor-free path.
            # modeling_qwen2.py:639-660 / unsloth/llama.py:791-809 gate
            # time_emb on these attrs — all None means time_emb stays None.
            inner._gt_time_emb = None
            inner._gt_time_mask = None
            inner._cached_last_hidden = None

            # Step 1: backbone under no_grad to get last hidden states.
            # Calling `inner` directly (not `model`) skips the LM head entirely.
            # Gradient-wise this is numerically identical to the old path
            # because modeling_qwen2.py:716 / unsloth/llama.py:987 feeds
            # hidden_states.detach() into the predictor — the backbone was
            # never in the predictor's gradient graph. Under the outer
            # no_grad the internal predictor call at line 987 builds no
            # graph; we re-invoke the predictor explicitly below for grads.
            with torch.no_grad():
                out = inner(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                hidden_states = out.last_hidden_state  # (B, L, D), no grad

            # Step 2: run only the predictor with grads enabled.
            optimizer.zero_grad(set_to_none=True)
            pred_time = predictor(hidden_states)  # (B, L)

            gt_time = compute_gt_time(t_mask).to(pred_time.device, pred_time.dtype)
            tm = t_mask.float()
            dz = dead_zone_mse(pred_time, gt_time)
            loss = (dz * tm).sum() / tm.sum().clamp(min=1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            del full_ids, t_mask, attn_mask, hidden_states, pred_time, out

        avg = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{num_epochs}: time_pred_mse = {avg:.6f}")

    # Restore requires_grad
    for name, param in model.named_parameters():
        param.requires_grad_(saved_grad.get(name, True))

    # Cleanup
    inner._gt_time_emb = None
    inner._gt_time_mask = None
    inner._cached_last_hidden = None
    inner._last_time_pred = None
    del all_full_ids, all_thinking_masks, all_attention_masks
    import gc as _gc; _gc.collect()
    torch.cuda.empty_cache()

    model.train()
    print("[Phase 0] Time predictor pre-training complete.")
