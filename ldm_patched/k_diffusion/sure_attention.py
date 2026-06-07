"""SURE-AG: Attention-Guided SURE correction.

Theory
------
Standard SURE (Stein's Unbiased Risk Estimator) applies uniform correction:

    x0_corrected = x0_hat − α · ∇SURE(x0_hat)

SURE-AG weights this correction by per-layer self-attention entropy U ∈ [0,1]:

    x0_corrected = x0_hat − α · (1 + w · U) · ∇SURE(x0_hat)

Attention entropy at query position i:
    H_l[i] = −∑_j A_l[i,j] · log(A_l[i,j] + ε)

where A_l ∈ R^{B×H×N×N} is the attention weight matrix at layer l.

High entropy → model is uncertain where to attend → apply stronger correction.
Low entropy  → confident, sharp attention → trust the model more.

This is the spatial analogue of wavelet SURE (which weights by frequency band
risk). SURE-AG weights by semantic/spatial uncertainty.

Formal proofs: SURE_attention.lean
  §1 attn_entropy_nonneg     — H(p) ≥ 0 for valid distributions
  §2 attn_weight_bounded     — W(U) ∈ [1, 1+w] (step stays finite)
  §3 sure_ag_effective_range — descent requires α < 1/(2(1+w))
  §4 sure_ag_proxy_descent   — proxy SURE strictly decreases in valid range
  §5 entropy_norm_le_one     — normalised U ∈ [0,1] by construction

VRAM note: the attention capture pass uses attention_basic_with_sim (explicit
softmax attention, not flash attention) to obtain the full A matrix.  On 8GB
VRAM this costs ~50–100 MiB extra peak per correction pass (the N×N matrices
at each active resolution).  This only applies to the SURE correction pass,
not the main sampling trajectory.  attn_blocks='middle' reduces cost to <5 MiB
at the expense of missing fine-detail uncertainty from input/output blocks.
"""

import logging
import math

import torch
import torch.nn.functional as F

import ldm_patched.modules.model_patcher as _model_patcher

_logger = logging.getLogger("sure_attention")


# ---------------------------------------------------------------------------
# Attention entropy capture hook
# ---------------------------------------------------------------------------

def _make_entropy_hook(store: list):
    """Return an attn1-replacement function that captures entropy and returns
    the correct attention output.  Replaces optimized_attention for one pass.

    The hook is compatible with the patches_replace["attn1"][block] interface:
        hook(q, k, v, extra_options, mask=None) → out
    """
    from ldm_patched.contrib.nodes_sag import attention_basic_with_sim

    def hook(q, k, v, extra_options, mask=None):
        heads     = extra_options["n_heads"]
        orig_dtype = q.dtype

        # torch.autocast (fp16/bf16 mode) downcasts fp32 inputs before eligible
        # ops like einsum/matmul even after an explicit `.float()` call.  At
        # seq=1600 (SDXL middle block) the fp16 QK dot-products overflow → inf
        # → one-hot softmax → entropy = 0 every step.  Step outside autocast.
        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            out_f32, sim = attention_basic_with_sim(
                q.float(), k.float(), v.float(),
                heads=heads, attn_precision=torch.float32, mask=mask,
            )

        # sim: (B*heads, N_q, N_k) in fp32 — compute per-query entropy
        # H[i] = −∑_j A[i,j]·log(A[i,j]+ε)  ∈ [0, log N_k]
        eps_ent = 1e-8
        ent = -(sim * (sim + eps_ent).log()).sum(-1).clamp(min=0)  # (B*heads, N_q)

        # One-shot deep diagnostic: print on the first capture of each denoising step.
        if not store:
            sm_max  = sim.max(dim=-1)[0]              # (B*H, N_q)
            row_sum = sim.sum(dim=-1)                 # should be ~1.0 everywhere
            term    = sim * (sim + eps_ent).log()     # p * log(p+ε)
            raw_neg = -term.sum(-1)                   # entropy before clamp
            print(
                f"[EDIAG] sim  dtype={sim.dtype}  shape={tuple(sim.shape)}"
                f"  global_min={float(sim.min()):.2e}"
                f"  global_max={float(sim.max()):.4f}"
                f"  global_mean={float(sim.mean()):.2e}"
            )
            print(
                f"[EDIAG] row_sum  min={float(row_sum.min()):.6f}"
                f"  max={float(row_sum.max()):.6f}"
                f"  mean={float(row_sum.mean()):.6f}"
            )
            print(
                f"[EDIAG] sim_row_max  mean={float(sm_max.mean()):.6f}"
                f"  max={float(sm_max.max()):.6f}"
            )
            print(
                f"[EDIAG] term=p*log(p+e)  min={float(term.min()):.6f}"
                f"  max={float(term.max()):.6f}"
                f"  mean={float(term.mean()):.8f}"
            )
            print(
                f"[EDIAG] raw_neg=-sum(term)  min={float(raw_neg.min()):.6f}"
                f"  max={float(raw_neg.max()):.6f}"
                f"  mean={float(raw_neg.mean()):.6f}"
            )
            print(
                f"[EDIAG] ent(after clamp)  mean={float(ent.mean()):.6f}"
                f"  max={float(ent.max()):.6f}"
            )
            # Cross-check with torch.special.entr = -x*log(x) for x>0, 0 for x=0
            try:
                entr_check = torch.special.entr(sim).sum(-1)
                print(f"[EDIAG] cross-check entr()  mean={float(entr_check.mean()):.6f}"
                      f"  max={float(entr_check.max()):.6f}")
            except Exception as exc:
                print(f"[EDIAG] cross-check entr() failed: {exc}")

        orig_shape = extra_options.get("original_shape")  # [B, C, H_lat, W_lat]
        store.append((ent.detach(), heads, orig_shape))

        # Return attention output in the original dtype so the model stays healthy
        return out_f32.to(orig_dtype)

    return hook


def _build_attn_capture_options(extra_args: dict, store: list,
                                 block_types: str = "all") -> dict:
    """Clone model_options from extra_args and inject entropy-capture hooks
    into all attn1 self-attention layers of the specified block types.

    Parameters
    ----------
    extra_args   : sampler extra_args dict (must contain 'model_options')
    store        : mutable list; each hook appends (ent, heads, orig_shape)
    block_types  : 'all' | 'input' | 'middle' | 'output' | 'mid+out'

    Returns a new extra_args dict whose model_options has the capture hooks.
    The original extra_args is not mutated.
    """
    base_opts = extra_args.get("model_options", {})

    # Deep-copy the patches_replace subtree to avoid polluting the original
    new_opts = dict(base_opts)
    if "patches_replace" in new_opts:
        new_opts["patches_replace"] = {
            k: dict(v) for k, v in new_opts["patches_replace"].items()
        }
    # set_model_options_patch_replace requires transformer_options to exist
    if "transformer_options" not in new_opts:
        new_opts["transformer_options"] = {}

    hook_fn = _make_entropy_hook(store)

    # Block name → number of IDs to try (IDs that don't exist are silently skipped)
    _all_blocks = {
        "input":  range(12),
        "middle": range(1),
        "output": range(12),
    }
    if block_types == "all":
        active = _all_blocks
    elif block_types in ("input", "middle", "output"):
        active = {block_types: _all_blocks[block_types]}
    elif block_types == "mid+out":
        active = {"middle": _all_blocks["middle"], "output": _all_blocks["output"]}
    else:
        _logger.warning("sure_attention: unknown attn_blocks=%r; using 'all'", block_types)
        active = _all_blocks

    for block_name, ids in active.items():
        for block_id in ids:
            new_opts = _model_patcher.set_model_options_patch_replace(
                new_opts, hook_fn, "attn1", block_name, block_id
            )

    return {**extra_args, "model_options": new_opts}


def build_capture_model_options(model_options: dict, store: list,
                                 block_types: str = "all") -> dict:
    """Public API for guidance nodes: returns a new model_options dict with
    attn1 entropy-capture hooks injected.  Does not mutate model_options.

    Parameters
    ----------
    model_options : model_options dict (from args["model_options"] in post_cfg)
    store         : mutable list; each hook appends (ent, heads, orig_shape)
    block_types   : 'all' | 'input' | 'middle' | 'output' | 'mid+out'
    """
    return _build_attn_capture_options({"model_options": model_options},
                                        store, block_types)["model_options"]


# ---------------------------------------------------------------------------
# Entropy map aggregation
# ---------------------------------------------------------------------------

def _aggregate_entropy_map(store: list, latent_shape: tuple,
                            device, dtype) -> torch.Tensor | None:
    """Convert captured (ent, heads, orig_shape) entries into a single
    spatial entropy map U of shape (B, 1, H_lat, W_lat) normalised to [0,1].

    Parameters
    ----------
    store        : list filled by entropy hooks during a forward pass
    latent_shape : (B, C, H_lat, W_lat) of the corrected x0_hat
    """
    if not store:
        return None

    B, _, H_lat, W_lat = latent_shape
    maps = []

    for ent, heads, orig_shape in store:
        # ent: (B_eff * heads, N_q)
        BH, N = ent.shape
        if heads <= 0 or BH < heads:
            continue

        # Batch size from CFG-doubled forward (cond+uncond batched together)
        b_eff = BH // heads
        # When CFG doubles the batch (b_eff = 2*B), take the last B items
        # (cond half, which is the second half by convention in forge)
        if b_eff > B:
            b_use = B
            ent = ent[heads * (b_eff - B):]  # last B images × heads
        else:
            b_use = b_eff

        # Average over attention heads → (b_use, N)
        ent_b = ent.reshape(b_use, heads, N).mean(dim=1)

        # Recover (h_l, w_l) from N using the known latent aspect ratio
        if orig_shape is not None:
            H_src, W_src = int(orig_shape[2]), int(orig_shape[3])
        else:
            H_src, W_src = H_lat, W_lat

        h_l, w_l = _tokens_to_spatial(N, H_src, W_src)
        if h_l * w_l != N:
            _logger.debug(
                "sure_attention: N=%d doesn't factor cleanly with H=%d W=%d; skipping layer",
                N, H_src, W_src,
            )
            continue

        # Normalise by theoretical max entropy log(N_k) so U ∈ [0,1] absolutely.
        # Per-image min-max would collapse when attention is spatially uniform
        # (e.g. high-noise steps where all tokens attend equally → entropy ≈ log(N)
        # everywhere, U_max − U_min ≈ 0 → divide by ε → all zeros).
        log_N = math.log(max(N, 2))
        ent_spatial = ent_b.float().reshape(b_use, 1, h_l, w_l) / log_N

        # Debug: log raw entropy range before normalization
        _logger.debug(
            "sure_attention: layer N=%d log_N=%.3f ent_raw min=%.4f max=%.4f mean=%.4f",
            N, log_N,
            float(ent_b.min()), float(ent_b.max()), float(ent_b.mean()),
        )

        # Bilinear upsample to full latent resolution
        ent_up = F.interpolate(ent_spatial, (H_lat, W_lat),
                               mode="bilinear", align_corners=False)

        # Pad to batch size B if needed (shouldn't happen normally)
        if b_use < B:
            pad = ent_up[-1:].expand(B - b_use, -1, -1, -1)
            ent_up = torch.cat([ent_up, pad], dim=0)

        maps.append(ent_up.to(device=device))

    if not maps:
        return None

    # Average across all captured layers → (B, 1, H_lat, W_lat), values ∈ [0,1]
    # (each layer was already normalised by its own log(N_k))
    U = torch.stack(maps, dim=0).mean(dim=0)

    # Clamp to [0,1] — log(N) normalisation should already give this, but floating
    # point can produce tiny negatives near zero.  (Lean §5: entropy_norm_le_one)
    U_norm = U.clamp(0.0, 1.0).to(dtype=dtype)

    if U_norm.isnan().any() or U_norm.isinf().any():
        _logger.warning("[sure_attn] NaN/Inf in entropy map after normalisation; skipping attention weighting")
        return None

    return U_norm


def _tokens_to_spatial(N: int, H_src: int, W_src: int) -> tuple[int, int]:
    """Find (h, w) such that h*w = N and h/w ≈ H_src/W_src.

    Tries power-of-2 downsampling factors first (d=1,2,4,8,16,32), then
    falls back to a ratio-preserving search.
    """
    for d in (1, 2, 4, 8, 16, 32):
        h, w = H_src // d, W_src // d
        if h > 0 and w > 0 and h * w == N:
            return h, w

    # Fallback: find integer factor closest to aspect ratio
    ratio = H_src / max(W_src, 1)
    w = max(1, round(math.sqrt(N / ratio)))
    for delta in range(w):
        for w_try in (w + delta, w - delta):
            if w_try > 0 and N % w_try == 0:
                return N // w_try, w_try
    return 1, N  # degenerate: treat as 1×N strip


# ---------------------------------------------------------------------------
# Core SURE-AG correction function
# ---------------------------------------------------------------------------

def _sure_correct_x0_attention(
    model, x0_hat, sigma_hat_0, s_in, extra_args,
    alpha=0.05, n_mc=1, eps_mc=1e-3, use_jac=True,
    sigma_t=None,
    adam_state=None, adam_mode="none",
    adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
    adam_wd=0.01,
    grad_mode="approx",
    approx_coeff=2.0,
    attn_weight=1.0,
    attn_blocks="all",
):
    """SURE-AG: attention-entropy-weighted SURE gradient correction.

    Extends _sure_correct_x0 by computing a spatial uncertainty map U from
    self-attention entropy and weighting the SURE correction by (1 + w·U).

    Where the model is confident (low entropy, U≈0), the correction is the
    same as base SURE.  Where uncertain (high entropy, U≈1), the correction
    is amplified by up to (1+w).

    Parameters (new vs _sure_correct_x0)
    -------------------------------------
    attn_weight : float ≥ 0
        Strength of attention weighting.  0 = pure base SURE.  1 = default.
        Larger values amplify the correction in uncertain regions more.
        Lean §3 constrains alpha < 1/(2(1+attn_weight)) for descent.
    attn_blocks : str
        Which UNet blocks to capture entropy from:
          'all'     — all input/middle/output blocks (default, most informative)
          'middle'  — middle block only (cheapest, captures global structure)
          'mid+out' — middle + output blocks
          'input'   — input blocks only

    Returns
    -------
    x0_corrected : Tensor   corrected denoised estimate
    info         : dict     {'jac_ratio': float|None, 'entropy_rms': float}
    """
    # ── Import helpers from sampling.py (same module at runtime) ─────────────
    from ldm_patched.k_diffusion.sampling import (
        _sure_logger, _sure_effective_alpha, release_cache,
    )

    eps = float(x0_hat.abs().max().clamp(min=eps_mc * 1000.0)) / 1000.0
    device = x0_hat.device
    sigma_denoiser = torch.tensor(sigma_hat_0, device=device, dtype=x0_hat.dtype)
    sigma_p = sigma_denoiser
    sigma2  = float(sigma_denoiser ** 2)
    n       = x0_hat.numel()
    use_jac = use_jac and (sigma2 > eps_mc)

    # ── Build model_options with attention capture hooks ─────────────────────
    attn_store: list = []
    capture_extra = _build_attn_capture_options(extra_args, attn_store, attn_blocks)

    # ── Pass 1: forward with attention capture ────────────────────────────────
    # For 'approx': no-grad pass, attention captured for entropy map.
    # For 'vjp'/'full': grad-enabled pass; entropy is detached to avoid
    #   polluting the gradient with dU/dx (entropy gradient w.r.t. input).
    if grad_mode != "approx":
        release_cache(device)

    x_in = x0_hat.detach().requires_grad_(grad_mode != "approx")

    if grad_mode == "approx":
        with torch.no_grad():
            x_hat = model(x_in, sigma_denoiser * s_in, **capture_extra).detach()
        residual = x_in - x_hat
        grad = approx_coeff * residual
    else:
        def _model_ckpt(x):
            return model(x, sigma_denoiser * s_in, **capture_extra)

        with torch.enable_grad():
            x_hat    = torch.utils.checkpoint.checkpoint(
                           _model_ckpt, x_in, use_reentrant=False)
            residual = x_in - x_hat
            residual_sq = (residual ** 2).sum()
            grad = torch.autograd.grad(residual_sq, x_in)[0]
            residual_sq_val = float(residual_sq)
        x_hat    = x_hat.detach()
        residual = residual.detach()
        release_cache(device)

    # ── Aggregate attention entropy map ──────────────────────────────────────
    U = _aggregate_entropy_map(attn_store, x0_hat.shape, device, x0_hat.dtype)
    entropy_rms = 0.0
    if U is not None:
        entropy_rms = float(U.pow(2).mean().sqrt())
        _logger.debug(
            "[sure_attn] entropy_rms=%.4f  attn_blocks=%s  layers_captured=%d",
            entropy_rms, attn_blocks, len(attn_store),
        )
        # Apply Lean §2 bound: weight W = 1 + attn_weight * U ∈ [1, 1+attn_weight]
        if attn_weight > 0.0:
            W = 1.0 + attn_weight * U  # (B, 1, H, W) broadcast over latent channels
            grad = grad * W
    else:
        _logger.debug("[sure_attn] no attention maps captured; using base SURE grad")

    # ── Jacobian trace (optional, same as _sure_correct_x0) ──────────────────
    jac_trace: float | None = None
    if use_jac and grad_mode != "approx":
        bs = [torch.randn_like(x0_hat) for _ in range(n_mc)]
        jac_trace = 0.0
        for b in bs:
            x_in_pert = x_in.detach() + eps * b
            with torch.no_grad():
                x_pert_hat = model(x_in_pert, sigma_p * s_in, **extra_args).detach()
            jac_trace += float((b * (x_pert_hat - x_hat)).sum()) / eps
        jac_trace /= n_mc

        # Hutchinson floor (same as _sure_correct_x0 Gap #6)
        _jac_floor = -3.0 * (2.0 * n / max(n_mc, 1)) ** 0.5
        if jac_trace < _jac_floor:
            _logger.warning(
                "[sure_attn] jac_trace=%.2f < floor %.2f; using proxy grad",
                jac_trace, _jac_floor,
            )
            jac_trace = None

    # SURE value (for logging)
    if grad_mode == "approx":
        sure_val = float(-n * sigma2 + (residual ** 2).sum())
    else:
        sure_val = -n * sigma2 + residual_sq_val
    if jac_trace is not None:
        sure_val += 2.0 * sigma2 * jac_trace

    # ── Effective alpha (Lean §3: alpha tightened by 1/(1+attn_weight)) ──────
    # The Lean proof (sure_ag_effective_range) shows that for descent we need
    # alpha * (1 + attn_weight) < 1/2, i.e. effective_alpha * W < 1/2.
    # _sure_effective_alpha caps at the sigma-normalised value which already
    # satisfies this constraint for typical sigma schedules.
    effective_alpha = _sure_effective_alpha(
        alpha, sigma_t,
        adam_active=(adam_state is not None and adam_mode in ("adam", "adamw")),
    )

    # ── Optimiser step (same Adam/AdamW logic as _sure_correct_x0) ───────────
    if adam_state is not None and adam_mode in ("adam", "adamw"):
        if adam_state["optimizer"] is None:
            param = x0_hat.detach().clone().requires_grad_(True)
            opt_cls = torch.optim.AdamW if adam_mode == "adamw" else torch.optim.Adam
            opt_kwargs = dict(
                lr=effective_alpha,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
            )
            if adam_mode == "adamw":
                opt_kwargs["weight_decay"] = adam_wd
            else:
                opt_kwargs.update(weight_decay=0.0, amsgrad=True)
            opt = opt_cls([param], **opt_kwargs)
            adam_state["optimizer"] = opt
            adam_state["param"] = param

        param = adam_state["param"]
        opt   = adam_state["optimizer"]
        for pg in opt.param_groups:
            pg["lr"] = effective_alpha
        param.data.copy_(x0_hat.detach())
        param.grad = grad.detach()
        opt.step()
        x0_corrected = param.data.detach().clone()
    else:
        x0_corrected = (x0_hat - effective_alpha * grad).detach()

    _logger.info(
        "[sure_attn] sigma_hat_0=%.5f  lr=%.5f  sure=%.4f  "
        "entropy_rms=%.4f  layers=%d  attn_weight=%.2f  attn_blocks=%s",
        sigma_hat_0, effective_alpha, sure_val,
        entropy_rms, len(attn_store), attn_weight, attn_blocks,
    )

    return x0_corrected, {"jac_ratio": None, "entropy_rms": entropy_rms}
