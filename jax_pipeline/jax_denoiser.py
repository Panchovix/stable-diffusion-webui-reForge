"""
jax_pipeline/jax_denoiser.py — JAX-native CFG denoiser for SDXL.

Mirrors mlx_pipeline/mlx_denoiser.py exactly — same conditioning
extraction, same batched-CFG data flow, same float32-for-CFG-math
rationale — but wraps jax_pipeline's pure-functional UNet
(``forward_fn(params, x, t, encoder_hidden_states, added_cond_kwargs)``,
see jax_pipeline.unet.unet_forward) instead of a stateful mlx.nn.Module,
so construction takes ``(forward_fn, params)`` rather than a single
callable module.

Pre-converts conditioning tensors (cross-attention, pooled CLIP-G) to JAX
*once* at construction time. Each denoising call stays entirely in JAX:

    Conditioning pre-built once
         v
    __call__(x, sigma)           <- all jnp.ndarray
         +-- EPS preconditioning  (x / sqrt(sigma^2+1))
         +-- sigma -> timestep    (nearest-neighbour in log-sigma table)
         +-- batch cond || uncond (single UNet forward, batch=2B)
         +-- CFG                  (uncond + scale*(cond-uncond), float32)
         +-- EPS postconditioning (x - out*sigma)
         v
    denoised: jnp.ndarray         <- no torch round-trip

Conditioning extraction
------------------------
Reads ``model_conds`` from the ldm cond-list format that
KDiffusionSampler stores in ``sampler_extra_args`` — this extraction
logic is framework-agnostic (pure torch-side dict/tensor plumbing); only
the final array conversion differs from mlx_pipeline's:

    cond_list[0]['model_conds']['crossattn'].cond     -> [1, S, 2048] torch
    cond_list[0]['model_conds']['pooled_output'].cond -> [1, 1280+] torch

cond/uncond sequence-length mismatch
--------------------------------------
Positive and negative prompts are chunked into 77-token blocks
independently, so they routinely encode to different sequence lengths
(e.g. a longer positive prompt needing 2 chunks against a 1-chunk
negative prompt) — concatenating them along the batch axis for a single
UNet call requires the other axes to match first. This is exactly what
``shared.opts.pad_cond_uncond`` / ``pad_cond_uncond_v0`` (Settings >
Stable Diffusion > "Pad prompt/negative prompt") already solve for the
standard PyTorch path (``modules/sd_samplers_cfg_denoiser.py``) —
``_reconcile_cond_uncond_seq_len`` mirrors both of those exactly, so
enabling that setting gets identical batching behavior here. If neither
is enabled, this warns once and ``JAXCFGDenoiser`` falls back to two
separate (still fully JAX-accelerated) forward calls per step instead of
one batched call — see ``JAXCFGDenoiser.__call__``.

batch=2B vs two batch=B calls on VRAM-constrained cards
----------------------------------------------------------
Independent of the above: when the shared ``PhaseManager`` reports
``.enabled`` (a <20GB-VRAM card — the same signal that gates UNet/CLIP/VAE
phase-cycling), ``JAXCFGDenoiser`` always runs cond/uncond as two
sequential batch=B forward calls, never one batch=2B call, even when their
sequence lengths already match. A single 2B-batch UNet forward roughly
doubles peak activation/workspace memory versus two B-batch calls (params
are shared either way — this is about activations, not weights), which is
enough to OOM on small cards even though the circuit breaker
(``host_offload.handle_jax_oom``) recovers cleanly from it. See
``JAXCFGDenoiser.__init__``'s ``vram_constrained`` check.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)

_warned_unpadded_mismatch = False


# -- conditioning extraction (framework-agnostic torch-side logic) --------------

def _get_raw_cond_dict(cond_obj: Any) -> Optional[Dict]:
    """Extract the raw conditioning dict from whatever format
    KDiffusionSampler stores in ``sampler_extra_args``.

    Two formats are encountered at the sampler level:

    * ``MulticondLearnedConditioning`` (cond):
        ``.batch[0][0].schedules[0].cond``
        -> dict ``{'crossattn': Tensor[1,S,2048], 'pooled_output': Tensor[1,1280]}``

    * ``list[list[ScheduledPromptConditioning]]`` (uncond):
        ``[0][0].cond``
        -> same dict format

    Both return a plain-tensor dict — NOT the ``model_conds`` / ``CONDRegular``
    wrapping used deeper inside ``sampling_function``. Identical to
    ``mlx_pipeline.mlx_denoiser._get_raw_cond_dict`` — this step never
    touches MLX or JAX, only torch/webui plumbing.
    """
    try:
        from modules.prompt_parser import MulticondLearnedConditioning
        if isinstance(cond_obj, MulticondLearnedConditioning):
            raw = cond_obj.batch[0][0].schedules[0].cond
        elif isinstance(cond_obj, list) and cond_obj and cond_obj[0]:
            raw = cond_obj[0][0].cond
        else:
            return None
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _extract_cond_jax(
    cond_obj: Any,
    B: int,
    latent_hw: Tuple[int, int],
) -> Tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
    """Extract encoder_hidden_states [B, S, 2048], text_embeds [B, 1280],
    and time_ids [B, 6] as JAX bfloat16 / float32 arrays from the raw
    conditioning object stored in ``sampler_extra_args``.

    Parameters
    ----------
    cond_obj   : ``MulticondLearnedConditioning`` or ``list[list[SPC]]``
    B          : batch size (number of samples)
    latent_hw  : (H, W) of the *latent* — used to build time_ids
    """
    import jax.numpy as jnp

    enc_hs_t = text_embeds_t = None

    raw = _get_raw_cond_dict(cond_obj)
    if raw is not None:
        ca = raw.get("crossattn", None)      # Tensor [1, S, 2048]
        po = raw.get("pooled_output", None)  # Tensor [1, >=1280]
        if ca is not None:
            enc_hs_t = ca.detach().cpu().float()
        if po is not None:
            text_embeds_t = po.detach().cpu().float()[:, :1280]

    enc_hs = (
        jnp.asarray(enc_hs_t.expand(B, -1, -1).numpy(), dtype=jnp.bfloat16)
        if enc_hs_t is not None
        else jnp.zeros((B, 1, 2048), dtype=jnp.bfloat16)
    )
    text_embeds = (
        jnp.asarray(text_embeds_t.expand(B, -1).numpy(), dtype=jnp.bfloat16)
        if text_embeds_t is not None
        else jnp.zeros((B, 1280), dtype=jnp.bfloat16)
    )
    h_px = latent_hw[0] * 8
    w_px = latent_hw[1] * 8
    time_ids = jnp.asarray(
        [[float(h_px), float(w_px), 0.0, 0.0, float(h_px), float(w_px)]] * B,
        dtype=jnp.float32,
    )

    return enc_hs, text_embeds, time_ids


# -- cond/uncond sequence-length reconciliation ---------------------------------
# See the module docstring's "cond/uncond sequence-length mismatch" section.

def _empty_prompt_to_jax(empty_torch: "torch.Tensor") -> "jnp.ndarray":
    import jax.numpy as jnp
    arr = empty_torch.detach().cpu().float().numpy()
    return jnp.asarray(arr, dtype=jnp.bfloat16)


def _get_empty_prompt_embedding(sd_model) -> Optional["torch.Tensor"]:
    """Get (and cache, on ``sd_model``) the empty-prompt ('') CLIP
    crossattn embedding.

    Reuses the exact same sd_model-level cache
    (``cond_stage_model_empty_prompt``) that
    ``modules.sd_samplers_cfg_denoiser.CFGDenoiser.
    _compute_and_cache_empty_prompt`` uses, so it's computed at most once
    per model load regardless of which code path (the standard torch
    CFGDenoiser, or this one) needs it first.
    """
    empty = getattr(sd_model, "cond_stage_model_empty_prompt", None)
    if empty is not None:
        return empty
    try:
        if hasattr(sd_model, "get_learned_conditioning"):
            with torch.no_grad():
                d = sd_model.get_learned_conditioning([""])
        elif hasattr(sd_model, "cond_stage_model"):
            with torch.no_grad():
                d = sd_model.cond_stage_model([""])
        else:
            return None
        if isinstance(d, dict):
            d = d["crossattn"]
        sd_model.cond_stage_model_empty_prompt = d
        return d
    except Exception as e:
        log.warning("[JAX denoiser] Could not compute empty-prompt embedding: %s", e)
        return None


def _pad_seq_v0(enc_c: "jnp.ndarray", enc_u: "jnp.ndarray") -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """Mirrors ``CFGDenoiser.pad_cond_uncond_v0`` exactly: repeat uncond's
    last token vector to extend it, or truncate it, so its sequence length
    matches cond's. Only ever adjusts uncond — matches the reference
    exactly, including its "truncates negative prompt if too long" caveat.
    """
    import jax.numpy as jnp

    if enc_u.shape[1] < enc_c.shape[1]:
        last = enc_u[:, -1:, :]
        pad = jnp.tile(last, (1, enc_c.shape[1] - enc_u.shape[1], 1))
        enc_u = jnp.concatenate([enc_u, pad], axis=1)
    elif enc_u.shape[1] > enc_c.shape[1]:
        enc_u = enc_u[:, :enc_c.shape[1], :]
    return enc_c, enc_u


def _pad_seq_with_empty(
    enc_c: "jnp.ndarray", enc_u: "jnp.ndarray", empty_jax: "jnp.ndarray",
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """Mirrors ``CFGDenoiser.pad_cond_uncond`` / the module-level
    ``pad_cond`` exactly: extend the shorter of cond/uncond with whole
    tiled copies of the empty-prompt embedding until their lengths match.

    Falls back to ``_pad_seq_v0`` if the length difference isn't a clean
    multiple of the empty embedding's length — shouldn't happen in
    practice (webui's chunking always produces multiples of one chunk
    length) but stay safe rather than raise.
    """
    import jax.numpy as jnp

    diff = enc_c.shape[1] - enc_u.shape[1]
    empty_len = empty_jax.shape[1]
    if empty_len == 0 or diff % empty_len != 0:
        return _pad_seq_v0(enc_c, enc_u)

    num_repeats = diff // empty_len
    if num_repeats > 0:
        tiled = jnp.tile(empty_jax, (enc_u.shape[0], num_repeats, 1))
        enc_u = jnp.concatenate([enc_u, tiled], axis=1)
    elif num_repeats < 0:
        tiled = jnp.tile(empty_jax, (enc_c.shape[0], -num_repeats, 1))
        enc_c = jnp.concatenate([enc_c, tiled], axis=1)
    return enc_c, enc_u


def _reconcile_cond_uncond_seq_len(
    enc_c: "jnp.ndarray", enc_u: "jnp.ndarray",
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """Make ``enc_c``/``enc_u``'s sequence lengths match, so they CAN be
    batched into a single UNet forward call, by mirroring webui's own
    ``shared.opts.pad_cond_uncond`` / ``pad_cond_uncond_v0`` settings
    (Settings > Stable Diffusion > "Pad prompt/negative prompt") —
    including their exact precedence (v0 overrides the other when both are
    set) — so enabling that setting gets identical batching behavior here
    as in the standard PyTorch CFGDenoiser path.

    If already equal, or if neither setting is enabled, returns the arrays
    unchanged (a no-mismatch call is nearly free — one shape comparison).
    When neither setting is enabled AND the lengths differ, warns once per
    process — this is a real, common situation (differently-sized positive
    /negative prompts), not an edge case — and leaves reconciliation to the
    caller, which falls back to two separate forward calls per step
    instead of one batched call (see ``JAXCFGDenoiser.__call__``).
    """
    global _warned_unpadded_mismatch

    if enc_c.shape[1] == enc_u.shape[1]:
        return enc_c, enc_u

    import modules.shared as shared

    v0 = bool(getattr(shared.opts, "pad_cond_uncond_v0", False))
    v1 = bool(getattr(shared.opts, "pad_cond_uncond", False))

    if v0:
        return _pad_seq_v0(enc_c, enc_u)

    if v1:
        empty = _get_empty_prompt_embedding(shared.sd_model)
        if empty is not None:
            return _pad_seq_with_empty(enc_c, enc_u, _empty_prompt_to_jax(empty))
        return _pad_seq_v0(enc_c, enc_u)  # matches pad_cond_uncond's own fallback

    if not _warned_unpadded_mismatch:
        log.warning(
            "[JAX Pipeline] cond and uncond prompts encode to different "
            "lengths (%d vs %d tokens) and neither 'Pad prompt/negative "
            "prompt' setting is enabled (Settings > Stable Diffusion) — "
            "falling back to two separate JAX UNet forward calls per step "
            "instead of one batched call for this generation (still fully "
            "JAX-accelerated, just not batched). Enable 'Pad prompt/"
            "negative prompt' to restore single-call batching.",
            enc_c.shape[1], enc_u.shape[1],
        )
        _warned_unpadded_mismatch = True

    return enc_c, enc_u


# -- JAX CFG denoiser ----------------------------------------------------------

class JAXCFGDenoiser:
    """JAX-native CFG denoiser for SDXL.

    Conditioning (encoder hidden states + pooled CLIP-G) is pre-converted
    to JAX at construction time. The denoising call runs entirely in JAX
    — no torch tensors are created inside ``__call__``.

    Parameters
    ----------
    forward_fn     : callable matching ``unet_forward``'s call signature,
                      ``forward_fn(params, x, t, encoder_hidden_states,
                      added_cond_kwargs) -> jnp.ndarray`` — either
                      ``jax.jit``-compiled directly (see
                      ``jax_pipeline.host_offload.make_forward``) or the
                      block-streaming wrapper (``make_streaming_forward``)
                      around ``jax_pipeline.unet.unet_forward_streaming``.
    params         : the UNet's current params — either a plain pytree
                      (device-resident by the time this is constructed on
                      large-VRAM cards) or, when block-streaming is active
                      (VRAM-constrained cards — see
                      ``JAXSDXLPipeline.__init__``), a
                      ``jax_pipeline.block_cache.BlockParamCache`` that
                      stages each block onto device on demand inside
                      ``forward_fn`` itself. Either way this is just
                      passed straight through to ``forward_fn``'s first
                      argument — ``JAXCFGDenoiser`` never inspects it.
    model_sampling : ldm model_sampling object for sigma <-> timestep
    extra_args     : k-diffusion sampler extra_args dict
                      (keys: ``cond``, ``uncond``, ``cond_scale``, ...)
    latent_shape   : ``(B, 4, H, W)`` of the initial noisy latent
    phase_manager  : optional ``host_offload.PhaseManager`` — when its
                      ``.enabled`` is True (VRAM-constrained card, <20GB),
                      cond/uncond are ALWAYS run as two sequential batch=B
                      forward calls instead of one batch=2B call, even when
                      their sequence lengths already match. See the
                      "batch=2B vs two batch=B calls" note below.
    """

    def __init__(
        self,
        forward_fn,
        params: Any,  # flat pytree OR a jax_pipeline.block_cache.BlockParamCache — see docstring
        model_sampling: Any,
        extra_args: Dict,
        latent_shape: Tuple[int, ...],
        phase_manager: Any = None,
    ) -> None:
        import jax.numpy as jnp

        self._forward = forward_fn
        self._params = params
        self._ms = model_sampling
        self.cond_scale = float(extra_args.get("cond_scale", 7.5))
        B = latent_shape[0]
        latent_hw = (latent_shape[2], latent_shape[3])  # (H, W)

        cond_list = extra_args.get("cond", [])
        uncond_list = extra_args.get("uncond", [])

        enc_c, te_c, ti = _extract_cond_jax(cond_list, B, latent_hw)
        enc_u, te_u, _ = _extract_cond_jax(uncond_list, B, latent_hw)

        # cond/uncond routinely encode to different sequence lengths (see
        # module docstring) — try to reconcile them (respecting the same
        # webui settings the standard PyTorch path uses) so they CAN be
        # batched into one [2B, ...] UNet call; if that's not possible
        # (setting disabled), keep them separate and run two forward calls
        # per step instead — see __call__.
        enc_c, enc_u = _reconcile_cond_uncond_seq_len(enc_c, enc_u)

        # batch=2B vs two batch=B calls: params are identical either way
        # (shared, not duplicated), but a single 2B-batch UNet forward
        # roughly doubles peak *activation*/workspace memory versus two
        # sequential B-batch calls — SDXL's cross-attention maps and conv
        # intermediates scale with batch size. On a VRAM-constrained card
        # (phase_manager.enabled — the same <20GB signal used to decide
        # whether to phase-cycle UNet/CLIP/VAE at all) that extra headroom
        # is worth more than the one-time extra jax.jit compile and the
        # small per-step dispatch overhead of a second forward call, so
        # always take the unbatched path there regardless of whether
        # lengths already matched.
        vram_constrained = phase_manager is not None and getattr(phase_manager, "enabled", False)
        self._batched = (enc_c.shape[1] == enc_u.shape[1]) and not vram_constrained

        if self._batched:
            # Build batched conditioning: cond first, uncond second -> [2B, ...]
            self._enc_b = jnp.concatenate([enc_c, enc_u], axis=0)  # [2B, S, 2048]
            self._te_b = jnp.concatenate([te_c, te_u], axis=0)     # [2B, 1280]
            self._ti_b = jnp.concatenate([ti, ti], axis=0)         # [2B, 6]
        else:
            # Two separate B-sized calls per step instead of one 2B-sized
            # call. Safe regardless of the (now possibly still mismatched)
            # sequence lengths, because the UNet's output shape depends
            # only on the latent x, never on encoder_hidden_states'
            # sequence length.
            self._enc_c, self._enc_u = enc_c, enc_u
            self._te_c, self._te_u = te_c, te_u
            self._ti_single = ti

        # Cache log-sigma table for fast timestep lookup (avoids torch round-trip)
        log_sigmas_np = model_sampling.log_sigmas.detach().cpu().float().numpy()
        self._log_sigmas = jnp.asarray(log_sigmas_np)  # [N]

        log.debug(
            "[JAX denoiser] Conditioning pre-built: enc_c=%s enc_u=%s batched=%s cond_scale=%.1f",
            enc_c.shape, enc_u.shape, self._batched, self.cond_scale,
        )

    # -- sigma ops (all in JAX) ------------------------------------------------

    def _timestep(self, sigma: "jnp.ndarray") -> "jnp.ndarray":
        """sigma [B] float32 -> nearest discrete timestep [B] float32."""
        import jax.numpy as jnp
        log_s = jnp.log(sigma)                                      # [B]
        diff = jnp.abs(log_s[:, None] - self._log_sigmas[None, :])  # [B, N]
        return jnp.argmin(diff, axis=1).astype(jnp.float32)          # [B]

    @staticmethod
    def _calc_input(sigma: "jnp.ndarray", x: "jnp.ndarray") -> "jnp.ndarray":
        """EPS preconditioning: xc = x / sqrt(sigma^2+1)."""
        import jax.numpy as jnp
        s = sigma.reshape((-1, 1, 1, 1)).astype(jnp.float32)
        return (x.astype(jnp.float32) / jnp.sqrt(s * s + 1.0)).astype(jnp.bfloat16)

    @staticmethod
    def _calc_denoised(sigma: "jnp.ndarray", model_out: "jnp.ndarray", x: "jnp.ndarray") -> "jnp.ndarray":
        """EPS postconditioning: denoised = x - model_out * sigma.

        Returns float32 to match the reference (model_base.py:
        ``model_output = diffusion_model(...).float()``). The sampler
        loop and ODE/SDE arithmetic then operate in full float32 precision.
        """
        import jax.numpy as jnp
        s = sigma.reshape((-1, 1, 1, 1)).astype(jnp.float32)
        return x.astype(jnp.float32) - model_out.astype(jnp.float32) * s  # float32

    # -- forward ---------------------------------------------------------------

    def __call__(
        self,
        x: "jnp.ndarray",
        sigma: "jnp.ndarray",
        **_kwargs: Any,  # absorb extra_args kwargs forwarded by sampler
    ) -> "jnp.ndarray":
        """
        x     : [B, 4, H, W]  bfloat16 (or float32 — cast happens in _calc_input)
        sigma : [B]           float32
        -> denoised [B, 4, H, W] float32
        """
        import jax.numpy as jnp

        B = x.shape[0]

        # Preconditioning (stays in JAX)
        xc = self._calc_input(sigma, x)  # [B, 4, H, W]
        ts = self._timestep(sigma)       # [B]

        if self._batched:
            # Single batched UNet forward (batch=2B) — no redundant
            # torch<->JAX conversions, and only one jax.jit executable.
            xc_b = jnp.concatenate([xc, xc], axis=0)  # [2B, 4, H, W]
            ts_b = jnp.concatenate([ts, ts], axis=0)  # [2B]
            added = {
                "text_embeds": self._te_b,  # [2B, 1280]
                "time_ids": self._ti_b,     # [2B, 6]
            }
            out = self._forward(self._params, xc_b, ts_b, self._enc_b, added)  # [2B, 4, H, W]
            out_c = out[:B].astype(jnp.float32)  # cond   [B, 4, H, W] f32
            out_u = out[B:].astype(jnp.float32)  # uncond [B, 4, H, W] f32
        else:
            # cond/uncond couldn't be reconciled to the same sequence
            # length (see _reconcile_cond_uncond_seq_len) — run them as
            # two separate batch=B calls instead. jax.jit caches each
            # shape's compiled executable independently, so this costs one
            # extra one-time compile at the start of the loop, not a
            # per-step penalty; the whole-loop residency optimization
            # (no torch round-trip) is still fully in effect either way.
            added_c = {"text_embeds": self._te_c, "time_ids": self._ti_single}
            added_u = {"text_embeds": self._te_u, "time_ids": self._ti_single}
            out_c = self._forward(self._params, xc, ts, self._enc_c, added_c).astype(jnp.float32)
            out_u = self._forward(self._params, xc, ts, self._enc_u, added_u).astype(jnp.float32)

        # CFG: uncond + scale*(cond - uncond)
        #
        # MUST be computed in float32 — see mlx_denoiser.py's identical
        # note: bfloat16's 7 mantissa bits quantise cond-uncond differences
        # catastrophically, producing isolated bright-spot artifacts that
        # are most visible at low CFG scales.
        cfg_out = out_u + self.cond_scale * (out_c - out_u)  # f32

        # Postconditioning (stays in JAX, _calc_denoised handles f32 input)
        return self._calc_denoised(sigma, cfg_out, x)
