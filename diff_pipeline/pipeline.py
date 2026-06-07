"""
diff_pipeline/pipeline.py — Diffusers-based SDXL pipeline.

Replaces the ldm-patched ``BaseModel.apply_model()`` forward path so that a
HuggingFace ``UNet2DConditionModel`` is used instead.

Activated when BOTH conditions are met at load time:
  - ``sd_model.is_sdxl`` is True
  - ``--forge-diffusers-pipeline`` CLI flag is set

Architecture overview
---------------------
::

    DiffPipeline.apply_model(x, t, c_crossattn, y, control, transformer_options)
        │
        ├── sigma preconditioning  (model_sampling.calculate_input / timestep)
        ├── conditioning bridge    (c_crossattn → encoder_hidden_states,
        │                           adm_text_embeds / adm_time_ids → added_cond_kwargs)
        ├── ControlNet mapping     (control["input"] / ["middle"] →
        │                           down_block_additional_residuals / mid_block_additional_residual)
        ├── HF UNet forward        (cross_attention_kwargs carries transformer_options
        │                           for ForgeAttnProcessor to dispatch patch hooks)
        └── sigma postconditioning (model_sampling.calculate_denoised)

ForgeAttnProcessor
------------------
Installed on every ``attn2`` (cross-attention) sub-module of the HF UNet at
construction time.  Receives ``transformer_options`` via
``cross_attention_kwargs`` and dispatches:

  * ``patches_replace["attn2"][(block_name, block_idx, transformer_idx)]``
    — full Q/K/V replacement (IP-Adapter, etc.)
  * ``patches["attn2_patch"]``
    — Q/K/V input modifier before standard attention (ControlLLite, etc.)

Known limitations
-----------------
* LoRA patches (Phase 5): ``_sync_lora()`` reads ``unet_patcher.patches`` on
  every ``apply_model()`` call and installs PEFT adapters on the HF UNet via
  ``load_lora_adapter()``.  DoRA (``dora_scale``) and tucker mid weights are
  not yet handled.
* Block modifier hooks (``add_block_modifier`` / ``add_block_inner_modifier``)
  are Phase 4 work; not yet wired.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from diff_pipeline._cache import lru_cached
from diff_pipeline.vram_allocator import VRAMAllocator

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from modules_forge.unet_patcher import UnetPatcher

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded optimisation helpers (mirrors modules_forge/stream.py and
# ldm_patched/modules/model_management.py).  Imported once at first use so
# that the module can be imported without a running webui environment.
# ---------------------------------------------------------------------------
@lru_cached
def _get_stream_module():
    """Return modules_forge.stream, or None if unavailable."""
    try:
        import modules_forge.stream as _s
        return _s
    except Exception:
        return None

@lru_cached
def _get_pin_shared_memory() -> bool:
    """Return True when --pin-shared-memory was requested."""
    try:
        from ldm_patched.modules.model_management import PIN_SHARED_MEMORY
        return bool(PIN_SHARED_MEMORY)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hardcoded SDXL UNet2DConditionModel config
# Source: stabilityai/stable-diffusion-xl-base-1.0 — unet/config.json
# ---------------------------------------------------------------------------

_SDXL_HF_UNET_CONFIG: dict = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 2048,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ],
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ],
    "use_linear_projection": True,
}

# Full ldm unet_config used for unet_to_diffusers() key mapping.
# Must have num_res_blocks / channel_mult / transformer_depth* fields.
# These match the values detect_unet_config() derives from SDXL weights.
_SDXL_LDM_UNET_CONFIG: dict = {
    "num_res_blocks": [2, 2, 2],
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "channel_mult": [1, 2, 4],
    "transformer_depth_middle": 10,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
}

# ---------------------------------------------------------------------------
# LDM → HF UNet config derivation
# ---------------------------------------------------------------------------

def _derive_hf_config_from_ldm(merged_unet_cfg: dict) -> tuple[dict, list[str]]:
    """Derive HF UNet2DConditionModel config overrides from a merged ldm unet_config.

    Reads keys that are actually present in the merged config (model-embedded
    values take precedence over our hardcoded defaults) and translates them to
    the HF naming convention.  Returns:
        overrides  — dict to merge on top of _SDXL_HF_UNET_CONFIG
        report     — human-readable list of detected → applied mappings

    Only keys that differ from _SDXL_HF_UNET_CONFIG are included in overrides.
    """
    overrides: dict = {}
    report: list[str] = []

    base = _SDXL_HF_UNET_CONFIG

    # in_channels / out_channels
    for ldm_key, hf_key in (("in_channels", "in_channels"), ("out_channels", "out_channels")):
        v = merged_unet_cfg.get(ldm_key)
        if v is not None and v != base.get(hf_key):
            overrides[hf_key] = v
            report.append(f"  {ldm_key}={v}  →  hf:{hf_key}={v}  (was {base.get(hf_key)})")

    # context_dim → cross_attention_dim
    v = merged_unet_cfg.get("context_dim")
    if v is not None and v != base.get("cross_attention_dim"):
        overrides["cross_attention_dim"] = v
        report.append(f"  context_dim={v}  →  hf:cross_attention_dim={v}  (was {base.get('cross_attention_dim')})")

    # adm_in_channels → projection_class_embeddings_input_dim
    v = merged_unet_cfg.get("adm_in_channels")
    if v is not None and v != base.get("projection_class_embeddings_input_dim"):
        overrides["projection_class_embeddings_input_dim"] = v
        report.append(f"  adm_in_channels={v}  →  hf:projection_class_embeddings_input_dim={v}  (was {base.get('projection_class_embeddings_input_dim')})")

    # model_channels → block_out_channels[0] (leading channel width)
    mc = merged_unet_cfg.get("model_channels")
    cm = merged_unet_cfg.get("channel_mult")
    if mc is not None and cm is not None:
        derived_boc = [mc * m for m in cm]
        if derived_boc != list(base.get("block_out_channels", [])):
            overrides["block_out_channels"] = derived_boc
            report.append(f"  model_channels={mc} × channel_mult={cm}  →  hf:block_out_channels={derived_boc}  (was {base.get('block_out_channels')})")

    # transformer_depth → transformer_layers_per_block
    # ldm layout: [d_down0, d_down0_skip, d_down1, d_down1_skip, d_down2, d_down2_skip]
    # HF layout: one entry per down-block (use first non-zero depth per pair)
    td = merged_unet_cfg.get("transformer_depth")
    if td is not None and len(td) >= 6:
        hf_td = [td[0] or td[1], td[2] or td[3], td[4] or td[5]]
        # Include middle block depth
        tdm = merged_unet_cfg.get("transformer_depth_middle")
        if tdm is not None:
            hf_td = hf_td  # HF transformer_layers_per_block is per down-block only
        if hf_td != list(base.get("transformer_layers_per_block", [])):
            overrides["transformer_layers_per_block"] = hf_td
            report.append(f"  transformer_depth={td}  →  hf:transformer_layers_per_block={hf_td}  (was {base.get('transformer_layers_per_block')})")

    # num_res_blocks → layers_per_block (use first value; SDXL has uniform depth)
    nr = merged_unet_cfg.get("num_res_blocks")
    if nr is not None:
        first = nr[0] if isinstance(nr, (list, tuple)) else nr
        if first != base.get("layers_per_block"):
            overrides["layers_per_block"] = first
            report.append(f"  num_res_blocks[0]={first}  →  hf:layers_per_block={first}  (was {base.get('layers_per_block')})")

    return overrides, report


# ---------------------------------------------------------------------------
# V-prediction + ZeroSNR helpers
# ---------------------------------------------------------------------------

def _rescale_zero_terminal_snr_sigmas(sigmas: "torch.Tensor") -> "torch.Tensor":
    """Rescale a sigma schedule to enforce zero terminal SNR (Lin et al. 2023).

    Equivalent to nodes_model_advanced.rescale_zero_terminal_snr_sigmas.
    Transforms the schedule so alphas_cumprod[-1] → 0 (sigma[-1] → ∞),
    matching checkpoints that were trained with a zero-SNR noise schedule.
    """
    alphas_cumprod = 1.0 / ((sigmas ** 2) + 1.0)
    ab_sqrt = alphas_cumprod.sqrt()
    ab_sqrt_0 = ab_sqrt[0].clone()
    ab_sqrt_T = ab_sqrt[-1].clone()
    ab_sqrt = (ab_sqrt - ab_sqrt_T) * ab_sqrt_0 / (ab_sqrt_0 - ab_sqrt_T)
    alphas_bar = ab_sqrt ** 2
    alphas_bar[-1] = 4.8973451890853435e-08   # numerical zero (from ComfyUI)
    return ((1.0 - alphas_bar) / alphas_bar) ** 0.5


def _ensure_vpred_zsnr(model_sampling) -> None:
    """Auto-apply ZeroSNR to a V_PREDICTION model_sampling when it is not yet active.

    Mirrors the 'Advanced Model Sampling for reForge' script behaviour
    (Discrete / v_prediction / Zero SNR) so DiffPipeline works correctly for
    v-pred checkpoints without requiring the user to enable that script.

    Logic
    -----
    - If model_sampling is not V_PREDICTION → no-op.
    - If ZSNR is already active (model_sampling.zsnr or sigma_max > 100) → log and return.
    - Otherwise rescale the sigma schedule in-place and stamp zsnr = True.

    Why in-place?
    The model_sampling object is shared with unet_patcher.model.model_sampling, so
    the ldm sampler path also benefits from the corrected schedule automatically.
    The object is recreated per model load, so there is no cross-model contamination.
    """
    # Check via MRO names — V_PREDICTION is a mixin so isinstance() narrows
    # the type and mypy loses sigma_max / sigmas / set_sigmas from the
    # ModelSamplingDiscrete base.  Checking the name avoids that narrowing.
    is_vpred = any(
        c.__name__ == "V_PREDICTION"
        for c in type(model_sampling).__mro__
    )
    if not is_vpred:
        return

    sigma_max = float(getattr(model_sampling, "sigma_max", 0.0))
    zsnr_flag = getattr(model_sampling, "zsnr", False)

    # Trust sigma_max, not the flag.  _build_model_sampling_from_pipe can set
    # zsnr=True from scheduler.config.rescale_betas_zero_snr but then call
    # set_sigmas(scheduler.alphas_cumprod) which overwrites with non-ZSNR sigmas,
    # leaving the flag set but the schedule un-rescaled (sigma_max ≈ 14.6).
    if sigma_max > 100.0:
        log.info(
            "DiffPipeline v-pred: ZeroSNR already active in sigmas "
            "(sigma_max=%.2f) — no rescaling needed.",
            sigma_max,
        )
        return
    if zsnr_flag:
        log.info(
            "DiffPipeline v-pred: zsnr flag is True but sigma_max=%.4f (≤100) — "
            "sigmas were overwritten after ZSNR init; will re-apply rescaling.",
            sigma_max,
        )

    log.info(
        "DiffPipeline v-pred: sigma_max=%.4f — ZeroSNR not detected; "
        "auto-applying rescaling (mirrors Advanced Model Sampling / Zero SNR).",
        sigma_max,
    )
    sigmas = getattr(model_sampling, "sigmas", None)
    if sigmas is None:
        log.warning("DiffPipeline v-pred: model_sampling has no sigmas — skipping ZSNR.")
        return
    new_sigmas = _rescale_zero_terminal_snr_sigmas(sigmas)
    model_sampling.set_sigmas(new_sigmas)   # type: ignore[union-attr]
    model_sampling.zsnr = True              # type: ignore[union-attr]
    log.info(
        "DiffPipeline v-pred: ZeroSNR applied — new sigma_max=%.2f.",
        float(getattr(model_sampling, "sigma_max", float("nan"))),
    )


# ---------------------------------------------------------------------------
# Block address table: HF (b_idx, a_idx) → ldm block index
# Derived from unet_to_diffusers() with num_res_blocks=[2,2,2].
#   down: n = 1 + 3*b_idx + a_idx   (only b_idx=1,2 have attentions)
#   up:   n = 3*b_idx + a_idx        (only b_idx=0,1 have attentions)
# ---------------------------------------------------------------------------
_DOWN_ATTN_LDM_IDX: dict = {
    (1, 0): 4, (1, 1): 5,
    (2, 0): 7, (2, 1): 8,
}
_UP_ATTN_LDM_IDX: dict = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2,
    (1, 0): 3, (1, 1): 4, (1, 2): 5,
}


# ---------------------------------------------------------------------------
# ForgeAttnSelfProcessor  (replaces PassthroughAttnProcessor)
# ---------------------------------------------------------------------------

class ForgeAttnSelfProcessor:
    """HF attention processor for attn1 (self-attention) blocks.

    When ``patches_replace["attn1"]`` carries a hook for this block, it
    computes Q/K/V explicitly and calls the hook — enabling entropy-capture
    passes (SURE-AGWAV, SAG-style guidance) on the Diffusers pipeline path.

    Without a hook it delegates to ``AttnProcessor2_0``, identical to the
    old ``PassthroughAttnProcessor`` behaviour.

    Hook signature (same as ldm_patched attn1 replace):
        hook(q, k, v, extra_options, mask=None) → out
        q/k/v:  (B, N, heads*dim_head)  — flat pre-SDPA format
        out:    (B, N, heads*dim_head)
    """

    def __init__(self, block_name: str, ldm_idx: int, t_idx: int) -> None:
        from diffusers.models.attention_processor import AttnProcessor2_0
        self._inner     = AttnProcessor2_0()
        self.block_name = block_name
        self.ldm_idx    = ldm_idx
        self.t_idx      = t_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,  # noqa: ARG002 — self-attn never uses this
        attention_mask=None,
        temb=None,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}

        patches_replace = transformer_options.get("patches_replace", {})
        attn1_replace   = patches_replace.get("attn1", {})

        block_tuple = (self.block_name, self.ldm_idx, self.t_idx)
        block_pair  = (self.block_name, self.ldm_idx)
        hook_fn = attn1_replace.get(block_tuple) or attn1_replace.get(block_pair)

        if hook_fn is None:
            return self._inner(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        # ── Hook path: compute Q/K/V explicitly so hook sees full attention ──
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # Self-attention: K, V project from the same hidden_states as Q
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size
            )

        extra_options = {
            "block":          (self.block_name, self.ldm_idx),
            "block_index":    self.t_idx,
            "n_heads":        attn.heads,
            "dim_head":       attn.to_q.out_features // attn.heads,
            "original_shape": transformer_options.get("original_shape"),
            "attn_precision": transformer_options.get("attn_precision"),
        }
        for k, v in transformer_options.items():
            if k not in ("patches", "patches_replace") and k not in extra_options:
                extra_options[k] = v

        # q/k/v in (B, N, heads*dim_head) — matches attention_basic_with_sim
        hidden_states = hook_fn(query, key, value, extra_options,
                                mask=attention_mask)

        # Output projection (same path as ForgeAttnProcessor)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Keep old name as alias so any external code that references it still works
PassthroughAttnProcessor = ForgeAttnSelfProcessor


# ---------------------------------------------------------------------------
# ForgeAttnProcessor
# ---------------------------------------------------------------------------

class ForgeAttnProcessor:
    """HF attention processor that dispatches Forge patch hooks.

    Installed on every ``attn2`` module in the HF UNet at construction time.
    The LDM block address is burned in so per-call lookup is O(1).

    Supported hooks (same keys as ldm attention.py):
      * ``transformer_options["patches_replace"]["attn2"][(block, idx, t)]``
        Full Q/K/V replacement.  Fn signature: ``(q, k, v, extra_options) → out``
        where tensors are in flat ``(B, seq, heads*dim)`` format.
      * ``transformer_options["patches"]["attn2_patch"]``
        Q/K/V modifier before standard attention.
        Fn signature: ``(q, k, v, extra_options) → (q, k, v)``
      * ``transformer_options["clip_attn_norm"]`` (bool, default False)
        Apply soft per-token L2 normalisation to encoder_hidden_states before the
        K/V projections: divide each token embedding by its own norm, but only when
        that norm exceeds 1.0 — leaving normally-scaled tokens untouched while
        clamping outliers (e.g. from high prompt weights) to the unit sphere.
        Preserves directional/semantic content; prevents extreme Q·K dot-products.
        Enabled via ``--forge-diffusers-clip-attn-norm``.
    """

    def __init__(self, block_name: str, block_idx: int, transformer_idx: int) -> None:
        self.block_name = block_name
        self.block_idx = block_idx
        self.transformer_idx = transformer_idx


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        transformer_options=None,
        **_cross_attention_kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        transformer_patches = transformer_options.get("patches", {})
        transformer_patches_replace = transformer_options.get("patches_replace", {})
        # extra_options mirrors ldm's convention so patch fns get the same dict
        extra_options = {
            "block": (self.block_name, self.block_idx),
            "block_index": self.transformer_idx,
            "n_heads": attn.heads,
            "dim_head": attn.to_q.out_features // attn.heads,
        }
        for k, v in transformer_options.items():
            if k not in ("patches", "patches_replace"):
                extra_options[k] = v

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        # clip_attn_norm: normalization is applied in apply_model() before the
        # UNet forward pass so it runs outside the compiled graph region.

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # --- Forge patch dispatch ---
        block_tuple = (self.block_name, self.block_idx, self.transformer_idx)
        block_pair = (self.block_name, self.block_idx)

        attn2_replace = transformer_patches_replace.get("attn2", {})
        replace_key = (
            block_tuple if block_tuple in attn2_replace else
            block_pair if block_pair in attn2_replace else
            None
        )

        if replace_key is not None:
            # Full replacement — patch fn owns the attention computation.
            # Q/K/V are in flat (B, seq, inner_dim) format, matching ldm convention.
            hidden_states = attn2_replace[replace_key](
                query, key, value, extra_options
            )
        else:
            if "attn2_patch" in transformer_patches:
                for p in transformer_patches["attn2_patch"]:
                    query, key, value = p(query, key, value, extra_options)

            # Standard scaled-dot-product attention
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if getattr(attn, "norm_q", None) is not None:
                query = attn.norm_q(query)
            if getattr(attn, "norm_k", None) is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, inner_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        # Projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Inference headroom helper used by _should_use_lru
# ---------------------------------------------------------------------------

def _inference_headroom(x: Optional["torch.Tensor"] = None) -> int:
    """Return bytes to reserve for activations during inference.

    Base = reForge's minimum_inference_memory() (~1 GiB), overridable via
    ``--forge-diffusers-lru-headroom <MB>``.  If the latent tensor *x* is
    provided we add a resolution-scaled estimate for the largest intermediate
    feature maps (batch × 320 channels × H × W, fp16 = 2 bytes, counted
    twice for input + output of each residual block).
    """
    try:
        from modules.shared import cmd_opts
        override_mb = getattr(cmd_opts, "forge_diffusers_lru_headroom", 0)
    except Exception:
        override_mb = 0

    if override_mb > 0:
        base = override_mb * 1024 * 1024
    else:
        try:
            from ldm_patched.modules.model_management import minimum_inference_memory
            base = int(minimum_inference_memory())
        except Exception:
            base = 1 * 1024 * 1024 * 1024  # 1 GiB fallback

    if x is not None:
        # Factor of 16: skip-connection tensors from all encoder blocks are held
        # in VRAM until consumed by the matching decoder block.  At 1280×1280
        # (SDXL) this is ~172 MB of skips + ~200 MB of intermediate activations;
        # factor=16 provides ~524 MB extra on top of the 1 GiB base.
        b, _c, h, w = x.shape
        base += b * 320 * h * w * 2 * 16
    return base


# ---------------------------------------------------------------------------
# Standalone auto-offload helper (used by both DiffPipeline and the hijack path)
# ---------------------------------------------------------------------------

def apply_auto_offload_to_unet(hf_unet, device: "torch.device"):
    """Partition an HF UNet2DConditionModel into Group A (device) and Group B (CPU)
    using infer_auto_device_map, then install forward pre/post hooks on Group B so
    each block is streamed to the compute device only while it runs.

    Returns a list of (pre_handle, post_handle) pairs so the caller can remove them.
    """
    from accelerate import infer_auto_device_map

    device_key = str(device)
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]]
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        max_memory = {device_key: int(free_bytes * 0.85), "cpu": "48GiB"}
    elif device.type == "mps":
        max_memory = {device_key: "8GiB", "cpu": "48GiB"}
    else:
        max_memory = None

    device_map = infer_auto_device_map(
        hf_unet,
        max_memory=max_memory,
        no_split_module_classes=["Transformer2DModel", "ResnetBlock2D"],
    )
    log.info("apply_auto_offload_to_unet: raw device_map=%s", device_map)

    # Collect block-level children (expand ModuleLists like down_blocks/up_blocks)
    def iter_blocks(unet):
        for name, child in unet.named_children():
            if isinstance(child, torch.nn.ModuleList):
                for idx in range(len(child)):
                    yield f"{name}.{idx}", child[idx]
            else:
                yield name, child

    group_a = []
    group_b = []
    for path, module in iter_blocks(hf_unet):
        is_cpu = any(
            str(dev) == "cpu"
            for key, dev in device_map.items()
            if key == path or key.startswith(path + ".")
        )
        (group_b if is_cpu else group_a).append((path, module))

    log.info("apply_auto_offload_to_unet: Group A (device)=%s", [p for p, _ in group_a])
    log.info("apply_auto_offload_to_unet: Group B (CPU offload)=%s", [p for p, _ in group_b])

    # Move Group A to device; Group B stays on CPU
    for _, module in group_a:
        module.to(device=device)

    # Install load/unload hooks on Group B
    hook_handles = []
    unet_module_map = {n: m for n, m in hf_unet.named_modules() if n}

    for path, _ in group_b:
        current = unet_module_map.get(path)
        if current is None:
            log.warning("apply_auto_offload_to_unet: could not resolve path '%s' — skipping hooks", path)
            continue

        def make_hooks(dev):
            def pre_hook(m, inp):
                m.to(device=dev)
            def post_hook(m, inp, out):
                m.to(device="cpu")
            return pre_hook, post_hook

        pre_fn, post_fn = make_hooks(device)
        hook_handles.append(current.register_forward_pre_hook(pre_fn))
        hook_handles.append(current.register_forward_hook(post_fn))

    log.info(
        "apply_auto_offload_to_unet: done — %d Group-A blocks, %d Group-B blocks, %d hooks installed",
        len(group_a), len(group_b), len(hook_handles),
    )
    return hook_handles


# ---------------------------------------------------------------------------
# DiffPipeline
# ---------------------------------------------------------------------------

class DiffPipeline():
    """Diffusers-based SDXL pipeline.

    Wraps a ``UnetPatcher`` and replaces the ldm-patched
    ``BaseModel.apply_model()`` forward path so that a HuggingFace
    ``UNet2DConditionModel`` is used instead.

    Attached to ``sd_model.diff_pipeline`` by
    ``forge_loader.load_model_for_a1111()`` when both conditions are met:
      - the loaded checkpoint is SDXL (``sd_model.is_sdxl``)
      - ``--forge-diffusers-pipeline`` CLI flag is set

    Attributes:
        unet_patcher: The ``UnetPatcher`` that owns model_options, LoRA
            patches, ControlNet linked list, and block modifier hooks.
        sd_model: The a1111 shell object (``StableDiffusionModel``).
        model_sampling: Sigma ↔ timestep conversion (reused from ldm).
    """

    # ------------------------------------------------------------------
    # Alternative constructor: wrap an already-built HF UNet
    # (used by the diffusers hijack path in DiffusersModelAdapter)
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_unet(cls, hf_unet, unet_patcher, sd_model) -> "DiffPipeline":
        """Create a DiffPipeline from an already-loaded HF UNet2DConditionModel.

        Unlike the normal constructor this skips the ldm→HF weight conversion
        (``_build_hf_unet``).  Used when a StableDiffusionXLPipeline has been
        loaded by the diffusers path-hijack and we want DiffPipeline.apply_model
        to be the active forward path so ControlNet mapping, ForgeAttnProcessor
        dispatch, LoRA sync, and offload logic all work correctly.
        """
        instance = cls.__new__(cls)
        instance.unet_patcher = unet_patcher
        instance.sd_model = sd_model
        instance.model_sampling = unet_patcher.model.model_sampling
        _ensure_vpred_zsnr(instance.model_sampling)

        from modules.shared_cmd_options import cmd_opts
        instance._auto_offload = getattr(cmd_opts, 'forge_diffusers_auto_offload', False)
        instance._sequential_offload = (
            getattr(cmd_opts, 'forge_diffusers_sequential_offload', False)
            and not instance._auto_offload
        )
        instance._offload = (
            getattr(cmd_opts, 'forge_diffusers_offload', False)
            and not instance._sequential_offload
            and not instance._auto_offload
        )
        instance._seq_hooks_installed = False
        instance._compiled = False
        instance._mps_optimized = False
        instance._compile = getattr(cmd_opts, 'forge_diffusers_compile', False)
        instance._clip_attn_norm = getattr(cmd_opts, 'forge_diffusers_clip_attn_norm', False)
        instance._auto_offload_ready = False
        instance._b_hooks = []
        instance._b_block_paths = []
        instance._vram_at_partition = 0
        instance._regions_installed = False
        instance._tc_ready = False
        instance._autocast_dtype = None
        instance._vram_allocator = None
        instance._lru_block_names = []
        instance._lru_ready = False
        instance._block_paths = set()
        instance._non_blocks_on_device = False

        instance._hf_unet = hf_unet
        instance._install_attn_processors(hf_unet)

        # No ldm key map available on this path — LoRA sync via PEFT is skipped.
        instance._ldm_to_hf = {}
        instance._synced_patches_uuid = None
        instance._active_adapters = []

        # HF UNet summary (from_single_file path — _build_hf_unet was skipped)
        try:
            hf_cfg = hf_unet.config
            hf_keys = set(hf_unet.state_dict().keys())
            hf_dtype = next(hf_unet.parameters()).dtype
            _unet_lines = [
                "",
                "╔══════════════════════════════════════════════════════════════╗",
                "║         DiffPipeline — HF UNet load report (hijack)         ║",
                "╚══════════════════════════════════════════════════════════════╝",
                "  Source              : diffusers from_single_file (weights pre-loaded)",
                f"  Dtype               : {hf_dtype}",
                f"  Total HF keys       : {len(hf_keys)}",
                f"  in_channels         : {getattr(hf_cfg, 'in_channels', '?')}",
                f"  out_channels        : {getattr(hf_cfg, 'out_channels', '?')}",
                f"  cross_attention_dim : {getattr(hf_cfg, 'cross_attention_dim', '?')}",
                f"  projection_class_embeddings_input_dim : "
                f"{getattr(hf_cfg, 'projection_class_embeddings_input_dim', '?')}",
                f"  transformer_layers_per_block : "
                f"{getattr(hf_cfg, 'transformer_layers_per_block', '?')}",
                f"  layers_per_block    : {getattr(hf_cfg, 'layers_per_block', '?')}",
                f"  block_out_channels  : {getattr(hf_cfg, 'block_out_channels', '?')}",
                "",
            ]
            print("\n".join(_unet_lines))
        except Exception as _e:
            log.warning("DiffPipeline.from_hf_unet: could not read HF UNet config: %s", _e)

        instance._log_sampling_report()
        log.info(
            "DiffPipeline.from_hf_unet: attached to existing HF UNet — "
            "diffusers hijack path is ACTIVE."
        )
        return instance

    def __init__(self, unet_patcher: UnetPatcher, sd_model: object) -> None:
        self.unet_patcher = unet_patcher
        self.sd_model = sd_model
        self.model_sampling = unet_patcher.model.model_sampling
        _ensure_vpred_zsnr(self.model_sampling)

        from modules.shared_cmd_options import cmd_opts
        # Priority: auto_offload > sequential_offload > offload > default (whole-model on device).
        self._auto_offload: bool = getattr(cmd_opts, 'forge_diffusers_auto_offload', False)
        self._sequential_offload: bool = (
            getattr(cmd_opts, 'forge_diffusers_sequential_offload', False)
            and not self._auto_offload
        )
        self._offload: bool = (
            getattr(cmd_opts, 'forge_diffusers_offload', False)
            and not self._sequential_offload
            and not self._auto_offload
        )
        self._seq_hooks_installed: bool = False
        self._compiled: bool = False
        self._mps_optimized: bool = False
        self._compile: bool = getattr(cmd_opts, 'forge_diffusers_compile', False)
        self._clip_attn_norm: bool = getattr(cmd_opts, 'forge_diffusers_clip_attn_norm', False)

        # Auto-offload state — populated lazily on first apply_model() call.
        self._auto_offload_ready: bool = False
        self._b_hooks: list = []        # registered hook handles (removed on LoRA change)
        self._b_block_paths: list = []  # Group B block paths (for logging)
        self._vram_at_partition: int = 0  # free bytes (reForge formula) at last partition
        self._regions_installed: bool = False  # True once nested_compile_region / per-block compile has run

        # VRAMAllocator — used on the default path when VRAM is insufficient.
        self._vram_allocator: Optional[VRAMAllocator] = None
        self._lru_block_names: list = []
        self._lru_ready: bool = False  # True once _setup_lru_offload has run
        # Set of top-level HF UNet child names that are managed as LRU blocks.
        # Populated by _setup_lru_offload; used by _restore_non_blocks().
        self._block_paths: set = set()
        # True while non-block UNet children (conv_in, time_embedding, …) are
        # resident on the compute device.  Set False by full_unload_to_cpu() so
        # apply_model() knows to restore them before the next forward pass.
        self._non_blocks_on_device: bool = False

        # Tensor Core optimisation state — populated lazily on first apply_model() call.
        self._tc_ready: bool = False
        self._autocast_dtype: Optional[torch.dtype] = None  # set by _maybe_setup_tensor_core_opts

        self._hf_unet: UNet2DConditionModel
        log.info("DiffPipeline: building HF UNet2DConditionModel from checkpoint weights …")
        self._hf_unet = self._build_hf_unet(unet_patcher.model)
        self._install_attn_processors(self._hf_unet)

        # Phase 5: LoRA sync state
        self._ldm_to_hf: dict = self._build_ldm_to_hf_map(unet_patcher.model)
        self._synced_patches_uuid = None
        self._active_adapters: list = []   # list of (adapter_name, strength)

        self._log_sampling_report()
        log.info(
            "DiffPipeline attached to sd_model — "
            "Diffusers SDXL path is ACTIVE."
        )

    # ------------------------------------------------------------------
    # Sampling compliance report (shared by __init__ and from_hf_unet)
    # ------------------------------------------------------------------

    def _log_sampling_report(self) -> None:
        """Print and log a sampling-parameter compliance summary.

        Verifies that the model_sampling object is in the expected state after
        apply_checkpoint_sampling_params() has run.  Confirms that schedule,
        prediction type, and ZTSNR are properly obeyed by the three delegation
        points in apply_model():
          calculate_input    → preconditioning  (EPS/V_PREDICTION/EDM/CONST …)
          timestep           → sigma → integer timestep via model's sigma table
          calculate_denoised → postconditioning (prediction-type-aware)
        """
        ms = self.model_sampling
        ms_cls   = type(ms)
        ms_bases = [c.__name__ for c in ms_cls.__mro__
                    if c.__name__ not in ("object", ms_cls.__name__)]

        _pred_names = {
            "EPS":          "epsilon (ε) — standard noise prediction",
            "V_PREDICTION": "v-prediction (velocity parameterization)",
            "EDM":          "EDM (Karras et al. 2022) — c_skip/c_out preconditioning",
            "CONST":        "constant (flow-matching / rectified flow)",
            "X0":           "x0 prediction",
            "IMG_TO_IMG":   "img2img epsilon",
            "COSMOS_RFLOW": "Cosmos rectified-flow",
        }
        pred_type = next(
            (_pred_names[base] for base in ms_bases if base in _pred_names),
            f"unknown ({ms_bases})"
        )

        try:
            sig_min  = float(ms.sigma_min)
            sig_max  = float(ms.sigma_max)
            n_sigmas = len(ms.sigmas) if hasattr(ms, "sigmas") else "?"
        except Exception:
            sig_min = sig_max = float("nan")
            n_sigmas = "?"

        zsnr_active = getattr(ms, "zsnr", False)
        _ztsnr_heuristic = sig_max > 100.0
        if zsnr_active:
            zsnr_note = "YES (flag set on model_sampling)"
        elif _ztsnr_heuristic:
            zsnr_note = "YES (inferred from sigma_max > 100 — ZTSNR rescaling applied)"
        else:
            zsnr_note = "No"

        # Prefer the tagged source set by _build_model_sampling_from_pipe.
        # Fall back to a sigma-range heuristic for the legacy ldm path.
        if hasattr(ms, "_schedule_source"):
            schedule_source = ms._schedule_source
        else:
            _default_sigma_max = 14.615
            schedule_source = (
                "checkpoint alphas_cumprod (ldm path)"
                if abs(sig_max - _default_sigma_max) > 0.1
                else "hardcoded scaled_linear default (no schedule tensors in checkpoint)"
            )

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║         DiffPipeline — Sampling compliance report           ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"  model_sampling type : {ms_cls.__name__}",
            f"  MRO mixins          : {ms_bases}",
            f"  Prediction type     : {pred_type}",
            f"  Schedule source     : {schedule_source}",
            f"  σ_min               : {sig_min:.6f}",
            f"  σ_max               : {sig_max:.6f}",
            f"  Schedule steps (T)  : {n_sigmas}",
            f"  ZTSNR               : {zsnr_note}",
            "",
            "  Delegation points in apply_model():",
            "    calculate_input    → preconditioning   ✓ (delegates to model_sampling)",
            "    timestep(sigma)    → integer timestep  ✓ (uses model's sigma table)",
            "    calculate_denoised → postconditioning  ✓ (delegates to model_sampling)",
            "",
        ]
        print("\n".join(lines))
        log.info(
            "DiffPipeline sampling: pred=%s  σ=[%.4f, %.4f]  T=%s  zsnr=%s  "
            "schedule_source=%s",
            pred_type, sig_min, sig_max, n_sigmas, zsnr_note, schedule_source,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_hf_unet(self, ldm_model) -> "UNet2DConditionModel":
        """Convert ldm SDXL state dict → HF UNet2DConditionModel."""
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        from ldm_patched.modules.utils import unet_to_diffusers

        # Merge detected config (has num_res_blocks etc.) with our known full
        # SDXL config so unet_to_diffusers() gets all required keys.
        unet_cfg = dict(_SDXL_LDM_UNET_CONFIG)
        model_embedded_cfg = dict(ldm_model.model_config.unet_config)
        unet_cfg.update(model_embedded_cfg)

        # --- Report what the model carries vs our hardcoded defaults ----------
        # Keys present in the model-embedded config that differ from our base.
        cfg_diffs = {
            k: (v, _SDXL_LDM_UNET_CONFIG.get(k, "<not in base>"))
            for k, v in model_embedded_cfg.items()
            if _SDXL_LDM_UNET_CONFIG.get(k) != v
        }
        cfg_same_count = len(model_embedded_cfg) - len(cfg_diffs)

        # --- Derive HF UNet config overrides from model-embedded parameters --
        hf_overrides, override_report = _derive_hf_config_from_ldm(unet_cfg)
        hf_cfg = dict(_SDXL_HF_UNET_CONFIG)
        hf_cfg.update(hf_overrides)

        key_map = unet_to_diffusers(unet_cfg)  # hf_key → ldm_key

        ldm_sd = ldm_model.diffusion_model.state_dict()

        # Instantiate on CPU; moved to compute device in apply_model().
        hf_unet = UNet2DConditionModel(**hf_cfg)

        # unet_to_diffusers() generates entries for every possible resnet key
        # (conv_shortcut, downsamplers, upsamplers) regardless of whether the
        # block actually has one.  Filter key_map against the HF model's real
        # parameter set so we only copy weights that exist on both sides, and
        # only warn about keys that are genuinely absent.
        hf_model_keys = set(hf_unet.state_dict().keys())

        hf_sd = {}
        missing_ldm = []
        for hf_key, ldm_key in key_map.items():
            if hf_key not in hf_model_keys:
                # Architectural mismatch (e.g. same-channel resnet has no
                # conv_shortcut, last down/up block has no sampler) — skip silently.
                continue
            if ldm_key in ldm_sd:
                hf_sd[hf_key] = ldm_sd[ldm_key]
            else:
                missing_ldm.append(hf_key)

        if missing_ldm:
            log.warning(
                "DiffPipeline: %d HF keys had no matching LDM key (first 5: %s)",
                len(missing_ldm), missing_ldm[:5],
            )

        missing, unexpected = hf_unet.load_state_dict(hf_sd, strict=False)

        if missing:
            log.warning(
                "DiffPipeline: HF UNet missing %d keys (first 5: %s)",
                len(missing), missing[:5],
            )
        if unexpected:
            log.warning(
                "DiffPipeline: HF UNet unexpected %d keys", len(unexpected)
            )

        # Match dtype from the ldm diffusion model
        ref_dtype = None
        try:
            ref_dtype = next(ldm_model.diffusion_model.parameters()).dtype
            hf_unet = hf_unet.to(dtype=ref_dtype)
        except StopIteration:
            pass

        # --- Load report (printed once at construction) -----------------------
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║            DiffPipeline — HF UNet load report               ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"  Weights loaded : {len(hf_sd)} / {len(hf_model_keys)} HF parameter tensors",
            f"  Dtype          : {ref_dtype} (matched from ldm diffusion_model)",
        ]

        # Model-embedded config summary
        lines.append(f"\n  Model-embedded unet_config  ({len(model_embedded_cfg)} keys read from checkpoint):")
        if cfg_diffs:
            for k, (model_val, base_val) in sorted(cfg_diffs.items()):
                lines.append(f"    {k}: {model_val!r}  (base default was {base_val!r})")
        else:
            lines.append("    (all values match hardcoded base — standard SDXL-base-1.0)")
        if cfg_same_count:
            lines.append(f"    … plus {cfg_same_count} keys matching base defaults (omitted)")

        # HF config adjustments
        lines.append(f"\n  HF UNet2DConditionModel config adjustments ({len(hf_overrides)} override(s)):")
        if override_report:
            for r in override_report:
                lines.append(r)
        else:
            lines.append("    (none — using hardcoded SDXL-base-1.0 HF config as-is)")

        # Weight transfer issues
        if missing_ldm or missing or unexpected:
            lines.append("\n  Weight transfer issues:")
            if missing_ldm:
                lines.append(f"    {len(missing_ldm)} HF keys had no matching LDM key  (first 3: {missing_ldm[:3]})")
            if missing:
                lines.append(f"    {len(missing)} HF UNet keys left uninitialised  (first 3: {list(missing)[:3]})")
            if unexpected:
                lines.append(f"    {len(unexpected)} unexpected keys ignored")
        else:
            lines.append("\n  Weight transfer : clean (no missing or unexpected keys)")

        lines.append("")
        print("\n".join(lines))

        return hf_unet

    def _build_ldm_to_hf_map(self, ldm_model) -> dict:
        """Build reverse key map: 'diffusion_model.X.weight' → HF parameter path.

        ``unet_to_diffusers()`` returns ``{hf_key: ldm_rel_key}``.  We invert it
        so that ``unet_patcher.patches`` keys (which use the full
        ``diffusion_model.`` prefix) can be looked up in O(1).
        """
        from ldm_patched.modules.utils import unet_to_diffusers

        unet_cfg = dict(_SDXL_LDM_UNET_CONFIG)
        unet_cfg.update(ldm_model.model_config.unet_config)
        key_map = unet_to_diffusers(unet_cfg)   # hf_key → ldm_rel_key

        result: dict = {}
        for hf_key, ldm_rel in key_map.items():
            result["diffusion_model." + ldm_rel] = hf_key
        return result

    def _install_attn_processors(self, hf_unet) -> None:
        """Install ForgeAttnProcessor on every attn2 and ForgeAttnSelfProcessor
        on every attn1 in the HF UNet.

        Both processors declare ``transformer_options`` explicitly in their
        signatures so diffusers does not emit "not expected … will be ignored"
        warnings when cross_attention_kwargs is forwarded from apply_model().

        ForgeAttnSelfProcessor dispatches patches_replace["attn1"] hooks when
        present (e.g. SURE-AGWAV entropy capture), falling back to AttnProcessor2_0.
        """
        def _install_block_processors(attn_mod, block_name, ldm_idx):
            for t_idx, tb in enumerate(attn_mod.transformer_blocks):
                tb.attn2.set_processor(
                    ForgeAttnProcessor(block_name, ldm_idx, t_idx)
                )
                tb.attn1.set_processor(
                    ForgeAttnSelfProcessor(block_name, ldm_idx, t_idx)
                )

        # Down blocks
        for b_idx, down_block in enumerate(hf_unet.down_blocks):
            for a_idx, attn_mod in enumerate(getattr(down_block, "attentions", [])):
                ldm_idx = _DOWN_ATTN_LDM_IDX.get((b_idx, a_idx))
                if ldm_idx is None:
                    continue
                _install_block_processors(attn_mod, "input", ldm_idx)

        # Mid block
        for attn_mod in getattr(hf_unet.mid_block, "attentions", []):
            _install_block_processors(attn_mod, "middle", 0)

        # Up blocks
        for b_idx, up_block in enumerate(hf_unet.up_blocks):
            for a_idx, attn_mod in enumerate(getattr(up_block, "attentions", [])):
                ldm_idx = _UP_ATTN_LDM_IDX.get((b_idx, a_idx))
                if ldm_idx is None:
                    continue
                _install_block_processors(attn_mod, "output", ldm_idx)

    def _install_sequential_offload_hooks(self, device) -> None:
        """Install accelerate per-block CPU offload hooks on the HF UNet.

        Each direct child of the UNet (conv_in, down_blocks, mid_block,
        up_blocks, conv_out, …) gets an AlignDevicesHook that moves its
        parameters to *device* just before its forward pass and returns them
        to CPU immediately after.  Peak VRAM ≈ largest single block.

        Called lazily on the first apply_model() invocation so the compute
        device is known.
        """
        from accelerate import cpu_offload

        for child in self._hf_unet.children():
            cpu_offload(child, execution_device=device, offload_buffers=True)

        self._seq_hooks_installed = True
        log.info(
            "DiffPipeline: sequential CPU offload hooks installed (execution device: %s)", device
        )

    # ------------------------------------------------------------------
    # Auto device-map offload (--forge-diffusers-auto-offload)
    # ------------------------------------------------------------------

    def _iter_unet_blocks(self):
        """Yield (path, setter, module) for each block-level child of the HF UNet.

        For ModuleList children (down_blocks, up_blocks) yields one entry per
        item so the granularity matches infer_auto_device_map output.
        ``setter(m)`` replaces the entry in the UNet in-place (used when
        swapping a block for its torch.compile'd wrapper).
        """
        # Only yield block-level submodules suitable for per-block torch.compile.
        # ModuleList children (down_blocks, up_blocks) → yield each item.
        # Singleton children → only mid_block qualifies; conv_in, conv_out,
        # conv_norm_out, time_embedding etc. must NOT be compiled individually
        # because diffusers uses them in boolean context (``if self.conv_norm_out:``)
        # and OptimizedModule's __len__ raises TypeError in that case.
        _SINGLETON_BLOCK_NAMES = {"mid_block"}
        for name, child in self._hf_unet.named_children():
            if isinstance(child, torch.nn.ModuleList):
                for idx in range(len(child)):
                    path = f"{name}.{idx}"
                    # capture loop variables with distinct helper names
                    def make_list_setter(lst, i):
                        def setter(m): lst[i] = m
                        return setter
                    yield path, make_list_setter(child, idx), child[idx]
            elif name in _SINGLETON_BLOCK_NAMES:
                def make_attr_setter(attr):
                    def setter(m): setattr(self._hf_unet, attr, m)
                    return setter
                yield name, make_attr_setter(name), child

    def _setup_auto_offload(self, device: torch.device) -> None:
        """Partition UNet blocks into Group A (device) and Group B (CPU),
        compile each group regionally, and install load/unload hooks on Group B.

        Called lazily on the first ``apply_model()`` invocation so the compute
        device is known.  Must be re-called (via ``_reset_auto_offload``) after
        any structural change to the UNet (e.g. LoRA adapter swap).
        """
        _is_repartition = self._regions_installed  # True on 2nd+ call
        log.info(
            "DiffPipeline auto-offload: %s (device=%s)",
            "re-partitioning" if _is_repartition else "initial partition",
            device,
        )
        from accelerate import infer_auto_device_map

        # --- 1. Determine max_memory budget ---
        # infer_auto_device_map expects Dict[int | str, int | str] keys.
        # Use str(device) (e.g. "cuda:0") rather than a torch.device object.
        #
        # We use model_management.get_free_memory() rather than
        # torch.cuda.mem_get_info() because the latter reports PyTorch's
        # allocator cache as consumed.  get_free_memory adds back
        # (mem_reserved − mem_active) — the portion of that cache PyTorch can
        # reuse immediately — giving a more accurate picture of what
        # infer_auto_device_map can actually place on device.  We store the
        # measurement so _reset_auto_offload can detect significant pressure
        # changes and trigger a re-partition.
        device_key = str(device)
        if device.type == "cuda":
            # Mirrors model_management.get_free_memory(): adds the PyTorch
            # allocator cache (reserved − active) back to the OS-visible free
            # bytes so infer_auto_device_map sees memory that PyTorch can
            # reuse immediately, not just what the OS considers unallocated.
            _stats = torch.cuda.memory_stats(device)
            _cuda_free, _ = torch.cuda.mem_get_info(device)
            free_bytes: int = int(
                _cuda_free
                + _stats["reserved_bytes.all.current"]
                - _stats["active_bytes.all.current"]
            )
            max_vram = int(free_bytes * 0.85)
            max_memory: Optional[dict] = {device_key: max_vram, "cpu": "48GiB"}
            self._vram_at_partition = free_bytes
            log.info(
                "DiffPipeline auto-offload: VRAM free=%.0f MB  "
                "budget (85%%)=%.0f MB  allocator_cache=%.0f MB",
                free_bytes / (1024 ** 2),
                max_vram / (1024 ** 2),
                (_stats["reserved_bytes.all.current"] - _stats["active_bytes.all.current"]) / (1024 ** 2),
            )
        elif device.type == "mps":
            # MPS doesn't expose free memory; use a conservative fixed budget.
            max_memory = {device_key: "8GiB", "cpu": "48GiB"}
            self._vram_at_partition = 0
            log.info("DiffPipeline auto-offload: MPS — fixed budget 8GiB")
        else:
            max_memory = None
            self._vram_at_partition = 0
            log.info("DiffPipeline auto-offload: CPU device — no memory budget")

        # --- 2. Infer device map ---
        # no_split_module_classes prevents splitting a Transformer or ResnetBlock
        # across two devices, which would require cross-device tensor copies mid-block.
        device_map: dict = infer_auto_device_map(
            self._hf_unet,
            max_memory=max_memory,
            no_split_module_classes=["Transformer2DModel", "ResnetBlock2D"],
        )
        log.info("DiffPipeline auto-offload: raw device_map = %s", device_map)

        # --- 3. Classify each block into Group A or Group B ---
        # A block is Group B if *any* of its leaf entries in device_map is "cpu".
        group_a: list[tuple] = []   # (path, setter, module)
        group_b: list[tuple] = []

        for path, setter, module in self._iter_unet_blocks():
            is_cpu = any(
                str(dev) == "cpu"
                for key, dev in device_map.items()
                if key == path or key.startswith(path + ".")
            )
            if is_cpu:
                group_b.append((path, setter, module))
            else:
                group_a.append((path, setter, module))

        log.info(
            "DiffPipeline auto-offload: Group A (device) = %s",
            [p for p, _, _ in group_a],
        )
        log.info(
            "DiffPipeline auto-offload: Group B (CPU offload) = %s",
            [p for p, _, _ in group_b],
        )

        # --- 4. Move Group A to device; Group B stays on CPU ---
        for _, _, module in group_a:
            module.to(device=device)

        # --- 5. Per-block compile (first partition only, requires --forge-diffusers-compile) ---
        if self._compile and device.type != "mps":
            if not self._regions_installed:
                log.info(
                    "DiffPipeline auto-offload: installing compile regions "
                    "(%d blocks) …", len(group_a) + len(group_b)
                )
            else:
                log.info(
                    "DiffPipeline auto-offload: compile regions already installed "
                    "— skipping recompile (re-partition pass)"
                )
            self._install_compile_regions()
        elif not self._compile:
            log.info(
                "DiffPipeline auto-offload: per-block compile skipped "
                "(pass --forge-diffusers-compile to enable)."
            )

        # --- 6. Install load/unload hooks on Group B ---
        # Re-fetch modules after possible compile replacement.
        self._b_hooks.clear()
        self._b_block_paths.clear()

        # Build a name→module lookup after compile replacements.
        # named_modules() walks the entire tree; we only need direct matches.
        unet_module_map: dict[str, torch.nn.Module] = {
            n: m for n, m in self._hf_unet.named_modules() if n
        }

        for path, *_ in group_b:
            current = unet_module_map.get(path)
            if current is None:
                log.warning(
                    "DiffPipeline auto-offload: could not resolve Group B path '%s' "
                    "after compile — skipping hooks for this block", path
                )
                continue

            def make_hooks(dev: torch.device):
                def pre_hook(m: torch.nn.Module, inp: Any) -> None:
                    m.to(device=dev)
                def post_hook(m: torch.nn.Module, inp: Any, out: Any) -> None:
                    m.to(device="cpu")
                return pre_hook, post_hook

            pre_fn, post_fn = make_hooks(device)
            pre_handle = current.register_forward_pre_hook(pre_fn)
            post_handle = current.register_forward_hook(post_fn)
            self._b_hooks.extend([pre_handle, post_handle])
            self._b_block_paths.append(path)

        self._auto_offload_ready = True

        # Compute on-device vs offloaded memory totals for the summary log.
        def _block_mb(blocks):
            total = 0
            for _, _, m in blocks:
                for p in m.parameters():
                    total += p.numel() * p.element_size()
            return total / (1024 ** 2)

        group_a_mb = _block_mb(group_a)
        group_b_mb = _block_mb(group_b)
        log.info(
            "DiffPipeline auto-offload: partition complete — "
            "Group A %d blocks (%.0f MB on %s)  |  "
            "Group B %d blocks (%.0f MB CPU-hooked)  |  "
            "%d hooks installed",
            len(group_a), group_a_mb, device,
            len(group_b), group_b_mb,
            len(self._b_hooks),
        )

    def _reset_auto_offload(self) -> None:
        """Remove Group B hooks and mark auto-offload as needing re-setup.

        Called whenever the UNet structure changes (e.g. LoRA adapter swap)
        so that ``_setup_auto_offload`` re-runs on the next forward pass.
        """
        for handle in self._b_hooks:
            handle.remove()
        self._b_hooks.clear()
        self._b_block_paths.clear()
        self._auto_offload_ready = False

    # ------------------------------------------------------------------
    # LRU dynamic block loading (default path when VRAM is insufficient)
    # ------------------------------------------------------------------

    def _should_use_lru(self, device: torch.device, x: Optional["torch.Tensor"] = None) -> bool:
        """Return True when the whole HF UNet does not fit in available VRAM.

        Uses the same free-memory formula as model_management.get_free_memory().
        Headroom = reForge's minimum_inference_memory() (1 GiB) + a
        resolution-scaled activation estimate derived from x.shape.
        """
        if device.type != "cuda":
            return False
        try:
            stats = torch.cuda.memory_stats(device)
            cuda_free, _ = torch.cuda.mem_get_info(device)
            # Subtract inactive_split_bytes: fragmented reserved blocks that the
            # allocator cannot use for large (100s-MB) block allocations.
            fragmentation = stats.get("inactive_split_bytes.all.current", 0)
            free = int(
                cuda_free
                + stats.get("reserved_bytes.all.current", 0)
                - stats.get("active_bytes.all.current", 0)
                - fragmentation
            )
        except Exception:
            return False

        unet_bytes = sum(
            p.numel() * p.element_size() for p in self._hf_unet.parameters()
        )
        unet_bytes += sum(
            b.numel() * b.element_size()
            for b in self._hf_unet.buffers()
            if b is not None
        )
        headroom = _inference_headroom(x)
        needed = unet_bytes + headroom
        if needed > free:
            log.info(
                "DiffPipeline: UNet=%.0f MB + headroom=%.0f MB > free=%.0f MB — "
                "switching to LRU block loading.",
                unet_bytes / (1024 ** 2),
                headroom / (1024 ** 2),
                free / (1024 ** 2),
            )
            return True
        log.info(
            "DiffPipeline: UNet=%.0f MB fits in free=%.0f MB (headroom=%.0f MB) — "
            "whole-model placement.",
            unet_bytes / (1024 ** 2),
            free / (1024 ** 2),
            headroom / (1024 ** 2),
        )
        return False

    def _setup_lru_offload(self, device: torch.device, x: Optional["torch.Tensor"] = None) -> None:
        """Set up VRAMAllocator-based LRU block loading.

        Non-block UNet modules (conv_in, time_embedding, add_embedding,
        conv_norm_out, conv_out) are moved to device permanently — they are
        small and always needed.  Block-level modules (down_blocks, mid_block,
        up_blocks) are registered with :class:`VRAMAllocator`; forward
        pre-hooks activate each block just before it runs, evicting LRU
        residents as needed.

        Called once on the first ``apply_model()`` when VRAM is insufficient
        for the full UNet.
        """
        block_list = [
            (path, module)
            for path, _setter, module in self._iter_unet_blocks()
        ]
        block_paths = {path for path, _ in block_list}
        self._block_paths = block_paths  # saved for _restore_non_blocks()

        # 1. Move non-block children to device permanently (always resident).
        for name, child in self._hf_unet.named_children():
            if name in block_paths or any(p.startswith(name + ".") for p in block_paths):
                continue
            log.info("DiffPipeline LRU: moving non-block '%s' to %s", name, device)
            child.to(device=device)
        self._non_blocks_on_device = True

        # 2. Move all blocks to CPU so they start evicted (allocator will
        #    load them on demand via forward pre-hooks).
        for _, module in block_list:
            module.to("cpu")

        # 3. Evict CLIP and any other tracked models so their VRAM is
        #    available for UNet blocks during sampling.
        try:
            from ldm_patched.modules import model_management as _mm
            _mm.free_memory(1e30, device)
        except Exception:
            pass

        # Sync allocator device-state after the free_memory() call — it may
        # have silently moved CLIP/VAE blocks to CPU without notifying us.
        allocator = VRAMAllocator.get(device)
        allocator.sync_device_state()

        # Pre-evict CLIP blocks so their ~1.6 GB is immediately available
        # for UNet inference.  Without this, the LRU would burn CLIP during
        # step 1 (when up_blocks.0 needs 2.6 GB) but by then it has already
        # evicted smaller UNet blocks (down_blocks.0/1) to close the last gap,
        # causing needless reloads on every subsequent step.
        allocator.evict_by_prefix("clip_l.")
        allocator.evict_by_prefix("clip_g.")

        # 4. Reserve activation headroom BEFORE registering blocks.
        #    skip-connection tensors from down_blocks stay alive in VRAM while
        #    up_blocks runs; _inference_headroom() accounts for those plus the
        #    minimum_inference_memory() baseline.  Subtracting this from
        #    _effective_free() forces the allocator to evict earlier.
        headroom = _inference_headroom(x)
        allocator.reserve("inference_headroom", headroom)
        self._vram_allocator = allocator

        print(
            f"[VRAM] LRU setup: inference_headroom reserved = {headroom >> 20} MB  "
            f"(base=1 GiB + resolution-scaled skip-connection estimate)"
        )

        # 5. Register every block with the shared VRAMAllocator.
        for path, module in block_list:
            name = f"unet.{path}"
            allocator.register_module(name, module)
            self._lru_block_names.append(name)

        self._lru_ready = True

        # Register a pressure hook so model_management.free_memory() can
        # evict LRU blocks when needed by other model_management consumers.
        try:
            from ldm_patched.modules import model_management as _mm
            _mm.register_vram_pressure_hook(self.flush_lru_to_cpu)
        except Exception:
            pass

        log.info(
            "DiffPipeline: VRAMAllocator LRU active — %d blocks registered on %s "
            "(headroom=%d MB).",
            len(block_list), device, headroom >> 20,
        )

    def flush_lru_to_cpu(self, device: Optional[torch.device] = None) -> None:
        """Evict all LRU-resident UNet blocks back to CPU.

        Called by model_management's VRAM pressure hook when VAE or CLIP
        needs GPU memory after sampling.  Hooks stay installed so the next
        ``apply_model()`` reloads blocks normally via the forward pre-hook.
        Non-block modules (conv_in, time_embedding, …) remain on device.
        """
        if self._vram_allocator is not None:
            self._vram_allocator.evict_by_prefix("unet.")

    def full_unload_to_cpu(self) -> None:
        """Move the ENTIRE HF UNet to CPU — blocks AND non-block parts.

        Used when the allocator's block eviction alone is insufficient.
        After this call ``_non_blocks_on_device`` is False; the next
        ``apply_model()`` automatically restores non-block modules before
        the forward pass.
        """
        if self._vram_allocator is not None:
            self._vram_allocator.flush_all()
        self._hf_unet.to("cpu")
        self._non_blocks_on_device = False
        log.info("DiffPipeline: full UNet unload to CPU (VAE/CLIP VRAM reclaim).")

    def _restore_non_blocks(self, device: torch.device) -> None:
        """Move non-block UNet children back to device after full_unload_to_cpu().

        Called automatically by apply_model() when _non_blocks_on_device is False.
        """
        for name, child in self._hf_unet.named_children():
            if name in self._block_paths or any(
                p.startswith(name + ".") for p in self._block_paths
            ):
                continue
            child.to(device=device)
        self._non_blocks_on_device = True
        log.debug("DiffPipeline: non-block modules restored to %s.", device)

    def _reset_lru_offload(self) -> None:
        """Unregister all LRU blocks and destroy the allocator references.

        Called when the UNet structure changes (LoRA swap) so that the next
        ``apply_model()`` rebuilds with fresh parameter references.
        """
        try:
            from ldm_patched.modules import model_management as _mm
            _mm.unregister_vram_pressure_hook(self.flush_lru_to_cpu)
        except Exception:
            pass
        if self._vram_allocator is not None:
            self._vram_allocator.free_reservation("inference_headroom")
            for name in list(self._lru_block_names):
                self._vram_allocator.unregister(name)
        self._lru_block_names.clear()
        self._vram_allocator = None
        self._lru_ready = False
        self._non_blocks_on_device = False

    # ------------------------------------------------------------------
    # Compile-region installation (called once from _setup_auto_offload)
    # ------------------------------------------------------------------

    def _install_compile_regions(self) -> None:
        """Compile each UNet block individually with torch.compile.

        Called once from _setup_auto_offload (first partition only).
        After this call, re-partitioning (_reset_auto_offload +
        _setup_auto_offload) only repositions pre/post hooks — the
        _regions_installed guard prevents any recompilation.

        Note: torch.compiler.nested_compile_region was attempted here but
        is not usable — it generates get_attr nodes with dotted paths
        (e.g. repeated_subgraph5.repeated_subgraph0) that inductor's
        aot_autograd cannot resolve on nn.Module hierarchies.  Per-block
        torch.compile is the reliable alternative: each block is compiled
        separately and replaced via setter(), giving the same stable graph
        cache across re-partitions.

        No-op after the first call (_regions_installed guard).
        """
        if self._regions_installed:
            return

        # Set early so an exception in torch.compile cannot cause re-entry.
        self._regions_installed = True

        n_compiled = 0
        for path, setter, module in self._iter_unet_blocks():
            log.debug(
                "DiffPipeline [compile-regions]: compiling block '%s'", path
            )
            setter(torch.compile(module, mode="default", fullgraph=False))
            n_compiled += 1

        log.info(
            "DiffPipeline [compile-regions]: %d blocks compiled (per-block).",
            n_compiled,
        )

    # ------------------------------------------------------------------
    # Phase 5 — LoRA synchronisation
    # ------------------------------------------------------------------

    def _remove_lora_adapters(self) -> None:
        """Delete all PEFT adapters previously loaded onto the HF UNet."""
        for name, _ in self._active_adapters:
            try:
                self._hf_unet.delete_adapter(name)
            except Exception:
                pass
        self._active_adapters.clear()

    def _sync_lora(self) -> None:
        """Sync ldm LoRA patches → PEFT adapters on the HF UNet.

        Called at the top of every ``apply_model()`` invocation.  No-ops when
        ``unet_patcher.patches_uuid`` has not changed since the last call.

        Each "depth" in the stacked patch list becomes one named PEFT adapter
        (``forge_lora_0``, ``forge_lora_1``, …).  After loading all adapters
        ``set_adapters()`` applies the per-adapter strength weights.
        """
        from ldm_patched.modules.weight_adapter.lora import LoRAAdapter

        # The sampler clones the unet_patcher and adds LoRA patches to the clone,
        # giving it a new patches_uuid.  self.unet_patcher still points to the
        # original (unpatched) patcher whose uuid never changes.  Use
        # model.current_patcher (set by ModelPatcher.pre_run() to the active clone)
        # so we see the real patch state.
        # forge_objects.unet is updated by LoRA loading (networks.py clones + patches it).
        # self.unet_patcher is the original at init time and never gets LoRA patches.
        # current_patcher is only set by ModelPatcher.pre_run(), which is NOT called in
        # the k-diffusion sampler path — so we cannot rely on it.
        # Reading forge_objects.unet directly gives us the live, LoRA-patched patcher.
        forge_objects = getattr(self.sd_model, "forge_objects", None)
        active_patcher = (
            forge_objects.unet
            if forge_objects is not None and getattr(forge_objects, "unet", None) is not None
            else self.unet_patcher
        )

        patches = getattr(active_patcher, "patches", {})
        patches_uuid = getattr(active_patcher, "patches_uuid", None)

        log.debug(
            "[_sync_lora] active_patcher id=%d patches_keys=%d patches_uuid=%s synced_uuid=%s same=%s",
            id(active_patcher), len(patches), patches_uuid, self._synced_patches_uuid,
            patches_uuid == self._synced_patches_uuid,
        )

        if patches_uuid == self._synced_patches_uuid:
            return   # nothing changed

        self._remove_lora_adapters()
        self._synced_patches_uuid = patches_uuid
        # LoRA adapter changes invalidate any compiled graph — force recompile.
        # LRU hooks are NOT reset: PEFT load_lora_adapter / delete_adapter only
        # adds/removes adapter sub-layers inside the block modules; it never
        # replaces the DownBlock2D / MidBlock2D / UpBlock2D objects themselves,
        # so the pre-hooks installed on those objects remain valid.
        self._compiled = False

        if not patches:
            return

        max_depth = max(len(v) for v in patches.values())

        for depth in range(max_depth):
            state_dict: dict = {}
            network_alphas: dict = {}
            adapter_strength = None

            for ldm_key, patch_list in patches.items():
                if depth >= len(patch_list):
                    continue
                strength_patch, adapter, _strength_model, _offset, _fn = patch_list[depth]
                if not isinstance(adapter, LoRAAdapter):
                    continue
                hf_key = self._ldm_to_hf.get(ldm_key)
                if hf_key is None:
                    continue

                lora_up    = adapter.weights[0]  # lora_up.weight   → PEFT lora_B  [out, r]
                lora_down  = adapter.weights[1]  # lora_down.weight → PEFT lora_A  [r, in]
                alpha      = adapter.weights[2]  # scalar or None
                # weights[3] = mid (tucker), weights[4] = dora_scale, weights[5] = reshape
                dora_scale = adapter.weights[4]  # [1, out_dim] or None (DoRA magnitude)
                r = lora_down.shape[0]
                alpha_val = float(alpha) if alpha is not None else float(r)

                module_path = hf_key[: -len(".weight")]
                state_dict[f"{module_path}.lora_A.weight"] = lora_down
                state_dict[f"{module_path}.lora_B.weight"] = lora_up
                network_alphas[module_path] = alpha_val

                # DoRA: add magnitude vector so PEFT detects use_dora=True and applies
                # weight decomposition.  Kohya/A1111 stores it as [1, out_dim]; squeeze
                # to [out_dim] which is the shape PEFT's lora_magnitude_vector expects.
                if dora_scale is not None:
                    mag = dora_scale.squeeze(0) if dora_scale.dim() > 1 and dora_scale.shape[0] == 1 else dora_scale
                    state_dict[f"{module_path}.lora_magnitude_vector"] = mag

                if adapter_strength is None:
                    adapter_strength = float(strength_patch)

            if not state_dict:
                continue

            adapter_name = f"forge_lora_{depth}"
            self._hf_unet.load_lora_adapter(
                state_dict,
                network_alphas=network_alphas,
                adapter_name=adapter_name,
                low_cpu_mem_usage=False,
            )


            self._active_adapters.append((adapter_name, adapter_strength or 1.0))

        if self._active_adapters:
            names = [n for n, _ in self._active_adapters]
            weights = [w for _, w in self._active_adapters]
            self._hf_unet.set_adapters(names, weights)
            log.info("DiffPipeline: activated %d LoRA adapter(s): %s", len(names), names)
    # ------------------------------------------------------------------
    # Public interface (mirrors BaseModel.apply_model signature)
    # ------------------------------------------------------------------

    def _maybe_setup_tensor_core_opts(self, device: "torch.device", unet_dtype: "torch.dtype") -> None:
        """Detect GPU Tensor Core capability and apply compatible optimisations once.

        Called lazily on the first ``apply_model()`` after device placement so
        the compute device is known.

        Actions taken (each guarded so it's only applied when appropriate):

        * **Ampere+ (sm_80+) with fp32 UNet** — enables TF32 for matmul and
          cuDNN convolutions.  TF32 uses Tensor Cores internally while keeping
          fp32 inputs/outputs, so existing precision is preserved at higher
          throughput.

        * **Volta+ (sm_70+) with fp16/bf16 UNet** — stores the half-precision
          dtype so ``apply_model()`` wraps the UNet forward in
          ``torch.autocast``.  The UNet weights are already in the target dtype;
          the autocast context ensures *all* intermediate ops (time-step
          embedding, norms, etc.) also stay in half-precision and hit the TC
          path instead of silently widening to fp32.

        SDXL channel sizes (320 / 640 / 1280) are all divisible by 8, and the
        attention head-dim is 64 — both satisfy the minimum alignment required
        for fp16/bf16 Tensor Core GEMMs, so no padding is needed.
        """
        if self._tc_ready:
            return
        self._tc_ready = True

        # --- MPS path: enable fp16 autocast to minimise memory and reduce swap ---
        # torch.autocast on MPS keeps intermediate activations in float16 instead of
        # silently widening to float32, halving the active-memory footprint and
        # reducing the chance that macOS swaps UNet tensors to the unified memory
        # pool's slow region.  We guard with is_autocast_available() so this is
        # skipped on PyTorch builds that do not yet expose MPS autocast (pre-2.3).
        if device.type == "mps":
            mps_autocast_ok = False
            try:
                mps_autocast_ok = torch.amp.autocast_mode.is_autocast_available("mps")
            except Exception:
                pass  # older PyTorch — attribute not present

            if mps_autocast_ok:
                # Always prefer float16 on MPS regardless of the stored UNet dtype.
                # A float32 UNet still benefits: autocast casts eligible ops to fp16
                # so peak live memory is cut roughly in half, reducing swap pressure.
                self._autocast_dtype = torch.float16

                # Log the recommended memory ceiling so the user can see the budget.
                try:
                    rec_mem_bytes = torch.mps.recommended_max_memory()
                    rec_mem_gib = rec_mem_bytes / (1024 ** 3)
                    mem_note = f"  (torch.mps.recommended_max_memory = {rec_mem_gib:.1f} GiB)"
                except Exception:
                    mem_note = ""

                msg = (
                    f"MPS autocast enabled — UNet forward wrapped in "
                    f"torch.autocast(device_type='mps', dtype=float16) "
                    f"to keep activations in fp16 and minimise swap{mem_note}"
                )
                print(f"\n[DiffPipeline] {msg}\n")
                log.info("DiffPipeline: %s", msg)
            else:
                self._autocast_dtype = None
                log.info(
                    "DiffPipeline: MPS autocast not available on this PyTorch build "
                    "(torch.amp.autocast_mode.is_autocast_available('mps') = False) "
                    "— running without autocast."
                )
            return

        if device.type != "cuda" or not torch.cuda.is_available():
            self._autocast_dtype = None
            return

        major, minor = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
        msgs: list[str] = []

        # --- Ampere+ (sm_80+): TF32 for fp32 UNet ---
        if major >= 8 and unet_dtype == torch.float32:
            if not torch.backends.cuda.matmul.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                msgs.append("matmul TF32 enabled  (torch.backends.cuda.matmul.allow_tf32 = True)")
            if not torch.backends.cudnn.allow_tf32:
                torch.backends.cudnn.allow_tf32 = True
                msgs.append("cuDNN conv TF32 enabled  (torch.backends.cudnn.allow_tf32 = True)")

        # --- Volta+ (sm_70+): autocast for fp16 / bf16 UNet ---
        if major >= 7 and unet_dtype in (torch.float16, torch.bfloat16):
            self._autocast_dtype = unet_dtype
            msgs.append(
                f"UNet forward wrapped in torch.autocast({unet_dtype})  "
                f"— keeps time-embed / norm ops in half-precision for consistent TC usage"
            )
        else:
            self._autocast_dtype = None

        # --- Alignment note ---
        if major >= 7:
            msgs.append(
                "SDXL channel dims 320/640/1280 are ÷8 and head-dim=64 is ÷8  "
                "— fp16/bf16 TC alignment satisfied ✓"
            )

        if msgs:
            lines = "\n".join(f"  • {m}" for m in msgs)
            print(
                f"\n[DiffPipeline] Tensor Core optimisations applied"
                f" ({gpu_name}, sm_{major}{minor}, unet_dtype={unet_dtype}):\n"
                f"{lines}\n"
            )
            log.info(
                "DiffPipeline: TC opts (%s sm_%d%d dtype=%s): %s",
                gpu_name, major, minor, unet_dtype, "; ".join(msgs),
            )
        else:
            log.info(
                "DiffPipeline: No TC optimisations applied"
                " (device=%s, sm_%d%d, unet_dtype=%s — no matching rule)",
                gpu_name, major, minor, unet_dtype,
            )

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        """Forward pass through the HF Diffusers UNet.

        Accepts the same arguments as ``BaseModel.apply_model()`` so the
        sampler calling convention is unchanged.
        """
        if transformer_options is None:
            transformer_options = {}

        if self._clip_attn_norm:
            transformer_options["clip_attn_norm"] = True

        log.debug("[DiffPipeline] apply_model called — sigma=%s, x.shape=%s", t, x.shape)

        sigma = t

        # --- 1. Sigma preconditioning (identical to ldm _apply_model) ---
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        # Integer timestep for HF UNet
        timestep = self.model_sampling.timestep(sigma).float()

        # --- 2. Device placement / offload setup ---
        device = x.device
        if self._auto_offload:
            # Partition UNet into Group A (device) / Group B (CPU) with hooks.
            # Done lazily on the first call; re-checks VRAM pressure every step
            # and re-partitions when it shifts significantly (mirrors reForge's
            # dynamic load_models_gpu budget).  Because _install_compile_regions
            # is a no-op after the first call, re-partition only moves hooks.
            if not self._auto_offload_ready:
                self._setup_auto_offload(device)
            elif device.type == "cuda" and self._vram_at_partition > 0:
                _stats = torch.cuda.memory_stats(device)
                _cuda_free, _ = torch.cuda.mem_get_info(device)
                _free_now: int = int(
                    _cuda_free
                    + _stats["reserved_bytes.all.current"]
                    - _stats["active_bytes.all.current"]
                )
                _delta_mb = (_free_now - self._vram_at_partition) / (1024 ** 2)
                log.debug(
                    "DiffPipeline VRAM pressure: free_now=%.0f MB  "
                    "at_partition=%.0f MB  delta=%+.0f MB",
                    _free_now / (1024 ** 2),
                    self._vram_at_partition / (1024 ** 2),
                    _delta_mb,
                )
                if abs(_free_now - self._vram_at_partition) > 512 * 1024 * 1024:
                    log.info(
                        "DiffPipeline: VRAM pressure shifted %+.0f MB — "
                        "re-partitioning (Group A/B boundary will move).",
                        _delta_mb,
                    )
                    self._reset_auto_offload()
                    self._setup_auto_offload(device)
        elif self._sequential_offload:
            # Install per-block accelerate hooks on the first call.
            # After that the hooks handle GPU↔CPU movement automatically;
            # the UNet itself stays on CPU between blocks.
            if not self._seq_hooks_installed:
                self._install_sequential_offload_hooks(device)
        else:
            # Default path: whole-model on device, or LRU block loading when
            # VRAM is insufficient (mirrors reForge's dynamic model loading).
            if self._lru_ready:
                # After full_unload_to_cpu() the non-block parts (conv_in,
                # time_embedding, …) were moved to CPU.  Restore them before
                # the forward pass; block activation is handled by the hooks.
                if not self._non_blocks_on_device:
                    self._restore_non_blocks(device)
            elif self._should_use_lru(device, x):
                self._setup_lru_offload(device, x)
            else:
                # Whole-model placement: move to compute device if needed.
                if next(self._hf_unet.parameters()).device != device:
                    self._hf_unet.to(device=device)

                # MPS-specific optimizations — applied once after the UNet lands on device.
                if device.type == "mps" and not self._mps_optimized:
                    if hasattr(self._hf_unet, "enable_attention_slicing"):
                        self._hf_unet.enable_attention_slicing()
                        log.info("DiffPipeline: MPS — enabled attention slicing")
                    self._mps_optimized = True


        # --- 2b. Tensor Core setup (runs once per session) ---
        dtype = next(self._hf_unet.parameters()).dtype
        self._maybe_setup_tensor_core_opts(device, dtype)

        # Dtype from HF UNet (works regardless of current parameter device).
        xc = xc.to(dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype)

        # --- 3. Conditioning bridge ---
        encoder_hidden_states = c_crossattn  # (B, seq, 2048)

        # Resolve text_embeds (1280-dim CLIP-G pooled output):
        #   Priority 1 — adm_text_embeds: set by SDXL.extra_conds() (ldm non-hijack path)
        #                 or by cond_from_a1111_to_patched_ldm (hijack path, after fix).
        #   Priority 2 — y: raw pooled output stored as model_conds["y"] by
        #                 cond_from_a1111_to_patched_ldm (shape B×1280 in diffusers path,
        #                 shape B×2816 Fourier-embedded in pure ldm path — take [:1280]).
        #   Fallback    — zeros: warns loudly; causes visible quality degradation.
        adm_text_embeds = kwargs.get("adm_text_embeds", None)
        adm_time_ids    = kwargs.get("adm_time_ids", None)
        y               = kwargs.get("y", None)

        if adm_text_embeds is not None:
            text_embeds = adm_text_embeds.to(dtype)
        elif y is not None:
            text_embeds = y[:, :1280].to(dtype)
        else:
            log.warning(
                "[DiffPipeline] No pooled conditioning found "
                "(adm_text_embeds/y both absent) — using zero fallback. "
                "Quality will be significantly degraded."
            )
            text_embeds = torch.zeros(x.shape[0], 1280, device=device, dtype=dtype)

        # Resolve time_ids ([orig_h, orig_w, crop_top, crop_left, target_h, target_w]):
        #   Priority 1 — adm_time_ids: set by SDXL.extra_conds() (ldm non-hijack path).
        #   Priority 2 — derive from latent shape; correct for standard (uncropped) generation.

        if adm_time_ids is not None:
            time_ids = adm_time_ids.to(dtype)
        else:
            _h_px = x.shape[2] * 8
            _w_px = x.shape[3] * 8
            time_ids = torch.tensor(
                [[_h_px, _w_px, 0, 0, _h_px, _w_px]],
                device=device, dtype=dtype,
            ).expand(text_embeds.shape[0], -1)

        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        # --- 3b. CLIP attention norm soft-clamping (outside compiled graph) ---
        # CLIP token embeddings have an ~8x natural norm variance (empirically
        # 1.2–212 with mean ~26).  High-norm tokens attract disproportionate
        # cross-attention solely because of embedding magnitude, not semantics.
        # Clamp any token whose L2 norm exceeds 3× the batch mean to that
        # threshold; tokens at or below the threshold are left unchanged.
        # Done here (not inside ForgeAttnProcessor) to avoid dynamo graph breaks.
        if self._clip_attn_norm and encoder_hidden_states is not None:
            norms_kd = encoder_hidden_states.norm(dim=-1, keepdim=True)
            threshold = norms_kd.mean() * 3.0
            encoder_hidden_states = (
                encoder_hidden_states / norms_kd.clamp(min=threshold) * threshold
            )

        # --- 4. ControlNet residual mapping ---
        # ldm: control["input"] is a list that gets pop()d (reverse order).
        # HF:  down_block_additional_residuals is a tuple in forward order.
        down_block_residuals: Optional[tuple] = None
        mid_block_residual = None

        if control is not None:
            raw_input = control.get("input", [])
            if raw_input:
                # Reverse the list (ldm pops from end = first block first).
                reversed_residuals = [r for r in reversed(raw_input) if r is not None]
                if reversed_residuals:
                    down_block_residuals = tuple(reversed_residuals)

            raw_middle = control.get("middle", [])
            if raw_middle and raw_middle[0] is not None:
                mid_block_residual = raw_middle[0]

        # --- 5. LoRA sync (Phase 5) ---
        self._sync_lora()

        # --- 5b. Diffusers optimization injection ---
        # Override apply_diffusers_optimization() or monkey-patch it to inject
        # custom Diffusers optimizations (xformers, flash-attn, torch.compile, etc.).
        # Called after device placement and LoRA sync, before the forward pass.
        self.apply_diffusers_optimization(self._hf_unet)

        # --- 6. Whole-model torch.compile (single-device path, --forge-diffusers-compile only) ---
        # Compile is safe when the UNet runs in bf16 or fp32 — both have the
        # same fp32 exponent range so inductor kernel fusion cannot overflow.
        # fp16 is excluded: its narrow exponent range (~65504 max) causes
        # silent NaN when inductor fuses ops, as observed with NoobAI XL.
        # On offload paths blocks are spread across devices so whole-model compile
        # does not apply. LRU path also excluded (block-level param redirects
        # invalidate compiled graphs).
        if (
            self._compile
            and not self._compiled
            and not self._auto_offload
            and not self._sequential_offload
            and not self._lru_ready
        ):
            unet_dtype = next(self._hf_unet.parameters()).dtype
            if unet_dtype == torch.float16:
                log.info(
                    "DiffPipeline: torch.compile skipped — UNet is in fp16 "
                    "(narrow exponent range risks NaN under inductor fusion). "
                    "Switch to bf16/fp32 hardware to enable compile.",
                )
                self._compiled = True  # mark so we don't re-check every step
            else:
                # Point the inductor cache at a persistent, model-specific directory
                # so compiled kernel artifacts survive process restarts.  The cache
                # is keyed by the WebUI model hash; a device fingerprint check voids
                # stale entries (different GPU / PyTorch version / CUDA).
                # LoRA hotswap is handled automatically: inductor creates separate
                # cache entries for each distinct graph (base model vs. each LoRA
                # combination) within the same directory.
                from diff_pipeline import compile_cache as _cc
                _model_hash = getattr(self.sd_model, "sd_model_hash", None)
                _cache_dir = _cc.activate(_model_hash, device)

                _cache_note = (
                    f"Compiled artifacts cached at:\n    {_cache_dir}\n"
                    f"    Subsequent loads of the same model will skip most of this compile."
                    if _cache_dir else
                    "No persistent cache — artifacts will not survive process restart."
                )
                log.warning(
                    "DiffPipeline: torch.compile is about to begin (UNet dtype=%s). "
                    "The UI will appear frozen for 1–3 minutes while the inductor "
                    "captures and optimises the compute graph — this is normal.\n%s",
                    unet_dtype,
                    _cache_note,
                )
                print(
                    f"\n[DiffPipeline] *** torch.compile starting (dtype={unet_dtype}) ***\n"
                    f"    The UI will be unresponsive for ~1-3 min during graph capture.\n"
                    f"    {_cache_note}\n"
                )
                if device == "mps":
                    # MPS don't have max auto tune, inductor use as default
                    self._hf_unet = torch.compile(
                        self._hf_unet, mode="default", fullgraph=False
                    )
                else:
                    self._hf_unet = torch.compile(
                        self._hf_unet, mode="default", fullgraph=False
                    )
                self._compiled = True
                log.warning("DiffPipeline: torch.compile finished — graph cached, normal speed resumes.")
                print("\n[DiffPipeline] *** torch.compile done — generation will proceed normally. ***\n")

        # Debug: log added_cond_kwargs shapes before the forward pass.
        log.debug(
            "[DiffPipeline] added_cond_kwargs keys=%s text_embeds.shape=%s time_ids.shape=%s",
            list(added_cond_kwargs.keys()) if added_cond_kwargs is not None else "None",
            tuple(added_cond_kwargs["text_embeds"].shape) if added_cond_kwargs and "text_embeds" in added_cond_kwargs else "absent",
            tuple(added_cond_kwargs["time_ids"].shape)    if added_cond_kwargs and "time_ids"    in added_cond_kwargs else "absent",
        )

        # --- 6b. cuda-stream: run HF UNet forward on the dedicated compute stream ---
        # Mirrors the modules_forge/stream.py approach used by the ldm path.
        # When --cuda-stream is active, modules_forge.stream.current_stream holds
        # a pre-warmed torch.cuda.Stream.  Wrapping the forward pass in that
        # stream context lets the CUDA scheduler overlap this compute with any
        # async memory transfers happening on the mover_stream.
        _stream_mod = _get_stream_module()
        _use_stream = (
            _stream_mod is not None
            and getattr(_stream_mod, "using_stream", False)
            and getattr(_stream_mod, "current_stream", None) is not None
        )

        def _unet_forward():
            return self._hf_unet(
                sample=xc,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_residuals,
                mid_block_additional_residual=mid_block_residual,
                cross_attention_kwargs={"transformer_options": transformer_options},
                return_dict=False,
            )

        # Wrap forward in autocast when _maybe_setup_tensor_core_opts decided it
        # would help (fp16/bf16 UNet on Volta+).  This keeps time-embed and norm
        # ops in half-precision so they hit the TC path rather than widening to fp32.
        def _run_forward():
            if self._autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=self._autocast_dtype):
                    return _unet_forward()
            return _unet_forward()

        # When LRU is active, wrap the forward pass in tracking_context() so that
        # large intermediate tensors (skip-connection activations) are tracked via
        # weakref.  This gives _effective_free() real-time accounting: skip tensors
        # from the down pass show as reserved VRAM while the up pass evicts/loads
        # blocks, preventing over-commit.  The static inference_headroom reservation
        # remains as a floor for when TorchDispatchMode is unavailable.
        _tracking_ctx = (
            self._vram_allocator.tracking_context()
            if self._lru_ready and self._vram_allocator is not None
            else contextlib.nullcontext()
        )

        if _use_stream:
            with torch.cuda.stream(_stream_mod.current_stream), _tracking_ctx:
                unet_output = _run_forward()
            # Synchronise so downstream ops on the default stream see the result.
            _stream_mod.current_stream.synchronize()
        else:
            with _tracking_ctx:
                unet_output = _run_forward()

        model_output = unet_output[0].float()

        # Debug: check for NaN in UNet output.
        if torch.isnan(model_output).any():
            nan_count = torch.isnan(model_output).sum().item()
            valid = model_output[~torch.isnan(model_output)]
            log.warning(
                "[DiffPipeline] NaN in UNet output! nan_count=%d  valid_min=%.3f  valid_max=%.3f",
                nan_count,
                valid.min().item() if valid.numel() > 0 else float("nan"),
                valid.max().item() if valid.numel() > 0 else float("nan"),
            )
        else:
            log.debug(
                "[DiffPipeline] UNet output OK: min=%.3f  max=%.3f",
                model_output.min().item(), model_output.max().item(),
            )

        # --- 7. Sigma postconditioning (identical to ldm _apply_model) ---
        result = self.model_sampling.calculate_denoised(sigma, model_output, x)

        # Debug: check for NaN after calculate_denoised.
        nan_result = torch.isnan(result).sum().item()
        if nan_result > 0:
            print(
                f"[DiffPipeline] NaN after calculate_denoised! nan={nan_result}  "
                f"sigma={sigma.item() if sigma.numel()==1 else float(sigma.max())}  "
                f"x_nan={torch.isnan(x).sum().item()}"
            )

        # --- 8. Offload HF UNet back to CPU if requested ---
        if self._offload:
            # Use non_blocking when cuda-stream is active so the CPU←GPU copy
            # is enqueued asynchronously and doesn't stall the compute stream.
            self._hf_unet = self._hf_unet.to(device="cpu", non_blocking=_use_stream)

            # --pin-shared-memory: pin the offloaded weights to page-locked
            # (pinned) CPU memory so the next GPU upload uses DMA and avoids
            # a kernel-to-user copy.  Mirrors model_patcher.py's usage of
            # PIN_SHARED_MEMORY.  We only pin once — if the parameters are
            # already pinned this is a no-op.
            if _get_pin_shared_memory():
                for p in self._hf_unet.parameters():
                    if not p.data.is_pinned():
                        p.data = p.data.pin_memory()

        return result

    # ------------------------------------------------------------------
    # Diffusers optimization injection point
    # ------------------------------------------------------------------

    def get_pipeline(self) -> "UNet2DConditionModel":
        """Return the underlying HF ``UNet2DConditionModel``.

        Use this to apply Diffusers-level optimizations before inference,
        for example::

            pipeline = diff_pipeline.get_pipeline()
            pipeline.enable_xformers_memory_efficient_attention()
            # or: pipeline.enable_flash_attn_2(), torch.compile(pipeline), etc.

        To future developer: DON'T DELETE THIS!
        UNTIL I FIND A WAY PROPERLY LABEL WHAT THE HECK THE HF NET TYPE IS
        """
        return self._hf_unet

    def apply_diffusers_optimization(self, _hf_unet: "UNet2DConditionModel") -> None:
        """Stub — override or monkey-patch this to inject custom Diffusers optimizations.

        Called once per ``apply_model()`` invocation, after the UNet has been
        moved to the correct compute device and LoRA adapters have been synced,
        but before the HF UNet forward pass.  The ``hf_unet`` argument is the
        same object as ``self.get_pipeline()``.

        Example usage (monkey-patch from outside)::

            from diff_pipeline.pipeline import DiffPipeline

            def my_opt(self, unet):
                unet.enable_xformers_memory_efficient_attention()

            DiffPipeline.apply_diffusers_optimization = my_opt

        Or subclass ``DiffPipeline`` and override this method directly.
        """
        #EXPERIEMENT: TaylorSeer Cache


        pass  # no-op by default

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True once the HF UNet is wired and apply_model() is usable."""
        return self._hf_unet is not None

    def __repr__(self) -> str:
        status = "active" if self.is_active() else "scaffold-only"
        return f"DiffPipeline({status})"
