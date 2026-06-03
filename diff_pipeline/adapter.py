"""
diff_pipeline/adapter.py

DiffusersModelAdapter — wraps a diffusers pipeline (e.g. StableDiffusionXLPipeline)
in an object that satisfies the ldm-style interface expected by stable-diffusion-webui-reForge.

The webui accesses dozens of attributes on `shared.sd_model`. When the diffusers path
hijack replaces normal ldm loading, the raw pipeline object lacks all of these. This
adapter bridges the gap so the UI and processing code don't crash.

Generation path
---------------
All actual denoising is delegated to the underlying diffusers pipeline via
`apply_model`.  The VAE and text-encoding methods wrap the diffusers components
so that `processing.py` (which calls `encode_first_stage`, `decode_first_stage`,
`get_learned_conditioning`) works without modification.

Usage
-----
Wrap the loaded pipeline before returning it from a path-hijack loader:

    from diff_pipeline.adapter import DiffusersModelAdapter
    pipe = StableDiffusionXLPipeline.from_single_file(...)
    return DiffusersModelAdapter(pipe, checkpoint_info, model_type="sdxl")
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch


class _FakeEmaScope:
    """No-op context manager satisfying the sd_model.ema_scope() interface."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


class _FakeModel:
    """Minimal stand-in for sd_model.model (the ldm DiffusionWrapper).

    processing.py reads .conditioning_key and extensions read .diffusion_model.
    After the UnetPatcher is built we point diffusion_model at the same
    DiffusersUnetModel wrapper so both paths see the same object.
    """
    def __init__(self, pipe):
        self._pipe = pipe
        self.conditioning_key = "crossattn"   # SDXL always uses cross-attention

    @property
    def diffusion_model(self):
        # Prefer the patcher's model wrapper if already built (set by adapter init)
        return getattr(self, "_diffusion_model", getattr(self._pipe, "unet", None))

    @diffusion_model.setter
    def diffusion_model(self, val):
        self._diffusion_model = val


class _FakeCondStageModel:
    """Thin wrapper around diffusers text encoders that satisfies the
    cond_stage_model interface expected by processing.py and extensions.

    The token-counting methods (process_texts / get_target_prompt_token_count)
    use the pipeline's tokenizer so the UI token counter shows accurate counts
    instead of crashing.  The standard CLIP chunk length is 75 tokens (77
    including BOS/EOS); SDXL uses the same limit.
    """

    CHUNK_LENGTH = 75  # standard CLIP tokens per chunk (excl. BOS/EOS)

    def __init__(self, pipe):
        self._pipe = pipe

    # ---- token counting --------------------------------------------------

    def process_texts(self, texts: list[str]):
        """Tokenise *texts* and return (batch_chunks, max_token_count).

        The UI only needs the token count; we return a minimal structure so
        the caller doesn't crash.  Each 'chunk' is a plain object with a
        ``tokens`` attribute.
        """
        tokenizer = getattr(self._pipe, "tokenizer", None)
        token_count = 0

        class _Chunk:
            def __init__(self, tokens):
                self.tokens = tokens
                self.multipliers = [1.0] * len(tokens)

        batch_chunks = []
        for text in texts:
            if tokenizer is not None:
                try:
                    ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                    ).input_ids
                    n = len(ids)
                except Exception:
                    n = 0
            else:
                # rough estimate: ~0.75 tokens per char
                n = max(0, int(len(text) * 0.75))

            token_count = max(token_count, n)
            batch_chunks.append([_Chunk(list(range(n)))])

        return batch_chunks, token_count

    def get_target_prompt_token_count(self, token_count: int) -> int:
        """Return the next chunk-aligned token ceiling for the counter display."""
        return math.ceil(max(token_count, 1) / self.CHUNK_LENGTH) * self.CHUNK_LENGTH

    # ---- tokenize --------------------------------------------------------

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Return raw token-id lists for *texts* using pipe.tokenizer."""
        tokenizer = getattr(self._pipe, "tokenizer", None)
        result = []
        for text in texts:
            if tokenizer is not None:
                try:
                    ids = tokenizer(text, add_special_tokens=False,
                                    truncation=False).input_ids
                    result.append(ids)
                    continue
                except Exception:
                    pass
            result.append([])
        return result

    # ---- textual inversion support ---------------------------------------

    def encode_embedding_init_text(self, init_text: str, nvpt: int) -> "torch.Tensor":
        """Return the token embedding(s) for *init_text*, truncated/padded to *nvpt*.

        Used by textual_inversion to determine the expected embedding shape.
        For SDXL (CLIP-L 768 + CLIP-G 1280 = 2048 dims) we replicate what the
        real GeneralConditioner does: concatenate embeddings from both encoders.
        """
        import torch
        results = []

        # CLIP-L (text_encoder, tokenizer)
        te1 = getattr(self._pipe, "text_encoder", None)
        tok1 = getattr(self._pipe, "tokenizer", None)
        if te1 is not None and tok1 is not None:
            try:
                enc = tok1(init_text, return_tensors="pt",
                           add_special_tokens=True, truncation=True,
                           max_length=tok1.model_max_length)
                ids = enc.input_ids.to(next(te1.parameters()).device)
                # token embeddings before transformer
                emb = te1.text_model.embeddings.token_embedding(ids)  # (1, T, 768)
                # take up to nvpt non-special tokens
                emb = emb[:, 1:nvpt + 1, :]  # skip BOS
                results.append(emb)
            except Exception:
                results.append(torch.zeros(1, nvpt, 768))

        # CLIP-G (text_encoder_2, tokenizer_2)
        te2 = getattr(self._pipe, "text_encoder_2", None)
        tok2 = getattr(self._pipe, "tokenizer_2", None)
        if te2 is not None and tok2 is not None:
            try:
                enc = tok2(init_text, return_tensors="pt",
                           add_special_tokens=True, truncation=True,
                           max_length=tok2.model_max_length)
                ids = enc.input_ids.to(next(te2.parameters()).device)
                emb = te2.text_model.embeddings.token_embedding(ids)  # (1, T, 1280)
                emb = emb[:, 1:nvpt + 1, :]
                results.append(emb)
            except Exception:
                results.append(torch.zeros(1, nvpt, 1280))

        if not results:
            # Fallback: return a zero tensor with SDXL combined dim
            return torch.zeros(1, nvpt, 2048)

        return torch.cat(results, dim=2).squeeze(0)  # (nvpt, 768+1280) — matches ldm GeneralConditioner shape

    # ---- conditioning ----------------------------------------------------

    def encode(self, tokens):
        raise NotImplementedError(
            "DiffusersModelAdapter: cond_stage_model.encode() not yet implemented. "
            "Use get_learned_conditioning() on the adapter instead."
        )

    def get_learned_conditioning(self, text: list[str]):
        # Returns a dict {'crossattn': (B,seq,hidden), 'vector': (B,pooled)}
        # so prompt_parser can index it per-prompt and reconstruct_*_batch
        # stacks them back into a DictWithShape for cond_from_a1111_to_patched_ldm.
        return _encode_prompts(self._pipe, text)

    # Extensions may try to iterate over embedders
    @property
    def embedders(self):
        return []


class _DiffusersUnetModel:
    """Minimal model container that ModelPatcher / UnetPatcher expect on .model.

    ModelPatcher.__init__ reads several attributes from the wrapped model object
    (device, model_loaded_weight_memory, lowvram_patch_counter, model_lowvram,
    current_weight_patches_uuid) and calls model.memory_required().  This class
    provides all of them so a real UnetPatcher can be constructed around the
    diffusers UNet2DConditionModel.
    """

    def __init__(self, diffusers_unet, device: torch.device, scheduler=None):
        self.diffusion_model = diffusers_unet   # accessed by forge extensions
        self.device = device

        # Attributes ModelPatcher.__init__ sets if missing — pre-set them so
        # the patcher doesn't have to patch the diffusers nn.Module directly.
        self.model_loaded_weight_memory = 0
        self.lowvram_patch_counter = 0
        self.model_lowvram = False
        self.current_weight_patches_uuid = None

        # Set by DiffusersModelAdapter after DiffPipeline construction.
        # When set, apply_model() delegates to DiffPipeline.apply_model() so
        # ControlNet mapping, ForgeAttnProcessor dispatch, and offload logic work.
        self._diff_pipeline = None

        # model_sampling — auto-detected from the diffusers UNet/scheduler config.
        # Provides timestep(sigma), calculate_input, calculate_denoised
        # which apply_model uses for correct sigma preconditioning.
        self.model_sampling = _build_model_sampling_from_pipe(diffusers_unet, scheduler=scheduler)
        self.model_config = _FakeModelConfig()
        self.latent_channels = 4

        # conditioning_key read by processing.py
        self.conditioning_key = "crossattn"

    def memory_required(self, input_shape, **kwargs):
        """Rough VRAM estimate: batch * channels * H * W * 2 bytes * 50 (unet factor).
        This will not use if user decide to use auto offload, that is handle by diffuser"""
        if not input_shape:
            return 0
        elems = 1
        for d in input_shape:
            elems *= d
        return elems * 2 * 50   # float16 bytes × empirical SDXL factor

    def state_dict(self):
        """Delegate to the underlying diffusers UNet so module_size() works."""
        return self.diffusion_model.state_dict()

    def extra_conds(self, **kwargs):
        """Build model_conds entries for the SDXL diffusers UNet.

        Called by encode_model_conds with params that include:
          cross_attn   — text encoder hidden states (from convert_cond)
          pooled_output — CLIP-G pooled embedding (from _encode_prompts)
          width / height / crop_w / crop_h / target_width / target_height
        """
        import ldm_patched.modules.conds as _conds
        out: dict = {}

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = _conds.CONDCrossAttn(cross_attn)

        pooled_output = kwargs.get("pooled_output", None)
        if pooled_output is not None:
            out["adm_text_embeds"] = _conds.CONDRegular(pooled_output)

            # Build the 6-element time_ids vector expected by SDXL.
            height        = kwargs.get("height", 1024)
            width         = kwargs.get("width", 1024)
            crop_h        = kwargs.get("crop_h", 0)
            crop_w        = kwargs.get("crop_w", 0)
            target_height = kwargs.get("target_height", height)
            target_width  = kwargs.get("target_width", width)
            time_ids = torch.tensor(
                [[height, width, crop_h, crop_w, target_height, target_width]],
                dtype=pooled_output.dtype,
                device=pooled_output.device,
            ).repeat(pooled_output.shape[0], 1)
            out["adm_time_ids"] = _conds.CONDRegular(time_ids)

        return out

    # ---- nn.Module compatibility (required by ModelPatcher) -----------------

    def named_modules(self, *args, **kwargs):
        """Delegate to the underlying diffusers UNet so ModelPatcher can iterate modules."""
        return self.diffusion_model.named_modules(*args, **kwargs)

    def modules(self, *args, **kwargs):
        """Delegate modules() iteration to the diffusers UNet (used by unpatch logic)."""
        return self.diffusion_model.modules(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.diffusion_model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.diffusion_model.parameters(*args, **kwargs)

    def to(self, *args, **kwargs):
        # With sequential CPU offload the UNet weights live on meta device and
        # are managed by diffusers hooks. Calling .to() on meta tensors raises
        # NotImplementedError, so skip the move when offload hooks are present.
        try:
            first_param = next(self.diffusion_model.parameters(), None)
        except StopIteration:
            first_param = None
        if first_param is None or first_param.device.type != "meta":
            self.diffusion_model.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.diffusion_model.train(mode)
        return self

    def eval(self):
        self.diffusion_model.eval()
        return self

    def get_dtype(self):
        """Return the dtype of the diffusers UNet (called by ModelPatcher.model_dtype)."""
        try:
            return next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ---- Generation interface (required by samplers.py) ---------------------

    def apply_model(self, x, t, c_crossattn=None, c_concat=None, **kwargs):
        """Forward pass called by samplers.py _calc_cond_batch.

        ``t`` arrives as a sigma tensor from the k-diffusion sampler.  The
        diffusers UNet expects integer timesteps (0–999), so we must:
          1. Scale the noisy input:   xc = calculate_input(sigma, x)
          2. Convert sigma → timestep: ts = model_sampling.timestep(sigma)
          3. Run the UNet:             eps = unet(xc, ts, ...)
          4. Recover denoised x:       x0 = calculate_denoised(sigma, eps, x)

        This mirrors DiffPipeline.apply_model() in pipeline.py exactly.
        """
        # Delegate to DiffPipeline when available — it handles ControlNet
        # mapping, ForgeAttnProcessor dispatch, LoRA sync, and offload modes.
        if self._diff_pipeline is not None:
            return self._diff_pipeline.apply_model(
                x, t,
                c_crossattn=c_crossattn,
                c_concat=c_concat,
                control=kwargs.get("control"),
                transformer_options=kwargs.get("transformer_options", {}),
                **{k: v for k, v in kwargs.items()
                   if k not in ("control", "transformer_options")},
            )

        sigma = t

        # --- sigma preconditioning (step 1 & 2) ----------------------------
        xc = self.model_sampling.calculate_input(sigma, x)
        timestep = self.model_sampling.timestep(sigma).float()

        # c_crossattn may arrive as a list of tensors — concatenate them
        if isinstance(c_crossattn, (list, tuple)):
            c_crossattn = torch.cat(c_crossattn, dim=1)

        # 'y' = pooled text embedding injected by cond_from_a1111_to_patched_ldm
        y = kwargs.pop("y", None)

        # Collect any pre-built added_cond_kwargs
        added_cond_kwargs = kwargs.pop("added_cond_kwargs", {}).copy()

        if y is not None and "text_embeds" not in added_cond_kwargs:
            added_cond_kwargs["text_embeds"] = y

        if "time_ids" not in added_cond_kwargs and "text_embeds" in added_cond_kwargs:
            # Build the standard SDXL 6-element time_ids from image dimensions.
            # x shape: (B, C, H_lat, W_lat) → pixel dims × 8
            batch = x.shape[0]
            h_px = x.shape[2] * 8
            w_px = x.shape[3] * 8
            time_ids = torch.tensor(
                [[h_px, w_px, 0, 0, h_px, w_px]],
                dtype=added_cond_kwargs["text_embeds"].dtype,
                device=added_cond_kwargs["text_embeds"].device,
            ).repeat(batch, 1)
            added_cond_kwargs["time_ids"] = time_ids

        # Ensure all inputs share the same dtype as the UNet weights
        model_dtype = self.get_dtype()
        xc = xc.to(dtype=model_dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype=model_dtype)
        added_cond_kwargs = {
            k: v.to(dtype=model_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in added_cond_kwargs.items()
        }

        # --- UNet forward (step 3) -----------------------------------------
        model_output = self.diffusion_model(
            xc,
            timestep,
            encoder_hidden_states=c_crossattn,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
        ).sample

        # --- recover denoised prediction (step 4) --------------------------
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    # ---- Latent pass-through (called on inner_model by KSampler) -----------

    def process_latent_in(self, latent):
        """No-op — diffusers handles latent scaling internally."""
        return latent

    def process_latent_out(self, latent):
        """No-op — diffusers handles latent scaling internally."""
        return latent

    # ---- Model object accessor (used for sigma schedule lookup) -------------

    def get_model_object(self, name: str):
        """Return named sub-objects; supports 'model_sampling' for sigma calc."""
        if name == "model_sampling":
            return self.model_sampling
        return getattr(self, name, None)

    def __call__(self, x, t, c_crossattn=None, **kwargs):
        """Forward pass — delegates to apply_model."""
        return self.apply_model(x, t, c_crossattn=c_crossattn, **kwargs)


def _build_model_sampling_from_pipe(diffusers_unet, scheduler=None):
    """Detect prediction type and beta schedule from a diffusers pipeline and
    return the matching ``ModelSamplingDiscrete`` subclass instance.

    This mirrors the logic in ``forge_loader.py`` / ``model_base.model_sampling()``
    which selects between ``EPS``, ``V_PREDICTION``, etc. based on the checkpoint's
    ``parameterization`` key — but reads equivalent information from diffusers
    config objects instead of ldm state-dict metadata.

    Prediction-type mapping
    -----------------------
    diffusers ``prediction_type``  →  ldm_patched prediction class
    ``"epsilon"``                  →  ``EPS``           (most SDXL / SD1.5 models)
    ``"v_prediction"``             →  ``V_PREDICTION``  (SD2.1, some SDXL variants)
    ``"sample"``                   →  ``X0``            (direct x₀ prediction, rare)

    The ``prediction_type`` is read from ``diffusers_unet.config`` first (most
    reliable for single-file checkpoints), then from ``scheduler.config`` as a
    fallback.

    Beta-schedule mapping
    ---------------------
    diffusers ``beta_schedule``    →  ldm_patched schedule name
    ``"scaled_linear"``            →  ``"linear"``  (both do linspace(√β₀,√β₁)²)
    ``"squaredcos_cap_v2"``        →  ``"cosine"``
    ``"linear"``                   →  ``"linear"``  (close enough for inference)

    ``beta_start``, ``beta_end``, ``num_train_timesteps``, and
    ``rescale_betas_zero_snr`` are also transplanted when present.
    """
    from ldm_patched.modules.model_sampling import ModelSamplingDiscrete, EPS, V_PREDICTION, X0

    # --- 1. Prediction type -------------------------------------------------
    prediction_type = "epsilon"   # safe default (most checkpoints)

    # UNet config is more reliable for from_single_file() loaded checkpoints
    unet_cfg = getattr(diffusers_unet, "config", None)
    if unet_cfg is not None:
        prediction_type = getattr(unet_cfg, "prediction_type", prediction_type)

    # Scheduler config as a secondary source (may override if more specific)
    if scheduler is not None:
        sched_cfg = getattr(scheduler, "config", None)
        if sched_cfg is not None:
            prediction_type = getattr(sched_cfg, "prediction_type", prediction_type)

    _PRED_MAP = {
        "epsilon":      EPS,
        "v_prediction": V_PREDICTION,
        "sample":       X0,
    }
    pred_class = _PRED_MAP.get(prediction_type, EPS)

    # --- 2. Beta schedule ---------------------------------------------------
    # Defaults match SDXL / SD1.5 (diffusers "scaled_linear" = ldm "linear")
    beta_schedule = "linear"
    linear_start  = 0.00085
    linear_end    = 0.012
    timesteps     = 1000
    zsnr          = False

    if scheduler is not None:
        sched_cfg = getattr(scheduler, "config", None)
        if sched_cfg is not None:
            _SCHED_MAP = {
                "scaled_linear":      "linear",   # SDXL default
                "squaredcos_cap_v2":  "cosine",   # SD2 cosine variant
                # diffusers "linear" ≈ ldm "linear" for inference purposes
            }
            diff_sched = getattr(sched_cfg, "beta_schedule", "scaled_linear")
            beta_schedule = _SCHED_MAP.get(diff_sched, "linear")
            linear_start  = getattr(sched_cfg, "beta_start",            linear_start)
            linear_end    = getattr(sched_cfg, "beta_end",              linear_end)
            timesteps     = getattr(sched_cfg, "num_train_timesteps",   timesteps)
            zsnr          = getattr(sched_cfg, "rescale_betas_zero_snr", zsnr)

    # --- 3. Build sampling class --------------------------------------------
    class _SamplingCfg:
        sampling_settings = {
            "beta_schedule": beta_schedule,
            "linear_start":  linear_start,
            "linear_end":    linear_end,
            "timesteps":     timesteps,
            "zsnr":          zsnr,
        }

    # type() avoids the mypy "Invalid base class variable" error while still
    # producing correct MRO (ModelSamplingDiscrete → pred_class).
    _DynSampling = type("_DynSampling", (ModelSamplingDiscrete, pred_class), {})
    instance = _DynSampling(model_config=_SamplingCfg())

    # --- 4. Override sigmas from scheduler.alphas_cumprod if available ------
    # The scheduler's alphas_cumprod is recomputed in float32 from the config
    # beta_start/beta_end — this is more accurate than any raw checkpoint tensors
    # which are often stored as BF16.
    #
    # BF16 storage problem (MODEL_META.md §BF16 / A1111 #14071):
    #   alphas_cumprod[0] stored as BF16 rounds to 1.0 exactly.
    #   This gives sigma_min = sqrt((1-1)/1) = 0 — catastrophically wrong.
    #   The float32 scheduler recomputation is immune to this.
    #
    # Note: if scheduler.alphas_cumprod is identical to what _register_schedule
    # already computed (most cases), this is a no-op — but it makes the schedule
    # source explicit and handles any case where the scheduler was configured with
    # non-default beta parameters.
    schedule_source = f"scheduler config (beta_start={linear_start}, beta_end={linear_end}, T={timesteps})"
    if scheduler is not None:
        sched_ac = getattr(scheduler, "alphas_cumprod", None)
        if sched_ac is not None:
            try:
                ac32 = sched_ac.float()
                sigmas = ((1.0 - ac32) / ac32.clamp(min=1e-10)) ** 0.5
                ztsnr_from_ac = bool(ac32[-1].item() < 1e-5)
                if ztsnr_from_ac and not zsnr:
                    # scheduler.alphas_cumprod encodes ZTSNR even though
                    # rescale_betas_zero_snr was not set — stamp the flag.
                    zsnr = True
                    instance.zsnr = True
                    print(
                        f"[DiffusersModelAdapter] ZTSNR detected from scheduler.alphas_cumprod[-1]"
                        f"={ac32[-1].item():.2e} — stamping zsnr=True"
                    )
                instance.set_sigmas(sigmas)

                # If ZSNR is required but alphas_cumprod is not ZSNR-rescaled
                # (rescale_betas_zero_snr=True in config but raw schedule stored),
                # apply the rescaling NOW — otherwise set_sigmas() overwrites the
                # ZSNR-correct schedule that _register_schedule computed internally.
                if zsnr and not ztsnr_from_ac:
                    ab = 1.0 / ((sigmas ** 2) + 1.0)
                    ab_sqrt = ab.sqrt()
                    ab_sqrt_0 = ab_sqrt[0].clone()
                    ab_sqrt_T = ab_sqrt[-1].clone()
                    ab_sqrt = (ab_sqrt - ab_sqrt_T) * ab_sqrt_0 / (ab_sqrt_0 - ab_sqrt_T)
                    ab_rescaled = ab_sqrt ** 2
                    ab_rescaled[-1] = 4.8973451890853435e-08
                    sigmas_zsnr = ((1.0 - ab_rescaled) / ab_rescaled) ** 0.5
                    instance.set_sigmas(sigmas_zsnr)
                    sigmas = sigmas_zsnr
                    print(
                        f"[DiffusersModelAdapter] ZSNR flag set but alphas_cumprod not rescaled — "
                        f"applied ZeroSNR rescaling: σ_max={sigmas_zsnr[-1].item():.1f}"
                    )

                schedule_source = (
                    f"scheduler.alphas_cumprod (float32, T={len(ac32)}, "
                    f"σ_min={sigmas[0].item():.4f}, σ_max={sigmas[-1].item():.4f}"
                    f"{', ZTSNR' if (ztsnr_from_ac or (zsnr and not ztsnr_from_ac)) else ''})"
                )
            except Exception as _e:
                schedule_source += f"  [alphas_cumprod override failed: {_e}]"

    # Tag the instance so _log_sampling_report can display the true source.
    instance._schedule_source = schedule_source  # type: ignore[attr-defined]

    print(
        f"[DiffusersModelAdapter] model_sampling: prediction={prediction_type} "
        f"schedule={beta_schedule} linear_start={linear_start} "
        f"linear_end={linear_end} timesteps={timesteps} zsnr={zsnr}  "
        f"schedule_source={schedule_source}"
    )
    return instance


class _FakeModelConfig:
    """Stub for unet.model.model_config used by forge_loader latent_format."""

    class _FakeLatentFormat:
        @staticmethod
        def process_out(x):
            return x

        @staticmethod
        def process_in(x):
            return x

    latent_format = _FakeLatentFormat()
    # lora.model_lora_keys_unet() calls model.model_config.unet_config to build
    # diffusers-format key mappings.  An empty dict means no diffusers-style keys
    # are added, but the standard lora_unet_* keys built earlier still work.
    unet_config = {}


def _build_unet_patcher(diffusers_unet, device: torch.device, scheduler=None) -> Any:
    """Wrap a diffusers UNet2DConditionModel in a real UnetPatcher.

    This gives extensions the full patcher API (clone(), model_options,
    wrappers, memory management) without any shimming.
    """
    from modules_forge.unet_patcher import UnetPatcher

    load_device   = device
    offload_device = torch.device("cpu")

    model_wrapper = _DiffusersUnetModel(diffusers_unet, device, scheduler=scheduler)
    patcher = UnetPatcher(
        model=model_wrapper,
        load_device=load_device,
        offload_device=offload_device,
        size=0,
    )
    return patcher


class _ForgeObjects:
    """forge_objects container holding a real UnetPatcher and VAE reference."""

    def __init__(self, unet_patcher, vae, clip=None, clipvision=None):
        self.unet = unet_patcher
        self.vae = vae
        self.clip = clip
        self.clipvision = clipvision

    def shallow_copy(self):
        copy = _ForgeObjects.__new__(_ForgeObjects)
        copy.unet = self.unet
        copy.vae = self.vae
        copy.clip = self.clip
        copy.clipvision = self.clipvision
        return copy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_dtype(device: torch.device) -> torch.dtype:
    """Return the best inference dtype for *device*.

    Delegates to ``preferred_unet_dtype`` (bf16 > fp32 > fp16) so the
    decision stays in sync with the rest of the forge backend.
    """
    from diff_pipeline.load_model import preferred_unet_dtype
    return preferred_unet_dtype(device)


def _pick_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_prompts_single(pipe, tokenizer, text_encoder, prompts: list[str], device, is_last: bool):
    """Encode *prompts* through one (tokenizer, text_encoder) pair using multi-chunk
    encoding so prompts longer than 77 tokens are never silently truncated.

    Returns:
        hidden  – (B, n_chunks * max_length, hidden_dim)  hidden states (penultimate layer)
        pooled  – (B, pooled_dim) or None; only meaningful when is_last=True (CLIP-G for SDXL)
    """
    max_length = tokenizer.model_max_length   # 77
    chunk_size  = max_length - 2              # 75 content tokens per chunk

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    # Tokenize without truncation to get the full id sequence for each prompt.
    raw = tokenizer(prompts, add_special_tokens=False, truncation=False, padding=False)

    batch_hidden = []
    batch_pooled = []

    for ids in raw.input_ids:
        # Split into chunks of chunk_size content tokens.
        chunks = [ids[i : i + chunk_size] for i in range(0, max(len(ids), 1), chunk_size)]
        if not chunks:
            chunks = [[]]

        chunk_hiddens = []
        last_pooled   = None
        for chunk in chunks:
            # Pad each chunk to exactly max_length: [BOS] + chunk + [EOS] + padding
            padded = [bos] + list(chunk) + [eos] + [pad] * (chunk_size - len(chunk))
            ids_t  = torch.tensor([padded], device=device)
            with torch.no_grad():
                out = text_encoder(ids_t, output_hidden_states=True)
            chunk_hiddens.append(out.hidden_states[-2])   # (1, max_length, hidden)
            last_pooled = out[0]                          # (1, pooled_dim)

        # Concatenate all chunks along the sequence dimension.
        batch_hidden.append(torch.cat(chunk_hiddens, dim=1))   # (1, n*max_length, hidden)
        if is_last:
            batch_pooled.append(last_pooled)

    # Pad to the same sequence length across the batch before stacking.
    max_seq = max(h.shape[1] for h in batch_hidden)
    padded_hidden = []
    for h in batch_hidden:
        if h.shape[1] < max_seq:
            pad_t = torch.zeros(1, max_seq - h.shape[1], h.shape[2], device=device, dtype=h.dtype)
            h = torch.cat([h, pad_t], dim=1)
        padded_hidden.append(h)

    hidden = torch.cat(padded_hidden, dim=0)                   # (B, n*max_length, hidden)
    pooled = torch.cat(batch_pooled, dim=0) if batch_pooled else None
    return hidden, pooled


def _encode_prompts(pipe, prompts: list[str]):
    """Encode a list of text prompts using multi-chunk encoding.

    Each prompt is split into 75-token windows so arbitrarily long prompts are
    supported without truncation — matching the behaviour of the forge/A1111 CLIP
    path in ldm_patched/modules/sd1_clip.py.

    Returns a dict compatible with the webui SDXL conditioning format:
      {'crossattn': tensor(B, n_chunks*77, hidden), 'vector': tensor(B, pooled)}

    For SD1.5/SD2 models (no tokenizer_2) only 'crossattn' is returned.
    """
    device = getattr(pipe, "_execution_device", None) or next(
        (p.device for p in pipe.unet.parameters() if p.device.type != "meta"), None
    ) or torch.device("cpu")

    tokenizers    = []
    text_encoders = []
    for tok_attr, te_attr in (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2")):
        tok = getattr(pipe, tok_attr, None)
        te  = getattr(pipe, te_attr,  None)
        if tok is not None and te is not None:
            tokenizers.append(tok)
            text_encoders.append(te)

    all_hidden = []
    pooled_embeds = None

    for i, (tok, te) in enumerate(zip(tokenizers, text_encoders, strict=False)):
        is_last = (i == len(tokenizers) - 1)
        hidden, pooled = _encode_prompts_single(pipe, tok, te, prompts, device, is_last)
        all_hidden.append(hidden)
        if pooled is not None:
            pooled_embeds = pooled

    # SDXL: concatenate CLIP-L and CLIP-G hidden states along the feature dimension.
    # SD1.5: only one encoder, nothing to concatenate.
    prompt_embeds = torch.cat(all_hidden, dim=-1) if len(all_hidden) > 1 else all_hidden[0]

    if pooled_embeds is not None:
        return {"crossattn": prompt_embeds, "vector": pooled_embeds}
    return {"crossattn": prompt_embeds}


# ---------------------------------------------------------------------------
# Textual inversion loader
# ---------------------------------------------------------------------------

def load_textual_inversion_embeddings(pipe, embeddings_dir: str) -> int:
    """Load all .safetensors textual inversion embeddings from *embeddings_dir* into *pipe*.

    Each .safetensors file is loaded via ``pipe.load_textual_inversion()`` using
    the filename stem as the trigger token.  SDXL pipelines with both
    ``text_encoder`` / ``text_encoder_2`` receive both ``clip_l`` and ``clip_g``
    weight slices automatically when present in the file.

    Multi-token embeddings (vec shape[0] > 1) are handled transparently by
    diffusers, which creates one special token per row and maps the trigger
    word to the full sequence at tokenization time.

    Returns the number of embeddings successfully loaded.
    """
    import os
    import logging

    log = logging.getLogger(__name__)
    loaded = 0

    if not os.path.isdir(embeddings_dir):
        log.warning("Textual inversion dir not found: %s", embeddings_dir)
        return 0

    for fn in sorted(os.listdir(embeddings_dir)):
        if not fn.lower().endswith(".safetensors"):
            continue
        token_name = os.path.splitext(fn)[0]
        path = os.path.join(embeddings_dir, fn)
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(path, device="cpu")

            clip_l = state_dict.get("clip_l")
            clip_g = state_dict.get("clip_g")

            if clip_l is not None or clip_g is not None:
                # SDXL dual-encoder format: load each encoder separately.
                if clip_l is not None and getattr(pipe, "text_encoder", None) is not None:
                    pipe.load_textual_inversion(
                        clip_l, token=token_name,
                        text_encoder=pipe.text_encoder,
                        tokenizer=pipe.tokenizer,
                    )
                if clip_g is not None and getattr(pipe, "text_encoder_2", None) is not None:
                    pipe.load_textual_inversion(
                        clip_g, token=token_name,
                        text_encoder=pipe.text_encoder_2,
                        tokenizer=pipe.tokenizer_2,
                    )
            elif len(state_dict) == 1:
                # Single-key format — pass tensor directly.
                tensor = next(iter(state_dict.values()))
                pipe.load_textual_inversion(tensor, token=token_name)
            else:
                # Fall back to path-based loading (handles string_to_param etc.).
                pipe.load_textual_inversion(path, token=token_name)

            log.info("Loaded textual inversion embedding: %s → '%s'", fn, token_name)
            loaded += 1
        except Exception as exc:
            log.warning("Failed to load embedding %s: %s", fn, exc)

    if loaded:
        print(f"[diffusers path hijack] Loaded {loaded} textual inversion embedding(s) "
              f"from {embeddings_dir}")
    return loaded


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class DiffusersModelAdapter:
    """ldm-compatible wrapper around a diffusers pipeline.

    Instantiate and return from a path-hijack loader instead of the raw pipeline:

        pipe = StableDiffusionXLPipeline.from_single_file(...)
        return DiffusersModelAdapter(pipe, checkpoint_info, model_type="sdxl")
    """

    def __init__(self, pipe, checkpoint_info, model_type: str = "sdxl"):
        self._pipe = pipe
        self._model_type = model_type  # "sdxl" | "sd1" | "sd2" | "sd3" | "flux"

        # ---- ldm boolean flags ------------------------------------------
        self.is_sdxl          = (model_type == "sdxl")
        self.is_sdxl_inpaint  = False   # would need UNet in_channels check
        self.is_ssd           = False
        self.is_sd1           = (model_type == "sd1")
        self.is_sd2           = (model_type == "sd2")
        self.is_sd3           = (model_type == "sd3")
        self.ztsnr            = False
        self.tiling_enabled   = False
        self._separate_te_loaded = False

        # ---- metadata ---------------------------------------------------
        self.sd_checkpoint_info   = checkpoint_info
        self.sd_model_checkpoint  = checkpoint_info.filename
        self.sd_model_hash        = getattr(checkpoint_info, 'hash', '')[:10]
        self.filename             = checkpoint_info.filename
        self.current_lora_hash    = ""
        self.latent_channels      = 16 if model_type == "sd3" else 4
        _unet_cfg = getattr(pipe.unet, "config", None)
        _pred_type = getattr(_unet_cfg, "prediction_type", "epsilon") if _unet_cfg else "epsilon"
        self.parameterization     = "v" if _pred_type == "v_prediction" else "eps"

        # ---- conditioning key -------------------------------------------
        self.cond_stage_key = "crossattn"

        # ---- sub-model wrappers -----------------------------------------
        self.model             = _FakeModel(pipe)
        self.cond_stage_model  = _FakeCondStageModel(pipe)
        self.first_stage_model = getattr(pipe, "vae", None)

        # ---- forge compatibility wrappers --------------------------------
        # Build a real UnetPatcher so extensions get the full patcher API
        # (clone(), model_options, wrappers, memory management).
        _device = _pick_device()
        _scheduler = getattr(pipe, "scheduler", None)
        _unet_patcher = _build_unet_patcher(pipe.unet, _device, scheduler=_scheduler)
        _fo = _ForgeObjects(_unet_patcher, getattr(pipe, "vae", None))
        self.forge_objects                     = _fo
        self.forge_objects_original            = _fo.shallow_copy()
        self.forge_objects_after_applying_lora = _fo.shallow_copy()

        # Point sd_model.model.diffusion_model at the patcher's model wrapper
        # so forge_loader / processing.py reads are consistent.
        self.model.diffusion_model = _unet_patcher.model

        # ---- noise schedule (minimal) ------------------------------------
        # Provide a plausible alphas_cumprod so samplers that read it don't crash.
        # The actual denoising uses the diffusers scheduler, so this is only
        # consulted by extensions that inspect the noise schedule.
        try:
            sc = getattr(pipe, "scheduler", None)
            if sc is not None and hasattr(sc, "alphas_cumprod"):
                self.alphas_cumprod = sc.alphas_cumprod
            else:
                self.alphas_cumprod = torch.linspace(1.0, 0.0, 1000)
        except Exception:
            self.alphas_cumprod = torch.linspace(1.0, 0.0, 1000)
        self.alphas_cumprod_original = self.alphas_cumprod
        # Initialise from the real model_sampling so set_sigmas(sigmas_original)
        # in processing.py never receives None.
        self.sigmas_original = _unet_patcher.model.model_sampling.sigmas

        # ---- misc -------------------------------------------------------
        self.cond_stage_model_empty_prompt = None

        # Build a DiffPipeline wrapping the already-loaded HF UNet so that
        # _DiffusersUnetModel.apply_model() delegates to DiffPipeline.apply_model().
        # This activates ControlNet mapping, ForgeAttnProcessor dispatch,
        # LoRA sync, and offload logic for the diffusers hijack path.
        try:
            from diff_pipeline.pipeline import DiffPipeline
            _dp = DiffPipeline.from_hf_unet(pipe.unet, _unet_patcher, self)
            _unet_patcher.model._diff_pipeline = _dp
            self.diff_pipeline = _dp
        except Exception as _dp_err:
            import logging as _log
            _log.getLogger(__name__).warning(
                "DiffusersModelAdapter: failed to build DiffPipeline (%s) — "
                "falling back to direct UNet forward", _dp_err
            )
            self.diff_pipeline = pipe

    # ---- dtype property -------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self._pipe.unet.parameters()).dtype
        except Exception:
            return torch.float32

    # ---- ldm-compatible methods -----------------------------------------

    def encode_first_stage(self, x: torch.Tensor) -> Any:
        """Encode an image batch to latents using the diffusers VAE."""
        vae = self._pipe.vae
        x = x.to(device=next(vae.parameters()).device, dtype=vae.dtype)
        dist = vae.encode(x)
        # diffusers VAE encode() returns an AutoencoderKLOutput with .latent_dist
        return getattr(dist, "latent_dist", dist)

    def get_first_stage_encoding(self, encoder_output) -> torch.Tensor:
        """Sample from the VAE posterior distribution."""
        if hasattr(encoder_output, "sample"):
            latents = encoder_output.sample()
        elif hasattr(encoder_output, "latent_dist"):
            latents = encoder_output.latent_dist.sample()
        else:
            latents = encoder_output
        return self._vae_scale(latents)

    def _vae_scale(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale latents after VAE encode (mirrors latent_formats.py process_in).

        Handles all known VAE latent format variants:
          Basic SDXL / SD 1.x:   latents * scaling_factor
          With shift_factor:      (latents - shift_factor) * scaling_factor
          With latents_mean/std:  (latents - mean) * scaling_factor / std
        """
        cfg = self._pipe.vae.config
        scaling_factor = getattr(cfg, "scaling_factor",  None)
        shift_factor   = getattr(cfg, "shift_factor",    None)
        latents_mean   = getattr(cfg, "latents_mean",    None)
        latents_std    = getattr(cfg, "latents_std",     None)

        if scaling_factor is None:
            raise ValueError(
                "DiffusersModelAdapter: vae.config.scaling_factor is missing. "
                "Cannot scale latents. Check the loaded checkpoint."
            )

        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
            std  = torch.tensor(latents_std,  device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
            return (latents - mean) * scaling_factor / std

        if shift_factor is not None:
            return (latents - shift_factor) * scaling_factor

        return latents * scaling_factor

    def _vae_unscale(self, z: torch.Tensor) -> torch.Tensor:
        """Unscale latents before VAE decode (mirrors latent_formats.py process_out).

        Handles all known VAE latent format variants:
          Basic SDXL / SD 1.x:   z / scaling_factor
          With shift_factor:      z / scaling_factor + shift_factor
          With latents_mean/std:  z * std / scaling_factor + mean

        The scaling_factor, shift_factor, latents_mean and latents_std are read
        from vae.config so every checkpoint carries its own correct formula.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)

        cfg = self._pipe.vae.config
        scaling_factor = getattr(cfg, "scaling_factor",  None)
        shift_factor   = getattr(cfg, "shift_factor",    None)
        latents_mean   = getattr(cfg, "latents_mean",    None)
        latents_std    = getattr(cfg, "latents_std",     None)

        if scaling_factor is None:
            raise ValueError(
                "DiffusersModelAdapter: vae.config.scaling_factor is missing. "
                "Cannot unscale latents. Check the loaded checkpoint."
            )

        import torch as _torch
        nan_in = int(_torch.isnan(z).sum())
        print(
            f"[VAE unscale] scaling_factor={scaling_factor:.5f}  "
            f"shift_factor={f'{shift_factor:.5f}' if shift_factor is not None else 'None'}  "
            f"has_mean_std={latents_mean is not None}  "
            f"z.shape={tuple(z.shape)}  z.dtype={z.dtype}  "
            f"nan_in_z={nan_in}  z_min={float(z[~_torch.isnan(z)].min()) if nan_in < z.numel() else 'all-nan'}  "
            f"z_max={float(z[~_torch.isnan(z)].max()) if nan_in < z.numel() else 'all-nan'}"
        )

        if latents_mean is not None and latents_std is not None:
            # Per-channel normalisation (Playground 2.5 / CogVideoX style)
            mean = torch.tensor(latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
            std  = torch.tensor(latents_std,  device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
            return z * std / scaling_factor + mean

        if shift_factor is not None:
            # Shifted latent format (SD3 / FLUX style)
            return z / scaling_factor + shift_factor

        # Standard (SDXL / SD 1.x)
        return z / scaling_factor

    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using the diffusers VAE.

        The VAE dtype is fixed at load time (see dummy_sdxl_hijack in load_model.py)
        to bf16 or fp32 to avoid fp16 overflow.  No runtime dtype switching is done
        here so this method is torch.compile / inductor CUDA-graph safe.
        """
        vae = self._pipe.vae
        z = self._vae_unscale(z)
        device = next(vae.parameters()).device
        # Cast latents to match the VAE's dtype (bf16 or fp32 after load-time upcast).
        z = z.to(device=device, dtype=vae.dtype)
        decoded = vae.decode(z)
        return getattr(decoded, "sample", decoded)

    def get_learned_conditioning(self, prompts: list[str]) -> Any:
        """Text-encode prompts using the pipeline's text encoders."""
        return _encode_prompts(self._pipe, prompts)

    def apply_model(self, x, t, cond, **kwargs):
        """Run the diffusion UNet forward pass.

        This is a minimal shim — actual generation should go through
        the diffusers pipeline scheduler loop.
        """
        # cond may be a dict or tensor; extract cross-attention context
        c_crossattn = cond
        if isinstance(cond, dict):
            c_crossattn = cond.get("crossattn", cond.get("c_crossattn", None))
            if isinstance(c_crossattn, list):
                c_crossattn = c_crossattn[0]
        unet = self._pipe.unet
        # MPS requires all inputs to share the same dtype as the UNet weights.
        model_dtype = self.dtype
        x = x.to(dtype=model_dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype=model_dtype)
        added_cond = {}
        return unet(x, t, encoder_hidden_states=c_crossattn, added_cond_kwargs=added_cond).sample

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        """No-op context manager (EMA weights not used in inference)."""
        yield

    def fix_dimensions(self, width: int, height: int):
        """Snap width/height to the nearest multiples of 8 (VAE requirement)."""
        width  = max(8, (width  // 8) * 8)
        height = max(8, (height // 8) * 8)
        return width, height

    def state_dict(self) -> dict:
        """Return an empty dict — state dict inspection not needed for inference."""
        return {}

    def to(self, device):
        self._pipe.to(device)
        return self

    def set_injections(self, name, lst):
        """No-op — injection management is handled by forge for ldm models."""
        pass

    # ---- __getattr__ fallback for anything not explicitly handled -------

    def __getattr__(self, name: str):
        # First try the underlying pipeline (handles .unet, .vae, .tokenizer, etc.)
        try:
            return getattr(self._pipe, name)
        except AttributeError:
            pass
        raise AttributeError(
            f"DiffusersModelAdapter has no attribute {name!r}. "
            f"If this is needed for generation, add it to diff_pipeline/adapter.py."
        )

    def __repr__(self):
        return (
            f"DiffusersModelAdapter("
            f"model_type={self._model_type!r}, "
            f"pipeline={type(self._pipe).__name__})"
        )
