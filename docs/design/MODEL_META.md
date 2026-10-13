# Model Metadata and Sampling Parameter Extraction

This document describes the sampling-relevant data that different model families embed
in their checkpoint files, how reForge reads (or should read) those values, and what
defaults are assumed when the data is absent.

---

## Background: Where Sampling Parameters Live

A Stable Diffusion checkpoint (`.safetensors` or `.ckpt`) contains more than just the
trained network weights. Models trained with the CompVis LDM framework save the entire
DDPM model object, which includes the noise schedule as PyTorch `register_buffer`
tensors. These buffers become part of the `state_dict` and therefore end up in the
checkpoint file.

At load time, reForge (and A1111) use `strict=False`, meaning unrecognised keys are
silently ignored. The important question is: **which of these extra keys carry
meaningful information that should change inference behaviour?**

There are two orthogonal concerns:

1. **Noise schedule** — the sigma sequence used for sampling.
2. **Prediction parameterization** — what the UNet was trained to predict (noise ε,
   velocity v, or a flow-matching vector field).

---

## Checkpoint Keys Relevant to Sampling

### Schedule tensors (CompVis DDPM `register_buffer`)

All of the following are precomputed from `betas` during training and saved as
float32 buffers. Only `alphas_cumprod` and `betas` carry independent information;
the rest are algebraic derivatives.

| Key | Shape | Independent? | How to use |
|-----|-------|--------------|------------|
| `alphas_cumprod` | [T] | **Yes** | `σ = sqrt((1−ā)/ā)` → `set_sigmas()` |
| `betas` | [T] | Yes (fallback) | `ā = cumprod(1−β)` → then as above |
| `alphas_cumprod_prev` | [T] | No | `ā` shifted right by one — ignore |
| `sqrt_alphas_cumprod` | [T] | No | `sqrt(ā)` — ignore |
| `sqrt_one_minus_alphas_cumprod` | [T] | No | `sqrt(1−ā)` — ignore |
| `log_one_minus_alphas_cumprod` | [T] | No | `log(1−ā)` — ignore |
| `sqrt_recip_alphas_cumprod` | [T] | No | `sqrt(1/ā)` — ignore |
| `sqrt_recipm1_alphas_cumprod` | [T] | No | `sqrt(1/ā−1)` — ignore |

**Precision**: always cast `alphas_cumprod` to float32 before converting to sigmas.
Bfloat16 round-off changes sigma values noticeably near the schedule boundaries
(A1111 issue #14071).

### Prediction-type flags

| Key | Meaning |
|-----|---------|
| `v_pred` | Model is v-prediction (velocity parameterization) |
| `ztsnr` | Zero terminal SNR training was used |
| `edm_vpred.sigma_max` | EDM-style v-prediction; gives σ_max directly |
| `edm_vpred.sigma_min` | EDM-style v-prediction; gives σ_min directly |
| `edm_mean` + `edm_std` | Playground V2.5 EDM normalisation constants |

### SAI ModelSpec safetensors header (optional)

Some training pipelines (kohya-ss `sd-scripts`, Diffusers `save_pretrained`) write
a `__metadata__` block into the safetensors header:

| Metadata key | Values | Used for |
|---|---|---|
| `modelspec.prediction_type` | `"epsilon"`, `"v_prediction"` | Prediction type |
| `modelspec.ztsnr` | `"true"` / `"false"` | Zero terminal SNR flag |
| `modelspec.architecture` | e.g. `"stable-diffusion-xl-v1-base"` | Architecture hint |
| `modelspec.resolution` | e.g. `"1024x1024"` | Native resolution |

**This is optional** — most fine-tunes do not include it. Never assume it is present.

### ZTSNR detection heuristic

If `alphas_cumprod` is present in the checkpoint and `alphas_cumprod[-1] < 1e-5`,
the training used zero-terminal SNR (the rescaling is already baked into the tensor).
Use the stored sigmas directly — do **not** apply `rescale_zero_terminal_snr_sigmas()`
again, as that would double-apply the transformation.

---

## Implementation: `apply_checkpoint_sampling_params`

The shared helper in `ldm_patched/modules/supported_models_base.py` handles all
CompVis-family schedule extraction and is called from `BASE.get_model()` (SD1.5,
SD2.x) and `SDXL.get_model()`. It:

1. Checks for `alphas_cumprod` → converts to sigmas → calls `model_sampling.set_sigmas()`.
2. Falls back to `betas` if `alphas_cumprod` is absent.
3. Logs what it did (or that it fell back to defaults) at INFO level.
4. Returns a `ztsnr_detected` bool.

`SDXL.model_type()` additionally reads `modelspec.prediction_type` from the
safetensors metadata dict when available.

---

## Model Family Reference

### SD1.x (SD 1.5)

| Property | Value |
|---|---|
| Forward process | DDPM, discrete, T=1000 |
| Beta schedule | `scaled_linear` (`linear_start=0.00085`, `linear_end=0.012`) |
| Prediction type | Epsilon (ε) |
| ZTSNR | No |
| Embeds `alphas_cumprod`? | Only if trained with full CompVis LDM save |
| Embeds `v_pred` / `ztsnr`? | Only if explicitly added by the training script |
| σ_min / σ_max (default) | ≈ 0.029 / ≈ 14.6 |
| Status | `apply_checkpoint_sampling_params` called via `BASE.get_model()` |

Most SD1.5 community fine-tunes do **not** embed schedule tensors. The default
`scaled_linear` schedule is correct for them.

---

### SD2.x (SD 2.0, 2.1)

| Property | Value |
|---|---|
| Forward process | DDPM, discrete, T=1000 |
| Beta schedule | `scaled_linear` (`linear_start=0.00085`, `linear_end=0.012`) |
| Prediction type | v-prediction (SD2.0 768px, SD2.1) — detected by weight std heuristic |
| ZTSNR | No |
| Embeds `alphas_cumprod`? | Rarely |
| σ_min / σ_max (default) | ≈ 0.029 / ≈ 14.6 |
| Status | `apply_checkpoint_sampling_params` called via `BASE.get_model()` |

SD2.0 512px inpainting models are epsilon; 768px models are v-prediction.
Detection uses a weight standard-deviation heuristic on `output_blocks.11.*`.

---

### SDXL (base 1.0, fine-tunes, Illustrious, NoobAI, Hassaku, etc.)

| Property | Value |
|---|---|
| Forward process | DDPM, discrete, T=1000 |
| Beta schedule | `scaled_linear` (`linear_start=0.00085`, `linear_end=0.012`) |
| Prediction type | Epsilon or v-prediction depending on variant |
| ZTSNR | Variant-dependent (see table below) |
| Embeds `alphas_cumprod`? | Yes — full CompVis-style training saves these buffers |
| σ_min / σ_max (default) | ≈ 0.029 / ≈ 14.6 |
| Status | `apply_checkpoint_sampling_params` called from `SDXL.get_model()` |

Known SDXL variants and their sampling parameters:

| Model | Prediction type | ZTSNR | Notes |
|---|---|---|---|
| SDXL base 1.0 (Stability) | Epsilon | No | Standard reference; no schedule tensors |
| Illustrious-XL v0–v2 (OnomaAI) | Epsilon | No | Embeds standard SDXL schedule |
| Illustrious-XL v3.0-vpred | v-prediction | Yes (buggy) | Colour collapse known; use v3.5 instead |
| Illustrious-XL v3.5-vpred | v-prediction | Yes (corrected) | CFG rescale ~0.7 recommended |
| NoobAI-XL epsilon 1.0/1.1 | Epsilon | No | Illustrious derivative; standard schedule |
| NoobAI-XL v-pred 1.0 | v-prediction | Yes | `v_pred`+`ztsnr` keys; Karras not recommended |
| Hassaku XL (Illust. base) | Epsilon | No | Confirmed: embeds standard schedule |
| Playground V2.5 | EDM | — | `edm_mean`/`edm_std` keys; σ_data=0.5 |
| SDXL-EDM v-pred | EDM v-prediction | — | `edm_vpred.sigma_max/min` keys |

For ZTSNR v-pred models, the start timestep must be T=1000 (not T=999) and a CFG
rescale factor of approximately 0.7 suppresses over-exposure.

---

### SD3 / SD3.5

| Property | Value |
|---|---|
| Forward process | **Rectified flow** (flow matching) — no DDPM, no `alphas_cumprod` |
| Parameterization | Velocity field (not ε, not v) |
| Schedule | Continuous time; recommended `shift=3.0` at 1024px |
| Key tensors in checkpoint | None of the DDPM buffer keys |
| Detection | `pos_embed_scaling_factor` UNet key + `joint_blocks.*` / `mmdit_*` weights |
| `sampling_settings` | `{"shift": 3.0}` |
| Status | **`apply_checkpoint_sampling_params` not applicable** — flow model |

SD3 uses `ModelSamplingDiscreteFlow` which operates in continuous time `[0, 1]`.
The `shift` parameter controls the timestep distribution; higher values push more
steps toward the high-noise end. No beta schedule, no `alphas_cumprod`.

**Planned support**: read `shift` from checkpoint metadata or embedded config if
training scripts begin storing it.

---

### FLUX.1 (dev / schnell)

| Property | Value |
|---|---|
| Forward process | **Rectified flow** — no `alphas_cumprod` |
| Parameterization | Velocity field |
| Schedule | Continuous; `shift` controls distribution |
| FLUX.1-dev | Guidance distilled; `guidance_embed=True`; uses CFG-like guidance scale |
| FLUX.1-schnell | Flow-matching distilled (few steps); `shift=1.0`, `ModelType.FLOW` |
| Key detection | `image_model=flux` + `guidance_embed` flag |
| `sampling_settings` | dev: `{}` (shift inferred per-run); schnell: `{"multiplier":1.0,"shift":1.0}` |
| Status | **`apply_checkpoint_sampling_params` not applicable** — flow model |

No `alphas_cumprod` exists. Sigma schedule is derived from the flow-matching
continuous timestep parameterization, not a discrete DDPM schedule.

**Planned support**: FLUX fine-tunes may embed a `shift` scalar; read from
checkpoint metadata if present.

---

### AuraFlow

| Property | Value |
|---|---|
| Forward process | Rectified flow |
| `sampling_settings` | `{"multiplier":1.0, "shift":1.73}` |
| Status | No schedule tensors expected |

---

### PixArt-α / PixArt-Σ

| Property | Value |
|---|---|
| Forward process | DDPM, but `sqrt_linear` schedule (`linear_start=0.0001`, `linear_end=0.02`) |
| Prediction type | Epsilon |
| Status | Fixed schedule in `sampling_settings`; `apply_checkpoint_sampling_params` called via `BASE.get_model()` — will log defaults |

Different beta schedule from SDXL/SD1.x. Schedule tensors in the checkpoint (if
present) should override these, which the helper handles automatically.

---

### HunyuanDiT

| Property | Value |
|---|---|
| Forward process | DDPM |
| `sampling_settings` | `{"linear_start":0.00085, "linear_end":0.018}` (v1) or `0.03` (v1.1) |
| Status | Fixed schedule; `apply_checkpoint_sampling_params` called via `BASE.get_model()` |

---

### Stable Cascade (Würstchen v3)

| Property | Value |
|---|---|
| Forward process | Continuous EDM-like; Prior (C) and Decoder (B) have separate schedules |
| Prior `sampling_settings` | `{"shift":2.0, "multiplier":0.538}` |
| Decoder `sampling_settings` | Inherits from Prior |
| Status | Proprietary schedule; no DDPM buffers expected |

---

### SVD (Stable Video Diffusion) / SV3D

| Property | Value |
|---|---|
| Forward process | EDM continuous |
| `sampling_settings` | `{"sigma_min":0.002, "sigma_max":700.0, "sigma_data":1.0}` |
| Status | `apply_checkpoint_sampling_params` called via `BASE.get_model()` — will log defaults |

---

### Video / Audio Models (Mochi, LTXV, HunyuanVideo, Cosmos, Wan 2.1, etc.)

All of these use rectified flow or EDM-style continuous processes:

| Model | Forward process | `shift` | Notes |
|---|---|---|---|
| GenmoMochi | Flow matching | 6.0 | |
| LTXV | Flow matching | — | |
| HunyuanVideo | Flow matching | 7.0 (t2v) | |
| CosmosT2V | EDM continuous | — | σ_max=80, σ_min=0.002 |
| Wan 2.1 T2V/I2V | Flow matching | 8.0 / 5.0 | resolution-dependent in practice |
| Lumina2 | Flow matching | 6.0 | |
| HiDream | Flow matching | — | |
| Chroma | Flow matching | — | FLUX derivative |
| ACEStep | — | — | Audio generation |

None of these embed `alphas_cumprod`. `apply_checkpoint_sampling_params` will log
that it fell back to defaults (which is correct — no override needed).

---

## Decision Tree: What to Do at Load Time

```
Does the checkpoint have `alphas_cumprod`?
│
├─ YES → convert to sigmas, call set_sigmas()
│        check alphas[-1] < 1e-5 → log ZTSNR detected
│
└─ NO → Does it have `betas`?
         │
         ├─ YES → cumprod → alphas_cumprod → sigmas → set_sigmas()
         │
         └─ NO → use hardcoded sampling_settings defaults, log the values

Does the safetensors metadata have `modelspec.prediction_type`?
│
├─ "v_prediction" → ModelType.V_PREDICTION
│                   check `modelspec.ztsnr == "true"` → set zsnr=True
├─ "epsilon"      → ModelType.EPS
└─ absent         → fall through to key-based heuristics below

Does state_dict have `edm_mean` + `edm_std`?     → ModelType.EDM  (Playground V2.5)
Does state_dict have `edm_vpred.sigma_max`?       → ModelType.V_PREDICTION_EDM
Does state_dict have `v_pred`?
│   Does state_dict also have `ztsnr`?            → set zsnr=True
└─                                                → ModelType.V_PREDICTION
Default                                           → ModelType.EPS

Is it a flow-matching model (SD3, FLUX, AuraFlow, video models)?
└─ No DDPM schedule at all — alphas_cumprod never expected
```

---

## Pending Work

- **SD3 / FLUX `shift` from metadata**: some training tools will start embedding the
  recommended `shift` value; read from checkpoint or metadata when present.
- **ZTSNR + v-pred from `alphas_cumprod` content alone**: when neither `v_pred` nor
  `ztsnr` keys are present but `alphas_cumprod[-1] ≈ 0`, the model is almost
  certainly v-prediction ZTSNR. Automatic reclassification from EPS → V_PREDICTION
  in this case is a future improvement.
- **`modelspec.resolution` → native resolution hint**: could set default HR-fix
  scale or aspect-ratio guidance automatically.
- **`modelspec.prediction_type` for SD15/SD20**: extend `SD20.model_type()` to
  check metadata in the same way SDXL does.
