# SDXLDiff — SDXL ↔ Diffusers Compatibility Layer Analysis

Feasibility study and design for a shim that lets reForge load a HuggingFace
`UNet2DConditionModel` (Diffusers SDXL) while keeping the full extension hook
surface intact.

---

## Hook Surface Map

Every extension attachment point and where it lives in the codebase:

| Hook | Set via | Consumed in | Used by |
|---|---|---|---|
| `model_function_wrapper` | `UnetPatcher.set_model_unet_function_wrapper()` | [ldm_patched/modules/samplers.py:331](ldm_patched/modules/samplers.py#L331) — wraps entire `apply_model()` | MultiDiffusion, tiled diffusion |
| `patches_replace["attn2"]` | `ModelPatcher.set_model_patch_replace()` | [ldm_patched/ldm/modules/attention.py:812](ldm_patched/ldm/modules/attention.py#L812) — replaces Q/K/V at `(block, idx, t_idx)` tuple | IP-Adapter |
| `patches["attn1_patch"]` / `attn2_patch` | `ModelPatcher.set_model_patch()` | [ldm_patched/ldm/modules/attention.py:834](ldm_patched/ldm/modules/attention.py#L834) — modifies attention inputs before projection | ControlLLite, Reference |
| `patches["input_block_patch"]` / `output_block_patch` | same | openaimodel.py residual outputs | Tile preprocessor |
| `add_block_modifier` / `add_block_inner_modifier` | [modules_forge/unet_patcher.py:246](modules_forge/unet_patcher.py#L246) | unet_patcher block hooks | ControlNet Forge |
| `control` residuals | `control.get_control()` in sampler | [ldm_patched/modules/model_base.py:191](ldm_patched/modules/model_base.py#L191) — passed to `diffusion_model()` as `control` kwarg | All ControlNets |
| LoRA weight patches | `ModelPatcher.add_patches()` | `ModelPatcher.calculate_weight()` — merged into tensors before inference | LoRA, LyCORIS |
| `sampler_cfg_function` / `post_cfg_function` | `ModelPatcher.set_model_sampler_*` | [ldm_patched/modules/samplers.py:372](ldm_patched/modules/samplers.py#L372) `cfg_function()` | DynamicCFG, PAG, SEG |

---

## Conditioning Format Difference

The sampler calls `model.apply_model(x, t, **c)`. The shim must translate:

```
ldm / reForge format                     HF Diffusers format
──────────────────────────────────────────────────────────────────────
x          (latent noisy tensor)    →    sample
t          (sigma)                  →    timestep  (via model_sampling.timestep())
c_crossattn (B, 77×n, 2048)        →    encoder_hidden_states
y          (B, 2816)                →    added_cond_kwargs = {
                                           "text_embeds": y[:, :1280],
                                           "time_ids":    decode_size_embeds(y)
                                         }
control    (dict of residual blobs) →    down_block_additional_residuals (list of 12)
                                         mid_block_additional_residual   (1 tensor)
transformer_options (patch dict)    →    cross_attention_kwargs["transformer_options"]
```

### ADM reverse-decode problem

`y` (2816-dim) is already Fourier-embedded by `SDXL.encode_adm()`:
- `y[:, :1280]` = pooled CLIP-G
- `y[:, 1280:]` = 6 size scalars each run through `Timestep(256)` → 256 dims each = 1536 dims

HF `added_cond_kwargs["time_ids"]` expects the **raw** 6-scalar tensor
`[orig_h, orig_w, crop_h, crop_w, tgt_h, tgt_w]`, not the Fourier-embedded version.

**Cleanest fix**: carry the raw scalars as a parallel cond key (e.g. `"adm_raw"`)
alongside `y` so the shim's `apply_model()` can assemble `added_cond_kwargs`
directly, bypassing `encode_adm()` for the Diffusers path entirely. This
requires a one-line change in `extend_sdxl()` where conditioning is built.

---

## Attention Block Address Mapping

reForge/ldm uses tuple keys `(block_name, block_idx, transformer_idx)` to
address individual attention modules. Diffusers uses named sub-module paths.

For SDXL the mapping is fully deterministic (same architecture):

| ldm key | HF module path |
|---|---|
| `("input", 4, 0..1)` | `down_blocks[2].attentions[0].transformer_blocks[0..1].attn2` |
| `("input", 5, 0..9)` | `down_blocks[2].attentions[1].transformer_blocks[0..9].attn2` |
| `("middle", 0, 0..9)` | `mid_block.attentions[0].transformer_blocks[0..9].attn2` |
| `("output", 0..1, 0..9)` | `up_blocks[0].attentions[0..1].transformer_blocks[0..9].attn2` |
| `("output", 2..3, 0..1)` | `up_blocks[1].attentions[0..1].transformer_blocks[0..1].attn2` |
| `("output", 4..5, —)` | `up_blocks[2].attentions[0..1]` (0 transformer blocks — pure resnet) |

Total ~34 transformer blocks. The full table is hard-codeable in a ~50-line
dict. Must be regenerated if Diffusers changes the model class (unlikely for
base SDXL).

---

## What Works Without Change

These hooks live above the UNet and require zero modification:

- **`sampler_cfg_function` / `post_cfg_function`** — entirely in
  [ldm_patched/modules/samplers.py:cfg_function()](ldm_patched/modules/samplers.py#L372),
  never touches the UNet.
- **`model_function_wrapper`** — wraps `apply_model()` which the shim defines.
  Transparent.
- **VRAM lifecycle** — `UnetPatcher` offload/load operates on the underlying
  `diffusion_model` module reference. Swap in the HF module; same interface.
- **LoRA weight patches** — reForge applies LoRA by merging weight deltas
  directly onto tensors before inference. HF `UNet2DConditionModel` has the same
  weight tensors under `down_blocks.N.attentions.M.*` key paths
  (post-`unet_to_diffusers()` conversion). `ModelPatcher.calculate_weight()`
  doesn't care whose module it is.
- **VAE, CLIP, sampler, extensions** — entirely unchanged.

---

## Proposed Layer Structure

```
ForgeSDXLDiffusersUNet                         (new class)
├── .hf_unet          : UNet2DConditionModel   (HF weights, loaded from Diffusers)
├── .model_sampling   : SDXLModelSampling      (reused from ldm_patched unchanged)
├── .adm_channels     : 2816
│
├── apply_model(x, t, c_crossattn, control, transformer_options, y, adm_raw=None)
│     ├── sigma → timestep via model_sampling.timestep()
│     ├── y / adm_raw → added_cond_kwargs {"text_embeds", "time_ids"}
│     ├── control dict → down_block_additional_residuals + mid_block_additional_residual
│     └── hf_unet(sample, timestep, encoder_hidden_states,
│                  added_cond_kwargs, cross_attention_kwargs={"transformer_options": to})
│
├── ForgeAttnProcessor  (installed on all ~34 attn sub-modules at load time)
│     ├── reads cross_attention_kwargs["transformer_options"]
│     ├── checks patches_replace["attn2"][(block_name, block_idx, t_idx)]
│     │     → delegates to replacement fn if present
│     ├── checks patches["attn2_patch"]
│     │     → pipes Q/K/V inputs through patch fn if present
│     └── otherwise: standard scaled-dot-product attention
│
└── block_hooks  (register_forward_hook on down_blocks[N], mid_block, up_blocks[N])
      └── positional map: ("input", N) ↔ down_blocks index
          consumed by add_block_modifier / add_block_inner_modifier
```

`UnetPatcher` wraps `ForgeSDXLDiffusersUNet` exactly as it currently wraps the
ldm diffusion model. From the rest of the pipeline's perspective, nothing
changes.

---

## ControlNet Residual Mapping

reForge's ControlNet returns a dict keyed by block index:
```python
{"input": [t0, t1, ..., t11], "middle": [t_mid]}
```
HF expects flat positional lists:
```python
down_block_additional_residuals = [t0, t1, ..., t11]   # 12 tensors for SDXL
mid_block_additional_residual   = t_mid
```
Shapes are identical (same architecture). This is a 3-line reshape in
`apply_model()`.

---

## Effort & Risk Assessment

| Component | Effort | Risk |
|---|---|---|
| `apply_model()` shim + conditioning translation | Small | Low — pure format conversion |
| Inverse ADM decode or parallel `adm_raw` cond key | Small | Low — one-line change in conditioning build |
| `ForgeAttnProcessor` class | Medium | Medium — must stay in sync with ldm attention patch protocol |
| SDXL block address table (~34 rows) | Small | Medium — fragile if HF renames sub-modules |
| ControlNet residual list mapping | Small | Low — same tensors, different container |
| `block_modifier` `register_forward_hook()` wiring | Medium | Medium — positional, not named; breaks if HF reorders blocks |
| LoRA key remapping via `unet_to_diffusers()` | Small | Low — function already exists |

**Total**: ~300–500 lines of new code, no changes to existing sampler/extension
code, one small change to conditioning assembly in `extend_sdxl()`.

---

## Hard Constraints

1. **`block_modifier` forward hooks are positional** — they enumerate HF
   sub-modules in the same order as ldm `input_blocks[N]`. For SDXL this is
   deterministic and stable, but any future HF refactor of `UNet2DConditionModel`
   block ordering silently breaks it. Mitigation: pin the HF `diffusers` version.

2. **`ForgeAttnProcessor` is called per-module** — it receives
   `cross_attention_kwargs` from HF's attention dispatch. This dict must be
   threaded through correctly; some HF attention implementations filter kwargs
   before passing them. Verify with `diffusers>=0.21` where
   `cross_attention_kwargs` pass-through is guaranteed.

3. **ADM time_ids** — if code anywhere encodes size embeddings before the shim
   sees them (e.g. in `extra_conds()` during conditioning build), the shim
   receives already-Fourier-embedded values with no way to recover the scalars.
   The `adm_raw` parallel key approach is the only clean solution.

---

## Key Files for Implementation

| File | Role |
|---|---|
| [modules_forge/forge_loader.py:389](modules_forge/forge_loader.py#L389) | Add Diffusers load branch in `load_model_for_a1111()` |
| [modules_forge/unet_patcher.py](modules_forge/unet_patcher.py) | `UnetPatcher` wraps the new shim unchanged |
| [ldm_patched/modules/model_base.py:439](ldm_patched/modules/model_base.py#L439) | `SDXL.encode_adm()` — add `adm_raw` parallel cond key |
| [modules/sd_models_xl.py](modules/sd_models_xl.py) | `extend_sdxl()` — carry raw size scalars alongside `y` |
| [ldm_patched/modules/model_detection.py:694](ldm_patched/modules/model_detection.py#L694) | `unet_config_from_diffusers_unet()` — already detects HF SDXL |
| [ldm_patched/modules/utils.py:266](ldm_patched/modules/utils.py#L266) | `unet_to_diffusers()` — existing key map, reuse for LoRA |
