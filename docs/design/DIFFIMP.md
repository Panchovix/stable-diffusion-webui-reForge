# DIFFIMP — Diffusers Pipeline Implementation Record

Documents every concrete code change made to implement the Diffusers-based SDXL
pipeline (Phases 1–3).  Companion to [SDXLDiff.md](SDXLDiff.md) (design) and
[SDXLGEN.md](SDXLGEN.md) (inference flow).

---

## What Was Implemented

| Phase | Description | Status |
|---|---|---|
| 1 | Conditioning bridge — `(c_crossattn, y)` → `(encoder_hidden_states, added_cond_kwargs)` | Done |
| 2 | `ForgeAttnProcessor` — dispatches `patches_replace["attn2"]` / `attn2_patch` hooks | Done |
| 3 | ControlNet residual mapping — `control["input"/"middle"]` → HF positional lists | Done |
| 4 | Block modifier hooks — `add_block_modifier` / `add_block_inner_modifier` | Pending |
| 5 | LoRA — PEFT adapters synced from `unet_patcher.patches` on every `apply_model()` call | Done |

---

## File 1 — `ldm_patched/modules/model_base.py`

### Change: `SDXL.extra_conds()` override (lines 445–463)

**Problem**: `encode_adm()` Fourier-embeds the 6 size scalars into a 1536-dim
blob before concatenating with `clip_pooled`.  The HF UNet's
`added_cond_kwargs["time_ids"]` expects the **raw** 6 scalars, not the
Fourier-embedded version.  Inverting Fourier embedding is lossy, so a parallel
cond key was the only clean approach (as described in SDXLDiff.md §ADM reverse-decode).

**Solution**: Override `extra_conds()` in `SDXL` to emit two extra cond keys
alongside the pre-existing `y`:

```python
# ldm_patched/modules/model_base.py:445
def extra_conds(self, **kwargs):
    out = super().extra_conds(**kwargs)   # still emits 'y' for the ldm path
    clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
    ...
    time_ids = torch.tensor([[h, w, crop_h, crop_w, th, tw]], dtype=clip_pooled.dtype)
                .repeat(batch, 1)
    out["adm_text_embeds"] = CONDRegular(clip_pooled)   # 1280-dim pooled CLIP-G
    out["adm_time_ids"]    = CONDRegular(time_ids)      # (B, 6) raw scalars
    return out
```

`adm_text_embeds` and `adm_time_ids` are passed to `apply_model()` as normal
`**kwargs` through the existing `_calc_cond_batch()` path in
[ldm_patched/modules/samplers.py:213](ldm_patched/modules/samplers.py#L213).
The ldm path ignores unknown kwargs; the Diffusers path reads them.

**No change to `encode_adm()`** — the ldm `y` path is completely unaffected.

---

## File 2 — `diff_pipeline/pipeline.py`

Full rewrite replacing the previous stub.  Three components:

### Component A — Module-level constants

```
_SDXL_HF_UNET_CONFIG   dict   Hardcoded HF UNet2DConditionModel constructor args
                               (stabilityai/stable-diffusion-xl-base-1.0 config.json)
_SDXL_LDM_UNET_CONFIG  dict   ldm-side unet_config for unet_to_diffusers() key map
                               (matches detect_unet_config() output for SDXL)
_DOWN_ATTN_LDM_IDX     dict   (b_idx, a_idx) → ldm input_block index for down attn
_UP_ATTN_LDM_IDX       dict   (b_idx, a_idx) → ldm output_block index for up attn
```

The block address dicts are derived from the formula in `unet_to_diffusers()`:
- down: `ldm_idx = 1 + 3*b + a` (only `b=1,2` have attention)
- up: `ldm_idx = 3*b + a` (only `b=0,1` have attention)

### Component B — `ForgeAttnProcessor`

Installed on every `attn2` (cross-attention) sub-module at construction.
Burns the LDM address `(block_name, block_idx, transformer_idx)` in at install
time so the per-call path is O(1).

**Call signature** (HF `AttnProcessor` protocol):
```python
def __call__(self, attn, hidden_states, encoder_hidden_states=None,
             attention_mask=None, temb=None, **cross_attention_kwargs)
```

`transformer_options` arrives in `cross_attention_kwargs["transformer_options"]`,
threaded by `DiffPipeline.apply_model()` via the HF UNet's
`cross_attention_kwargs` pass-through path.

**Dispatch logic** (mirrors [ldm_patched/ldm/modules/attention.py:807](ldm_patched/ldm/modules/attention.py#L807)):

1. Build `block_tuple = (block_name, block_idx, transformer_idx)` and
   `block_pair = (block_name, block_idx)`.
2. Look up `patches_replace["attn2"]` — try tuple key first, fall back to pair
   key (same fallback ldm uses).
3. If replace patch found: call it with `(q, k, v, extra_options)` in flat
   `(B, seq, inner_dim)` format; use its return value as `hidden_states`.
4. Else if `patches["attn2_patch"]` present: run each patch as
   `(q, k, v) = p(q, k, v, extra_options)`, then standard SDPA.
5. Else: standard `F.scaled_dot_product_attention`.

`extra_options` carries the same keys as ldm's `BasicTransformerBlock.forward`:
`block`, `block_index`, `n_heads`, `dim_head`, plus everything from
`transformer_options` (so `sigmas`, `cond_or_uncond`, `cond_mark`, etc. are
all accessible to patch fns).

### Component C — `DiffPipeline`

#### Construction (`__init__`)

1. Stores `unet_patcher`, `sd_model`, `model_sampling` (reused for sigma math).
2. Calls `_build_hf_unet(unet_patcher.model)`:
   - Calls `unet_to_diffusers(_SDXL_LDM_UNET_CONFIG)` to get the full
     `hf_key → ldm_key` map.
   - Reads `ldm_model.diffusion_model.state_dict()` and remaps tensors.
   - Instantiates `UNet2DConditionModel(**_SDXL_HF_UNET_CONFIG)` on CPU.
   - Loads remapped state dict with `strict=False`; logs missing/unexpected.
   - Casts to the same dtype as the ldm diffusion model.
3. Calls `_install_attn_processors(hf_unet)`:
   - Iterates `down_blocks[b].attentions[a].transformer_blocks[t].attn2` for
     `b ∈ {1,2}`, `a ∈ {0,1}`, all `t`.
   - Sets `ForgeAttnProcessor("input", ldm_idx, t)` on each.
   - Same for `mid_block.attentions[0].transformer_blocks[t]` → `"middle", 0, t`.
   - Same for `up_blocks[b].attentions[a].transformer_blocks[t].attn2` for
     `b ∈ {0,1}`, `a ∈ {0,1,2}`, all `t`.

#### `apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)`

Full pipeline forward — same signature as `BaseModel.apply_model()`:

```
Step 1: sigma preconditioning
    xc = model_sampling.calculate_input(sigma, x)
    timestep = model_sampling.timestep(sigma).float()

Step 2: lazy device migration
    if hf_unet.device != x.device: hf_unet.to(device=x.device)

Step 3: conditioning bridge
    encoder_hidden_states = c_crossattn            # (B, 77n, 2048) — no change

    if adm_text_embeds and adm_time_ids in kwargs:
        added_cond_kwargs = {
            "text_embeds": adm_text_embeds,        # (B, 1280) pooled CLIP-G
            "time_ids":    adm_time_ids,           # (B, 6)   raw size scalars
        }
    else:  # fallback for non-Forge callers
        added_cond_kwargs = {
            "text_embeds": y[:, :1280],
            "time_ids":    zeros(B, 6),            # imprecise but non-crashing
        }

Step 4: ControlNet residual mapping
    control["input"]  (list, pop order = first-block-first after reverse)
    → down_block_additional_residuals  tuple of up-to-12 tensors

    control["middle"][0]
    → mid_block_additional_residual    single tensor

Step 5: HF UNet forward
    hf_unet(
        sample=xc, timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        down_block_additional_residuals=...,
        mid_block_additional_residual=...,
        cross_attention_kwargs={"transformer_options": transformer_options},
        return_dict=False,
    )

Step 6: sigma postconditioning
    return model_sampling.calculate_denoised(sigma, model_output, x)
```

#### ControlNet mapping detail

ldm's `apply_control()` in
[openaimodel.py:366](ldm_patched/ldm/modules/diffusionmodules/openaimodel.py#L366)
pops from the *end* of `control["input"]`, so the list is stored
**reverse-first** — `control["input"][-1]` is the residual for `input_block[0]`.
`DiffPipeline` reverses the list before passing as `down_block_additional_residuals`
to restore natural forward order.

---

## File 3 — `requirements.txt`

Added `diffusers` on the line after `accelerate`.

---

## Known Gaps

### LoRA (Phase 5 — implemented)

`_sync_lora()` is called at the top of every `apply_model()` invocation.  It
reads `unet_patcher.patches` and `patches_uuid`; if the uuid has not changed
since the last call it returns immediately (O(1) fast path).

When the uuid changes:
1. Removes all previously installed PEFT adapters via `delete_adapter()`.
2. Iterates the patch list by depth (stacked LoRAs → multiple adapters).
3. For each depth, builds a PEFT-format state dict:
   - `module_path.lora_A.weight` ← `LoRAAdapter.weights[1]` (lora_down)
   - `module_path.lora_B.weight` ← `LoRAAdapter.weights[0]` (lora_up)
4. Calls `hf_unet.load_lora_adapter(state_dict, network_alphas=..., adapter_name=f"forge_lora_{depth}")`.
5. Calls `hf_unet.set_adapters(names, adapter_weights=strengths)`.

Key map: `_ldm_to_hf` (built once at construction) maps
`"diffusion_model.X.weight"` → HF parameter path via reverse of
`unet_to_diffusers()`.

**Remaining gaps**: DoRA (`dora_scale`) and tucker mid (`lora_mid`) are skipped.

### Block modifier hooks not forwarded

`add_block_modifier` / `add_block_inner_modifier` store callables in
`transformer_options["block_modifiers"]` /  `["block_inner_modifiers"]`.
These are consumed by the ldm UNetModel `_forward()` loop (calling them
`before` and `after` each block).  The HF UNet has no equivalent loop.

**Fix (Phase 4)**: Register `torch.nn.Module.register_forward_hook` on each
`down_blocks[N]` / `mid_block` / `up_blocks[N]` that calls the modifier list
with the matching `("input", ldm_idx)` / `"middle"` / `"output"` block tuple.
The `_DOWN_ATTN_LDM_IDX` / `_UP_ATTN_LDM_IDX` tables already encode this
mapping.

---

## Extension Hook Coverage After Implementation

| Hook | Works with DiffPipeline? | Notes |
|---|---|---|
| `model_function_wrapper` | Yes | Wraps `apply_model()` — transparent |
| `patches_replace["attn2"]` | Yes | `ForgeAttnProcessor` dispatches it |
| `patches["attn2_patch"]` | Yes | `ForgeAttnProcessor` dispatches it |
| `sampler_cfg_function` / `post_cfg_function` | Yes | Above-UNet, unaffected |
| ControlNet residuals | Yes | Phases 3 mapping implemented |
| LoRA weight patches | Yes | Phase 5: PEFT adapters synced per `patches_uuid` |
| `add_block_modifier` | No | Phase 4 gap — see above |
| `add_block_inner_modifier` | No | Phase 4 gap — see above |
| `patches["input_block_patch"]` | No | Not dispatched (Phase 4) |
| `patches["output_block_patch"]` | No | Not dispatched (Phase 4) |
| VAE / CLIP / sampler | Yes | Entirely unchanged |

---

---

## Bug Fix — Wrapper Not Dispatched at Sampling Time

### Symptom

`DiffPipeline.apply_model()` was never called despite the wrapper being registered
at model load time.  Generation ran silently through the ldm UNet.

### Root Cause (discovered 2026-04-04)

`KDiffusionSampler` (and other sampler classes) cache a reference to the
`CompVisDenoiser` the first time `CFGDenoiserKDiffusion.inner_model` is accessed:

```python
# modules/sd_samplers_kdiffusion.py — CFGDenoiserKDiffusion
@property
def inner_model(self):
    if self.model_wrap is None:
        self.model_wrap = denoiser(shared.sd_model, ...)   # captured once
    return self.model_wrap
```

When a model is loaded, `shared.sd_model` is **replaced** with a new Python
object (a fresh `ForgeSD` / ldm model).  The sampler's cached denoiser still
holds a reference to the **old** sd_model.  Both objects are alive, so no error
is raised.

At sampling time, `self.model_wrap.inner_model.forge_objects.unet` therefore
resolves to the *old* model's UnetPatcher — a different Python object from the
one that received `add_wrapper_with_key()` in `forge_loader.py`.

Confirmed with `id()` tracing:

```
[forge_loader]     wrapper on unet id=…393072   wrappers=['apply_model']
[processing:1041]  forge_objects.unet id=…393072  ← correct
[processing:1463]  forge_objects.unet id=…393072  ← still correct
[kdiffusion.sample] unet id=…969088 wrappers=[]  ← STALE object, no wrapper
[sampling_prepare]  unet id=…969088 wrappers=[]  ← same stale object
```

### Fix — `_ensure_diffusers_wrapper()` in `modules_forge/forge_sampler.py`

Rather than relying on the wrapper being present from load time, re-register it
lazily in `sampling_prepare()` on the exact UnetPatcher instance received:

```python
# modules_forge/forge_sampler.py

def _ensure_diffusers_wrapper(unet):
    key = patcher_extension.WrappersMP.APPLY_MODEL
    if "forge_diffusers" in unet.wrappers.get(key, {}):
        return  # already present — idempotent guard

    from modules import sd_models
    _dp = getattr(sd_models.model_data.get_sd_model(), "diff_pipeline", None)
    if _dp is None:
        return

    def _diff_apply_model_wrapper(executor, x, t, ...):
        return _dp.apply_model(x, t, ...)

    unet.add_wrapper_with_key(key, "forge_diffusers", _diff_apply_model_wrapper)

def sampling_prepare(unet, x):
    ...
    _ensure_diffusers_wrapper(unet)   # ← lazy re-registration

    # Propagate wrappers into transformer_options for _calc_cond_batch
    unet.model_options.setdefault("transformer_options", {})["wrappers"] = \
        patcher_extension.copy_nested_dicts(unet.wrappers)
    ...
```

The wrapper propagation into `transformer_options` was the *other* half of the
original fix (already committed before this investigation): `BaseModel.apply_model()`
reads wrappers from `transformer_options["wrappers"]`, not from `unet.wrappers`
directly.  Both halves are required.

### Additional fixes in the same session

| File | Change |
|---|---|
| [modules_forge/forge_sampler.py](modules_forge/forge_sampler.py) | `sampling_cleanup()` now pops `transformer_options["wrappers"]` to prevent stale state on model switch |
| [ldm_patched/modules/model_base.py](ldm_patched/modules/model_base.py) | Replaced `print()` debug with `logging.debug()` conditional on wrappers present |
| [diff_pipeline/pipeline.py](diff_pipeline/pipeline.py) | Added `log.debug()` entry in `apply_model()` for per-step tracing |

### Verification

Both plain and LoRA generation completed without errors.  The HF Diffusers
attention path was confirmed active by the
`cross_attention_kwargs ['transformer_options'] are not expected by AttnProcessor2_0`
warning — this fires because the default `AttnProcessor2_0` receives
`transformer_options` via `cross_attention_kwargs` before `ForgeAttnProcessor`
takes over.  (Phase 4 note: this warning will disappear once `ForgeAttnProcessor`
is installed on `attn1` as well, or once the passthrough is filtered at the
`DiffPipeline` level.)

---

## Key Files Quick Reference

| File | Change |
|---|---|
| [diff_pipeline/pipeline.py](diff_pipeline/pipeline.py) | Full implementation: `DiffPipeline`, `ForgeAttnProcessor`, block address tables |
| [ldm_patched/modules/model_base.py:445](ldm_patched/modules/model_base.py#L445) | `SDXL.extra_conds()` — emits `adm_text_embeds` + `adm_time_ids` |
| [modules_forge/forge_loader.py:531](modules_forge/forge_loader.py#L531) | Instantiation: `sd_model.diff_pipeline = DiffPipeline(...)` |
| [modules/cmd_args.py:152](modules/cmd_args.py#L152) | `--forge-diffusers-pipeline` flag |
| [requirements.txt](requirements.txt) | Added `diffusers` |
| [ldm_patched/modules/utils.py:266](ldm_patched/modules/utils.py#L266) | `unet_to_diffusers()` — key map reused at construction (no change) |
| [modules_forge/forge_sampler.py](modules_forge/forge_sampler.py) | `_ensure_diffusers_wrapper()` + wrapper propagation + cleanup |
