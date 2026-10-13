# NORPIP — Normal Load Pipeline

Deep-dive into how a checkpoint is loaded in reForge, from the user's model selection to a ready `sd_model` object.

---

## Overview

```
load_model()                          [modules/sd_models.py:1089]
  └─ get_checkpoint_state_dict()      [modules/sd_models.py:705]
  └─ load_model_for_a1111()           [modules_forge/forge_loader.py:389]
        ├─ instantiate_from_config()  (a1111 YAML shell, U-Net/VAE/CLIP = FakeObject)
        └─ load_checkpoint_guess_config()
              └─ load_state_dict_guess_config()
                    ├─ model_detection.model_config_from_unet()
                    ├─ model_config.get_model() + load_model_weights()
                    ├─ VAE(sd=vae_sd)
                    ├─ CLIP(clip_target, ...)
                    └─ UnetPatcher(model, ...)  →  ForgeSD
        └─ wire FakeObject slots back with real Forge components
  └─ sd_vae.load_vae()
  └─ script_callbacks.model_loaded_callback()
  └─ sd_text_encoder.apply_text_encoder()
  └─ get_empty_cond()
```

---

## Phase 1 — Entry Point: `load_model()` [modules/sd_models.py:1089](modules/sd_models.py#L1089)

1. **Cache check** (1097–1105): Scans `model_data.loaded_sd_models` for a matching filename. On hit, moves it to the front (LRU promotion) and returns immediately — no disk I/O.
2. **Forced reload** (1108–1114): If `forced_reload=True`, evicts the existing copy first via `complete_model_teardown()`.
3. **Limit enforcement** (1117–1118): If `len(loaded_sd_models) >= sd_checkpoints_limit`, calls `unload_first_loaded_model()` to evict the oldest model.
4. **Memory cleanup** (1121): `force_memory_deallocation()` — Python GC + `torch.cuda.empty_cache()`.
5. **State dict load** (1133): `get_checkpoint_state_dict(checkpoint_info, timer)`.
6. **Model construction** (1136): `forge_loader.load_model_for_a1111(timer, checkpoint_info, state_dict)`.
7. **Post-load wiring** (1149–1184): VAE, text encoder, callbacks, empty-prompt pre-computation, final GC.

The state dict is deleted in the `finally` block immediately after `load_model_for_a1111` returns to free RAM.

---

## Phase 2 — State Dict Loading: `get_checkpoint_state_dict()` [modules/sd_models.py:705](modules/sd_models.py#L705)

Calculates a short SHA hash for logging, then reads weights from disk:

| Format | Method |
|--------|--------|
| `.safetensors` (normal) | `safetensors.torch.load_file(filename, device=target_device)` — zero-copy mmap directly to device |
| `.safetensors` (disable_mmap) | Reads raw bytes → `safetensors.torch.load()` → moves tensors to device; then deletes raw bytes |
| `.ckpt` / `.pt` | `torch.load(map_location=target_device)` + `get_state_dict_from_checkpoint()` to unwrap nested `state_dict` key |

Target device is `shared.weight_load_location` if set, otherwise `model_management.get_torch_device()`.

---

## Phase 3 — Model Construction: `load_model_for_a1111()` [modules_forge/forge_loader.py:389](modules_forge/forge_loader.py#L389)

### Key innovation: FakeObject substitution

The a1111 YAML config is loaded via `OmegaConf`, but before calling `instantiate_from_config()`, three config targets are replaced with `FakeObject`:

```python
a1111_config.model.params.network_config.target  = 'FakeObject'  # U-Net
a1111_config.model.params.unet_config.target      = 'FakeObject'  # U-Net alt
a1111_config.model.params.first_stage_config.target = 'FakeObject'  # VAE
```

`FakeObject` is a stub that accepts any `__init__` args, has a no-op `eval()`, and returns `[]` from `parameters()`. CLIP is blocked via a `no_clip()` context manager that patches the CLIP import.

This produces a lightweight `sd_model` shell (scheduler config, parameterization, etc.) **without** allocating any real tensors for the heavy components.

**SD3 exception**: if `model.diffusion_model.x_embedder.proj.weight` is in the state dict, skip config entirely — use a bare `torch.nn.Module()`.

### Optional CLIP skip

If a separate text encoder is configured (or automatically matched by model name), `output_clip=False` is passed to `load_checkpoint_guess_config()` to avoid loading checkpoint CLIP weights that will be replaced anyway.

---

## Phase 4 — Forge Backend Loading: `load_state_dict_guess_config()` [modules_forge/forge_loader.py:234](modules_forge/forge_loader.py#L234)

This is where Forge's `ldm_patched/` backend takes over entirely.

### 4a — Architecture detection
```python
diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
```
Inspects state dict keys to identify architecture: SD1.x, SD2.x, SDXL, SSD-1B, SD3, Flux, etc. Returns an architecture-specific `model_config` object that knows VAE key prefixes, CLIP targets, supported dtypes, and latent format.

### 4b — dtype selection
```python
unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=..., weight_dtype=...)
manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, ...)
model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
```
Picks the best dtype (fp16/bf16/fp8/fp32) based on available VRAM and model parameter count. `manual_cast_dtype` handles cases where the model runs in one dtype but activations need casting.

### 4c — U-Net construction
```python
inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
model.load_model_weights(sd, diffusion_model_prefix)
```
Architecture-specific model is instantiated directly on the initial device (may be CPU to avoid OOM during construction).

### 4d — VAE construction
```python
vae_sd = utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
vae_sd = model_config.process_vae_state_dict(vae_sd)
vae = VAE(sd=vae_sd, metadata=metadata)
```
VAE keys are stripped of their prefix, optionally remapped, then loaded into a `VAE` object.

### 4e — CLIP construction
```python
clip_target = model_config.clip_target(state_dict=sd)
clip_sd = model_config.process_clip_state_dict(sd)
clip = CLIP(clip_target, embedding_directory=..., tokenizer_data=clip_sd, parameters=parameters)
clip.load_sd(clip_sd, full_model=True)
```
Architecture determines which CLIP variant(s) are instantiated. Missing `.logit_scale` and `.text_projection` keys are silently accepted (expected for some architectures).

### 4f — UnetPatcher wrapping
```python
model_patcher = UnetPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
if inital_load_device != torch.device("cpu"):
    model_management.load_models_gpu([model_patcher], force_full_load=True)
```
`UnetPatcher` wraps the U-Net and is the entry point for LoRA patching, weight injection, and VRAM lifecycle management. If initial load was on GPU, it's formally registered with the VRAM manager.

Returns `ForgeSD(model_patcher, clip, vae, clipvision)`.

---

## Phase 5 — Wiring FakeObject Slots: [modules_forge/forge_loader.py:443](modules_forge/forge_loader.py#L443)

The `sd_model` shell created in Phase 3 has its stubs replaced:

```python
sd_model.first_stage_model         = forge_objects.vae.first_stage_model
sd_model.model.diffusion_model      = forge_objects.unet.model.diffusion_model
sd_model.forge_objects              = forge_objects
sd_model.forge_objects_original     = forge_objects.shallow_copy()
sd_model.forge_objects_after_applying_lora = forge_objects.shallow_copy()
```

CLIP is wired per architecture:

| Architecture | Embedder class | Forge CLIP |
|---|---|---|
| SD 1.5 | `FrozenCLIPEmbedder` | `CLIP_SD_15_L` |
| SD 2.1 | `FrozenOpenCLIPEmbedder` | `CLIP_SD_21_H` |
| SDXL | `FrozenCLIPEmbedder` + `FrozenOpenCLIPEmbedder2` | `CLIP_SD_XL_L` + `CLIP_SD_XL_G` |

Tokenizers and transformers are swapped from the Forge CLIP objects. `EmbeddingsWithFixes` is injected at `token_embedding` to enable textual inversions.

`decode_first_stage` and `encode_first_stage` are monkey-patched as `@torch.inference_mode()` closures routing through `forge_objects.vae`, bypassing the original autoencoder code entirely.

Architecture flags set on `sd_model`:

| Flag | Condition |
|---|---|
| `is_sd1` | not sdxl, not sd2, not sd3 |
| `is_sd2` | not sdxl, not sd3, has `.model` on cond_stage |
| `is_sdxl` | has `conditioner`, not sd3 |
| `is_sd3` | has `x_embedder.proj.weight` in state dict |
| `is_ssd` | sdxl but missing middle block transformer key |
| `ztsnr` | has `ztsnr` key in state dict |

---

## Phase 6 — Post-load: [modules/sd_models.py:1149](modules/sd_models.py#L1149)

1. **VAEStructurePreserver** injected onto `sd_model.forge_objects` to protect VAE structure through LoRA application.
2. **VAE resolution**: `sd_vae.resolve_vae()` checks for a model-matched VAE file; `sd_vae.load_vae()` loads and applies it.
3. **Extension callbacks**: `script_callbacks.model_loaded_callback(sd_model)` — all extensions are notified synchronously.
4. **Text encoder swap**: `sd_text_encoder.apply_text_encoder()` — replaces CLIP with a separately-loaded TE if configured. Skipped if `_separate_te_loaded` was set during Phase 3.
5. **Empty prompt**: `get_empty_cond(sd_model)` pre-computes and caches the unconditional conditioning vector on `sd_model.cond_stage_model_empty_prompt`.
6. **Final GC**: `force_memory_deallocation()` clears any temporaries.

---

## Key Design Principle

The "normal" load is not using the original a1111 model construction path for the heavyweight components. The YAML config exists only to create the **outer shell** (scheduler, parameterization, conditioning wrappers). All actual weight tensors — U-Net, VAE, CLIP — are loaded and managed entirely by `ldm_patched/`, which owns:

- Architecture detection
- dtype/device selection
- VRAM lifecycle (load, offload, evict)
- LoRA/weight patching surface (`UnetPatcher`)

---

---

# SDXL Load Pipeline

SDXL is a first-class path that diverges from SD1.x/SD2.x at every stage: config selection, architecture fingerprint, dual-CLIP wiring, and conditioning.

---

## SDXL Detection Chain

### Step 1 — a1111 YAML config: `guess_model_config_from_state_dict()` [modules/sd_models_config.py:72](modules/sd_models_config.py#L72)

SDXL is detected by checking for the CLIP-G embedder key:

```python
if sd.get('conditioner.embedders.1.model.ln_final.weight', None) is not None:
    if diffusion_model_input.shape[1] == 9:
        return config_sdxl_inpainting   # sd_xl_inpaint.yaml
    else:
        return config_sdxl              # sd_xl_base.yaml
if sd.get('conditioner.embedders.0.model.ln_final.weight', None) is not None:
    return config_sdxl_refiner          # sd_xl_refiner.yaml
```

The discriminating key is `conditioner.embedders.1.model.ln_final.weight` — the layer-norm final weight of CLIP-G in the SDXL conditioner block. SD1/SD2 models have `cond_stage_model.*`, SDXL has `conditioner.embedders.*`.

### Step 2 — Forge architecture fingerprint: `model_config_from_unet_config()` [ldm_patched/modules/model_detection.py:608](ldm_patched/modules/model_detection.py#L608)

The `SDXL` class in `supported_models.py` registers this U-Net fingerprint:

```python
unet_config = {
    "model_channels": 320,
    "use_linear_in_transformer": True,
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "context_dim": 2048,
    "adm_in_channels": 2816,
    "use_temporal_attention": False,
}
```

`detect_unet_config()` reads these values from the state dict (channel shapes, transformer block counts), and `model_config_from_unet_config()` matches against the registered list. This is fully independent of the a1111 YAML.

### Variants matched by the same fingerprint

| Class | Key difference |
|---|---|
| `SDXL` | base config |
| `SDXL_flux2` | `in_channels=32, out_channels=32` |
| `SSD1B` | missing middle block key (`is_ssd` flag) |
| `Segmind_Vega`, `KOALA_700M/1B` | identical unet_config, different clip targets |
| `SDXL_instructpix2pix` | `in_channels=8` |
| `SDXLRefiner` | `model_channels=384`, `adm_in_channels=2560`, `context_dim=1280` |

---

## SDXL-Specific: Dual CLIP

SDXL uses two text encoders simultaneously. Their keys are split during `process_clip_state_dict()`:

```python
replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"
replace_prefix["conditioner.embedders.1.model."] = "clip_g."
state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
```

The `clip_target` returns `SDXLClipModel` wrapping both `clip_l` (OpenAI CLIP ViT-L/14, 768-dim) and `clip_g` (OpenCLIP ViT-bigG/14, 1280-dim). Their concatenated output (2048-dim) feeds the cross-attention context.

In Phase 5 wiring, each embedder is detected by class name and replaced:

```python
'FrozenCLIPEmbedder'     → CLIP_SD_XL_L  (clip_l tokenizer + transformer)
'FrozenOpenCLIPEmbedder2' → CLIP_SD_XL_G  (clip_g tokenizer + transformer + text_projection)
```

`text_projection` (1280→1280) is wired only for CLIP-G and is used to project the pooled text embedding into the ADM conditioning vector.

---

## SDXL-Specific: ADM Conditioning (Size Embeddings)

SDXL's `adm_in_channels=2816` means the U-Net receives an additional conditioning vector of 2816 dims beyond the text cross-attention. This encodes:
- `original_size_as_tuple` (H, W) — Fourier-embedded → 512 dims × 2
- `crop_coords_top_left` (top, left) — Fourier-embedded → 512 dims × 2
- `target_size_as_tuple` (H, W) — Fourier-embedded → 512 dims × 2
- pooled CLIP-G text embedding → 1280 dims

Total: 512×4 + 256×4 = 2048 + 768 = 2816. This is handled inside `extend_sdxl()` [modules/sd_models_xl.py], called at the end of Phase 5.

---

## SDXL-Specific: `extend_sdxl()` [modules/sd_models_xl.py](modules/sd_models_xl.py)

Called only when `sd_model.is_sdxl` is True. Patches the a1111 pipeline methods that need to know about SDXL's dual conditioning:
- `get_learned_conditioning()` — routes through dual CLIP
- `conditioner` interface — the SDXL outer model uses `conditioner.embedders[]` instead of a single `cond_stage_model`

---

## Diffusers SDXL: The Already-Existing Conversion Path

A complete Diffusers-format SDXL checkpoint (e.g. from HuggingFace `stabilityai/stable-diffusion-xl-base-1.0`) can already be loaded. The path activates in `load_diffusion_model_state_dict()` [modules_forge/forge_loader.py:313](modules_forge/forge_loader.py#L313):

```
Normal checkpoint state dict
  → model_config_from_unet()  ← tries ldm key prefix first
  → None (no "model.diffusion_model." prefix found)
  → convert_diffusers_mmdit()  ← try mmdit (SD3/Flux/AuraFlow) — fails for SDXL UNet
  → model_config_from_diffusers_unet()  ← detects "conv_in.weight" (Diffusers format)
      → unet_config_from_diffusers_unet()  ← counts down_blocks/attentions to rebuild unet_config
      → model_config_from_unet_config()    ← matches → SDXL class
  → unet_to_diffusers(model_config.unet_config)  ← builds ldm↔diffusers key map
  → remaps: down_blocks.N.resnets.M.* → input_blocks.N.M.*
            up_blocks.N.attentions.M.* → output_blocks.N.1.*
            mid_block.*               → middle_block.*
  → loads remapped sd into SDXL model as normal
```

`unet_config_from_diffusers_unet()` [model_detection.py:694](ldm_patched/modules/model_detection.py#L694) detects Diffusers SDXL by:
1. Presence of `conv_in.weight`
2. Counting transformer depth via `down_blocks.{i}.attentions.{j}.transformer_blocks.*`
3. Reading `context_dim` from `down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight.shape[1]`
4. Reading `adm_in_channels` from `add_embedding.linear_1.weight.shape[1]`
5. Matching the assembled config → `SDXL` class (transformer_depth=[0,0,2,2,10,10], context_dim=2048, adm_in_channels=2816)

### Limitation of this path

`load_diffusion_model_state_dict()` is only triggered as a **fallback** when `load_state_dict_guess_config()` cannot find a `model_config` — i.e. when the checkpoint has no `model.diffusion_model.*` prefix. It is **not** triggered during the normal `load_model_for_a1111()` path (Phase 3→4), which always receives a full SDXL checkpoint containing U-Net + VAE + CLIP.

For a pure Diffusers SDXL UNet-only file (no VAE/CLIP keys), the fallback path works and returns `ForgeSD(model_patcher, None, None, None)` — VAE and CLIP would need to be loaded separately.

### What would a full Diffusers pipeline switch require

To replace the entire load path with `diffusers.StableDiffusionXLPipeline.from_pretrained()`, the following substitutions would be needed:

| Current | Diffusers equivalent |
|---|---|
| `load_checkpoint_guess_config()` | `StableDiffusionXLPipeline.from_pretrained(repo_id)` |
| `ForgeSD.unet` (`UnetPatcher` around ldm U-Net) | `pipeline.unet` (HF `UNet2DConditionModel`) |
| `ForgeSD.vae` | `pipeline.vae` (HF `AutoencoderKL`) |
| `ForgeSD.clip` (`SDXLClipModel`) | `pipeline.text_encoder` + `pipeline.text_encoder_2` |
| `UnetPatcher` LoRA/weight patching | HF PEFT / `pipeline.load_lora_weights()` |
| `model_management` VRAM lifecycle | `pipeline.enable_model_cpu_offload()` / `enable_sequential_cpu_offload()` |
| `forge_objects.unet.model.model_config.latent_format` | `pipeline.vae.config.scaling_factor` |

The biggest friction points are:
1. **UnetPatcher has no HF equivalent** — all LoRA, ControlNet, and extension hooks attach to it. Replacing it would break the entire extension system.
2. **Conditioning format differs** — reForge encodes ADM conditioning inline in the sampler loop; HF pipelines encode it inside `pipeline.__call__()`.
3. **`decode_first_stage` / `encode_first_stage` monkey patches** in Phase 5 would need to route through `pipeline.vae` instead of `forge_objects.vae`.
4. **Key conversion already exists** (`unet_to_diffusers`) but only goes one direction (ldm→diffusers renaming for saving); loading Diffusers checkpoints requires the inverse mapping which is already implemented in the fallback path.
