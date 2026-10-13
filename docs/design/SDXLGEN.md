# SDXLGEN — Generate Button → SDXL Inference Path

Full trace of the code path from clicking "Generate" in the UI through to the SDXL
UNet forward pass and back.

---

## Phase 1: UI Entry

[modules/ui.py](modules/ui.py) wires the Generate button to `txt2img()`.

`txt2img()` dispatches via `main_thread.run_and_wait_result()` → `txt2img_function()`.

---

## Phase 2: Processing Object Build

[modules/txt2img.py:108](modules/txt2img.py#L108) `txt2img_function()`:

1. Calls `txt2img_create_processing()` → constructs
   `StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, width, height, cfg_scale, ...)`
2. Attaches extension scripts: `p.scripts = modules.scripts.scripts_txt2img`
3. Runs `modules.scripts.scripts_txt2img.run(p, *args)` — extension early-exit hook
4. Falls through to `processing.process_images(p)`

---

## Phase 3: Main Inference Loop

[modules/processing.py:938](modules/processing.py#L938) `process_images_inner()`:

1. `p.setup_prompts()` — expands prompt styles
2. `p.scripts.process(p)` — extension hook (before conditioning)
3. `sd_unet.apply_unet()` — applies any unet overrides
4. Per-batch iteration:
   - `extra_networks.activate(p, p.extra_network_data)` — activates LoRA/TI from prompt tags
   - `p.sd_model.forge_objects = p.sd_model.forge_objects_after_applying_lora.shallow_copy()`
   - `p.scripts.process_batch(p, ...)` — extension hook
   - **`p.setup_conds()`** — encodes prompts via both CLIP encoders (see below)
   - Optional ZeroSNR sigma rescale
   - **`samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, ...)`**

### setup_conds detail (SDXL-specific)

[modules/processing.py:560](modules/processing.py#L560) `setup_conds()` calls
`get_conds_with_caching()` → `prompt_parser.get_multicond_learned_conditioning()` which
drives `sd_hijack` through both SDXL CLIP encoders, producing:

```
p.c  = {"crossattn": Tensor(B, 77×n, 2048),   # clip_l(768) ++ clip_g(1280) concatenated
         "vector":   Tensor(B, 1280)}           # pooled CLIP-G output
p.uc = same structure for negative prompt
```

---

## Phase 4: Sampler Setup

[modules/processing.py:1427](modules/processing.py#L1427) `StableDiffusionProcessingTxt2Img.sample()`:

1. `sd_samplers.create_sampler(name, sd_model)` — creates a `CFGDenoiserKDiffusion`
   wrapping a k-diffusion schedule sampler (euler, dpmpp_2m, etc.)
2. `scripts.process_before_every_sampling(p, x=noise, ...)` — extension hook
3. **`samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, ...)`**
   — launches the denoising loop over all timesteps

---

## Phase 5: Denoiser Forward (called once per diffusion step)

[modules/sd_samplers_cfg_denoiser.py:113](modules/sd_samplers_cfg_denoiser.py#L113)
`CFGDenoiser.forward(x, sigma, uncond, cond, cond_scale, ...)`:

1. Reconstructs prompt-schedule cond/uncond for current step via
   `prompt_parser.reconstruct_multicond_batch(cond, self.step)`
2. Fires `cfg_denoiser_callback` (extension hook)
3. Optional dynamic CLIP skip swap
4. Padding of cond/uncond to equal token length if needed
5. **Format conversion** via [modules_forge/forge_sampler.py:8](modules_forge/forge_sampler.py#L8)
   `cond_from_a1111_to_patched_ldm()`:

   ```
   a1111 format                        patched-ldm format
   ─────────────────────────────────────────────────────
   cond["crossattn"]  (B, 77n, 2048)  → model_conds["c_crossattn"]  CONDCrossAttn
   cond["vector"]     (B, 1280)        → model_conds["y"]            CONDRegular
   ```

6. If ControlNet active: attaches `control` linked list to each cond dict
7. Runs `conditioning_modifiers` (PAG, SEG, etc.) registered in `model_options`
8. **`sampling_function(model, x, sigma, uncond, cond, cond_scale, model_options, seed)`**

---

## Phase 6: Conditioning Encoding & Batch Assembly

[ldm_patched/modules/samplers.py:985](ldm_patched/modules/samplers.py#L985)
`process_conds()`:

- Calls `encode_model_conds(model.extra_conds, conds[k], noise, device, k, ...)`
- This invokes **`SDXL.extra_conds()`** →
  **`SDXL.encode_adm()`** ([ldm_patched/modules/model_base.py:445](ldm_patched/modules/model_base.py#L445)):

  ```python
  # Builds the 2816-dim ADM conditioning vector y:
  clip_pooled  = model_conds["y"]          # 1280-dim pooled CLIP-G
  fourier_dims = []
  for scalar in [orig_h, orig_w, crop_h, crop_w, tgt_h, tgt_w]:
      fourier_dims.append(Timestep(256)(scalar))  # each → 256-dim
  y = concat(clip_pooled, *fourier_dims)   # 1280 + 6×256 = 2816
  ```

  Result stored back as `model_conds["y"]` (CONDRegular).

[ldm_patched/modules/samplers.py:213](ldm_patched/modules/samplers.py#L213)
`_calc_cond_batch()` then:

1. Batches all cond/uncond chunks together
2. Assembles `transformer_options`:
   - `patches` — LoRA/hook weight patches
   - `patches_replace` — per-block attention replacements (IP-Adapter etc.)
   - `cond_or_uncond` — batch index array
   - `sigmas` — current timestep
3. ControlNet forward: `c["control"] = control.get_control(input_x, timestep, c, ...)`
4. Dispatches to model — **extension intercept point**:

   ```python
   # Line 331-334
   if 'model_function_wrapper' in model_options:
       output = model_options['model_function_wrapper'](
           model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, ...}
       )
   else:
       output = model.apply_model(input_x, timestep_, **c)
   ```

---

## Phase 7: Model apply_model

[ldm_patched/modules/model_base.py:153](ldm_patched/modules/model_base.py#L153)
`SDXL.apply_model()` — wrapped by `WrapperExecutor` for `APPLY_MODEL` wrappers —
calls `_apply_model()`:

1. `xc = model_sampling.calculate_input(sigma, x)` — sigma preconditioning (EDM/EPS)
2. `t = model_sampling.timestep(sigma).float()` — sigma → integer timestep
3. Casts `xc`, `context`, `extra_conds` to model dtype (bf16/fp16/fp32)
4. Collects extra conds from kwargs: `c_crossattn` (2048-dim), `y` (2816-dim)
5. **`model_output = self.diffusion_model(xc, t, context=c_crossattn, control=control, transformer_options=transformer_options, y=y)`**
   — fires the SDXL UNet
6. `return model_sampling.calculate_denoised(sigma, model_output, x)`

---

## Phase 8: SDXL UNet Forward

`ldm_patched/ldm/modules/openaimodel.py` `UNetModel.forward()`:

1. **Time embedding**: `t_emb = timestep_embedding(t)` → 1280-dim via sinusoidal + MLP
2. **ADM embedding**: `label_emb(y)` — 2816-dim linear → adds to time embedding
3. **Encoder path** (12 input blocks):
   - Blocks 0–3: pure ResNet (no attention), spatial dims 128→64
   - Blocks 4–5: ResNet + 2 transformer layers (2048-dim cross-attn)
   - Blocks 6–8: ResNet, spatial 64→32
   - Blocks 9–11: ResNet + 10 transformer layers (2048-dim cross-attn)
   - At each transformer block: checks `transformer_options["patches_replace"]["attn2"][(block_name, block_idx, t_idx)]` → replaces attention if present
   - Checks `transformer_options["patches"]["attn2_patch"]` → modifies Q/K/V inputs
   - `add_block_modifier` forward hooks fire on each block output
4. **Middle block**: 10 transformer layers, 2048-dim cross-attn
5. **Decoder path** (12 output blocks): symmetric to encoder, skip connections concatenated
6. ControlNet residuals from `control["input"][N]` / `control["middle"][0]` added at matching block outputs

---

## Phase 9: CFG Blending

Back in [ldm_patched/modules/samplers.py:372](ldm_patched/modules/samplers.py#L372)
`cfg_function()`:

1. `sampler_cfg_function` hook fires if registered (DynamicCFG, PAG, SEG)
2. Default CFG blend: `cfg_result = uncond_pred + (cond_pred - uncond_pred) × scale`
3. `post_cfg_function` hooks fire (one per registered fn)

---

## Phase 10: Return Path

Denoised latent bubbles back through `sampler.sample()` → `p.sample()` →
`process_images_inner()`:

- `p.scripts.post_sample(p, ps)` — extension hook
- `devices.test_for_nans(samples_ddim, "unet")`
- `decode_latent_batch(sd_model, samples_ddim, ...)` → VAE decode → `[-1,1]` pixel tensor
- Clamp + rescale → uint8 numpy → `PIL.Image`
- `images.save_image(...)` → PNG saved to `outdir_txt2img_samples/`
- Returns `processed.images` to Gradio → displayed in UI gallery

---

## SDXL vs SD1.5 Key Differences

| Point | SD1.5 | SDXL |
|---|---|---|
| CLIP encoders | 1 (clip_l, 768-dim) | 2 (clip_l 768 + clip_g 1280, concat → 2048-dim) |
| `cond["vector"]` | absent | pooled CLIP-G (1280-dim) |
| `encode_adm()` called | no | yes — builds 2816-dim `y` from pooled + size Fourier embeddings |
| ADM feeding | — | `label_emb` linear in UNet adds to time embedding |
| Transformer depth | 1 per block | up to 10 per block (middle + top decoder blocks) |
| Attention context dim | 768 | 2048 |
| Model channels | 320 | 320 (base) / 384 (refiner) |

---

## Extension Intercept Points (in order)

| Step | Hook | File |
|---|---|---|
| Before conditioning | `scripts.process(p)` | [modules/processing.py:993](modules/processing.py#L993) |
| LoRA activation | `extra_networks.activate()` | [modules/processing.py:1039](modules/processing.py#L1039) |
| Per-batch | `scripts.process_batch(p)` | [modules/processing.py:1044](modules/processing.py#L1044) |
| Before sampling | `scripts.process_before_every_sampling()` | [modules/processing.py:1462](modules/processing.py#L1462) |
| Per denoiser step | `cfg_denoiser_callback` | [modules/sd_samplers_cfg_denoiser.py:173](modules/sd_samplers_cfg_denoiser.py#L173) |
| Conditioning modifiers | `conditioning_modifiers` list | [modules/sd_samplers_cfg_denoiser.py:239](modules/sd_samplers_cfg_denoiser.py#L239) |
| Entire UNet call | `model_function_wrapper` | [ldm_patched/modules/samplers.py:331](ldm_patched/modules/samplers.py#L331) |
| Per attention block | `patches_replace["attn2"]` | [ldm_patched/ldm/modules/attention.py:812](ldm_patched/ldm/modules/attention.py#L812) |
| Per attention inputs | `patches["attn2_patch"]` | [ldm_patched/ldm/modules/attention.py:834](ldm_patched/ldm/modules/attention.py#L834) |
| Per block output | `add_block_modifier` hooks | [modules_forge/unet_patcher.py:246](modules_forge/unet_patcher.py#L246) |
| CFG | `sampler_cfg_function` | [ldm_patched/modules/samplers.py:375](ldm_patched/modules/samplers.py#L375) |
| Post CFG | `post_cfg_function` | [ldm_patched/modules/samplers.py:396](ldm_patched/modules/samplers.py#L396) |
| Post sample | `scripts.post_sample(p, ps)` | [modules/processing.py:1099](modules/processing.py#L1099) |
| Post batch | `scripts.postprocess_batch()` | [modules/processing.py:1122](modules/processing.py#L1122) |
