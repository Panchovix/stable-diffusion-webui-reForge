#!/usr/bin/env python3
"""
tools/verify_jax_unet.py — Numeric-parity harness for jax_pipeline's SDXL UNet.

Loads a real SDXL checkpoint, builds fixed-seed synthetic conditioning, and
runs one apply_model() forward pass through both:
  * the reference PyTorch UNet   (unet_patcher.model._apply_model — the
    private method, called directly so this is always the "vanilla" path
    regardless of what WrappersMP.APPLY_MODEL wrappers happen to be
    registered)
  * the new JAX UNet             (jax_pipeline.pipeline.JAXSDXLPipeline)

and reports max-abs-diff / mean-abs-diff / cosine similarity between the two
denoised outputs. This is the required gate before jax_pipeline.apply_model
is ever wired into real sampling — run it after any change to
jax_pipeline/unet.py or jax_pipeline/convert.py.

This script performs a minimal headless bootstrap (parse args, initialize
shared state, load one checkpoint synchronously) rather than the full
modules.initialize.initialize() sequence — we only need model weights
loaded, not the Gradio UI, extensions, or forge's async main_thread
dispatch.

Usage
-----
    python tools/verify_jax_unet.py <checkpoint_name_or_path> [options]

``checkpoint_name_or_path`` is matched the same way the webui UI resolves a
checkpoint (title, filename, or path) via modules.sd_models; omit it to use
whatever checkpoint is already configured as default.

Options: --height/--width (default 128x128 px -> fast 16x16-latent check;
use 1024x1024 for a slower full-resolution check), --batch, --seed, --atol.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _bootstrap_webui(checkpoint):
    """Minimal headless bootstrap: parse cmd args, initialize shared state,
    load one checkpoint synchronously on the calling thread. No Gradio
    server, no extension/script loading, no forge main_thread dispatch.
    """
    from modules import cmd_args, shared_cmd_options

    argv = ["--skip-prepare-environment", "--skip-torch-cuda-test", "--skip-python-version-check"]
    if checkpoint:
        argv += ["--ckpt", checkpoint]
    shared_cmd_options.cmd_opts = cmd_args.parser.parse_args(argv)

    from modules import initialize
    initialize.imports()
    initialize.check_versions()

    from modules import sd_models
    sd_models.setup_model()
    sd_models.list_models()

    checkpoint_info = sd_models.select_checkpoint()
    return sd_models.load_model(checkpoint_info)


def _make_fixed_inputs(sd_model, batch, height, width, seed):
    """Fixed-seed synthetic latent + conditioning. Default 128x128 px keeps
    the check fast (16x16 latent); pass --height/--width 1024 for a
    full-resolution check once the fast check passes.
    """
    import torch

    device = sd_model.forge_objects.unet.load_device
    g = torch.Generator(device="cpu").manual_seed(seed)

    latent_h, latent_w = height // 8, width // 8
    x = torch.randn(batch, 4, latent_h, latent_w, generator=g).to(device=device, dtype=torch.float32)
    sigma = torch.full((batch,), 7.5, device=device, dtype=torch.float32)
    context = torch.randn(batch, 77, 2048, generator=g).to(device=device, dtype=torch.float32)
    pooled = torch.randn(batch, 1280, generator=g).to(device=device, dtype=torch.float32)
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=torch.float32, device=device,
    ).expand(batch, -1)

    return dict(
        x=x, sigma=sigma, c_crossattn=context,
        kwargs=dict(adm_text_embeds=pooled, adm_time_ids=time_ids),
    )


def _run_reference(sd_model, inputs):
    """Vanilla PyTorch UNet, bypassing WrappersMP.APPLY_MODEL entirely."""
    unet_model = sd_model.forge_objects.unet.model
    return unet_model._apply_model(
        inputs["x"], inputs["sigma"],
        c_crossattn=inputs["c_crossattn"],
        transformer_options={},
        **inputs["kwargs"],
    )


def _run_jax(sd_model, inputs):
    from jax_pipeline.pipeline import JAXSDXLPipeline

    jax_pipe = JAXSDXLPipeline(sd_model.forge_objects.unet, sd_model)
    return jax_pipe.apply_model(
        inputs["x"], inputs["sigma"],
        c_crossattn=inputs["c_crossattn"],
        transformer_options={},
        **inputs["kwargs"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=None,
                         help="Checkpoint name/title/path (default: currently configured checkpoint)")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.05, help="Max-abs-diff threshold for a PASS verdict")
    args = parser.parse_args()

    print(f"[verify_jax_unet] Loading checkpoint: {args.checkpoint or '(default)'}")
    sd_model = _bootstrap_webui(args.checkpoint)

    if not getattr(sd_model, "is_sdxl", False):
        print("[verify_jax_unet] ERROR: loaded checkpoint is not SDXL - jax_pipeline v1 is SDXL-only.")
        sys.exit(1)

    print(f"[verify_jax_unet] Building fixed inputs (seed={args.seed}, {args.height}x{args.width}, batch={args.batch})")
    inputs = _make_fixed_inputs(sd_model, batch=args.batch, height=args.height, width=args.width, seed=args.seed)

    print("[verify_jax_unet] Running reference PyTorch UNet...")
    ref = _run_reference(sd_model, inputs).float()

    print("[verify_jax_unet] Running JAX UNet (first call includes JIT compile time)...")
    out = _run_jax(sd_model, inputs)

    if out is None:
        print("[verify_jax_unet] ERROR: JAX apply_model returned None (unexpected - no ControlNet was passed).")
        sys.exit(1)
    out = out.float()

    import torch

    diff = (ref - out).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()

    print("\n[verify_jax_unet] Results")
    print(f"  max_abs_diff  = {max_abs:.6f}")
    print(f"  mean_abs_diff = {mean_abs:.6f}")
    print(f"  cosine_sim    = {cos_sim:.6f}")
    print(f"  ref stats: mean={ref.mean().item():.6f} std={ref.std().item():.6f}")
    print(f"  jax stats: mean={out.mean().item():.6f} std={out.std().item():.6f}")

    verdict = "PASS" if max_abs <= args.atol else "FAIL"
    print(f"\n[verify_jax_unet] {verdict} (atol={args.atol})")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
