# Comfy Kitchen INT8 Attention

Port of upstream ComfyUI's Comfy Kitchen integration into reForge, tracking:

- **PR #15479** — "Implement comfy kitchen attention" (comfyanonymous/ComfyUI,
  now Comfy-Org/ComfyUI) — original `--use-ck-attention` flag, the
  `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE` probe, `attention_comfy_kitchen_int8`,
  and the `ModelAttentionBackend` node.
- **PR #16154** — "chore: Harmonize model attention nodes" — migrated the node
  to the newer `io.ComfyNode`/schema API; no behavioral change to the attention
  path itself.

Reference commits used while porting (from a throwaway clone of
`comfyanonymous/ComfyUI`, not committed to this repo):
`bf4c9a08` (#15479) and `5bbdf8a7` (#16154).

## What Comfy Kitchen is

[`comfy-kitchen`](https://github.com/Comfy-Org/comfy-kitchen) is Comfy-Org's
kernel library for diffusion inference: eager/cuda/triton/hip backends
providing quantized GEMMs (fp8/NVFP4/MXFP8/INT8), fused RoPE/AdaLN, and INT8
attention. This port only wires in **INT8 attention** (the part upstream
exposes as a selectable attention backend); it does not port the
quantized-GEMM or fused-RoPE/AdaLN kernels.

## Files changed

- `requirements.txt` — `comfy-kitchen` added as a **commented-out, optional**
  dependency with an explanation, not an unconditional line (see "Install
  requirements" below for why).
- `ldm_patched/modules/args_parser.py` — new flags: `--use-ck-attention`,
  `--enable-triton-backend`, `--disable-triton-backend` (mutually exclusive
  with each other).
- `ldm_patched/modules/model_management.py` — `comfy_kitchen_attention_enabled()`
  and `comfy_kitchen_triton_backend_enabled()` helpers (mirrors the existing
  `sage_attention_enabled()` / `flash_attention_enabled()` pattern).
- `ldm_patched/ldm/modules/attention.py` — availability probe
  (`COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE`), `attention_comfy_kitchen_int8`,
  a minimal `REGISTERED_ATTENTION_FUNCTIONS` / `register_attention_function` /
  `get_attention_function` registry (reForge didn't have one; upstream's is
  ported in simplified form, without the tensor-container/prequantization fast
  path since reForge doesn't yet have the wrap_attn container plumbing
  upstream added alongside it), and backend-selection wiring.
- `modules/shared_items.py` — `dit_attention_backend_choices()`, following the
  existing `sd_vae_items()` / `sd_unet_items()` "dynamic choices" convention.
- `modules/shared_options.py` — `dit_attention_backend` dropdown under
  Settings → Optimizations, appearing next to the (currently commented-out)
  `cross_attention_optimization` option. The "Comfy Kitchen (INT8)" choice is
  only added to the dropdown when `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE`
  is true, matching how upstream's `ModelAttentionBackend` node only lists
  "comfy kitchen attention" when available.
- `docs/comfy_kitchen_attention.md` — this file.

## Install requirements

`comfy-kitchen` is listed on PyPI with wheels for:

| Python | Linux (manylinux x86_64 / aarch64) | Windows (amd64 / arm64) | macOS |
|---|---|---|---|
| 3.10 | yes | yes | **no** |
| 3.11 | yes | yes | **no** |
| 3.12–3.14 (`cp312-abi3`) | yes | yes (amd64 + arm64) | **no** |

There is also a `py3-none-any` wheel published, but the project's own
cp3xx-tagged wheels take precedence for compiled extension loading; the
`py3-none-any` one is not a substitute for the native kernels on unsupported
platforms.

**Gap:** there is no macOS/Darwin wheel at all (checked PyPI JSON API,
`comfy-kitchen==0.2.36`, 2026-09-30). Since reForge explicitly targets Python
3.11–3.14 across platforms including macOS (this dev machine included, Apple
Silicon, MPS backend), `comfy-kitchen` **cannot** be a hard/unconditional
requirement — `pip install -r requirements.txt` would fail on macOS. It is
therefore commented out in `requirements.txt` with install instructions,
exactly like reForge already treats `sageattention` and `flash-attn` (neither
of which are in `requirements.txt` either — they're documented "install
yourself if you want the flag" optional deps).

Install manually on a supported machine:

```bash
pip install comfy-kitchen
```

## Hardware compatibility matrix

| Feature | Requirement | Vendor |
|---|---|---|
| INT8 attention (ported here) | works on both CUDA and HIP backends; no specific compute-capability floor called out upstream | Nvidia (any CUDA-capable card) and AMD (ROCm/HIP) |
| fp8 compute (quantized GEMM, not ported) | SM ≥ 8.9 (Ada / RTX 40-series and newer) | Nvidia only |
| NVFP4 / MXFP8 (quantized GEMM, not ported) | SM ≥ 10.0 (Blackwell / RTX 50-series) | Nvidia only |

This means INT8 attention specifically is relevant across the broad
RTX 20–50 (and equivalent AMD) install base, not just the newest cards — which
is why it was prioritized over the fp8/NVFP4/MXFP8 GEMM kernels for this pass.

## CLI usage

```bash
# Use Comfy Kitchen INT8 attention as the default attention backend
python launch.py --use-ck-attention

# Optional: force/deny the triton backend within comfy_kitchen
python launch.py --use-ck-attention --enable-triton-backend
python launch.py --use-ck-attention --disable-triton-backend
```

These flags live in `ldm_patched/modules/args_parser.py`, the same
argparse namespace as `--use-sage-attention`, `--use-sage-attention3`, and
`--use-flash-attention` — i.e. they govern the ldm_patched/DiT attention
backend selection, not the legacy SD1.x/SDXL U-Net path in
`modules/cmd_args.py` (`--xformers`, `--opt-sdp-attention`, etc). Like those
existing flags, `--use-ck-attention` is placed in a mutually-exclusive
`attn_group` so multiple forced backends can't be requested at once.

## Exit-vs-fallback decision (required design call)

**Upstream ComfyUI behavior:** if `--use-ck-attention` is passed but
`comfy_kitchen.int8_attention_is_available()` is False, ComfyUI logs an error
and calls `exit(-1)` — the whole process dies.

**reForge behavior (this port): logs a warning and falls back** to reForge's
normal attention auto-selection (pytorch attention if available, otherwise
split/sub-quadratic), the process keeps running.

**Why we diverged:**

1. **Existing reForge convention already does this for its other optional
   attention backends.** `attention_sage()` and `attention_sage3()` in
   `ldm_patched/ldm/modules/attention.py` both wrap their call in
   `try/except` and fall back to `attention_pytorch` with a logged error on
   failure — they never raise or exit. Only the *import-time* dependency
   check for `--use-sage-attention`/`--use-flash-attention` still calls
   `exit(-1)` today if the package is flatly missing (that's upstream
   behavior reForge inherited unchanged; we did not "fix" it as it's out of
   scope for this task, but note it's the one place reForge's own pattern is
   actually closer to upstream's hard-exit style). For the *runtime*
   attention-selection path (which is what `--use-ck-attention` drives), the
   established reForge convention is clearly "log and fall back", so we
   followed that for consistency rather than porting upstream's exit(-1)
   uncritically.
2. **reForge is a single-user, interactive webui**, not a
   batch/server/headless pipeline runner like ComfyUI. Killing the entire
   application because one performance flag's dependency is missing is a
   much worse failure mode here than it is for a workflow-queue backend:
   users would lose their whole session over what is, functionally, an
   opt-in speed knob.
3. **Platform gap makes a hard exit especially harsh.** Since `comfy-kitchen`
   has no macOS wheel, any script, saved launch config, or shared
   command-line flags file that includes `--use-ck-attention` (e.g. copied
   from a Linux/Windows setup) would permanently break the webui on macOS
   with no recovery path other than editing the launch flags file.
4. We still *preserve the signal*: the warning at startup is a
   `logging.warning(...)` (not silent), names the exact reason (`comfy-kitchen`
   not installed / no wheel for the platform / hardware check failed), and
   points the user at `pip install comfy-kitchen` and this document.

If this project ever wants upstream's stricter fail-fast behavior back (e.g.
for a dedicated server/headless deployment mode), the right place to add it is
a new explicit flag (e.g. `--ck-attention-strict`) that opts into `exit(-1)`,
rather than changing the default — so the default UX for the general webui
stays fault-tolerant.

## Known limitations

- **No GPU in this development environment.** This port was authored and
  smoke-tested on Apple Silicon macOS with the MPS backend and no Nvidia/AMD
  GPU available. `comfy_kitchen` is not installed here (it has no macOS
  wheel — see above), so:
  - The actual INT8 attention numerics, the triton/cuda/hip kernel dispatch,
    and any real performance/quality characteristics are **unverified** —
    there are no benchmark numbers in this document and none should be
    assumed or fabricated.
  - What *is* verified: the module imports and initializes cleanly with
    `comfy_kitchen` absent, `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE`
    correctly evaluates to `False`, `--use-ck-attention` does not crash or
    exit the process and correctly falls back with a warning, and the
    `dit_attention_backend` Settings dropdown only offers
    "Automatic"/"PyTorch" (no "Comfy Kitchen (INT8)" entry) when the package
    is unavailable, exactly as intended.
- **The Gradio dropdown is currently informational only.** It reports what
  would be available and documents the intended pairing with
  `--use-ck-attention`, but wiring a live per-generation backend switch
  (equivalent to upstream's `ModelAttentionBackend` node, which patches a
  specific `ModelPatcher` instance via `set_model_optimized_attention`) into
  reForge's UI for the DiT pipeline was left out of scope for this task; the
  CLI flag is the supported way to actually enable the backend right now.
- **The tensor-container / prequantization fast path** upstream added in
  #15479 (`AttentionTensorContainer`, `container_function` on `wrap_attn`,
  `prequantize_int8_attention` / `int8_attention_from_prequantized`) was not
  ported — it depends on upstream's `wrap_attn` container-passing machinery,
  which reForge's `attention.py` does not currently have. The ported
  `attention_comfy_kitchen_int8` calls `comfy_kitchen.int8_attention(...)`
  directly, which is functionally correct but skips that specific
  memory-optimization path.
- No automated test suite in this repo currently exercises the ldm_patched
  attention backends (see "Testing" below) — the verification here is a
  manual smoke test, not a `pytest` regression test.

## Testing performed

- `python -m py_compile` on every modified file: passed.
- Direct import + exercise of `ldm_patched.modules.args_parser`,
  `ldm_patched.modules.model_management`, and
  `ldm_patched.ldm.modules.attention` in the repo's `venv` (Python 3.13,
  torch 2.11, macOS/MPS, `comfy_kitchen` NOT installed): module import
  succeeds, `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE` is `False`,
  `get_attention_function('pytorch')` resolves, `get_attention_function`
  with an unregistered name returns the provided default instead of
  raising, and passing `--use-ck-attention` logs the fallback warning and
  leaves `optimized_attention` pointing at the auto-selected default
  (`attention_sub_quad` on this machine) without exiting the process.
- `pytest test/ --no-server --collect-only`: 101 tests collected (repo does
  have a `test/` suite with `conftest.py`; correcting an earlier assumption).
  Several of those (`test_txt2img.py`, `test_img2img.py`, `test_extras.py`,
  `test_face_restorers.py`, `test_utils.py`) need a live `webui.py` server
  even with `--no-server` (that flag only skips *auto-launching* one) and
  were not run here — starting the full webui server was out of scope for a
  no-GPU environment and this attention-only change.
- Ran the subset that does not need a server:
  `pytest test/test_diff_pipeline_pipeline.py test/test_torch_utils.py
  test/test_jax_pipeline_convert.py --no-server`: **16 failed, 54 passed, 1
  skipped**, identical before and after this change (verified by running the
  same command against `git stash` of these edits, then restoring them) —
  i.e. all 16 failures are pre-existing and unrelated to this port (they're
  in `diff_pipeline/pipeline.py`'s `PassthroughAttnProcessor` /
  `_apply_model_conditioning_bridge` / `_apply_model_controlnet_mapping` and
  a `torch_utils.get_param` mock-patching issue in `test_torch_utils.py`, none
  of which touch `ldm_patched/ldm/modules/attention.py`,
  `ldm_patched/modules/args_parser.py`, or
  `ldm_patched/modules/model_management.py`). This change introduces **zero**
  new test failures.
