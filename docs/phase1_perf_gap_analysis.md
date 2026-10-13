# Phase 1 Backend Performance Gap Analysis: reForge vs. Upstream ComfyUI

**Task:** t_03efa024 — Phase 1: Backend 性能对齐 ComfyUI 上游
**Scope:** `ldm_patched/modules/{model_management.py, ops.py, args_parser.py}` only. No new model
architectures, no video pipelines.
**Method:** `git clone --depth 1 https://github.com/comfyanonymous/ComfyUI` into
`/Users/nickzeng/.hermes/cache/scratch/ComfyUI_upstream` (read-only reference, untouched reForge
tree) at commit `b65d1ff` (2026-09-30). Upstream repo has moved to the `Comfy-Org/ComfyUI` org;
motivations below are sourced from PR titles/descriptions found via web search since the shallow
clone has only 1 commit of history (no local `git blame` history available for the changed lines).
**Target hardware:** RTX 20-series (Turing, sm_75) through RTX 50-series (Blackwell, sm_120) — every
finding below is evaluated for coverage across that whole range, not just the newest cards.

File sizes for scale: reForge `model_management.py` 1568 lines / upstream 2199 lines; reForge
`ops.py` 493 lines / upstream 1820 lines; reForge `args_parser.py` 209 lines / upstream `cli_args.py`
321 lines. Upstream has grown substantially — much of that growth is new-architecture support
(explicitly out of scope, see Exclusions), but several generically-useful performance/VRAM
mechanisms are also missing from reForge.

---

## 1. Dynamic VRAM management (comfy-aimdo)

- **Upstream:** `comfy/model_management.py:1114,1222,1233,1403-1490,1797`; flags in
  `comfy/cli_args.py:170,180-181,318-321`.
  - `--enable-dynamic-vram` / `--disable-dynamic-vram`, gated by `enables_dynamic_vram()`
    (`cli_args.py:318-321`): **on by default** for CUDA when not `--highvram/--gpu-only/--novram/--cpu`,
    off for AMD unless explicitly enabled (`Fix/aimdo windows pin budget` PR context; AMD support
    added in comfy-aimdo v0.3.0 but is opt-in via `--enable-dynamic-vram`).
  - Depends on external package `comfy_aimdo` (`import comfy_aimdo.host_buffer`,
    `comfy_aimdo.vram_buffer`, `comfy_aimdo.control` — separate pip wheel, not vendored).
  - Swaps `comfy.model_patcher.CoreModelPatcher` for `ModelPatcherDynamic` when aimdo initializes
    successfully; falls back to the legacy/estimate-based ModelPatcher otherwise (needs PyTorch
    ≥2.8; older torch or failed native-hook install logs a warning and disables it, so it degrades
    gracefully on older setups — relevant for Turing/Ampere cards run with older CUDA/torch stacks).
  - Windows-specific: sets `MIMALLOC_PURGE_DELAY=0` env var before aimdo init; separate
    `should_free_pins_for_ram_pressure()` (`model_management.py:734`) uses a different (swap%-based)
    trigger on Windows vs. immediate release on Linux, because Windows doesn't reliably report
    `virtual_memory_available()` under memory pressure the same way.
  - **Motivation (from PR history — rattus128, multiple PRs #12368, #13419, #13604, #14489,
    #16333):** replaces static, model-size-estimate-based VRAM accounting with a native library that
    hooks CUDA allocator calls directly, giving exact live VRAM tracking instead of heuristics. This
    fixes chronic under/over-estimation bugs (the old code's comment at `model_management.py:1797`
    literally flags the old smart-memory accounting as inconsistent with dynamic VRAM). Also unlocks
    multi-GPU symmetric device init and per-GPU (not global-pool) free-memory reporting.
- **reForge equivalent:** does not exist. No `comfy_aimdo` import, no `--enable-dynamic-vram`/
  `--disable-dynamic-vram` flags, no `ModelPatcherDynamic`. reForge's VRAM accounting is still the
  legacy estimate-based path (`free_memory()` at `ldm_patched/modules/model_management.py:622`).
- **Priority: HIGH.** This is now upstream's *default* path on CUDA and materially changes
  peak-VRAM behavior and OOM robustness — directly relevant across the whole RTX 20-50 range,
  especially lower-VRAM Turing/Ampere cards where estimate error causes avoidable OOMs or
  under-utilization.
- **Migration complexity: HIGH.** Requires vendoring or depending on the external `comfy-aimdo`
  native package (has its own release cadence, platform-specific wheels, ARM/AMD variants), plus
  porting `ModelPatcherDynamic` (not in the three files under audit — lives in `comfy/model_patcher.py`,
  a large surrounding change). This is a multi-file, multi-package undertaking, not a localized patch.
  Recommend tracking as its own follow-up epic rather than folding into this phase.

## 2. LoRA requantization for quantized (fp8/INT8) weights

- **Upstream:** `comfy/ops.py:270-340` (`resolve_cast_module_with_vbar`, `cast_bias_weight` with
  `want_requant` param), `comfy/ops.py:882,1017-1020,1404-1507` (usages in linear/conv paths),
  using `orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)`.
  - Lets a LoRA (or other weight patch) be correctly applied on top of a quantized (fp8 or
    Nunchaku-style INT8/"vbar") base weight by dequantizing, applying the patch in float, then
    re-quantizing with `stochastic_rounding` to reduce quantization-bias error, keeping compute on
    the fast INT8/FP8 path (`comfy/ops.py:1017` comment: "want_requant keeps a vbar-streamed layer
    on the INT8 path when a LoRA is [attached]").
  - **Motivation:** without this, applying a LoRA to an fp8-stored checkpoint either forces a full
    fp32/fp16 fallback for that layer (losing the fp8 speed/memory win) or silently drifts weight
    values across repeated requant cycles. Stochastic rounding specifically targets seed-dependent
    quantization bias buildup when a LoRA is toggled/changed across generations.
- **reForge equivalent:** does not exist. `grep -n requant ldm_patched/modules/ops.py` returns
  nothing. reForge's `fp8_linear`/`fp8_ops` (`ops.py:337-448`) supports fp8-only compute but has no
  requantization path for stacking LoRA on quantized weights — a LoRA use case with fp8 checkpoints
  either isn't supported cleanly or forces a cast-up.
- **Priority: MID-HIGH.** LoRA + fp8 checkpoints is a common real-world combo (SDXL/Flux fp8 +
  community LoRAs) and affects every card where fp8 compute is used (Ada/Hopper/Blackwell, i.e.
  the top half of the RTX 20-50 range where `supports_fp8_compute()` returns true — Turing/most
  Ampere don't hit this path since fp8 compute itself isn't available there, so the practical
  hardware footprint is RTX 40/50-series only). Given that hardware constraint, priority is MID for
  full sm_75-sm_120 coverage but HIGH for the Ada+/Blackwell subset.
- **Migration complexity: MID.** Confined mostly to `ops.py`; requires the base tensor class to
  expose a `requantize_from_float(...)` method (reForge's `QuantizedTensor`/fp8 wrapper equivalent
  would need this method added) plus threading a `want_requant` flag through the cast/forward call
  sites already present in `ops.py:53` (`cast_bias_weight`). No external dependency needed.

## 3. `--fast autotune` as a 4th PerformanceFeature

- **Upstream:** `comfy/cli_args.py:195-201` — `class PerformanceFeature(enum.Enum)` has 4 members:
  `Fp16Accumulation`, `Fp8MatrixMultiplication`, `CublasOps`, `AutoTune = "autotune"`. Consumed at
  `comfy/model_management.py:565`: `torch.backends.cudnn.benchmark = PerformanceFeature.AutoTune in args.fast`.
  - **Motivation:** simple, low-risk win — cuDNN's autotuner benchmarks multiple conv algorithms on
    first call and picks the fastest for the given input shape, at the cost of extra startup latency
    per distinct shape. Introduced alongside the original `--fast` mechanism (commit `9953f22`,
    "Add --fast argument to enable experimental optimizations") and formalized into the enum in PR
    #7024 ("Use enum list for --fast options"). Off by default because it can add latency for
    workflows with constantly-varying resolutions (e.g. interactive resizing) — hence gated behind
    `--fast`, not a bare default.
- **reForge equivalent:** `ldm_patched/modules/args_parser.py:157-162` — `PerformanceFeature` enum
  has only 3 members (`Fp16Accumulation`, `Fp8MatrixMultiplication`, `CublasOps`). No `AutoTune`
  member, and reForge never sets `torch.backends.cudnn.benchmark` from a `--fast` flag anywhere in
  `model_management.py`.
- **Priority: LOW-MID.** Trivial one-line functional gain, benefits every CUDA card in the sm_75-120
  range uniformly (cuDNN benchmark mode is architecture-agnostic), but the performance upside is
  workload-dependent (helps only when shapes are static/repeated across many iterations, e.g. batch
  video/animation runs; can hurt with highly variable resolutions).
- **Migration complexity: LOW.** Add the enum member + one conditional line
  (`torch.backends.cudnn.benchmark = PerformanceFeature.AutoTune in args.fast`) near reForge's
  existing `Fp16Accumulation` handling. Update the `--fast` help string. No dependencies, no other
  file touches needed. Good "quick win" candidate.

## 4. SageAttention / FlashAttention version detection & dispatch (sm_75 → sm_120)

- **Upstream:** primarily in `comfy/ldm/modules/attention.py:25-50,685,803,829-845,893-929`
  (outside the three audited files — the *detection flags* it depends on
  (`sage_attention_enabled()`, `flash_attention_enabled()`) live in
  `comfy/model_management.py:1723-1724` and mirror reForge's existing
  `ldm_patched/modules/model_management.py:1152-1156`).
  - Upstream additionally detects **SageAttention 3** (`SAGE_ATTENTION3_IS_AVAILABLE`, importing
    `sageattn3_blackwell` — a Blackwell(sm_120)-specific kernel package) and gates a fallback path
    with a try/except that logs and falls back to PyTorch attention on failure (`attention.py:803`).
    It also wraps FlashAttention with a `torch.library.custom_op` (`attention.py:829-845`) for
    `torch.compile` graph compatibility, and imports a `comfy_kitchen` module providing
    `int8_attention_is_available()` for INT8-quantized attention paths.
  - **Motivation:** SageAttention 3 targets Blackwell's native fp4/fp8 tensor-core paths that older
    SageAttention (targeting Ampere/Ada int8) can't use optimally; the version-tiered try/except
    dispatch lets one codebase serve Turing (fallback to PyTorch/xformers attention, no SageAttention
    support at all on sm_75) through Blackwell (SageAttention 3) without hard failures on
    unsupported hardware. The `torch.library.custom_op` wrapper around `flash_attn_func` fixes
    `torch.compile` graph breaks/recompilation triggered by calling an external non-traceable
    function directly inside a compiled region.
- **reForge equivalent:** `ldm_patched/modules/model_management.py:1152-1156` has only the basic
  `sage_attention_enabled()`/`sage_attention3_enabled()` flag functions (flag plumbing exists, note
  reForge already anticipated a "sage_attention3" toggle name) but the actual dispatch/detection
  logic that decides *which* SageAttention version is importable and used lives in reForge's own
  attention module (not one of the 3 audited files) — out of direct scope for this file-level diff,
  but flagged since model_management.py's flag surface is the integration point.
- **Priority: MID.** Directly affects performance scaling across the full requested hardware range
  (low-end Turing needs correct graceful fallback; Blackwell needs the SA3 kernel to get its
  intended speedup) but the bulk of the actual dispatch code lives outside the 3 audited files
  (in `attention.py`), so from a strict model_management.py/ops.py/args_parser.py lens the concrete
  gap is small: reForge is missing the `sage_attention3` *availability detection* wiring
  (`SAGE_ATTENTION3_IS_AVAILABLE`) that upstream's model_management-adjacent code assumes exists.
- **Migration complexity: LOW for the args/flag surface** (mirroring `sage_attention3_enabled()`
  is already done in reForge) **but MID-HIGH to fully port** since the real kernel-dispatch/fallback
  logic and the `torch.library.custom_op` compile-safety wrapper live in the attention module, which
  is explicitly outside this phase's file scope. Recommend a follow-up ticket scoped to
  `ldm_patched/ldm/modules/attention.py` rather than expanding this phase's scope.

## 5. torch.compile bug fixes / "Comfy Compiler"

- **Upstream:** `comfy/cli_args.py:185-186` — upstream has *replaced* the old granular
  `--torch-compile-*` flag family with a single `--disable-comfy-compiler` /
  `--assert-graph-breaks` pair (`cli_args.py:297` wires `args.disable_comfy_compiler`), backed by a
  new "Comfy Compiler" subsystem (PR #15861, "Introduce Comfy Compiler (CORE-389)", plus follow-up
  #16148 "Pause comfy compiler for long lived sparse allocations", both by rattus128, and #16072
  overlapping "Add Sparse Attention node"). It bundles CUDA graph capture as a documented
  "subfeature" of the same on/off switch, and specifically pauses compilation/caching for
  "long-lived sparse allocations" — i.e. a **fix for a torch.compile-vs-caching conflict** matching
  what this task asked to confirm.
  - **Motivation:** the previous fine-grained `--torch-compile-mode/backend/epilogue-fusion/...`
    flag surface (which reForge still has verbatim at `args_parser.py:92-99`) required per-user
    tuning and was fragile across model/attention combinations; "Comfy Compiler" is an
    upstream-managed, model-graph-aware compile wrapper that auto-selects safe boundaries and
    disables itself around allocation patterns (e.g. KV/sparse-attention caches) that previously
    caused recompilation storms or CUDA graph capture failures.
- **reForge equivalent:** `ldm_patched/modules/args_parser.py:92-99` — the full legacy
  `--torch-compile`, `--torch-compile-backend`, `--torch-compile-mode`,
  `--torch-compile-epilogue-fusion`, `--torch-compile-max-autotune`, `--torch-compile-fallback-random`,
  `--torch-compile-shape-padding`, `--torch-compile-cudagraphs`, `--torch-compile-trace`,
  `--torch-compile-graph-diagram` flags are all present — this is exactly the surface upstream
  moved away from. No `--disable-comfy-compiler`/`--assert-graph-breaks` equivalent, no
  sparse-allocation compile-pause logic.
- **Priority: MID.** The reForge flags still work as a manually-tuned baseline, so this isn't a
  correctness gap, but reForge will miss upstream's automatic graph-break/recompile mitigations
  going forward, and any new node types upstream ships tuned for "Comfy Compiler" semantics won't
  have a clean reForge equivalent. Applies uniformly across the hardware range (torch.compile
  behavior is not GPU-generation-specific beyond backend/codegen support already gated elsewhere).
- **Migration complexity: HIGH.** "Comfy Compiler" is a new subsystem (introduced across many files,
  not just `cli_args.py`) with its own pause/resume heuristics tied into the model-patching and
  attention/caching layers — a full port is a project-sized effort, not a localized fix. Recommend
  tracking as a separate epic; for this phase, at most note the flag-surface delta.

## 6. VRAM estimation / OOM-prevention (fragmentation-aware accounting)

- **Upstream:** `comfy/model_management.py:877-891` (`EXTRA_RESERVED_VRAM`,
  `extra_reserved_memory()`, `minimum_inference_memory()`), `:893-1025` (`free_memory()` /
  `load_models_gpu()` — dynamic-VRAM-aware unload logic, `for_dynamic` param, `pins_required`,
  `ram_required`), `:734-761` (`should_free_pins_for_ram_pressure()`, `ensure_pin_budget()` — RAM,
  not VRAM, pressure-based pin eviction with Windows-specific swap-percent heuristic), `:1675`
  (`extra_ram_release()` callback hook).
  - **Motivation:** once dynamic VRAM (comfy-aimdo) is active, the old flat
    "minimum_memory_required" estimate isn't the deciding factor anymore — real allocator-reported
    free memory is. The RAM-side pin-eviction logic (`should_free_pins_for_ram_pressure`) was added
    because pinned-host-memory buffers (used for async offload, same mechanism reForge already has)
    can starve system RAM on Windows in ways `virtual_memory_available()` doesn't reflect promptly,
    hence the swap-percentage fallback trigger.
- **reForge equivalent:** `ldm_patched/modules/model_management.py:608-672` — reForge has
  `EXTRA_RESERVED_VRAM`/`extra_reserved_memory()`/`minimum_inference_memory()`/`free_memory()`
  (functionally close to upstream's pre-dynamic-VRAM baseline) **plus** an already-present,
  reForge-original extension at `model_management.py:658-670`: a `_vram_pressure_hooks` registry
  that calls into custom eviction hooks (used by `diff_pipeline`) when standard model unloading
  isn't enough — this is *not* in upstream at all.
  - **reForge's own `diff_pipeline/vram_allocator.py`** (759 lines) implements a considerably more
    sophisticated **generation-based adaptive LRU allocator** with fragmentation-ratio-triggered
    defragmentation (`FRAG_THRESHOLD = 0.25`), free-list tracking via weakref finalizers, and a
    `TorchDispatchMode`-based live tensor tracker (`VRAMTrackerDispatch`) — none of this exists
    upstream in any form; it is a reForge-original design that is arguably *more* fragmentation-aware
    than upstream's dynamic-VRAM approach (which relies on the external aimdo library doing exact
    accounting rather than an in-Python generational allocator).
- **Judgment (as requested):** **do not duplicate — consolidate.** reForge already has two parallel
  VRAM-pressure mechanisms: (1) the legacy `free_memory()`/`current_loaded_models` unload path in
  `model_management.py`, extended with the `_vram_pressure_hooks` callback so `diff_pipeline` can
  plug in, and (2) `diff_pipeline/vram_allocator.py`'s own independent generational-LRU allocator.
  Importing upstream's RAM-side pin-pressure heuristics (`should_free_pins_for_ram_pressure`,
  Windows swap-percent trigger) is a reasonable, small, additive change to `model_management.py`
  since reForge doesn't have RAM-pressure-triggered pin eviction at all today (it only has
  VRAM-pressure hooks). But do **not** attempt to port upstream's aimdo-based dynamic VRAM engine as
  a *replacement* for `vram_allocator.py`'s allocator — that would be redundant/competing logic for
  the same problem; the more sane target architecture is to keep `vram_allocator.py` as reForge's
  VRAM-accounting authority and, if aimdo is ever integrated (see item 1), have it feed exact
  free-memory numbers into `vram_allocator.py`'s existing pressure-hook mechanism rather than
  bypass it.
- **Priority: MID** for the RAM-pressure pin-eviction addition specifically (small, real gap,
  hardware-agnostic); the broader "should we adopt aimdo" question is covered by item 1 (HIGH
  priority but high complexity, and contingent on the consolidation judgment above).
- **Migration complexity:** RAM-pressure pin eviction alone: **LOW-MID** (self-contained functions,
  no external deps, can reuse reForge's existing pinned-memory wrapper at
  `model_management.py:1488` and its `NUM_STREAMS`/async-offload machinery already in place).

---

## Priority Summary (highest first)

| # | Feature | Upstream loc | reForge loc | Priority | Complexity |
|---|---------|-------------|--------------|----------|------------|
| 1 | Dynamic VRAM (comfy-aimdo) | `model_management.py:1114-1490`, `cli_args.py:170-181,318-321` | does not exist | **HIGH** | HIGH (external dep + ModelPatcherDynamic) |
| 2 | LoRA requant on quantized weights | `ops.py:270-340,882,1017-1507` | does not exist | **MID-HIGH** (HIGH on Ada/Hopper/Blackwell subset) | MID |
| 5 | Comfy Compiler (torch.compile fixes) | `cli_args.py:185-186` + external subsystem | legacy `--torch-compile-*` only, `args_parser.py:92-99` | MID | HIGH |
| 6 | RAM-pressure pin eviction | `model_management.py:734-761` | does not exist (VRAM-pressure hooks only) | MID | LOW-MID |
| 4 | SageAttention3/Blackwell dispatch wiring | `attention.py` (adjacent to `model_management.py:1723`) | flag stub only, `model_management.py:1152-1156` | MID | LOW (flags) / MID-HIGH (full, out of file scope) |
| 3 | `--fast autotune` (cudnn.benchmark) | `cli_args.py:195-201`, `model_management.py:565` | 3-member enum, no autotune, `args_parser.py:157-162` | LOW-MID | **LOW — quick win** |

---

## Exclusions (explicitly out of scope for this phase)

The following upstream changes were observed during the diff but are excluded because they depend
on model-architecture support, video-pipeline features, or subsystems reForge does not have and
this task's scope forbids adding:

- **`supports_nvfp4_compute()` / `supports_mxfp8_compute()`** (`comfy/model_management.py:2035-2056`)
  — NVFP4/MXFP8 quantization detection for Blackwell (sm_100+) native microscaling formats. Useful
  only if/when reForge adds NVFP4/MXFP8 model support; pure architecture-capability probing with no
  consumer in reForge's current `ops.py`. Excluded per "no new model architectures."
- **`comfy_kitchen.int8_attention_is_available()`** and the broader `comfy_kitchen` package
  dependency in `attention.py` — ties INT8 attention detection to an additional external package;
  bundled with new-architecture (quantized transformer) support, not a generic backend change.
- **MiniMax H3 / video-model-specific VRAM and attention changes** (seen throughout upstream commit
  history: `perf(minimax): make H3 QKV contiguous on ROCm`, H3 VAE temporal-pad fixes, etc.) — video
  pipeline work, explicitly excluded by task scope.
- **Multi-GPU / distributed queue support** (`--distributed-queue-*` flags, multi-GPU aimdo
  `init_devices`) bundled inside the dynamic-VRAM code path — reForge is single-GPU-per-process
  today; multi-GPU orchestration is a distinct, larger feature than "VRAM management alignment" and
  is not evaluated further here.
- **ROCm/AMD-specific contiguity and attention perf patches** — target hardware for this task is
  RTX 20-50 (NVIDIA only); AMD-path changes are noted in passing (item 1) only insofar as they
  affect the default-enablement logic for dynamic VRAM, not evaluated as their own gap.
- **Sparse Attention node** (#16072, bundled with the Comfy Compiler PRs) — new node/architecture
  surface, not a backend VRAM/perf primitive.

---

## Sanity check

`./venv/bin/python -m pytest test/ --collect-only -q` from the reForge repo root: **101 tests
collected, 0 errors.** Only `docs/phase1_perf_gap_analysis.md` was created; no source files were
modified, so this is expected and confirms the analysis work did not disturb the existing test
suite's collectability. (pytest was not pre-installed in `venv/`; it was installed into the existing
project venv to run this check — no other packages or files were changed.)
