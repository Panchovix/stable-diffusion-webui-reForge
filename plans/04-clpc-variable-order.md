# Plan 04: CLPC Configurable Maximum Order (UniPC-style)

## Status: Implemented

**Motivation**: UniPC (`ldm_patched/unipc/uni_pc.py`) exposes `order` (1-3) as a
user-facing parameter, with `lower_order_final` ramping it down near the end of the
schedule. CLPC instead hardcoded its order cap (predictor `min(len(history), 3)`,
corrector `min(len(history)+1, 4)`) — no way to raise or lower it without editing code.

- **Phase 1 (Lean)**: `lean_proofs_rfv/RFVProofs/VariableOrderGain.lean`, 5 theorems,
  0 sorries, full project builds clean (2635 jobs). See `THEOREM_BUFFER.md`'s
  2026-07-07 entry. Answers two questions:
  1. Does the AM corrector stay a *consistent* update (b-coefficients sum to 1) at any
     order, not just the orders 2/3 that were hand-verified in AdamsStability.lean?
     Yes — `am_b_coeffs_sum_to_one_general`, via Mathlib's `Lagrange.sum_basis`.
  2. Does raising the order actually reduce local truncation error?
     Yes, strictly, as h→0 — `order_gain_ratio_tendsto_zero`, generalising
     PECEOrderGain.lean's hardcoded n=1→2 case to any consecutive order pair.
  Caveat surfaced by the synthesis theorem
  (`variable_order_needs_chebyshev_beyond_three`): partition of unity only bounds the
  coefficients' *sum*; the only order-general *individual*-coefficient bound
  (`chebyshev_monic_minimax`, already in ChebyshevAdaptive.lean) requires Chebyshev
  node spacing. The codebase already applied Chebyshev selection to the predictor
  but not the corrector — a real gap once `max_order` becomes user-raisable.

- **Phase 2**: `ldm_patched/k_diffusion/clpc_sampler.py`
  - `_adams_predict` / `_adams_correct` / `_clpc_loop` / `sample_clpc_ode` /
    `sample_clpc_sde` all gained a `max_order: int = 3` parameter — the cap itself
    defaults to the old hardcoded value of 3.
  - `_adams_correct` now also applies `_select_chebyshev_history` (previously
    predictor-only) when the corrector's order exceeds 3 and `use_chebyshev=True` —
    closing the gap Phase 1 identified.
  - `lower_order_final: bool = True` (UniPC's own default) ramps the effective order
    down as remaining schedule steps shrink: `step_max_order = min(max_order, remaining)`.

- **Phase 3**: Exposed on both UIs, mirroring existing `use_chebyshev`/`use_kalman` controls:
  - `ldm_patched/contrib/nodes_clpc.py` (ComfyUI nodes): `max_order` INT slider (1-6),
    `lower_order_final` BOOLEAN.
  - `extensions-builtin/sd_forge_clpc/scripts/forge_clpc.py` (WebUI accordion): same,
    plus `clpc_max_order`/`clpc_lower_order_final` recorded in
    `p.extra_generation_params` for image metadata.

## Verification

- `lean_build` on `lean_proofs_rfv`: full project (2635 jobs) builds with 0 errors.
- `lean_verify` on all three new headline theorems: axioms = `{propext, Classical.choice,
  Quot.sound}` only — no `sorryAx`, no custom axioms.
- Python: `py_compile` on all three touched files; manual signature audit (positional
  argument order at every `_adams_predict`/`_adams_correct` call site) via `ast.parse`.
- NOT tested against a real model run (no GPU/torch execution in this session).
  `max_order=3` alone reproduces the prior hardcoded cap exactly. The new
  `lower_order_final=True` default (matching UniPC's own default) is a deliberate
  behavior change, not a no-op: it additionally ramps the order down over the last
  `max_order-1` steps of the schedule, where the old code — capped only by history
  length — would have kept using order 3. Users who need the exact prior trajectory
  can pass `lower_order_final=False`.
