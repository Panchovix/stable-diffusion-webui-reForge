# Plan 02: CLPC PECE Performance — Diagnosis & Redesign

## Status: Ready for Implementation

---

## Root Cause Analysis (Evidence-Based)

### Bug 1 — ODE error semantic mismatch (PRIMARY CAUSE)

`clpc_error.py:compute_ode_error` is called with `x_hi=x_corr, x_lo=x_pred`:

```python
# clpc_sampler.py:430-431
ode=compute_ode_error(x_hi=x_corr, x_lo=x_pred, ...)
```

This computes the **corrector–predictor gap** G = ‖x_corr − x_pred‖.

When PECE activates (σ ≤ 4.0), the Adams-Moulton corrector makes large corrections (that is the point of the corrector — it should move x significantly). So G is large. PID sees large error → rejects step.

**The semantic error:** G measures how much the corrector *improved* x, not how wrong the corrected solution is. A large G means the predictor was inaccurate and the corrector fixed it — a SUCCESS, not a failure. We should accept, not reject.

From `CorrectorConvergence.lean` (already proved):
```
‖x_corr - x_true‖ ≤ (1 + L·h/2) · e_pred     [corrector_error_bound]
G = ‖x_corr - x_pred‖ ≤ ‖x_pred - x_true‖ + ‖x_corr - x_true‖
                       ≤ e_pred · (2 + L·h/2)
```
So `e_pred ≥ G / (2 + L·h/2)`. G is a LOWER BOUND on predictor error, not an upper bound on corrector error. Using G to reject via PID is backwards.

### Bug 2 — Schedule-driven retry is a deadlock

When PID rejects, `i` does not advance:

```python
# clpc_sampler.py (reject branch):
else:
    step_count += 1
    if step_count >= max_steps:
        x = x_corr; i += 1   # forced only after max_steps retries
```

But `sigma_s = sigmas[i]`, `sigma_t = sigmas[i+1]` are fixed by the schedule. The model is deterministic → every retry produces *identical* composite error → perpetual rejection until `step_count = max_steps`.

With max_steps=1000 and 30 schedule steps, worst case: 970 wasted retries × 2 NFE each = 1940 wasted forward passes. This explains "even slower than Adaptive ODE samplers."

### Bug 3 — Float64 Vandermonde on FP32 hardware

`_sigma_to_lambda` returns float64 tensors (fixed in previous session), and all lambda tensors are float64. With a GPU constrained to FP32, float64 ops are emulated in software — typically 8–32× slower.

The `torch.linalg.solve` in `sa_solver.py:94` runs on float64. On FP32-only hardware this is the most expensive part of each step.

---

## Phase 0: Allowed APIs (Confirmed)

| What | Where | Confirmed |
|------|-------|-----------|
| `PIDStepSizeController.propose_step(error) -> bool` | `sampling.py:876` | ✓ always called; updates `self.h` |
| `compute_stochastic_adams_b_coeffs(sigma_next, curr_lambdas, lambda_s, lambda_t, tau_t, ...)` | `sa_solver.py:69` | ✓ all args must match dtype |
| `CLPCError.ode` | `clpc_error.py:CLPCError` | ✓ is the composite weight target |
| `torch.linalg.solve` dtype requirement | `sa_solver.py:94` | ✓ A and B must match |
| FP32 ceiling | GPU constraint | all GPU tensors must stay float32; float64 ops → CPU or eliminated |

---

## Phase 1: Fix the Deadlock (No-reject accept-always)

**Goal:** Stop the infinite retry loop. Schedule-driven samplers MUST always accept; the PID is meaningless for fixed schedules.

### Changes

#### `clpc_sampler.py` — Remove step rejection

Replace the PID accept/reject split with accept-always. Keep `pid.propose_step()` for monitoring only (its `self.h` field tracks ideal step size for display):

```python
# BEFORE (clpc_sampler.py ~line 448):
accepted = pid.propose_step(composite)
if accepted:
    ...advance...
else:
    step_count += 1
    if step_count >= max_steps:
        x = x_corr; i += 1

# AFTER:
pid.propose_step(composite)   # updates self.h for monitoring; return value ignored
accepted = True               # always accept on schedule-driven loop
# ... unified accept path only ...
```

Remove the `else` branch entirely. Remove `max_steps` retry guard (keep max_steps as a total accepted-step cap for safety).

#### `clpc_sampler.py` — Add PID-ideal-h to progress display

In `_advance_outer`, show `pid.h` alongside error so the user can see what step size the PID *would* want vs. what the schedule provides:

```python
"CLPC {:5.1f}%  σ={:.3f}  steps={}  E={:.3f}  pid_h={:.4f}".format(
    pct_display, sigma_val, accepted_count, composite, pid.h,
)
```

### Verification

Run CLPC ODE on 20 steps. Terminal inner bar should cycle Predict→Correct→Error→PID exactly 20 times with no retries. Total NFE = 2 × accepted steps + 1 (first Euler step costs 1 extra).

---

## Phase 2: Proper Embedded-Pair ODE Error

**Goal:** Replace the corrector–predictor gap with a legitimate ODE error estimate that doesn't penalise the corrector for doing its job.

### Design

Use an **order-1 Euler prediction** vs the **order-2 Adams prediction** as the embedded pair:

```
x_euler  = σ_t/σ_s · x + expm1(h) · denoised_prev     (1st order, cheap — reuse denoised_prev)
x_adams  = Adams predictor at order 2+                  (existing prediction path)
e_ode    = ‖x_adams - x_euler‖ / scale                  (Richardson pair)
```

This is the correct embedded-pair philosophy: two approximations of *different order* to the same ODE, not the corrector improvement over the predictor.

### Why this is valid

- Euler is O(h²) error; Adams order-2+ is O(h³) error.
- Gap ‖x_adams − x_euler‖ ~ O(h²) → Richardson gives O(h²) error estimate.
- The corrector is then a pure improvement layer on top of x_adams; it does not contribute to the ODE error estimate at all.

### Changes

#### `clpc_error.py` — New `compute_ode_error_embedded_pair`

```python
def compute_ode_error_embedded_pair(
    x_adams: torch.Tensor,   # higher-order Adams prediction
    x_euler: torch.Tensor,   # 1st-order Euler prediction (same starting point)
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> float:
    ...
```

#### `clpc_sampler.py` — Build Euler baseline inside loop

In the predict stage, always compute a cheap `x_euler` (reuse `history.latest().denoised` — zero extra NFE):

```python
# Cheap Euler baseline for embedded pair (reuse last denoised, no NFE)
if history.latest() is not None:
    denoised_prev_for_euler = history.latest().denoised
    h_signed = float(lam_t - lam_s)
    x_euler = (sigma_t / sigma_s) * x + math.expm1(h_signed) * denoised_prev_for_euler
else:
    x_euler = None
```

Pass `x_euler` and `x_adams` to `build_clpc_error`; the corrector output `x_corr` is no longer involved in ODE error at all.

### Verification

After this change: ODE error should be near zero on the first step (Euler = Adams when order=1), and grow informatively as Adams uses more history. The PID (monitoring only after Phase 1) should report smoothly decreasing h suggestions as sigma decreases.

---

## Phase 3: Lean Proofs — Corrector-Predictor Gap vs. True Error

**Goal:** Formally characterise when G = ‖x_corr − x_pred‖ is and is not a valid error proxy. Establish bound for the embedded-pair estimate.

### Lean theorems to explore

#### Theorem A: Gap overestimates corrector error

```lean
-- G = ‖x_corr - x_pred‖ overestimates ‖x_corr - x_true‖ when corrector helps
theorem corrector_gap_not_error_proxy
    (x_pred x_corr x_true : E) (e_pred G : ℝ)
    (hpred : ‖x_pred - x_true‖ ≤ e_pred)
    (hcorr : ‖x_corr - x_true‖ ≤ (1 + L * h / 2) * e_pred)   -- from CorrectorConvergence
    (hG : G = ‖x_corr - x_pred‖) :
    -- G can be as large as (2 + L*h/2)*e_pred even when x_corr is the better solution
    G ≤ (2 + L * h / 2) * e_pred := ...
```

Closing tactic: triangle inequality + CorrectorConvergence bound.

#### Theorem B: Embedded-pair gap IS a valid error proxy

```lean
-- For two methods of order p and p+1, the gap is O(h^{p+1}) = predictor error order
theorem embedded_pair_gap_valid
    (x_lo x_hi x_true : ℝ → E) (h C_lo C_hi : ℝ)
    (hlo : ‖x_lo h - x_true h‖ ≤ C_lo * h ^ 2)   -- Euler O(h²)
    (hhi : ‖x_hi h - x_true h‖ ≤ C_hi * h ^ 3)   -- Adams O(h³)
    (hh : 0 < h) :
    -- gap between lo and hi bounds the lo error up to constants
    ‖x_hi h - x_lo h‖ ≤ (C_lo + C_hi * h) * h ^ 2 := ...
-- → using gap/C_lo as error estimate gives O(h²) accuracy
```

#### Theorem C: Embedded-pair Richardson correction

```lean
-- Standard Richardson: e_true ≈ gap / (r^p - 1) where r = h_coarse/h_fine
-- For embedded pair with p=1 (Euler) vs p=2 (Adams): r→∞ so e_true ≈ gap
theorem richardson_embedded_pair (gap : ℝ) (hgap : 0 ≤ gap) :
    -- As order difference → ∞, Richardson correction factor → 1
    -- So gap IS the error estimate (no correction factor needed)
    True := trivial   -- placeholder; explore if exact factor matters
```

### Lean file: `lean_proofs_rfv/RFVProofs/EmbeddedPairError.lean`

Create new module. Register in `RFVProofs.lean` as:
```lean
import RFVProofs.EmbeddedPairError
```

### Success criterion

All three theorems proved with zero `sorry`. Theorem A establishes that the Phase 1 accept-always fix was mathematically necessary (rejecting on G was wrong). Theorem B establishes that the Phase 2 embedded pair is a valid error proxy.

---

## Phase 4: FP32 Lambda Arithmetic

**Goal:** Move all float64 lambda/Vandermonde computations off GPU (or replace with float32 + compensated summation).

### Constraint

GPU handles FP32 only. Float64 on GPU is software-emulated → 8–32× slower per op.

### Strategy

The Vandermonde solve in `sa_solver.py:94` **requires** double precision to be well-conditioned for order > 2. Solution: run it on CPU in float64, move result back to GPU as float32.

#### `clpc_sampler.py` — CPU lambda tensors in Adams calls

In `_adams_predict` and `_adams_correct`, move the `lambdas` tensor and `b_coeffs` solve to CPU:

```python
# Build lambdas on CPU (float64), solve there
lambdas_cpu = torch.tensor([e.lam for e in entries], dtype=torch.float64)   # CPU
lam_s_cpu = lam_s.cpu()
lam_t_cpu = lam_t.cpu()
sigma_t_cpu = sigma_t.double().cpu()

b_coeffs_cpu = _sa.compute_stochastic_adams_b_coeffs(
    sigma_t_cpu, lambdas_cpu, lam_s_cpu, lam_t_cpu, tau_t, ...
)
# Move result to GPU as float32 for actual latent arithmetic
b_coeffs = b_coeffs_cpu.to(device=x.device, dtype=x.dtype)
```

#### `clpc_sampler.py` — `_sigma_to_lambda` returns CPU float64

```python
def _sigma_to_lambda(sigma: torch.Tensor, is_flow: bool) -> torch.Tensor:
    sigma_cpu = sigma.double().cpu()   # always CPU float64; never on GPU
    ...
```

Lambdas are used only for scalar indexing and the CPU solve — they never participate in GPU tensor arithmetic directly.

### Verification

`torch.is_floating_point(lam_s)` is True and `lam_s.device == 'cpu'` in the corrected path. Run a 20-step generation and confirm step latency is uniform across sigma values (no slowdown at σ < 4).

---

## Phase 5: Verification

### End-to-end tests

1. **No retry deadlock**: add a debug counter to `_clpc_loop`; assert `step_count == accepted_count` after a run.
2. **ODE error is valid**: after Phase 2, confirm `err.ode` is monotonically decreasing with step size (smaller h → smaller error).
3. **Speed parity**: 20-step CLPC ODE should take ≤ 2× wall-time of 20-step Euler (both call model ~40 times).
4. **Lean zero-sorry**: `cd lean_proofs_rfv && lake build` exits 0 with no "declaration uses sorry" warnings.

### Anti-patterns to avoid

- Do NOT use `x_corr` vs `x_pred` as the ODE error estimate after Phase 2.
- Do NOT put float64 tensors on GPU — keep them on CPU.
- Do NOT reject steps in a schedule-driven loop; accept-always after Phase 1.
- Do NOT use `_sigma_to_lambda` result directly in GPU arithmetic — it must stay CPU.

---

## Summary of Changes

| File | Change | Phase |
|------|--------|-------|
| `clpc_sampler.py` | Remove accept/reject split; always accept | 1 |
| `clpc_sampler.py` | Build cheap `x_euler` for embedded pair; pass to error | 2 |
| `clpc_error.py` | Add `compute_ode_error_embedded_pair`; update `build_clpc_error` | 2 |
| `lean_proofs_rfv/RFVProofs/EmbeddedPairError.lean` | 3 new theorems, 0 sorry | 3 |
| `lean_proofs_rfv/RFVProofs.lean` | Add `import RFVProofs.EmbeddedPairError` | 3 |
| `clpc_sampler.py` | Move lambda solve to CPU; `_sigma_to_lambda` → CPU float64 | 4 |
