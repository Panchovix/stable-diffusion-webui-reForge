# RFVProofs

Formal (Lean 4 / Mathlib) exploration of the "stable velocity field" diagnostic
for rectified-flow / velocity-prediction samplers.

## Background claim

Rectified Flow (Liu et al., *Flow Straight and Fast*) trains a model to predict
a constant velocity field along noise→data trajectories that are encouraged to
be straight lines. In high-dimensional latent spaces this straightness is
empirically near-perfect (straightness ratio ≈ 1.0 — see
https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html). The same
constant-velocity assumption underlies VP/EPS-parameterized SDXL sampling.

The informal claim under test: **if the sampling trajectory is (near-)affine,
then the finite-difference velocity estimate computed from the sampling
history is (near-)constant across steps — so an observed history of
finite-difference velocities can be used as a diagnostic signal for whether
sampling is tracking the assumed straight-line path.**

## Formalization plan

- `RFVProofs/Straightness.lean` — the rigorous anchor result: path length
  ≥ chord length (straightness ratio ≥ 1), with equality exactly when the
  velocity field is constant. Built from `norm_integral_le_integral_norm`
  + the fundamental theorem of calculus.
- `RFVProofs/Stability.lean` — the core diagnostic theorems: on an affine
  trajectory, every two-point finite-difference velocity estimate equals the
  same constant `c` (history is perfectly stable); contrapositive: any
  observed disagreement between two finite-difference estimates proves the
  trajectory is *not* affine on that span.
- `RFVProofs/Perturbation.lean` — the open / `sorry`-bearing target for
  Lean's proof automation: a *quantitative* version bounding how far the
  finite-difference estimate can drift when the trajectory is affine plus a
  bounded perturbation ‖r t‖ ≤ δ.
- `RFVProofs/LocalLinearity.lean` — windowed generalization: replaces the
  global affine assumption with a local Lipschitz bound on velocity over a
  window, matching how multi-step/adaptive samplers actually behave (see
  "Constraints from real samplers" below). Also open / `sorry`-bearing.

These three `sorry`-bearing files (`Straightness`, `Perturbation`,
`LocalLinearity`) are what `explore.sh` hammers on, one theorem per
`Explore*.lean` file.

## Constraints from real samplers (refining the claim)

The first pass formalized the *global* claim — "the whole trajectory is
affine, so every finite-difference estimate equals the same constant." That
claim is too strong once you account for how real multi-step/adaptive
samplers actually behave:

1. **DPM++ 2M is multi-step** — it fits a low-order polynomial through a
   short window of recent (x, velocity) history rather than assuming one
   global straight line. So "is the trajectory affine?" is the wrong
   question; "is the trajectory affine *over the window the sampler is
   actually using*?" is the one that matches what these methods rely on.
2. **Windowing / local linearity** — a real EPS/VP trajectory is generally
   curved overall, but (assuming the U-Net is well-trained — garbage-in/
   garbage-out aside) any sufficiently small window should look near-linear:
   the velocity field changes slowly relative to the step size. This is
   formalized in `RFVProofs/LocalLinearity.lean` as a *local Lipschitz bound
   on velocity* (`LipschitzVelocityOn v window L`, in `Defs.lean`) rather
   than a global affine assumption — `L` is the local curvature bound, and
   `L = 0` recovers exactly the old affine case from `Stability.lean`. The
   new theorems bound how far the trajectory deviates from its tangent line
   (`affine_dev_le_of_lipschitz_velocity`, a windowed second-order Taylor
   remainder) and, as a corollary, how close a finite-difference estimate
   inside the window is to the true instantaneous velocity
   (`finDiff_window_near_velocity`, bound `L * h`). This is the quantitative
   diagnostic threshold: monitor `finDiffVelocity` history and flag drift
   once it exceeds what `L * h` predicts for the sampler's step size.
3. **Euler (ODE) vs Euler a (SDE), Normal-noise scheduler / SDXL U-Net** —
   this project only models the deterministic/ODE component of sampling.
   Stochastic samplers add a Brownian-scale noise increment each step, and
   that noise's contribution to `finDiffVelocity` scales like `noise / Δt`
   — it **diverges** as the window shrinks, independent of how straight the
   underlying drift is. That's a real, important caveat for using this
   diagnostic on Euler a / any SDE integrator (don't shrink the window
   indefinitely and expect stability), but it is **not formalized here**:
   Mathlib's probability/stochastic-calculus support isn't mature enough to
   make an automated-exploration pass on it worthwhile. Documented as a
   scope limitation rather than attempted.
4. **Bosh3 (Bogacki–Shampine, adaptive RK3) as the best pure-ODE sampler for
   SDXL** — informally corroborates point 2: an embedded adaptive RK pair's
   internal step-size controller is *already* estimating something like
   the same local curvature bound `L` (via the difference between its two
   embedded order estimates) to decide how big a step it can safely take.
   `affine_dev_le_of_lipschitz_velocity`'s `L * (t - t0)^2 / 2` term is the
   same kind of local-truncation-error quantity such controllers bound
   internally. No separate Lean theorem for this — it's a motivating
   analogy, not formalized.

`Explore4.lean` / `Explore5.lean` target the two new `LocalLinearity.lean`
theorems and currently have no hand-written attempt — per the "see where
Lean leads us" exploration goal, they're left to `exact?`/`apply?`/`aesop`
first; a hand-proof pass (chaining through the same FTC-norm bound used for
`chord_le_path_length`) is the fallback if nothing turns up.

## Exploration tooling

`mathlib` pulls in `aesop`, `Qq`, `batteries`, and `LeanSearchClient`
transitively — no extra `require` entries needed. `LeanSearchClient` exposes
`#leansearch "<query>"` / `exact%` which queries https://leansearch.net for
relevant Mathlib lemmas over the network; `exact?` / `apply?` do local proof
search against the imported library. `explore.sh` runs both kinds of search
non-interactively and logs every attempt.

## Running

See `explore.sh` at the repo root of this folder. First run fetches the
Mathlib `.olean` cache (`lake exe cache get`) — this alone can take a long
time on a slow connection since no cache is shared with the sibling
`lean_proofs/` project.
