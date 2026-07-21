"""
jax_pipeline/samplers.py — JAX-native k-diffusion sampler algorithms.

Mechanical, formula-faithful mirror of mlx_pipeline/samplers.py — every
``mx.array``/``mx.*`` call is replaced 1:1 with its JAX equivalent, and
every step formula is copied verbatim from that file (which itself
mirrors ``ldm_patched/k_diffusion/sampling.py``). See that file's
docstrings for the full derivation of each ODE/SDE step.

The one structural difference: JAX's PRNG is purely functional (an
explicit key must be threaded through every draw), unlike MLX's
``mx.random.normal`` or torch's ``randn_like``, which both read from
implicit global RNG state. ``_NoiseState`` is a small mutable key-splitting
wrapper that recovers that same "just call it, get independent noise"
ergonomics at each call site without restructuring every sampler's
signature to thread a key explicitly.

Data-flow per generation
--------------------------
::

    KDiffusionSampler.sample()           (torch setup: sigmas, x scaling)
        |
        +-- _make_func_wrapper()         (swaps self.func for duration)
                |
                +-- build JAXCFGDenoiser (cond/uncond pre-converted once)
                |
                +-- JAX sampler loop     (all arrays stay in JAX)
                        +-- model(x_jax, sigma_jax)  -> denoised_jax
                        +-- ODE/SDE step (JAX arithmetic)
                        +-- callback     (convert back to torch for webui)
                |
                +-- _to_torch(x_jax)    (single conversion at the end)

Implemented samplers
----------------------
* ``sample_euler``            — deterministic Euler ODE
* ``sample_euler_ancestral``  — Euler + ancestral noise
* ``sample_heun``              — 2nd-order Heun ODE
* ``sample_dpmpp_2m``          — DPM++ 2M multistep
* ``sample_dpmpp_sde``         — DPM++ SDE (2nd order stochastic)
* ``sample_dpmpp_2m_sde``      — DPM++ 2M SDE
* ``sample_dpmpp_3m_sde``      — DPM++ 3M SDE

All SDE samplers draw i.i.d. Gaussian noise via ``_NoiseState``.

The ``model`` argument must accept ``(x: jnp.ndarray, sigma: jnp.ndarray)``
and return ``denoised: jnp.ndarray`` — i.e. a ``JAXCFGDenoiser`` instance.

Synchronization policy: async end to end, no forced per-step sync
--------------------------------------------------------------------
Earlier versions of every sampler here called a ``_flush(x)`` helper
after each step (mirroring ``mlx_pipeline``'s per-step ``mx.eval(x)``),
purely to bound how far JAX's async dispatch queue could run ahead of
actual GPU execution — its own docstring said as much: "correctness
doesn't need this". That's been removed, and nothing replaces it:
``jax_pipeline.host_offload.PhaseManager``'s device<->host phase moves
(``activate``/``force_offload_all``/``force_release_all``) are ALSO
pure async ``jax.device_put`` calls with no explicit
``block_until_ready`` — the whole generation (CLIP encode, every
denoising step, VAE decode, every phase transition) forms one
continuous async dispatch chain from "user clicks generate" through to
the final image, relying on the same futures model JAX's own reference
host-offloading pattern uses (moving arrays between memory kinds every
step in a loop with no extra sync shown as necessary). Each step
already has a hard DATA dependency on the previous one (step N+1 reads
step N's ``x``), so XLA/CUDA's own stream-ordering makes the sequential
compute correct without a host-side wait; forcing one anyway just adds
a barrier that prevents the GPU from ever running more than one step's
work at a time.

The practical consequence: since nothing blocks, the Python dispatch
loop below races through issuing all of a generation's steps far faster
than the GPU actually executes them — "progress" can no longer be "the
loop is on iteration i" (meaningless once dispatch outruns execution).
``_ProgressState`` (using ``jax_pipeline._async.AsyncReadiness`` — an
epoll-style non-blocking poll, since JAX itself has no native
``is_ready()``) reports whichever step has ACTUALLY completed on the
GPU so far, falling back to the last known-good (possibly slightly
stale) values when nothing new is ready yet rather than blocking to
force freshness — live preview / progress bars don't need per-step
real-time accuracy. The interrupt check (``stop_at`` in webui's own
``callback_state``) is unaffected by any of this: it only reads the
plain Python step index ``d['i']``, which ``_ProgressState.report``
always passes through immediately, every step, regardless of whether
that step's tensor data happens to be ready yet.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# -- helpers ----------------------------------------------------------------

class _NoiseState:
    """Mutable JAX PRNG key holder, split on every draw so consecutive
    calls get independent noise — mirrors the ergonomics of MLX's/torch's
    implicit global RNG (``mx.random.normal`` / ``torch.randn_like`` need
    no explicit key) without needing to thread a key through every
    sampler's signature.

    Seeded from torch's current RNG state at construction (one draw from
    ``torch.randint``) so the JAX noise sequence still varies with the
    user's chosen seed, even though it won't be bit-exact with a pure
    PyTorch sampling run — an unavoidable consequence of switching RNG
    implementations, not something either mlx_pipeline or this port tries
    to paper over.
    """

    def __init__(self, seed: Optional[int] = None):
        import jax

        if seed is None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        self.key = jax.random.PRNGKey(seed)

    def normal(self, shape: Tuple[int, ...], dtype) -> Any:
        import jax
        self.key, subkey = jax.random.split(self.key)
        return jax.random.normal(subkey, shape, dtype=dtype)


def _to_torch(a: "Any") -> torch.Tensor:
    """JAX array -> CPU torch float32.

    ``np.asarray()`` blocks until the array is ready (no explicit
    ``block_until_ready`` needed for this specific conversion).
    """
    import jax.numpy as jnp
    return torch.from_numpy(np.asarray(a.astype(jnp.float32)).copy()).float()


def _randn_like(noise_state: _NoiseState, x: "Any") -> "Any":
    """Standard-Gaussian noise matching shape and dtype of x."""
    return noise_state.normal(x.shape, x.dtype)


class _ProgressState:
    """Epoll-style progress reporting for one generation's sampler loop —
    see this module's "Synchronization policy" docstring for why this
    exists instead of a plain per-step ``block_until_ready``.

    Exactly one background poll is ever outstanding at a time: submitting
    a new step's ``(x, denoised)`` before the previous one resolved just
    abandons the old poll (harmless — its thread finishes on its own,
    its result is simply never read) in favor of the newer one. The
    webui callback fires every step regardless of readiness, so
    interrupt-checking (``d['i']`` / ``stop_at``) and progress-bar step
    counting stay fully responsive; only the tensor VALUES (``x``/
    ``denoised``) are sometimes a step or two stale.
    """

    def __init__(self) -> None:
        self._last_x_torch: Optional[torch.Tensor] = None
        self._last_denoised_torch: Optional[torch.Tensor] = None
        self._pending = None  # Optional["AsyncReadiness"]

    def report(
        self, cb, i: int, sigma_t: torch.Tensor, x: "Any", denoised: "Any",
        force_sync: bool = False,
    ) -> None:
        if cb is None:
            return
        import jax
        from jax_pipeline._async import AsyncReadiness

        # First-ever step (nothing cached to fall back on yet) and the
        # final step (the report right before returning should reflect
        # the ACTUAL final state, not a stale one) both force a real
        # sync; every other step is a non-blocking poll.
        if force_sync or self._last_x_torch is None:
            x_r, denoised_r = jax.block_until_ready((x, denoised))
            self._last_x_torch = _to_torch(x_r)
            self._last_denoised_torch = _to_torch(denoised_r)
            self._pending = None
        else:
            if self._pending is not None and self._pending.ready():
                x_r, denoised_r = self._pending.get()
                self._last_x_torch = _to_torch(x_r)
                self._last_denoised_torch = _to_torch(denoised_r)
            self._pending = AsyncReadiness((x, denoised))

        # The standard PyTorch path updates sampler.last_latent every
        # step (modules.sd_samplers_cfg_denoiser.CFGDenoiser.forward:
        # "self.sampler.last_latent = denoised") — that's what
        # Sampler.launch_sampling returns when InterruptedException
        # unwinds the loop (see JAXCFGDenoiser.__call__'s interrupt
        # check). JAXCFGDenoiser bypasses CFGDenoiser.forward entirely,
        # so nothing else updates it; without this, interrupting mid-
        # generation would return the untouched initial noisy latent
        # instead of the actual partial progress. ``cb`` is
        # KDiffusionSampler.callback_state, a bound method — cb.__self__
        # is the sampler instance.
        sampler = getattr(cb, "__self__", None)
        if sampler is not None:
            sampler.last_latent = self._last_denoised_torch

        cb({
            "i": i,
            "sigma": sigma_t,
            "sigma_hat": sigma_t,
            "x": self._last_x_torch,
            "denoised": self._last_denoised_torch,
        })


def _ancestral_step(sigma: float, sigma_next: float, eta: float = 1.0) -> Tuple[float, float]:
    """Return (sigma_down, sigma_up) for one ancestral step."""
    if sigma_next == 0:
        return 0.0, 0.0
    var_ratio = sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2
    sigma_up = min(sigma_next, eta * math.sqrt(max(0.0, var_ratio)))
    sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
    return sigma_down, sigma_up


# -- sampler implementations -------------------------------------------------

def sample_euler(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    s_churn: float = 0.,
    s_tmin: float = 0.,
    s_tmax: float = float("inf"),
    s_noise: float = 1.,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """Euler ODE sampler (JAX). Mirrors ``k_diffusion.sampling.sample_euler``."""
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()
    n = len(sigmas) - 1
    progress = _ProgressState()
    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        # Optional stochastic churn
        gamma = min(s_churn / n, math.sqrt(2) - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat = sigma * (1 + gamma)
        if gamma > 0:
            eps = _randn_like(noise_state, x) * s_noise
            x = x + eps * math.sqrt(sigma_hat ** 2 - sigma ** 2)

        sig_jx = jnp.array([sigma_hat], dtype=jnp.float32)
        denoised = model(x, sig_jx)

        d = (x - denoised) / sigma_hat
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))
        x = x + d * (sigma_next - sigma_hat)

    return _to_torch(x)


def sample_euler_ancestral(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    noise_sampler: Optional[Callable] = None,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """Euler ancestral sampler (JAX)."""
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()
    n = len(sigmas) - 1
    progress = _ProgressState()
    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_jx = jnp.array([sigma], dtype=jnp.float32)
        denoised = model(x, sig_jx)

        sigma_down, sigma_up = _ancestral_step(sigma, sigma_next, eta)
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))

        d = (x - denoised) / sigma
        dt = sigma_down - sigma
        x = x + d * dt

        if sigma_next > 0 and sigma_up > 0:
            x = x + _randn_like(noise_state, x) * s_noise * sigma_up

    return _to_torch(x)


def sample_heun(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    s_churn: float = 0.,
    s_tmin: float = 0.,
    s_tmax: float = float("inf"),
    s_noise: float = 1.,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """Heun 2nd-order ODE sampler (JAX)."""
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()
    n = len(sigmas) - 1
    progress = _ProgressState()
    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        gamma = min(s_churn / n, math.sqrt(2) - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat = sigma * (1 + gamma)
        if gamma > 0:
            eps = _randn_like(noise_state, x) * s_noise
            x = x + eps * math.sqrt(sigma_hat ** 2 - sigma ** 2)

        sig_jx = jnp.array([sigma_hat], dtype=jnp.float32)
        denoised = model(x, sig_jx)

        d = (x - denoised) / sigma_hat
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))
        dt = sigma_next - sigma_hat
        x2 = x + d * dt

        if sigma_next > 0:
            sig_next_jx = jnp.array([sigma_next], dtype=jnp.float32)
            denoised2 = model(x2, sig_next_jx)
            d2 = (x2 - denoised2) / sigma_next
            d_avg = (d + d2) * 0.5
            x = x + d_avg * dt
        else:
            x = x2

    return _to_torch(x)


def sample_dpmpp_2m(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
) -> torch.Tensor:
    """DPM++ 2M multistep ODE sampler (JAX)."""
    import jax.numpy as jnp

    old_denoised: Optional[Any] = None
    h_last: Optional[float] = None
    n = len(sigmas) - 1
    progress = _ProgressState()

    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_jx = jnp.array([sigma], dtype=jnp.float32)
        denoised = model(x, sig_jx)
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))

        # t = -log sigma, h = t_next - t = log(sigma / sigma_next)
        t = -math.log(sigma)
        t_next = -math.log(sigma_next) if sigma_next > 0 else -math.log(1e-10)
        h = t_next - t  # positive (sigma decreasing)
        ratio = sigma_next / sigma  # < 1

        if old_denoised is None or sigma_next == 0:
            # 1st-order Euler step
            emh = math.expm1(-h)  # = exp(-h) - 1 < 0
            x = ratio * x - emh * denoised
        else:
            # 2nd-order correction using previous denoised
            r = h_last / h
            emh = math.expm1(-h)
            d_c = (1.0 + 1.0 / (2.0 * r)) * denoised
            d_p = (1.0 / (2.0 * r)) * old_denoised
            x = ratio * x - emh * (d_c - d_p)

        old_denoised = denoised
        h_last = h

    return _to_torch(x)


def sample_dpmpp_sde(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    r: float = 0.5,
    noise_sampler: Optional[Callable] = None,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """DPM++ SDE sampler (JAX).

    Mirrors ``sample_dpmpp_sde_classic`` from the reference (two ancestral
    half-steps with a blended denoised estimate).

    Step formula (t = -log sigma, h = log(sigma/sigma_next) > 0):
        sigma_mid = sigma * (sigma_next/sigma)^r          (midpoint sigma)
        Step 1:  x2 = (sd1/sigma)*x + (1 - sd1/sigma)*D  + su1*z1
        Step 2:  x  = (sd2/sigma)*x + (1 - sd2/sigma)*D_hat + su2*z2
    where D_hat = (1-fac)*D + fac*D2  (blended, fac = 1/(2r))
    and sd, su come from ``_ancestral_step`` (same as ``get_ancestral_step``).
    """
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()
    n = len(sigmas) - 1
    progress = _ProgressState()
    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_jx = jnp.array([sigma], dtype=jnp.float32)
        denoised = model(x, sig_jx)
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))

        if sigma_next == 0:
            x = denoised
        else:
            # sigma_mid = sigma * (sigma_next/sigma)^r (midpoint in log-sigma space)
            sigma_mid = sigma * (sigma_next / sigma) ** r

            # -- Step 1: sigma -> sigma_mid --
            sd1, su1 = _ancestral_step(sigma, sigma_mid, eta)
            ratio1 = sd1 / sigma
            x2 = ratio1 * x + (1.0 - ratio1) * denoised
            if su1 > 0.0:
                x2 = x2 + _randn_like(noise_state, x) * (su1 * s_noise)

            # -- Midpoint denoised --
            sig_mid_jx = jnp.array([sigma_mid], dtype=jnp.float32)
            denoised2 = model(x2, sig_mid_jx)

            # -- Step 2: sigma -> sigma_next (blended denoised) --
            fac = 1.0 / (2.0 * r)
            den_d = (1.0 - fac) * denoised + fac * denoised2

            sd2, su2 = _ancestral_step(sigma, sigma_next, eta)
            ratio2 = sd2 / sigma
            x = ratio2 * x + (1.0 - ratio2) * den_d
            if su2 > 0.0:
                x = x + _randn_like(noise_state, x) * (su2 * s_noise)


    return _to_torch(x)


def sample_dpmpp_2m_sde(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    solver_type: str = "midpoint",
    noise_sampler: Optional[Callable] = None,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """DPM++ 2M SDE sampler (JAX).

    Mirrors the reference ``sample_dpmpp_2m_sde`` with eta=1 (default).

    Let lam = log(sigma/sigma_next) > 0, ratio = sigma_next/sigma < 1,
        lam_eta = lam*(eta+1), r_x = ratio^(eta+1) = exp(-lam_eta).

    Base update (1st-order Euler-Maruyama):
        x = r_x*x + (1 - r_x)*D + sigma_next*sqrt(1 - ratio^(2 eta))*z*s_noise

    2nd-order correction when prev denoised D_prev is available:
        midpoint:  x += 0.5*(1 - r_x) / r_prev * (D - D_prev)
        heun:      x += (1 - (1-r_x)/lam_eta) / r_prev * (D - D_prev)
    where r_prev = lam_prev / lam.
    """
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()
    old_denoised: Optional[Any] = None
    lam_prev: Optional[float] = None
    n = len(sigmas) - 1
    progress = _ProgressState()

    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_jx = jnp.array([sigma], dtype=jnp.float32)
        denoised = model(x, sig_jx)
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))

        if sigma_next == 0:
            x = denoised
        else:
            ratio = sigma_next / sigma  # < 1
            lam = math.log(sigma / sigma_next)  # > 0 (log(1/ratio))
            lam_eta = lam * (eta + 1.0)
            r_x = ratio ** (eta + 1.0)  # = exp(-lam_eta) < 1

            # -- Noise --
            noise_scale = sigma_next * math.sqrt(max(0.0, 1.0 - ratio ** (2.0 * eta))) * s_noise
            noise = _randn_like(noise_state, x) * noise_scale

            # -- Base 1st-order update --
            x = r_x * x + (1.0 - r_x) * denoised + noise

            # -- 2nd-order correction --
            if old_denoised is not None and lam_prev is not None:
                r_prev = lam_prev / lam
                diff = denoised - old_denoised
                if solver_type == "heun":
                    coeff = (1.0 - (1.0 - r_x) / lam_eta) / r_prev
                else:
                    coeff = 0.5 * (1.0 - r_x) / r_prev
                x = x + coeff * diff

        old_denoised = denoised
        lam_prev = lam if sigma_next != 0 else lam_prev

    return _to_torch(x)


def sample_dpmpp_3m_sde(
    model: Callable,
    x: "Any",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    noise_sampler: Optional[Callable] = None,
    noise_state: Optional[_NoiseState] = None,
) -> torch.Tensor:
    """DPM++ 3M SDE sampler (JAX).

    Mirrors the reference ``sample_dpmpp_3m_sde``.

    Base update (same as 2M SDE):
        r_x = ratio^(eta+1), lam_eta = log(sigma/sigma_next)*(eta+1)
        x = r_x*x + (1 - r_x)*D + sigma_next*sqrt(1 - ratio^(2 eta))*z

    Higher-order corrections use denoised-output differences, not score:
        phi_2 = 1 - (1 - r_x) / lam_eta
        phi_3 = phi_2 / lam_eta - 0.5

        2nd-order (D_prev1 available):
            d = (D - D_prev1) / r0   where r0 = lam_prev1 / lam
            x += phi_2 * d

        3rd-order (D_prev1, D_prev2 available):
            d1_0 = (D - D_prev1) / r0,  d1_1 = (D_prev1 - D_prev2) / r1
            d1   = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2   = (d1_0 - d1_1) / (r0 + r1)
            x   += phi_2 * d1 - phi_3 * d2
    """
    import jax.numpy as jnp

    noise_state = noise_state or _NoiseState()

    # Denoised history (D_prev1 = last step, D_prev2 = two steps ago)
    D_prev1: Optional[Any] = None
    D_prev2: Optional[Any] = None
    lam_1: Optional[float] = None  # lam from previous step
    lam_2: Optional[float] = None  # lam from two steps ago
    n = len(sigmas) - 1
    progress = _ProgressState()

    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_jx = jnp.array([sigma], dtype=jnp.float32)
        denoised = model(x, sig_jx)
        progress.report(callback, i, sigmas[i], x, denoised, force_sync=(i == n - 1))

        if sigma_next == 0:
            x = denoised
        else:
            ratio = sigma_next / sigma
            lam = math.log(sigma / sigma_next)  # > 0
            lam_eta = lam * (eta + 1.0)
            r_x = ratio ** (eta + 1.0)  # < 1, attenuates x

            # -- Noise --
            noise_scale = sigma_next * math.sqrt(max(0.0, 1.0 - ratio ** (2.0 * eta))) * s_noise
            noise = _randn_like(noise_state, x) * noise_scale

            # -- Base 1st-order update --
            x = r_x * x + (1.0 - r_x) * denoised + noise

            # -- Higher-order corrections using denoised history --
            phi_2 = 1.0 - (1.0 - r_x) / lam_eta
            phi_3 = phi_2 / lam_eta - 0.5

            if D_prev1 is not None and lam_1 is not None:
                r0 = lam_1 / lam
                if D_prev2 is not None and lam_2 is not None:
                    # 3rd-order
                    r1 = lam_2 / lam
                    d1_0 = (denoised - D_prev1) / r0
                    d1_1 = (D_prev1 - D_prev2) / r1
                    d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                    d2 = (d1_0 - d1_1) / (r0 + r1)
                    x = x + phi_2 * d1 - phi_3 * d2
                else:
                    # 2nd-order
                    d = (denoised - D_prev1) / r0
                    x = x + phi_2 * d

        D_prev2 = D_prev1
        D_prev1 = denoised
        lam_2 = lam_1
        lam_1 = lam if sigma_next != 0 else lam_1

    return _to_torch(x)


# -- dispatch map --------------------------------------------------------------

#: Maps k-diffusion funcname -> JAX sampler function.
JAX_SAMPLER_MAP: Dict[str, Callable] = {
    "sample_euler": sample_euler,
    "sample_euler_ancestral": sample_euler_ancestral,
    "sample_heun": sample_heun,
    "sample_dpmpp_2m": sample_dpmpp_2m,
    "sample_dpmpp_sde": sample_dpmpp_sde,
    "sample_dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "sample_dpmpp_3m_sde": sample_dpmpp_3m_sde,
}


# -- bridge: torch-compatible wrapper ------------------------------------------

def make_jax_func(
    jax_sampler_fn: Callable,
    jax_denoiser: Any,
) -> Callable:
    """Wrap a JAX sampler function in a drop-in replacement for
    ``KDiffusionSampler.func``.

    The returned callable matches the k-diffusion signature::

        func(model_wrap_cfg, x_torch, sigmas=sigmas,
             extra_args=extra_args, callback=cb, disable=False, **kwargs)

    but internally converts x -> JAX, runs the JAX sampler, converts back.
    The ``model_wrap_cfg`` argument is accepted but ignored — the
    ``jax_denoiser`` (already constructed) handles conditioning.
    """
    def _wrapper(
        _model_wrap_cfg: Any,  # ignored — JAXCFGDenoiser handles this
        x_torch: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        extra_args: Optional[Dict] = None,
        callback: Optional[Callable] = None,
        disable: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        import jax.numpy as jnp

        # Keep x in float32 throughout the sampler loop — matches the
        # reference (k-diffusion samplers keep x as float32 torch tensor;
        # only the UNet input is cast to bfloat16 inside _calc_input).
        x_jx = jnp.asarray(x_torch.detach().float().cpu().numpy())  # float32

        result_torch = jax_sampler_fn(
            jax_denoiser,
            x_jx,
            sigmas.detach().cpu() if sigmas is not None else sigmas,
            extra_args={},  # JAXCFGDenoiser owns cond/uncond
            callback=callback,
            disable=disable,
            **kwargs,
        )

        return result_torch.to(device=x_torch.device, dtype=x_torch.dtype)

    return _wrapper
