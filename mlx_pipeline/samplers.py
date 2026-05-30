"""
mlx_pipeline/samplers.py — MLX-native k-diffusion sampler algorithms.

Each function mirrors its counterpart in
``ldm_patched/k_diffusion/sampling.py`` but operates entirely on
``mx.array`` objects, so the denoising loop itself never touches torch.

Data-flow per generation
------------------------
::

    KDiffusionSampler.sample()           (torch setup: sigmas, x scaling)
        │
        └── _make_func_wrapper()         (swaps self.func for duration)
                │
                ├── build MLXCFGDenoiser (cond/uncond pre-converted once)
                │
                └── MLX sampler loop     (all arrays stay in MLX)
                        ├── model(x_mlx, σ_mlx)  → denoised_mlx
                        ├── ODE/SDE step (MLX arithmetic)
                        └── callback     (convert back to torch for webui)
                │
                └── _to_torch(x_mlx)    (single conversion at the end)

Implemented samplers
--------------------
* ``sample_euler``            — deterministic Euler ODE
* ``sample_euler_ancestral``  — Euler + ancestral noise
* ``sample_heun``             — 2nd-order Heun ODE
* ``sample_dpmpp_2m``         — DPM++ 2M multistep
* ``sample_dpmpp_sde``        — DPM++ SDE (2nd order stochastic)
* ``sample_dpmpp_2m_sde``     — DPM++ 2M SDE
* ``sample_dpmpp_3m_sde``     — DPM++ 3M SDE

All SDE samplers use i.i.d. Gaussian noise (``mx.random.normal``).

The ``model`` argument must accept ``(x: mx.array, sigma: mx.array)``
and return ``denoised: mx.array`` — i.e. an ``MLXCFGDenoiser`` instance.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_torch(a: "mx.array") -> torch.Tensor:
    """MLX → CPU torch float32.

    np.array() implicitly evaluates the MLX array, so no explicit
    mx.eval() is needed here — Metal handles it.
    """
    import mlx.core as mx
    return torch.from_numpy(np.array(a.astype(mx.float32)).copy()).float()


def _randn_like(x: "mx.array") -> "mx.array":
    """Standard-Gaussian noise matching shape and dtype of x."""
    import mlx.core as mx
    return mx.random.normal(shape=x.shape).astype(x.dtype)


def _callback(cb, i: int, sigma_t: torch.Tensor, x: "mx.array", denoised: "mx.array"):
    """Fire a webui progress callback, converting MLX arrays back to torch."""
    if cb is None:
        return
    import mlx.core as mx
    mx.eval(x, denoised)
    cb({
        "i":         i,
        "sigma":     sigma_t,
        "sigma_hat": sigma_t,
        "x":         _to_torch(x),
        "denoised":  _to_torch(denoised),
    })


def _ancestral_step(sigma: float, sigma_next: float, eta: float = 1.0) -> Tuple[float, float]:
    """Return (sigma_down, sigma_up) for one ancestral step."""
    if sigma_next == 0:
        return 0.0, 0.0
    var_ratio = sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2
    sigma_up   = min(sigma_next, eta * math.sqrt(max(0.0, var_ratio)))
    sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
    return sigma_down, sigma_up


# ── sampler implementations ────────────────────────────────────────────────────

def sample_euler(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    s_churn: float = 0.,
    s_tmin: float = 0.,
    s_tmax: float = float("inf"),
    s_noise: float = 1.,
) -> torch.Tensor:
    """Euler ODE sampler (MLX).  Mirrors ``k_diffusion.sampling.sample_euler``."""
    import mlx.core as mx

    n = len(sigmas) - 1
    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        # Optional stochastic churn
        gamma = min(s_churn / n, math.sqrt(2) - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat = sigma * (1 + gamma)
        if gamma > 0:
            eps = _randn_like(x) * s_noise
            x   = (x.astype(mx.float32) + eps * math.sqrt(sigma_hat ** 2 - sigma ** 2)).astype(mx.bfloat16)

        sig_mx   = mx.array([sigma_hat], dtype=mx.float32)
        denoised = model(x, sig_mx)

        d = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma_hat
        _callback(callback, i, sigmas[i], x, denoised)
        x = (x.astype(mx.float32) + d * (sigma_next - sigma_hat)).astype(mx.bfloat16)
        mx.eval(x)   # flush graph each step — keeps graph small, lets Metal pipeline

    return _to_torch(x)


def sample_euler_ancestral(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    noise_sampler: Optional[Callable] = None,
) -> torch.Tensor:
    """Euler ancestral sampler (MLX)."""
    import mlx.core as mx

    n = len(sigmas) - 1
    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)

        sigma_down, sigma_up = _ancestral_step(sigma, sigma_next, eta)
        _callback(callback, i, sigmas[i], x, denoised)

        d  = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma
        dt = sigma_down - sigma
        x  = (x.astype(mx.float32) + d * dt).astype(mx.bfloat16)

        if sigma_next > 0 and sigma_up > 0:
            noise = _randn_like(x) * s_noise
            x     = (x.astype(mx.float32) + noise * sigma_up).astype(mx.bfloat16)
        mx.eval(x)   # flush graph each step

    return _to_torch(x)


def sample_heun(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    s_churn: float = 0.,
    s_tmin: float = 0.,
    s_tmax: float = float("inf"),
    s_noise: float = 1.,
) -> torch.Tensor:
    """Heun 2nd-order ODE sampler (MLX)."""
    import mlx.core as mx

    n = len(sigmas) - 1
    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        gamma      = min(s_churn / n, math.sqrt(2) - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat  = sigma * (1 + gamma)
        if gamma > 0:
            eps = _randn_like(x) * s_noise
            x   = (x.astype(mx.float32) + eps * math.sqrt(sigma_hat ** 2 - sigma ** 2)).astype(mx.bfloat16)

        sig_mx   = mx.array([sigma_hat], dtype=mx.float32)
        denoised = model(x, sig_mx)

        d  = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma_hat
        _callback(callback, i, sigmas[i], x, denoised)
        dt = sigma_next - sigma_hat
        x2 = (x.astype(mx.float32) + d * dt).astype(mx.bfloat16)

        if sigma_next > 0:
            sig_next_mx = mx.array([sigma_next], dtype=mx.float32)
            denoised2   = model(x2, sig_next_mx)
            d2          = (x2.astype(mx.float32) - denoised2.astype(mx.float32)) / sigma_next
            d_avg       = (d + d2) * 0.5
            x           = (x.astype(mx.float32) + d_avg * dt).astype(mx.bfloat16)
        else:
            x = x2
        mx.eval(x)   # flush graph each step

    return _to_torch(x)


def sample_dpmpp_2m(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
) -> torch.Tensor:
    """DPM++ 2M multistep ODE sampler (MLX)."""
    import mlx.core as mx

    old_denoised: Optional["mx.array"] = None
    h_last: Optional[float] = None
    n = len(sigmas) - 1

    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        # t = −log σ,  h = t_next − t = log(σ / σ_next)
        t      = -math.log(sigma)
        t_next = -math.log(sigma_next) if sigma_next > 0 else -math.log(1e-10)
        h      = t_next - t                             # positive (sigma decreasing)
        ratio  = sigma_next / sigma                     # < 1

        if old_denoised is None or sigma_next == 0:
            # 1st-order Euler step
            emh = math.expm1(-h)                        # = exp(-h) - 1 < 0
            x = (ratio * x.astype(mx.float32)
                 - emh * denoised.astype(mx.float32)).astype(mx.bfloat16)
        else:
            # 2nd-order correction using previous denoised
            r    = h_last / h
            emh  = math.expm1(-h)
            d_c  = (1.0 + 1.0 / (2.0 * r)) * denoised.astype(mx.float32)
            d_p  = (1.0 / (2.0 * r))        * old_denoised.astype(mx.float32)
            x = (ratio * x.astype(mx.float32) - emh * (d_c - d_p)).astype(mx.bfloat16)

        old_denoised = denoised
        h_last       = h
        mx.eval(x)   # flush graph each step

    return _to_torch(x)


def sample_dpmpp_sde(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    r: float = 0.5,
    noise_sampler: Optional[Callable] = None,
) -> torch.Tensor:
    """DPM++ SDE sampler (MLX).

    Mirrors ``sample_dpmpp_sde_classic`` from the reference (two ancestral
    half-steps with a blended denoised estimate).

    Step formula (t = −log σ, h = log(σ/σ_next) > 0):
        σ_mid = σ · (σ_next/σ)^r                      (midpoint sigma)
        Step 1:  x₂ = (σ↓₁/σ)·x + (1 − σ↓₁/σ)·D  + σ↑₁·z₁
        Step 2:  x  = (σ↓₂/σ)·x + (1 − σ↓₂/σ)·D̂  + σ↑₂·z₂
    where D̂ = (1−fac)·D + fac·D₂  (blended, fac = 1/(2r))
    and σ↓, σ↑ come from _ancestral_step (same as get_ancestral_step).
    """
    import mlx.core as mx

    n = len(sigmas) - 1
    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        if sigma_next == 0:
            # Final step: pure denoising (Euler to 0)
            x = denoised.astype(mx.bfloat16)
        else:
            # σ_mid = σ · (σ_next/σ)^r  (midpoint in log-σ space)
            sigma_mid = sigma * (sigma_next / sigma) ** r

            # ── Step 1: σ → σ_mid ────────────────────────────────────────
            sd1, su1 = _ancestral_step(sigma, sigma_mid, eta)
            ratio1   = sd1 / sigma
            x2 = (ratio1       * x.astype(mx.float32)
                  + (1.0 - ratio1) * denoised.astype(mx.float32)).astype(mx.bfloat16)
            if su1 > 0.0:
                x2 = (x2.astype(mx.float32) + _randn_like(x) * (su1 * s_noise)).astype(mx.bfloat16)

            # ── Midpoint denoised ────────────────────────────────────────
            sig_mid_mx = mx.array([sigma_mid], dtype=mx.float32)
            denoised2  = model(x2, sig_mid_mx)

            # ── Step 2: σ → σ_next (blended denoised) ───────────────────
            fac      = 1.0 / (2.0 * r)
            den_d    = ((1.0 - fac) * denoised.astype(mx.float32)
                        + fac       * denoised2.astype(mx.float32))

            sd2, su2 = _ancestral_step(sigma, sigma_next, eta)
            ratio2   = sd2 / sigma
            x = (ratio2       * x.astype(mx.float32)
                 + (1.0 - ratio2) * den_d).astype(mx.bfloat16)
            if su2 > 0.0:
                x = (x.astype(mx.float32) + _randn_like(x) * (su2 * s_noise)).astype(mx.bfloat16)

        mx.eval(x)   # flush graph each step

    return _to_torch(x)


def sample_dpmpp_2m_sde(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    solver_type: str = "midpoint",
    noise_sampler: Optional[Callable] = None,
) -> torch.Tensor:
    """DPM++ 2M SDE sampler (MLX).

    Mirrors the reference ``sample_dpmpp_2m_sde`` with η=1 (default).

    Let  lam = log(σ/σ_next) > 0,  ratio = σ_next/σ < 1,
         lam_η = lam·(η+1),  r_x = ratio^(η+1) = exp(−lam_η).

    Base update (1st-order Euler-Maruyama):
        x = r_x·x + (1 − r_x)·D + σ_next·√(1 − ratio^{2η})·z·s_noise

    2nd-order correction when prev denoised D₋₁ is available:
        midpoint:  x += 0.5·(1 − r_x) / r_prev · (D − D₋₁)
        heun:      x += (1 − (1−r_x)/lam_η) / r_prev · (D − D₋₁)
    where r_prev = lam_prev / lam.
    """
    import mlx.core as mx

    old_denoised:  Optional["mx.array"] = None
    lam_prev: Optional[float] = None
    n = len(sigmas) - 1

    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        if sigma_next == 0:
            x = denoised.astype(mx.bfloat16)
        else:
            ratio   = sigma_next / sigma               # < 1
            lam     = math.log(sigma / sigma_next)     # > 0  (log(1/ratio))
            lam_eta = lam * (eta + 1.0)
            r_x     = ratio ** (eta + 1.0)             # = exp(-lam_eta) < 1

            # ── Noise ──────────────────────────────────────────────────────
            # Reference: σ_next · √(1 − exp(−2·lam·η)) · s_noise
            noise_scale = sigma_next * math.sqrt(max(0.0, 1.0 - ratio ** (2.0 * eta))) * s_noise
            noise = _randn_like(x) * noise_scale

            # ── Base 1st-order update ───────────────────────────────────────
            x = (r_x           * x.astype(mx.float32)
                 + (1.0 - r_x) * denoised.astype(mx.float32)
                 + noise).astype(mx.bfloat16)

            # ── 2nd-order correction ────────────────────────────────────────
            if old_denoised is not None and lam_prev is not None:
                r_prev = lam_prev / lam
                diff   = (denoised.astype(mx.float32) - old_denoised.astype(mx.float32))
                if solver_type == "heun":
                    # phi_heun = 1 − (1 − r_x) / lam_eta
                    coeff = (1.0 - (1.0 - r_x) / lam_eta) / r_prev
                else:
                    # midpoint: 0.5 · (1 − r_x) / r_prev
                    coeff = 0.5 * (1.0 - r_x) / r_prev
                x = (x.astype(mx.float32) + coeff * diff).astype(mx.bfloat16)

        old_denoised = denoised
        lam_prev     = lam if sigma_next != 0 else lam_prev
        mx.eval(x)   # flush graph each step

    return _to_torch(x)


def sample_dpmpp_3m_sde(
    model: Callable,
    x: "mx.array",
    sigmas: torch.Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.,
    s_noise: float = 1.,
    noise_sampler: Optional[Callable] = None,
) -> torch.Tensor:
    """DPM++ 3M SDE sampler (MLX).

    Mirrors the reference ``sample_dpmpp_3m_sde``.

    Base update (same as 2M SDE):
        r_x = ratio^(η+1),  lam_η = log(σ/σ_next)·(η+1)
        x = r_x·x + (1 − r_x)·D + σ_next·√(1 − ratio^{2η})·z

    Higher-order corrections use denoised-output differences, not score:
        phi_2 = 1 − (1 − r_x) / lam_η
        phi_3 = phi_2 / lam_η − 0.5

        2nd-order (D₋₁ available):
            d = (D − D₋₁) / r0   where r0 = lam₋₁ / lam
            x += phi_2 · d

        3rd-order (D₋₁, D₋₂ available):
            d1_0 = (D − D₋₁) / r0,  d1_1 = (D₋₁ − D₋₂) / r1
            d1   = d1_0 + (d1_0 − d1_1) · r0 / (r0 + r1)
            d2   = (d1_0 − d1_1) / (r0 + r1)
            x   += phi_2 · d1 − phi_3 · d2
    """
    import mlx.core as mx

    # Denoised history (D_prev1 = last step, D_prev2 = two steps ago)
    D_prev1: Optional["mx.array"] = None
    D_prev2: Optional["mx.array"] = None
    lam_1:   Optional[float]      = None   # lam from previous step
    lam_2:   Optional[float]      = None   # lam from two steps ago
    n = len(sigmas) - 1

    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        if sigma_next == 0:
            x = denoised.astype(mx.bfloat16)
        else:
            ratio   = sigma_next / sigma
            lam     = math.log(sigma / sigma_next)     # > 0
            lam_eta = lam * (eta + 1.0)
            r_x     = ratio ** (eta + 1.0)             # < 1, attenuates x

            # ── Noise ──────────────────────────────────────────────────────
            noise_scale = sigma_next * math.sqrt(max(0.0, 1.0 - ratio ** (2.0 * eta))) * s_noise
            noise = _randn_like(x) * noise_scale

            # ── Base 1st-order update ───────────────────────────────────────
            x = (r_x           * x.astype(mx.float32)
                 + (1.0 - r_x) * denoised.astype(mx.float32)
                 + noise).astype(mx.bfloat16)

            # ── Higher-order corrections using denoised history ─────────────
            # phi_2 = h_eta.neg().expm1() / h_eta + 1  (reference, h_eta > 0)
            phi_2 = 1.0 - (1.0 - r_x) / lam_eta       # same thing, cleaner
            phi_3 = phi_2 / lam_eta - 0.5

            D  = denoised.astype(mx.float32)

            if D_prev1 is not None and lam_1 is not None:
                r0 = lam_1 / lam
                if D_prev2 is not None and lam_2 is not None:
                    # 3rd-order
                    r1   = lam_2 / lam
                    d1_0 = (D - D_prev1) / r0
                    d1_1 = (D_prev1 - D_prev2) / r1
                    d1   = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                    d2   = (d1_0 - d1_1) / (r0 + r1)
                    x    = (x.astype(mx.float32)
                            + phi_2 * d1 - phi_3 * d2).astype(mx.bfloat16)
                else:
                    # 2nd-order
                    d  = (D - D_prev1) / r0
                    x  = (x.astype(mx.float32) + phi_2 * d).astype(mx.bfloat16)

        D_prev2 = D_prev1
        D_prev1 = denoised.astype(mx.float32)
        lam_2   = lam_1
        lam_1   = lam if sigma_next != 0 else lam_1
        mx.eval(x)   # flush graph each step

    return _to_torch(x)


# ── dispatch map ───────────────────────────────────────────────────────────────

#: Maps k-diffusion funcname → MLX sampler function.
MLX_SAMPLER_MAP: Dict[str, Callable] = {
    "sample_euler":           sample_euler,
    "sample_euler_ancestral": sample_euler_ancestral,
    "sample_heun":            sample_heun,
    "sample_dpmpp_2m":        sample_dpmpp_2m,
    "sample_dpmpp_sde":       sample_dpmpp_sde,
    "sample_dpmpp_2m_sde":    sample_dpmpp_2m_sde,
    "sample_dpmpp_3m_sde":    sample_dpmpp_3m_sde,
}


# ── bridge: torch-compatible wrapper ──────────────────────────────────────────

def make_mlx_func(
    mlx_sampler_fn: Callable,
    mlx_denoiser: Any,
) -> Callable:
    """
    Wrap an MLX sampler function in a drop-in replacement for
    ``KDiffusionSampler.func``.

    The returned callable matches the k-diffusion signature::

        func(model_wrap_cfg, x_torch, sigmas=sigmas,
             extra_args=extra_args, callback=cb, disable=False, **kwargs)

    but internally converts x→MLX, runs the MLX sampler, converts back.
    The ``model_wrap_cfg`` argument is accepted but ignored — the
    ``mlx_denoiser`` (already constructed) handles conditioning.
    """
    def _wrapper(
        _model_wrap_cfg: Any,    # ignored — MLXCFGDenoiser handles this
        x_torch: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        extra_args: Optional[Dict] = None,
        callback: Optional[Callable] = None,
        disable: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        import mlx.core as mx

        x_mx = mx.array(x_torch.detach().float().cpu().numpy()).astype(mx.bfloat16)

        result_torch = mlx_sampler_fn(
            mlx_denoiser,
            x_mx,
            sigmas.detach().cpu() if sigmas is not None else sigmas,
            extra_args={},        # MLXCFGDenoiser owns cond/uncond
            callback=callback,
            disable=disable,
            **kwargs,
        )

        return result_torch.to(device=x_torch.device, dtype=x_torch.dtype)

    return _wrapper
