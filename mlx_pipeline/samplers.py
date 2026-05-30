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
    """DPM++ SDE sampler (MLX).  Uses i.i.d. Gaussian noise."""
    import mlx.core as mx

    n = len(sigmas) - 1
    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        if sigma_next == 0:
            d  = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma
            x  = (x.astype(mx.float32) + d * (sigma_next - sigma)).astype(mx.bfloat16)
        else:
            # 2nd-order DPM++ SDE step
            h         = math.log(sigma_next / sigma)          # negative
            sigma_mid = sigma * math.exp(r * h)

            # Midpoint denoised
            d1    = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma
            x_mid = (x.astype(mx.float32) + d1 * (sigma_mid - sigma)).astype(mx.bfloat16)

            sig_mid_mx = mx.array([sigma_mid], dtype=mx.float32)
            den_mid    = model(x_mid, sig_mid_mx)
            d2         = (x_mid.astype(mx.float32) - den_mid.astype(mx.float32)) / sigma_mid

            # Blend derivative
            fac    = 1.0 / (2.0 * r)
            d_blend = (1.0 - fac) * d1 + fac * d2

            # Stochastic noise injection
            noise_var = max(0.0, sigma_next ** 2 - sigma ** 2 * math.exp(2.0 * h * (1.0 - eta)))
            sigma_n   = math.sqrt(noise_var) * s_noise

            x = (sigma_next / sigma * x.astype(mx.float32)
                 + (sigma_next - sigma_next / sigma * sigma) * d_blend
                 + _randn_like(x) * sigma_n).astype(mx.bfloat16)
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
    """DPM++ 2M SDE sampler (MLX).  Uses i.i.d. Gaussian noise."""
    import mlx.core as mx

    old_denoised: Optional["mx.array"] = None
    n = len(sigmas) - 1

    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        if sigma_next == 0:
            d = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma
            x = (x.astype(mx.float32) + d * (sigma_next - sigma)).astype(mx.bfloat16)
        else:
            h          = math.log(sigma_next / sigma)          # negative
            alpha      = math.exp(-h)                          # = sigma / sigma_next  (>1 since h<0)
            phi_1      = math.expm1(-h)                        # exp(-h) - 1 > 0
            noise_var  = max(0.0, sigma_next ** 2 - alpha ** 2 * sigma ** 2)
            sigma_n    = math.sqrt(noise_var) * s_noise

            noise = _randn_like(x) * sigma_n

            if old_denoised is None:
                # Euler-Maruyama first step
                x = (alpha * x.astype(mx.float32)
                     - phi_1 * denoised.astype(mx.float32)
                     + noise).astype(mx.bfloat16)
            else:
                if solver_type == "heun":
                    # Heun correction
                    x = (sigma_next / sigma * x.astype(mx.float32)
                         + (1 - sigma_next / sigma) * denoised.astype(mx.float32)
                         + noise).astype(mx.bfloat16)
                else:
                    # Midpoint 2nd-order (default)
                    phi_2 = phi_1 / h + 1.0
                    d_c   = denoised.astype(mx.float32)
                    d_p   = old_denoised.astype(mx.float32)
                    x     = (alpha * x.astype(mx.float32)
                             - phi_1 * d_c
                             - phi_2 * (d_c - d_p)
                             + noise).astype(mx.bfloat16)

        old_denoised = denoised
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
    """DPM++ 3M SDE sampler (MLX).  Uses i.i.d. Gaussian noise."""
    import mlx.core as mx

    # Derivative history  (d_prev1 = one step ago, d_prev2 = two steps ago)
    d_prev1: Optional["mx.array"] = None
    d_prev2: Optional["mx.array"] = None
    n = len(sigmas) - 1

    for i in range(n):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sig_mx   = mx.array([sigma], dtype=mx.float32)
        denoised = model(x, sig_mx)
        _callback(callback, i, sigmas[i], x, denoised)

        d_cur = (x.astype(mx.float32) - denoised.astype(mx.float32)) / sigma

        if sigma_next == 0:
            x = (x.astype(mx.float32) + d_cur * (sigma_next - sigma)).astype(mx.bfloat16)
        else:
            h         = math.log(sigma_next / sigma)   # negative
            alpha     = math.exp(-h)
            phi_1     = math.expm1(-h)                 # > 0
            noise_var = max(0.0, sigma_next ** 2 - alpha ** 2 * sigma ** 2)
            sigma_n   = math.sqrt(noise_var) * s_noise
            noise     = _randn_like(x) * sigma_n

            if d_prev1 is None:
                # 1st-order
                x = (alpha * x.astype(mx.float32)
                     - phi_1 * denoised.astype(mx.float32)
                     + noise).astype(mx.bfloat16)
            elif d_prev2 is None:
                # 2nd-order
                phi_2 = phi_1 / h + 1.0
                x     = (alpha * x.astype(mx.float32)
                         - phi_1 * denoised.astype(mx.float32)
                         - phi_2 * (d_cur - d_prev1)
                         + noise).astype(mx.bfloat16)
            else:
                # 3rd-order
                phi_2 = phi_1 / h + 1.0
                phi_3 = phi_2 / h + 0.5
                x     = (alpha * x.astype(mx.float32)
                         - phi_1 * denoised.astype(mx.float32)
                         - phi_2 * (d_cur - d_prev1)
                         - phi_3 * (d_cur - 2.0 * d_prev1 + d_prev2)
                         + noise).astype(mx.bfloat16)

        d_prev2 = d_prev1
        d_prev1 = d_cur
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
