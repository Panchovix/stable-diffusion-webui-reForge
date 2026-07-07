"""sure_types.py — SURE sampler configuration and mutable state abstractions.

SUREConfig  — immutable dataclass holding all sure_* hyperparameters.
SUREState   — mutable per-run state (jac interval EMA, Adam, BO, sigma EMA).

Design goals
────────────
* Eliminate the 12-parameter sure_* sprawl duplicated in every sample_*_sure*
  function signature.
* Eliminate the 6-line state-init boilerplate copied verbatim across 9 samplers.
* Provide adapt_jac() so the 8-line EMA adaptation block is a one-liner at each
  call site.
* _sure_correct_step() wraps the 14-argument _sure_correct_x0() call in a thin
  dispatch that unpacks cfg + state fields — callers pass (model, …, cfg, state).

Import pattern (from sampling.py)
──────────────────────────────────
    from ldm_patched.k_diffusion.sure_types import SUREConfig, SUREState
    # _sure_correct_step is defined in sampling.py (needs _sure_correct_x0)
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Optional

_sure_logger = logging.getLogger("sure_sampler")

_JAC_EMA_A: float       = 0.35   # EMA decay for jac_ratio adaptation
_JAC_LO_THRESH: float   = 0.05   # ratio below which jac_interval is relaxed
_JAC_HI_THRESH: float   = 0.25   # ratio above which jac_interval is tightened
_JAC_MAX_INTERVAL: int  = 8      # upper cap on dyn_jac_interval


# ─────────────────────────────────────────────────────────────────────────────
# SUREConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SUREConfig:
    """All SURE hyperparameters in one place — replaces the sure_* kwarg sprawl.

    Fields map 1-to-1 to the old sure_* parameters:

        sure_alpha          → alpha
        sure_n_mc           → n_mc
        sure_eps            → eps
        sure_grad_mode      → grad_mode
        sure_approx_coeff   → approx_coeff
        sure_preheat_steps  → preheat_steps   (-1 = auto 15%)
        sure_preheat_frac   → preheat_frac    (adaptive sampler only)
        sure_jac_interval   → jac_interval    (-1 = adaptive EMA)
        sure_adam_mode      → adam_mode
        sure_adam_beta1     → adam_beta1
        sure_adam_beta2     → adam_beta2
        sure_adam_wd        → adam_wd
        sure_alpha_bo_trials    → alpha_bo_trials
        sure_alpha_bo_patience  → alpha_bo_patience
        sure_sigma_ema      → sigma_ema        (per-step sigma EMA; 0 = off)
        sure_wavelet        → wavelet
        sure_wavelet_level  → wavelet_level
        sure_wavelet_warmup_steps → wavelet_warmup_steps
        sure_wavelet_lp_frac      → wavelet_lp_frac
        sure_csv_path       → csv_path

    Default values match the historic defaults used in sampling.py so that
    SUREConfig() is a drop-in replacement for the old keyword-argument defaults.
    """

    # ── Core correction ───────────────────────────────────────────────────────
    alpha: float = 0.05
    n_mc: int = 1
    eps: float = 1e-3
    grad_mode: str = "vjp"
    approx_coeff: float = 2.0

    # ── Preheat ───────────────────────────────────────────────────────────────
    preheat_steps: int = -1       # -1 = auto ceil(15% of n_steps), min 2
    preheat_frac: float = 0.3     # fraction for adaptive (sigma-space) sampler

    # ── Jacobian scheduling ───────────────────────────────────────────────────
    jac_interval: int = 2         # -1 or 0 = adaptive EMA; ≥1 = fixed

    # ── Adam optimizer ────────────────────────────────────────────────────────
    adam_mode: str = "none"       # 'none' | 'adam' | 'adamw'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_wd: float = 0.01

    # ── Bayesian alpha search ─────────────────────────────────────────────────
    alpha_bo_trials: int = 0
    alpha_bo_patience: int = 4

    # ── Sigma EMA (wavelet / base SURE samplers) ──────────────────────────────
    sigma_ema: float = 0.0        # 0 = disabled; e.g. 0.9 for slow smoothing

    # ── Wavelet ───────────────────────────────────────────────────────────────
    wavelet: str = "db4"
    wavelet_level: int = 3
    wavelet_warmup_steps: int = 0
    wavelet_lp_frac: float = 1.0

    # ── CSV logging ───────────────────────────────────────────────────────────
    csv_path: Optional[str] = None

    # ── Validation ────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if self.alpha >= 0.5:
            _sure_logger.warning(
                "[sure_types] SUREConfig.alpha=%.4f ≥ 0.5; effective_alpha may "
                "exceed the Theorem A.3 bound (η < 0.5).  Reduce alpha to < 0.5.",
                self.alpha,
            )
        _VALID_ADAM = ("none", "adam", "adamw")
        if self.adam_mode not in _VALID_ADAM:
            raise ValueError(
                f"SUREConfig.adam_mode={self.adam_mode!r} not in {_VALID_ADAM}"
            )
        _VALID_GRAD = ("approx", "vjp", "full")
        if self.grad_mode not in _VALID_GRAD:
            raise ValueError(
                f"SUREConfig.grad_mode={self.grad_mode!r} not in {_VALID_GRAD}"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def preheat_count(self, n_steps: int) -> int:
        """Number of plain-denoise steps before SURE correction begins.

        preheat_steps >= 0  → use that value directly.
        preheat_steps = -1  → auto: ceil(15% of n_steps), minimum 2.
        """
        if self.preheat_steps >= 0:
            return self.preheat_steps
        return max(2, math.ceil(0.15 * n_steps))

    def initial_jac_interval(self) -> int:
        """Initial dyn_jac_interval value derived from jac_interval setting."""
        return max(1, self.jac_interval) if self.jac_interval >= 1 else 2


# ─────────────────────────────────────────────────────────────────────────────
# SUREState
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SUREState:
    """Mutable per-run state threaded through the SURE correction loop.

    All mutable variables that were previously scattered as locals across each
    sampler body are gathered here.  from_config() is the canonical constructor.

    adapt_jac(stats, step_gate)
        Updates jac_ratio EMA and adapts dyn_jac_interval in-place.
        step_gate: minimum corr_count before adaptation fires (default 3).
        Call this after every _sure_correct_step() call.

    use_jac() → bool
        Whether to run the full Jacobian at the current step.
    """

    dyn_jac_interval: int
    jac_ratio_ema: Optional[float] = None
    corr_count: int = 0
    adam_state: Optional[dict] = None
    alpha_bo_state: Optional[dict] = None
    sigma_hat_0_ema: Optional[float] = None   # for wavelet / base SURE sigma EMA
    adapt_enabled: bool = True                # False in adaptive sampler (fixed interval)

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: SUREConfig, *, adapt_enabled: bool = True) -> "SUREState":
        """Build initial state from a SUREConfig."""
        return cls(
            dyn_jac_interval=cfg.initial_jac_interval(),
            adam_state=(
                {"optimizer": None, "param": None}
                if cfg.adam_mode in ("adam", "adamw")
                else None
            ),
            alpha_bo_state={} if cfg.alpha_bo_trials > 0 else None,
            adapt_enabled=adapt_enabled,
        )

    # ── Step helpers ──────────────────────────────────────────────────────────
    def use_jac(self) -> bool:
        """True if a full Jacobian pass should run this correction step."""
        return (self.dyn_jac_interval <= 1) or (
            self.corr_count % self.dyn_jac_interval == 0
        )

    def adapt_jac(self, stats: dict, step_gate: int = 3) -> None:
        """Update jac EMA and adapt dyn_jac_interval in-place.

        args:
            stats:      dict returned by _sure_correct_x0 / _sure_correct_step
            step_gate:  do not adapt until corr_count >= step_gate (avoids
                        spurious early adaptation when EMA has not warmed up).
                        Use 3 for DPM++ samplers, 2 for base SURE samplers.
        """
        ratio = stats.get("jac_ratio")
        if ratio is not None:
            if self.jac_ratio_ema is None:
                self.jac_ratio_ema = ratio
            else:
                self.jac_ratio_ema = (
                    (1.0 - _JAC_EMA_A) * self.jac_ratio_ema + _JAC_EMA_A * ratio
                )

        if self.adapt_enabled and self.jac_ratio_ema is not None and self.corr_count >= step_gate:
            if self.jac_ratio_ema < _JAC_LO_THRESH and self.dyn_jac_interval < _JAC_MAX_INTERVAL:
                self.dyn_jac_interval += 1
                _sure_logger.info(
                    "[adapt] jac_interval -> %d  (ratio_ema=%.3f < %.2f)",
                    self.dyn_jac_interval, self.jac_ratio_ema, _JAC_LO_THRESH,
                )
            elif self.jac_ratio_ema > _JAC_HI_THRESH and self.dyn_jac_interval > 1:
                self.dyn_jac_interval -= 1
                _sure_logger.info(
                    "[adapt] jac_interval -> %d  (ratio_ema=%.3f > %.2f)",
                    self.dyn_jac_interval, self.jac_ratio_ema, _JAC_HI_THRESH,
                )

        _sure_logger.info(
            "[adapt] jac_ratio_ema=%s  dyn_jac_interval=%d",
            f"{self.jac_ratio_ema:.3f}" if self.jac_ratio_ema is not None else "n/a",
            self.dyn_jac_interval,
        )
        self.corr_count += 1
