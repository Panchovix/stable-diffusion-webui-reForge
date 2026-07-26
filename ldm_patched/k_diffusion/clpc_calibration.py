"""
CLPC online DC-Solver calibration module.

Learns corrector coefficients from the first few accepted steps using
torch.linalg.lstsq, then blends toward analytic SA-Solver coefficients.

DC-Solver insight: the analytic Adams-Moulton coefficients are derived under
a linearisation assumption.  Real models violate this; calibrated coefficients
from actual trajectory data improve accuracy.  We do this ONLINE (no pre-pass)
using the history accumulated during the first `warmup_steps` accepted steps.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class CLPCCalibrationState:
    warmup_steps: int = 6
    accepted_steps: int = 0
    calibrated: bool = False

    # Accumulated (A, b) for lstsq: A rows = feature vectors of lambda differences,
    # b rows = observed update targets.  Shape grows each step.
    A_rows: list = dataclasses.field(default_factory=list)
    b_rows: list = dataclasses.field(default_factory=list)

    # Calibrated coefficient tensors per order: {order: tensor shape (order,)}
    coeff_cache: dict = dataclasses.field(default_factory=dict)

    # Blend factor: 0.0 = fully analytic, 1.0 = fully calibrated
    blend: float = 0.0


def _build_feature_row(lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Feature vector for DC-Solver regression: differences of lambda values
    relative to the step [lambda_s, lambda_t].
    """
    h = lambda_t - lambda_s
    diffs = lambdas - lambda_s
    # Normalise by h so the regression is scale-invariant
    return (diffs / (h.abs() + 1e-8)).float()


def accumulate_calibration(
    state: CLPCCalibrationState,
    lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    update_target: torch.Tensor,
) -> None:
    """
    Add one accepted step to the regression buffer.
    update_target is the scalar "effective update" that was applied —
    in practice, the norm of the corrector residual tensor.
    """
    if state.calibrated:
        return
    row = _build_feature_row(lambdas, lambda_s, lambda_t)
    state.A_rows.append(row.cpu())
    state.b_rows.append(torch.tensor([float(update_target.norm())], dtype=torch.float32))
    state.accepted_steps += 1

    if state.accepted_steps >= state.warmup_steps:
        _try_calibrate(state)


def _try_calibrate(state: CLPCCalibrationState) -> None:
    if len(state.A_rows) < 2:
        return
    try:
        A = torch.stack(state.A_rows, dim=0)  # (N, K)
        b = torch.stack(state.b_rows, dim=0)  # (N, 1)
        # lstsq: min ||A @ x - b||; solution shape (K, 1)
        result = torch.linalg.lstsq(A, b, driver="gelsd")
        coeffs = result.solution.squeeze(-1)  # (K,)
        # Normalise so they sum to ~1 (blend-stability guard)
        coeff_sum = coeffs.abs().sum()
        if coeff_sum > 1e-6:
            coeffs = coeffs / coeff_sum
        state.coeff_cache[len(state.A_rows[0])] = coeffs
        state.calibrated = True
        # Ramp blend to 0.7 over warmup; leave room for analytic fallback
        state.blend = 0.7
    except Exception:
        # lstsq failure — stay with analytic coefficients
        state.calibrated = False


def get_corrector_coeffs(
    state: CLPCCalibrationState,
    analytic_coeffs: torch.Tensor,
) -> torch.Tensor:
    """
    Return blended corrector coefficients.

    analytic_coeffs: the SA-Solver Adams coefficients from
        compute_stochastic_adams_b_coeffs(), shape (K,).

    Returns tensor of same shape on same device.
    """
    if not state.calibrated or state.blend <= 0.0:
        return analytic_coeffs

    K = analytic_coeffs.shape[0]
    cal = state.coeff_cache.get(K)
    if cal is None:
        return analytic_coeffs

    cal = cal.to(analytic_coeffs.device, analytic_coeffs.dtype)
    return (1.0 - state.blend) * analytic_coeffs + state.blend * cal
