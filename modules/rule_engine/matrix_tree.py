"""Koo, Globerson, Carreras & Collins, "Structured Prediction Models via the
Matrix-Tree Theorem" (EMNLP-CoNLL 2007, ACL Anthology D07-1015) — the exact
partition-function and marginal formulas from their §3, implemented here for
the first time this session (Lean only proved abstract properties ABOUT
this algorithm's output — see `RFVProofs/MatrixTreeSoftOwnership.lean` —
this module is the algorithm itself).

Single-root setting only (`T^s_np(x)`, their §3.1-3.2): the virtual
root-symbol (node 0) has EXACTLY ONE child. This module does not implement
the multi-root case (§3.3) — not needed here, since a prompt's dependency
structure should have exactly one top-level entity/attribute root per
connected component in practice, and multi-root support would need the
extended-graph construction the paper describes separately.

All computation is plain CPU `numpy` linear algebra (small n×n matrices,
n = number of prompt segments — always tiny, single digits to low tens),
matching the "CPU vectorization preferred, stay off the GPU for this" design
choice: the diffusion model's GPU/VRAM stays free for the model itself, and
this is a one-time pre-sampling computation regardless (see
`MatrixTreeSoftOwnership.lean` Part 10: cost model, step-invariant, zero
extra NFE).

Node indexing convention used throughout this file: nodes are numbered
`0..n` where `0` is the virtual root and `1..n` are the `n` prompt segments
(matching the paper's own 1-indexed word convention exactly, so its
formulas can be transcribed directly without an off-by-one translation
layer at the call sites — only the matrix STORAGE here is 0-indexed numpy,
translated internally).
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class TreeMarginals:
    n: int                      # number of prompt segments (nodes 1..n; node 0 = root)
    Z: float                    # partition function (total weight over all valid trees)
    mu: np.ndarray               # (n+1, n+1) marginals; mu[h, m] = P(parent(m) = h), 0=root
    theta: np.ndarray            # (n+1, n+1) the raw potentials that produced this (debug)

    def parent_distribution(self, m: int) -> dict:
        """P(parent(m) = h) for every candidate h in {0, 1, ..., n} \\ {m},
        as a plain dict for readability/debug printing."""
        return {h: float(self.mu[h, m]) for h in range(self.n + 1) if h != m}


def _build_A_and_r(theta: np.ndarray) -> tuple:
    """theta: (n+1, n+1) potential matrix, theta[h, m] = θ_{h,m} for h,m in
    1..n (word-to-word), and theta[0, m] = θ_{0,m} (root-selection score for
    word m). theta[h, h] and the rest of row/col 0 are ignored.

    Returns (A, r): A is the (n, n) 1-indexed-as-0-indexed edge-weight
    matrix (A[h-1, m-1] = exp(theta[h,m]) for h != m, else 0), and r is the
    (n,) root-selection weight vector (r[m-1] = exp(theta[0,m]))."""
    n = theta.shape[0] - 1
    A = np.zeros((n, n), dtype=np.float64)
    for h in range(1, n + 1):
        for m in range(1, n + 1):
            if h != m:
                A[h - 1, m - 1] = np.exp(theta[h, m])
    r = np.array([np.exp(theta[0, m]) for m in range(1, n + 1)], dtype=np.float64)
    return A, r


def _build_laplacian(A: np.ndarray) -> np.ndarray:
    """L[h,m] = sum_h' A[h',m] if h==m else -A[h,m], for h,m = 1..n (paper's
    §3, "First, define the Laplacian matrix"). 0-indexed here: L[i,j]
    corresponds to the paper's L[i+1, j+1]."""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    col_sums = A.sum(axis=0)  # sum over h' of A[h', m], per column m
    for i in range(n):
        for j in range(n):
            L[i, j] = col_sums[j] if i == j else -A[i, j]
    return L


def _build_L_hat(L: np.ndarray, r: np.ndarray) -> np.ndarray:
    """L_hat = L with row 1 (0-indexed row 0) replaced by the root-selection
    scores r (paper's §3.1, "Define a new matrix L_hat...")."""
    L_hat = L.copy()
    L_hat[0, :] = r
    return L_hat


def compute_tree_marginals(theta: np.ndarray) -> TreeMarginals:
    """Compute the partition function and every edge marginal for the
    single-root non-projective dependency-tree distribution induced by
    `theta`, via Koo et al. 2007 §3.1-3.2: Z = det(L_hat) (Proposition 1),
    and every marginal from ONE matrix inversion of L_hat via Jacobi's
    formula (∂log|X|/∂X = (X^-1)^T) — closed-form, O(n^3), no iteration.
    """
    n = theta.shape[0] - 1
    A, r = _build_A_and_r(theta)
    L = _build_laplacian(A)
    L_hat = _build_L_hat(L, r)

    Z = float(np.linalg.det(L_hat))
    L_hat_inv = np.linalg.inv(L_hat)  # the ONE matrix inversion the paper's §3.2 relies on

    mu = np.zeros((n + 1, n + 1), dtype=np.float64)
    for m in range(1, n + 1):
        # mu_{0,m} = r_m * [L_hat^-1]_{m,1}  (paper's 1-indexed row/col 1 == 0-indexed [m-1, 0])
        mu[0, m] = r[m - 1] * L_hat_inv[m - 1, 0]
        for h in range(1, n + 1):
            if h == m:
                continue
            a_hm = A[h - 1, m - 1]
            term1 = 0.0 if m == 1 else a_hm * L_hat_inv[m - 1, m - 1]
            term2 = 0.0 if h == 1 else a_hm * L_hat_inv[m - 1, h - 1]
            mu[h, m] = term1 - term2

    return TreeMarginals(n=n, Z=Z, mu=mu, theta=theta.copy())
