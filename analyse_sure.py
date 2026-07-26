#!/usr/bin/env python3
"""
analyse_sure.py — GOF + Regression/ANOVA for the SURE approx-coeff experiment.

Inputs (relative to this file's directory):
  VJP-ref.txt         — INFO log from the VJP reference run
  approx-{tag}.csv    — per-step CSVs from approx-mode runs

Usage:
  python analyse_sure.py

Outputs (stdout):
  1. GOF table   — which distributions best fit residual_rms / sure_val per coeff
  2. OLS summary — sure_val ~ C(approx_coeff) + step + sigma_hat_0
  3. Type-II ANOVA table
  4. Pairwise KS — do different coefficients produce different sure_val distributions?
  5. Mean sure_val per coefficient (lower = better SURE loss)
"""

import re
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
VJP_LOG  = os.path.join(HERE, "VJP-ref.txt")
CSV_GLOB = os.path.join(HERE, "approx-*.csv")

# ─── 1. Parse VJP reference log ───────────────────────────────────────────────
_LOG_RE = re.compile(
    r"\[sure_x0\].*?"
    r"eps=([\d.]+).*?"
    r"sigma_hat_0=([\d.]+).*?"
    r"sigma_p=([\d.]+).*?"
    r"lr=([\d.]+).*?"
    r"step_rms=([\d.]+).*?"
    r"sure=(-?[\d.]+).*?"
    r"jac_trace=(-?[\d.]+).*?"
    r"residual_rms=([\d.]+).*?"
    r"grad_rms=([\d.]+).*?"
    r"eff_grad_rms=([\d.]+).*?"
    r"adam_ratio=([\d.]+)"
)

def parse_vjp_log(path: str) -> pd.DataFrame:
    rows = []
    step = 0
    with open(path) as fh:
        for line in fh:
            m = _LOG_RE.search(line)
            if not m:
                continue
            (eps, sigma_hat_0, sigma_p, lr, step_rms,
             sure, jac_trace, residual_rms,
             grad_rms, eff_grad_rms, adam_ratio) = m.groups()
            rows.append({
                "step":            step,
                "sigma_hat_0":     float(sigma_hat_0),
                "sigma_t":         float(sigma_p),
                "approx_coeff":    "vjp",
                "coeff_num":       float("nan"),
                "residual_rms":    float(residual_rms),
                "sure_val":        float(sure),
                "raw_grad_rms":    float(grad_rms),
                "eff_grad_rms":    float(eff_grad_rms),
                "effective_alpha": float(lr),
                "step_rms":        float(step_rms),
                "adam_ratio":      float(adam_ratio),
            })
            step += 1
    return pd.DataFrame(rows)


# ─── 2. Load approx CSVs ──────────────────────────────────────────────────────
def _coeff_label_from_path(p: str) -> str:
    tag = os.path.basename(p)[len("approx-"):-len(".csv")]
    return tag.replace("dot", ".")     # "0dot5" → "0.5"

def load_approx_csvs(glob_pat: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(glob_pat)):
        df = pd.read_csv(p)
        # approx_coeff column is already in the CSV (float); convert to label string
        df["approx_coeff"] = _coeff_label_from_path(p)
        df["coeff_num"]    = df["approx_coeff"].str.replace("dot", ".").astype(float)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── 3. GOF tests ─────────────────────────────────────────────────────────────
CANDIDATES = [
    ("normal",      stats.norm),
    ("lognormal",   stats.lognorm),
    ("gamma",       stats.gamma),
    ("exponential", stats.expon),
    ("logistic",    stats.logistic),
    ("laplace",     stats.laplace),
]

def _gof_one(x: np.ndarray, dist_name: str, dist_obj) -> dict:
    try:
        x_fit = x - x.min() + 1e-9 if (
            dist_name in ("lognormal", "gamma", "exponential") and x.min() <= 0
        ) else x
        params   = dist_obj.fit(x_fit)
        ks_s, ks_p  = stats.kstest(x_fit, dist_obj.cdf, args=params)
        cvm         = stats.cramervonmises(x_fit, dist_obj.cdf, args=params)
        return {
            "dist":     dist_name,
            "KS_stat":  round(ks_s, 5),
            "KS_p":     round(ks_p, 5),
            "CvM_stat": round(cvm.statistic, 5),
            "CvM_p":    round(cvm.pvalue, 5),
            "KS_reject": ks_p < 0.05,
        }
    except Exception as exc:
        return {"dist": dist_name, "KS_stat": float("nan"), "KS_p": float("nan"),
                "CvM_stat": float("nan"), "CvM_p": float("nan"),
                "KS_reject": None, "error": str(exc)}

def gof_group(series: pd.Series) -> pd.DataFrame:
    x = series.dropna().values
    rows = []
    # Shapiro-Wilk (normality, n ≤ 5000)
    if len(x) <= 5000:
        sw_s, sw_p = stats.shapiro(x)
        rows.append({"dist": "normal (Shapiro-Wilk)", "KS_stat": round(sw_s, 5),
                     "KS_p": round(sw_p, 5), "CvM_stat": float("nan"),
                     "CvM_p": float("nan"), "KS_reject": sw_p < 0.05})
    for dist_name, dist_obj in CANDIDATES:
        rows.append(_gof_one(x, dist_name, dist_obj))
    return pd.DataFrame(rows)

def best_fit_label(gof_df: pd.DataFrame) -> str:
    ks_rows = gof_df[gof_df["KS_reject"] == False]   # noqa: E712
    if ks_rows.empty:
        return "none (all rejected at α=0.05)"
    return ks_rows.sort_values("KS_p", ascending=False).iloc[0]["dist"]


# ─── 4. Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    # Load
    approx_df = load_approx_csvs(CSV_GLOB)
    vjp_df    = parse_vjp_log(VJP_LOG) if os.path.exists(VJP_LOG) else pd.DataFrame()

    if approx_df.empty:
        sys.exit(f"No approx CSVs found matching: {CSV_GLOB}")

    all_df   = pd.concat([approx_df, vjp_df], ignore_index=True)
    coeffs   = sorted(approx_df["approx_coeff"].unique(), key=float)
    has_vjp  = not vjp_df.empty

    sep = "=" * 72

    # ── GOF ──────────────────────────────────────────────────────────────────
    print(sep)
    print("SECTION 1 — GOODNESS-OF-FIT (GOF)")
    print(sep)
    print(f"  Candidate distributions : {[n for n,_ in CANDIDATES]}")
    print("  Tests                   : Shapiro-Wilk, KS (MLE-fit), Cramér-von Mises")
    print("  Rejection threshold     : α = 0.05")
    print()

    gof_summary_rows = []
    for coeff in coeffs + (["vjp"] if has_vjp else []):
        sub = all_df[all_df["approx_coeff"] == coeff]
        for var in ("residual_rms", "sure_val"):
            gof = gof_group(sub[var])
            best = best_fit_label(gof)
            gof_summary_rows.append({"coeff": coeff, "variable": var,
                                     "n": len(sub), "best_fit": best})
            print(f"  coeff={coeff}  |  {var}  (n={len(sub)})")
            print(gof.to_string(index=False))
            print(f"  → best non-rejected fit : {best}")
            print()

    print(sep)
    print("GOF SUMMARY")
    print(sep)
    print(pd.DataFrame(gof_summary_rows).to_string(index=False))
    print()

    # ── Regression + Type-II ANOVA ────────────────────────────────────────────
    reg_df = approx_df.copy()     # numeric coeff only (no VJP)
    reg_df["approx_coeff"] = reg_df["approx_coeff"].astype(str)

    print(sep)
    print("SECTION 2 — OLS REGRESSION")
    print("  Formula: sure_val ~ C(approx_coeff) + step + sigma_hat_0")
    print("  C(approx_coeff) uses default Treatment coding (ref = lowest coeff)")
    print(sep)
    formula = "sure_val ~ C(approx_coeff) + step + sigma_hat_0"
    lm      = ols(formula, data=reg_df).fit()
    print(lm.summary())
    print()

    print(sep)
    print("SECTION 3 — TYPE-II ANOVA")
    print("  Tests each factor after adjusting for all others (order-invariant)")
    print(sep)
    aov = anova_lm(lm, typ=2)
    print(aov.to_string())
    print()

    # ── Pairwise KS on sure_val distributions ────────────────────────────────
    print(sep)
    print("SECTION 4 — PAIRWISE KS TEST  (sure_val distributions)")
    print("  H0: the two coefficient groups are drawn from the same distribution")
    print(sep)
    hdr = f"{'coeff_A':>8}  {'coeff_B':>8}  {'KS stat':>8}  {'p-value':>10}  H0 rejected?"
    print(hdr)
    for i, ca in enumerate(coeffs):
        for cb in coeffs[i+1:]:
            a = all_df.loc[all_df["approx_coeff"] == ca, "sure_val"].dropna()
            b = all_df.loc[all_df["approx_coeff"] == cb, "sure_val"].dropna()
            ks_s, ks_p = stats.ks_2samp(a, b)
            print(f"{ca:>8}  {cb:>8}  {ks_s:>8.4f}  {ks_p:>10.4f}  "
                  f"{'YES (different)' if ks_p < 0.05 else 'no (same)'}")

    if has_vjp:
        print()
        print("  VJP reference vs each approx coeff:")
        vjp_sure = vjp_df["sure_val"].dropna()
        for ca in coeffs:
            a = all_df.loc[all_df["approx_coeff"] == ca, "sure_val"].dropna()
            ks_s, ks_p = stats.ks_2samp(vjp_sure, a)
            print(f"    vjp vs {ca:>5}  KS={ks_s:.4f}  p={ks_p:.4f}  "
                  f"{'DIFFERENT' if ks_p < 0.05 else 'same'}")
    print()

    # ── Mean sure_val summary (key result) ───────────────────────────────────
    print(sep)
    print("SECTION 5 — MEAN sure_val PER COEFFICIENT  (lower = better SURE loss)")
    print(sep)
    summary = (
        reg_df.groupby("approx_coeff")["sure_val"]
        .agg(mean="mean", std="std", median="median", n="count")
        .sort_values("mean")
    )
    print(summary.to_string())
    if has_vjp:
        vjp_mean   = vjp_df["sure_val"].mean()
        vjp_median = vjp_df["sure_val"].median()
        print(f"\n  VJP reference  mean={vjp_mean:.4f}  median={vjp_median:.4f}  "
              f"n={len(vjp_df)}")
    print()


if __name__ == "__main__":
    main()
