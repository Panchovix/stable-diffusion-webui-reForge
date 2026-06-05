"""SURE-AG: Attention-Guided SURE as a guidance node.

Patches any model to apply entropy-weighted SURE correction after every
CFG denoising step.  Compatible with all samplers (Euler, DPM++, etc.).

Cost: one extra UNet forward pass per step (same as SAG).

Algorithm per step
------------------
1. Run model(x0_hat, sigma) with attn1 entropy hooks → get sure_x0, capture U.
2. SURE residual:  r = x0_hat − sure_x0
3. Weight:         W = 1 + attn_weight · U   (U ∈ [0,1], W ∈ [1, 1+w])
4. Corrected:      x0_hat − alpha · approx_coeff · r · W

Alpha is automatically clamped to < 1/(2·(1+attn_weight)) (Lean §3).
"""

import logging

import torch
import ldm_patched.modules.samplers as _samplers
from ldm_patched.k_diffusion.sure_attention import (
    build_capture_model_options,
    _aggregate_entropy_map,
)

_logger = logging.getLogger("nodes_sure_ag")


class SureAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "alpha": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 0.49, "step": 0.001,
                    "tooltip": "SURE step size. Auto-clamped to 1/(2*(1+attn_weight)).",
                }),
                "attn_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Entropy amplification. 0 = uniform SURE. Higher = stronger correction in uncertain regions.",
                }),
                "attn_blocks": (["all", "middle", "mid+out", "input", "output"],),
                "approx_coeff": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Gradient scale (2.0 matches SURE theory).",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "guidance"

    def patch(self, model, alpha, attn_weight, attn_blocks, approx_coeff):
        m = model.clone()

        alpha = float(alpha)
        attn_weight = float(attn_weight)
        approx_coeff = float(approx_coeff)

        # Lean §3: descent requires alpha < 1 / (2 * (1 + attn_weight))
        alpha_max = 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4
        alpha_eff = min(alpha, alpha_max)
        if alpha > alpha_max:
            _logger.warning(
                "[sure_ag] alpha=%.4f exceeds safe range %.4f for attn_weight=%.2f; clamped",
                alpha, alpha_max, attn_weight,
            )

        def post_cfg_function(args):
            x0_hat        = args["denoised"]
            sigma         = args["sigma"]
            model_inner   = args["model"]
            model_options = args["model_options"]
            cond          = args["cond"]

            if min(x0_hat.shape[2:]) <= 4:
                return x0_hat

            # ── Extra SURE forward pass with entropy capture ──────────────────
            attn_store = []
            capture_opts = build_capture_model_options(model_options, attn_store, attn_blocks)

            (sure_x0,) = _samplers.calc_cond_batch(
                model_inner, [cond], x0_hat, sigma, capture_opts
            )

            # ── Aggregate entropy map U ∈ [0,1] ──────────────────────────────
            U = _aggregate_entropy_map(
                attn_store, x0_hat.shape, x0_hat.device, x0_hat.dtype
            )

            # ── SURE-AG weighted gradient ─────────────────────────────────────
            residual = x0_hat - sure_x0
            if U is not None and attn_weight > 0.0:
                grad = approx_coeff * residual * (1.0 + attn_weight * U)
            else:
                grad = approx_coeff * residual

            corrected = x0_hat - alpha_eff * grad

            _logger.debug(
                "[sure_ag] sigma=%.4f  alpha=%.4f  entropy_rms=%.4f  layers=%d  blocks=%s",
                float(sigma.max()), alpha_eff,
                float(U.pow(2).mean().sqrt()) if U is not None else 0.0,
                len(attn_store), attn_blocks,
            )

            return corrected

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SureAttentionGuidance": SureAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SureAttentionGuidance": "SURE Attention Guidance",
}
