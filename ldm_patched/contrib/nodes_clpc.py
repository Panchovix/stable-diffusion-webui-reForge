"""CLPC: Closed-Loop Predictor-Corrector adaptive sampler node.

Registers clpc_ode and clpc_sde as selectable samplers in ComfyUI.
Parameters are exposed as optional model-patch inputs so they can be
wired from other nodes without requiring a custom sampler wrapper.
"""

import ldm_patched.modules.samplers as _samplers
from ldm_patched.k_diffusion.clpc_sampler import (
    sample_clpc_ode, sample_clpc_sde,
    PREDICTOR_EULER, PREDICTOR_IMPLICIT_ADAMS,
)

_PREDICTOR_OPTIONS = [PREDICTOR_IMPLICIT_ADAMS, PREDICTOR_EULER]


class CLPCSamplerODE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "predictor": (_PREDICTOR_OPTIONS, {
                    "default": PREDICTOR_IMPLICIT_ADAMS,
                    "tooltip": (
                        "implicit_adams (default): AM-style; lam_t as node; bounded — "
                        "AdamsStability.lean. "
                        "euler: order-1 always; b∈(0,1); unconditionally safe — RungeGuard.lean."
                    ),
                }),
                "use_chebyshev": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Select history λ-nodes by Chebyshev spacing instead of recency. "
                        "Eliminates Runge-phenomenon risk for order≥2 without falling back "
                        "to Euler. Proved minimax-optimal in ChebyshevAdaptive.lean."
                    ),
                }),
                "use_kalman": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Blend predictor and corrector with Kalman gain "
                        "K = ode_err / (ode_err + wav_hf_err). "
                        "A=(σ_t/σ_s)·I proved contractive; K proved MMSE-optimal "
                        "in KalmanMatrixDesign.lean + KalmanFilter.lean."
                    ),
                }),
                "max_order": ("INT", {
                    "default": 3, "min": 1, "max": 6,
                    "tooltip": (
                        "Maximum predictor order (corrector gets max_order+1 nodes). "
                        "UniPC-style. Consistent (b-coeffs sum to 1) at ANY order and "
                        "strictly reduces local error as h→0 — VariableOrderGain.lean "
                        "(am_b_coeffs_sum_to_one_general, order_gain_ratio_tendsto_zero). "
                        "Above 3, individual-coefficient boundedness requires "
                        "use_chebyshev=True (now also gates the corrector)."
                    ),
                }),
                "lower_order_final": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ramp order down near the last step, like UniPC's lower_order_final.",
                }),
                "w_ode":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "ODE embedded-pair error weight."}),
                "w_ot":    ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "Optimal-transport drift error weight."}),
                "w_cfg":   ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "CFG guidance-drift error weight (zero NFE)."}),
                "atol":    ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "rtol":    ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 1e-1, "step": 1e-4}),
                "pece_sigma_threshold": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Disable PECE above this sigma (proved beneficial only at low σ)."}),
                "max_steps": ("INT", {"default": 1000, "min": 10, "max": 5000}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, predictor, use_chebyshev, use_kalman, max_order, lower_order_final,
                    w_ode, w_ot, w_cfg, atol, rtol,
                    pece_sigma_threshold, max_steps):
        sampler = _samplers.KSAMPLER(
            sample_clpc_ode,
            extra_options={
                "predictor": predictor,
                "use_chebyshev": use_chebyshev,
                "use_kalman": use_kalman,
                "max_order": max_order,
                "lower_order_final": lower_order_final,
                "w_ode": w_ode, "w_ot": w_ot, "w_sure": 0.0, "w_cfg": w_cfg,
                "atol": atol, "rtol": rtol,
                "pece_sigma_threshold": pece_sigma_threshold,
                "max_steps": max_steps,
            },
        )
        return (sampler,)


class CLPCSamplerSDE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "predictor": (_PREDICTOR_OPTIONS, {
                    "default": PREDICTOR_IMPLICIT_ADAMS,
                    "tooltip": (
                        "implicit_adams (default): AM-style; lam_t as node; bounded — "
                        "AdamsStability.lean. "
                        "euler: order-1 always; b∈(0,1); unconditionally safe — RungeGuard.lean."
                    ),
                }),
                "use_chebyshev": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Select history λ-nodes by Chebyshev spacing instead of recency. "
                        "Proved minimax-optimal in ChebyshevAdaptive.lean."
                    ),
                }),
                "use_kalman": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Blend predictor and corrector with Kalman gain K = ode_err/(ode_err+wav_hf). "
                        "Proved contractive + MMSE-optimal in KalmanMatrixDesign.lean."
                    ),
                }),
                "max_order": ("INT", {
                    "default": 3, "min": 1, "max": 6,
                    "tooltip": (
                        "Maximum predictor order (corrector gets max_order+1 nodes). "
                        "UniPC-style — VariableOrderGain.lean. Above 3, requires "
                        "use_chebyshev=True for individual-coefficient boundedness."
                    ),
                }),
                "lower_order_final": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ramp order down near the last step, like UniPC's lower_order_final.",
                }),
                "w_ode":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "w_ot":    ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "w_cfg":   ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05}),
                "atol":    ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "rtol":    ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 1e-1, "step": 1e-4}),
                "tau_eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "SDE noise strength (0 = ODE)."}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "adaptive_noise": ("BOOLEAN", {"default": True,
                                               "tooltip": "Ratchet noise up when ODE error > OT drift."}),
                "pece_sigma_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "max_steps": ("INT", {"default": 1000, "min": 10, "max": 5000}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, predictor, use_chebyshev, use_kalman, max_order, lower_order_final,
                    w_ode, w_ot, w_cfg, atol, rtol,
                    tau_eta, s_noise, adaptive_noise,
                    pece_sigma_threshold, max_steps):
        sampler = _samplers.KSAMPLER(
            sample_clpc_sde,
            extra_options={
                "predictor": predictor,
                "use_chebyshev": use_chebyshev,
                "use_kalman": use_kalman,
                "max_order": max_order,
                "lower_order_final": lower_order_final,
                "w_ode": w_ode, "w_ot": w_ot, "w_sure": 0.0, "w_cfg": w_cfg,
                "atol": atol, "rtol": rtol,
                "tau_eta": tau_eta, "s_noise": s_noise, "adaptive_noise": adaptive_noise,
                "pece_sigma_threshold": pece_sigma_threshold,
                "max_steps": max_steps,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "CLPCSamplerODE": CLPCSamplerODE,
    "CLPCSamplerSDE": CLPCSamplerSDE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLPCSamplerODE": "CLPC Sampler (ODE)",
    "CLPCSamplerSDE": "CLPC Sampler (SDE)",
}
