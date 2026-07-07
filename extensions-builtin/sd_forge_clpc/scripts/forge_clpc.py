"""CLPC: Closed-Loop Predictor-Corrector adaptive sampler — WebUI extension.

Adds "CLPC ODE" and "CLPC SDE" to the sampler dropdown.  The extension
accordion exposes all sampler parameters; changes are forwarded via
p.extra_generation_params so they appear in image metadata.
"""

import gradio as gr

from modules import scripts
from ldm_patched.k_diffusion.clpc_sampler import PREDICTOR_EULER, PREDICTOR_IMPLICIT_ADAMS


class CLPCForForge(scripts.Script):
    sorting_priority = 13.0

    def title(self):
        return "CLPC Sampler Settings"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown(
                "These settings only take effect when **CLPC ODE** or **CLPC SDE** "
                "is selected as the sampler."
            )

            # ── Predictor mode ────────────────────────────────────────────
            predictor = gr.Radio(
                choices=[PREDICTOR_IMPLICIT_ADAMS, PREDICTOR_EULER],
                value=PREDICTOR_IMPLICIT_ADAMS,
                label="Predictor mode",
                info=(
                    "implicit_adams (default): AM-style; lam_t as node; bounded — "
                    "AdamsStability.lean.  "
                    "euler: order-1 always; b∈(0,1); unconditionally safe — RungeGuard.lean."
                ),
            )

            # ── Lean-proved enhancements ──────────────────────────────────
            with gr.Row():
                use_chebyshev = gr.Checkbox(
                    label="Chebyshev node selection",
                    value=True,
                    info=(
                        "Select history λ-nodes by Chebyshev spacing (Priority 1). "
                        "Eliminates Runge risk for order≥2 without Euler fallback. "
                        "ChebyshevAdaptive.lean: adams_vs_chebyshev."
                    ),
                )
                use_kalman = gr.Checkbox(
                    label="Kalman-gain blend",
                    value=True,
                    info=(
                        "Blend pred/corr with K = ode_err/(ode_err+wav_hf) (Priority 3). "
                        "A=(σ_t/σ_s)·I proved contractive; K proved MMSE-optimal. "
                        "KalmanMatrixDesign.lean + KalmanFilter.lean."
                    ),
                )

            # ── Order (Priority 4) ─────────────────────────────────────────
            with gr.Row():
                max_order = gr.Slider(
                    label="Max order", minimum=1, maximum=6, step=1, value=3,
                    info=(
                        "Maximum predictor order (corrector gets max_order+1 nodes), "
                        "UniPC-style (Priority 4). Consistent at any order and strictly "
                        "reduces local error as h→0 — VariableOrderGain.lean "
                        "(am_b_coeffs_sum_to_one_general, order_gain_ratio_tendsto_zero). "
                        "Above 3, requires 'Chebyshev node selection' above for "
                        "individual-coefficient boundedness (now also gates the corrector, "
                        "not just the predictor)."
                    ),
                )
                lower_order_final = gr.Checkbox(
                    label="Lower order near end", value=True,
                    info="Ramp order down near the schedule's last step (UniPC's lower_order_final).",
                )

            # ── Token-space conditional guidance (separate error channels) ──
            with gr.Row():
                token_kalman_weight = gr.Slider(
                    label="Token-space error weight (G-score, monitoring only)",
                    minimum=0.0, maximum=2.0, step=0.05, value=0.5,
                    info=(
                        "Weights conditional-space error (1 - P_token) in the composite "
                        "G-score/progress display. Informational/monitoring ONLY — this "
                        "signal is deliberately NOT fed into the Kalman corrector-trust gain "
                        "(a live run showed doing so pinned corrector trust near maximum for "
                        "an entire generation, since vanish/leak/drift can plateau well below "
                        "'perfect' for a whole run instead of decaying like the ODE error does). "
                        "REQUIRES the separate 'SURE Token Subspace Guidance' extension to also "
                        "be enabled — this slider has no effect on its own. Check the console "
                        "for a '[TokenSubspaceGuidance] clpc_sampler: ENABLED/INACTIVE' line at "
                        "the start of each generation to confirm wiring."
                    ),
                )
                token_spatial_weight = gr.Slider(
                    label="Token-space spatial weight (region-mapped, GENUINELY active)",
                    minimum=0.0, maximum=2.0, step=0.05, value=0.3,
                    info=(
                        "Unlike the weight above, this one actually changes the corrector's "
                        "output: TSG's per-pixel LEAK correction magnitude is resampled onto "
                        "this latent's own grid (same recipe SURE-AG's entropy map uses) and "
                        "added directly to the Kalman gain — wherever a genuine leak correction "
                        "happened, the corrector is trusted more there. Kept OUT of ode_err/"
                        "wav_hf_err themselves (TokenAvoidSOC.lean's cluster_step_not_lipschitz "
                        "finding). Also requires 'SURE Token Subspace Guidance' enabled."
                    ),
                )

            # ── Error weights ─────────────────────────────────────────────
            with gr.Row():
                w_ode = gr.Slider(label="w_ode", minimum=0.0, maximum=5.0, step=0.05, value=1.0,
                                  info="ODE embedded-pair error weight.")
                w_ot = gr.Slider(label="w_ot", minimum=0.0, maximum=5.0, step=0.05, value=0.5,
                                 info="Optimal-transport drift error weight.")
                w_cfg = gr.Slider(label="w_cfg", minimum=0.0, maximum=5.0, step=0.05, value=0.3,
                                  info="CFG guidance-drift error weight (zero NFE).")

            # ── Tolerances + PECE ─────────────────────────────────────────
            with gr.Row():
                atol = gr.Number(label="atol", value=1e-4, precision=6)
                rtol = gr.Number(label="rtol", value=1e-3, precision=6)
                pece_sigma_threshold = gr.Slider(
                    label="PECE σ threshold", minimum=0.0, maximum=20.0, step=0.5, value=4.0,
                    info="Disable PECE corrector above this σ (only gain at low σ — PECEOrderGain.lean).",
                )

            # ── SDE controls ──────────────────────────────────────────────
            with gr.Row():
                tau_eta = gr.Slider(label="tau_eta (SDE only)", minimum=0.0, maximum=2.0,
                                    step=0.05, value=1.0,
                                    info="SDE noise strength. 0 = ODE mode regardless of sampler choice.")
                s_noise = gr.Slider(label="s_noise (SDE only)", minimum=0.5, maximum=2.0,
                                    step=0.05, value=1.0)
                adaptive_noise = gr.Checkbox(
                    label="Adaptive noise (SDE only)", value=True,
                    info="Ratchet SDE noise up when ODE error persistently exceeds OT drift.",
                )

            max_steps = gr.Slider(label="Max steps (hard limit)", minimum=10, maximum=5000,
                                  step=10, value=1000)

        return (predictor, use_chebyshev, use_kalman, max_order, lower_order_final,
                token_kalman_weight, token_spatial_weight,
                w_ode, w_ot, w_cfg, atol, rtol, pece_sigma_threshold,
                tau_eta, s_noise, adaptive_noise, max_steps)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (predictor, use_chebyshev, use_kalman, max_order, lower_order_final,
         token_kalman_weight, token_spatial_weight,
         w_ode, w_ot, w_cfg, atol, rtol, pece_sigma_threshold,
         tau_eta, s_noise, adaptive_noise, max_steps) = script_args

        sampler_name = getattr(p, "sampler_name", "")
        if "clpc" not in sampler_name.lower():
            return

        override = {
            "predictor": predictor,
            "use_chebyshev": bool(use_chebyshev),
            "use_kalman": bool(use_kalman),
            "max_order": int(max_order),
            "lower_order_final": bool(lower_order_final),
            "token_kalman_weight": float(token_kalman_weight),
            "token_spatial_weight": float(token_spatial_weight),
            "w_ode": float(w_ode),
            "w_ot": float(w_ot),
            "w_sure": 0.0,
            "w_cfg": float(w_cfg),
            "atol": float(atol),
            "rtol": float(rtol),
            "pece_sigma_threshold": float(pece_sigma_threshold),
            "max_steps": int(max_steps),
        }
        if "sde" in sampler_name.lower():
            override.update({
                "tau_eta": float(tau_eta),
                "s_noise": float(s_noise),
                "adaptive_noise": bool(adaptive_noise),
            })

        sampler_obj = getattr(p, "sampler", None)
        if sampler_obj is not None and hasattr(sampler_obj, "extra_options"):
            sampler_obj.extra_options.update(override)

        p.extra_generation_params.update({
            "clpc_predictor": predictor,
            "clpc_chebyshev": use_chebyshev,
            "clpc_kalman": use_kalman,
            "clpc_max_order": max_order,
            "clpc_lower_order_final": lower_order_final,
            "clpc_token_kalman_weight": token_kalman_weight,
            "clpc_token_spatial_weight": token_spatial_weight,
            "clpc_w_ode": w_ode,
            "clpc_w_ot": w_ot,
            "clpc_w_cfg": w_cfg,
            "clpc_atol": atol,
            "clpc_rtol": rtol,
            "clpc_pece_sigma_threshold": pece_sigma_threshold,
            "clpc_max_steps": max_steps,
        })
        if "sde" in sampler_name.lower():
            p.extra_generation_params.update({
                "clpc_tau_eta": tau_eta,
                "clpc_s_noise": s_noise,
                "clpc_adaptive_noise": adaptive_noise,
            })
