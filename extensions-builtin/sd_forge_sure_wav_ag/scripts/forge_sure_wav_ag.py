import gradio as gr

from modules import scripts
from ldm_patched.contrib.nodes_sure_wav_ag import SureWaveletAttentionGuidance

opSureWavAG = SureWaveletAttentionGuidance()


class SureWavAGForForge(scripts.Script):
    sorting_priority = 12.7

    def title(self):
        return "SURE Wavelet Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            with gr.Row():
                alpha = gr.Slider(
                    label="Alpha",
                    minimum=0.001, maximum=0.49, step=0.001, value=0.05,
                    info="Step size ceiling. Actual value set by Alpha Mode each step.",
                )
                alpha_mode = gr.Radio(
                    label="Alpha Mode",
                    choices=["sigma", "analytical", "bo", "fixed"],
                    value="sigma",
                    info=(
                        "sigma: scale by 1/(1+min(σ,1)), EDM-aware (recommended). "
                        "analytical: α*=⟨r,g⟩/‖g‖², free via Parseval. "
                        "bo: Bayesian optimisation, needs optuna. "
                        "fixed: static alpha."
                    ),
                )
            with gr.Row():
                attn_weight = gr.Slider(
                    label="Attention Weight",
                    minimum=0.0, maximum=4.0, step=0.05, value=1.0,
                    info="Per-subband entropy amplification. 0 = wavelet SURE only.",
                )
                attn_blocks = gr.Radio(
                    label="Attention Blocks",
                    choices=["all", "middle", "mid+out", "input", "output"],
                    value="middle",
                    info="UNet blocks to capture entropy from. 'middle' is cheapest.",
                )
            with gr.Row():
                approx_coeff = gr.Slider(
                    label="Gradient Coeff",
                    minimum=0.5, maximum=4.0, step=0.1, value=2.0,
                    info="Gradient approximation scale (2.0 = theory).",
                )
                wavelet = gr.Dropdown(
                    label="Wavelet",
                    choices=["db4", "db2", "haar", "sym4", "sym8", "bior2.2"],
                    value="db4",
                )
                wavelet_level = gr.Slider(
                    label="Wavelet Level",
                    minimum=1, maximum=5, step=1, value=3,
                    info="Decomposition depth.",
                )
            with gr.Row():
                lp_frac = gr.Slider(
                    label="LP Fraction",
                    minimum=0.0, maximum=1.0, step=0.1, value=1.0,
                    info="Fraction of detail subbands to correct. 1.0=all, 0.0=approx only.",
                )
                fft_scale = gr.Slider(
                    label="FFT Scale",
                    minimum=0.1, maximum=2.0, step=0.05, value=1.0,
                    info="FFT low-pass scale on approx gradient. 1.0=off.",
                )
                fft_threshold = gr.Slider(
                    label="FFT Threshold",
                    minimum=1, maximum=8, step=1, value=1,
                    info="Radius of FFT low-frequency window (latent pixels).",
                )

        return (enabled, alpha, alpha_mode, attn_weight, attn_blocks, approx_coeff,
                wavelet, wavelet_level, lp_frac, fft_scale, fft_threshold)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (enabled, alpha, alpha_mode, attn_weight, attn_blocks, approx_coeff,
         wavelet, wavelet_level, lp_frac, fft_scale, fft_threshold) = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        unet = opSureWavAG.patch(
            unet, alpha, alpha_mode, attn_weight, attn_blocks, approx_coeff,
            wavelet, int(wavelet_level), lp_frac, fft_scale, int(fft_threshold),
        )[0]
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            sure_wavag_enabled=True,
            sure_wavag_alpha=alpha,
            sure_wavag_alpha_mode=alpha_mode,
            sure_wavag_attn_weight=attn_weight,
            sure_wavag_attn_blocks=attn_blocks,
            sure_wavag_wavelet=wavelet,
            sure_wavag_level=int(wavelet_level),
            sure_wavag_lp_frac=lp_frac,
            sure_wavag_fft_scale=fft_scale,
        ))
