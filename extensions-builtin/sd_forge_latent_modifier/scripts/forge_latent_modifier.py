import gradio as gr
import sys
import traceback
from typing import Any
from functools import partial
from modules import scripts, script_callbacks
from modules.infotext_utils import PasteField

from lib_latent_modifier.sampler_mega_modifier import ModelSamplerLatentMegaModifier

opModelSamplerLatentMegaModifier = ModelSamplerLatentMegaModifier().mega_modify


class LatentModifierForForge(scripts.Script):
    sorting_priority = 16.01

    def title(self):
        return "LatentModifier Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Tab("Sharpness & Detail"):
                gr.Markdown("""
                    ### Sharpness Control
                    Sharpens the noise during diffusion for more perceptual detail. Particularly effective at higher strengths.
                    """)
                sharpness_enabled = gr.Checkbox(
                    label="Enable Sharpness Controls", value=False
                )
                with gr.Group(visible=True):
                    sharpness_multiplier = gr.Slider(
                        label="Sharpness Multiplier",
                        minimum=-100.0,
                        maximum=100.0,
                        step=0.1,
                        value=0.0,
                    )
                    sharpness_method = gr.Radio(
                        label="Sharpness Method",
                        choices=["anisotropic", "joint-anisotropic", "gaussian", "cas"],
                        value="anisotropic",
                    )
                    affect_uncond = gr.Radio(
                        label="Apply to Unconditional",
                        choices=["None", "Sharpness"],
                        value="None",
                        info="Whether to apply sharpness to the unconditional branch",
                    )

            with gr.Tab("Tone & Contrast"):
                gr.Markdown("""
                    ### Tonemap Control
                    Clamps conditioning noise (CFG) using various methods. Enables use of higher CFG values safely.
                    """)
                tone_enabled = gr.Checkbox(label="Enable Tone Controls", value=False)
                with gr.Group(visible=True):
                    tonemap_multiplier = gr.Slider(
                        label="Tonemap Multiplier",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.01,
                        value=0.0,
                    )
                    tonemap_method = gr.Radio(
                        label="Tonemap Method",
                        choices=[
                            "reinhard",
                            "reinhard_perchannel",
                            "arctan",
                            "quantile",
                            "gated",
                            "cfg-mimic",
                            "spatial-norm",
                        ],
                        value="reinhard",
                    )
                    tonemap_percentile = gr.Slider(
                        label="Tonemap Percentile",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.005,
                        value=100.0,
                    )
                    contrast_multiplier = gr.Slider(
                        label="Contrast Multiplier",
                        minimum=-100.0,
                        maximum=100.0,
                        step=0.1,
                        value=0.0,
                    )

            with gr.Tab("Noise & Combat"):
                noise_enabled = gr.Checkbox(
                    label="Enable Noise & Combat Controls", value=False
                )
                with gr.Group(visible=True):
                    gr.Markdown(
                        "### Extra Noise\nAdds controlled noise during diffusion for artistic effects."
                    )
                    extra_noise_type = gr.Radio(
                        label="Extra Noise Type",
                        choices=[
                            "gaussian",
                            "uniform",
                            "perlin",
                            "pink",
                            "green",
                            "pyramid",
                        ],
                        value="gaussian",
                    )
                    extra_noise_method = gr.Radio(
                        label="Extra Noise Method",
                        choices=[
                            "add",
                            "add_scaled",
                            "speckle",
                            "cads",
                            "cads_rescaled",
                            "cads_speckle",
                            "cads_speckle_rescaled",
                        ],
                        value="add",
                    )
                    extra_noise_multiplier = gr.Slider(
                        label="Extra Noise Multiplier",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=0.0,
                    )
                    extra_noise_lowpass = gr.Slider(
                        label="Extra Noise Lowpass",
                        minimum=0,
                        maximum=1000,
                        step=1,
                        value=100,
                    )

                    gr.Markdown(
                        "### CFG Combat & Rescaling\nHandles CFG-related artifacts and drift."
                    )
                    combat_method = gr.Radio(
                        label="Combat Method",
                        choices=[
                            "subtract",
                            "subtract_channels",
                            "subtract_median",
                            "sharpen",
                        ],
                        value="subtract",
                    )
                    combat_cfg_drift = gr.Slider(
                        label="Combat Cfg Drift",
                        minimum=-10.0,
                        maximum=10.0,
                        step=0.01,
                        value=0.0,
                    )
                    rescale_cfg_phi = gr.Slider(
                        label="Rescale Cfg Phi",
                        minimum=-10.0,
                        maximum=10.0,
                        step=0.01,
                        value=0.0,
                    )

            with gr.Tab("Advanced Processing"):
                advanced_enabled = gr.Checkbox(
                    label="Enable Advanced Processing", value=False
                )
                with gr.Group(visible=True):
                    gr.Markdown(
                        "### Divisive Normalization\nReduces noisy artifacts, especially useful with high sharpness."
                    )
                    divisive_norm_size = gr.Slider(
                        label="Divisive Norm Size",
                        minimum=1,
                        maximum=255,
                        step=1,
                        value=127,
                    )
                    divisive_norm_multiplier = gr.Slider(
                        label="Divisive Norm Multiplier",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                    )

                    gr.Markdown("""
                        ### Spectral Modulation
                        Frequency-domain processing to handle oversaturation from high CFG values while preserving median values.
                        """)
                    spectral_mod_mode = gr.Radio(
                        label="Spectral Mod Mode",
                        choices=["hard_clamp", "soft_clamp"],
                        value="hard_clamp",
                    )
                    spectral_mod_percentile = gr.Slider(
                        label="Spectral Mod Percentile",
                        minimum=0.0,
                        maximum=50.0,
                        step=0.01,
                        value=5.0,
                    )
                    spectral_mod_multiplier = gr.Slider(
                        label="Spectral Mod Multiplier",
                        minimum=-15.0,
                        maximum=15.0,
                        step=0.01,
                        value=0.0,
                    )

            with gr.Tab("Dynamic CFG"):
                dynamic_enabled = gr.Checkbox(label="Enable Dynamic CFG", value=False)
                with gr.Group(visible=True):
                    gr.Markdown(
                        "### Dynamic CFG Augmentation\nAdvanced CFG behavior modification."
                    )
                    dyn_cfg_augmentation = gr.Radio(
                        label="Dyn Cfg Augmentation",
                        choices=[
                            "None",
                            "dyncfg-halfcosine",
                            "dyncfg-halfcosine-mimic",
                        ],
                        value="None",
                    )

            self.infotext_fields = [
                PasteField(
                    sharpness_multiplier,
                    "latent_modifier_sharpness_multiplier",
                    api="latent_modifier_sharpness_multiplier",
                ),
                PasteField(
                    sharpness_method,
                    "latent_modifier_sharpness_method",
                    api="latent_modifier_sharpness_method",
                ),
                PasteField(
                    tonemap_multiplier,
                    "latent_modifier_tonemap_multiplier",
                    api="latent_modifier_tonemap_multiplier",
                ),
                PasteField(
                    tonemap_method,
                    "latent_modifier_tonemap_method",
                    api="latent_modifier_tonemap_method",
                ),
                PasteField(
                    tonemap_percentile,
                    "latent_modifier_tonemap_percentile",
                    api="latent_modifier_tonemap_percentile",
                ),
                PasteField(
                    contrast_multiplier,
                    "latent_modifier_contrast_multiplier",
                    api="latent_modifier_contrast_multiplier",
                ),
                PasteField(
                    combat_method,
                    "latent_modifier_combat_method",
                    api="latent_modifier_combat_method",
                ),
                PasteField(
                    combat_cfg_drift,
                    "latent_modifier_combat_cfg_drift",
                    api="latent_modifier_combat_cfg_drift",
                ),
                PasteField(
                    rescale_cfg_phi,
                    "latent_modifier_rescale_cfg_phi",
                    api="latent_modifier_rescale_cfg_phi",
                ),
                PasteField(
                    extra_noise_type,
                    "latent_modifier_extra_noise_type",
                    api="latent_modifier_extra_noise_type",
                ),
                PasteField(
                    extra_noise_method,
                    "latent_modifier_extra_noise_method",
                    api="latent_modifier_extra_noise_method",
                ),
                PasteField(
                    extra_noise_multiplier,
                    "latent_modifier_extra_noise_multiplier",
                    api="latent_modifier_extra_noise_multiplier",
                ),
                PasteField(
                    extra_noise_lowpass,
                    "latent_modifier_extra_noise_lowpass",
                    api="latent_modifier_extra_noise_lowpass",
                ),
                PasteField(
                    divisive_norm_size,
                    "latent_modifier_divisive_norm_size",
                    api="latent_modifier_divisive_norm_size",
                ),
                PasteField(
                    divisive_norm_multiplier,
                    "latent_modifier_divisive_norm_multiplier",
                    api="latent_modifier_divisive_norm_multiplier",
                ),
                PasteField(
                    spectral_mod_mode,
                    "latent_modifier_spectral_mod_mode",
                    api="latent_modifier_spectral_mod_mode",
                ),
                PasteField(
                    spectral_mod_percentile,
                    "latent_modifier_spectral_mod_percentile",
                    api="latent_modifier_spectral_mod_percentile",
                ),
                PasteField(
                    spectral_mod_multiplier,
                    "latent_modifier_spectral_mod_multiplier",
                    api="latent_modifier_spectral_mod_multiplier",
                ),
                PasteField(
                    affect_uncond,
                    "latent_modifier_affect_uncond",
                    api="latent_modifier_affect_uncond",
                ),
                PasteField(
                    dyn_cfg_augmentation,
                    "latent_modifier_dyn_cfg_augmentation",
                    api="latent_modifier_dyn_cfg_augmentation",
                ),
            ]
        self.paste_field_names = []
        for field in self.infotext_fields:
            self.paste_field_names.append(field.api)

        return (
            sharpness_enabled,
            sharpness_multiplier,
            sharpness_method,
            affect_uncond,
            tone_enabled,
            tonemap_multiplier,
            tonemap_method,
            tonemap_percentile,
            contrast_multiplier,
            noise_enabled,
            extra_noise_type,
            extra_noise_method,
            extra_noise_multiplier,
            extra_noise_lowpass,
            combat_method,
            combat_cfg_drift,
            rescale_cfg_phi,
            advanced_enabled,
            divisive_norm_size,
            divisive_norm_multiplier,
            spectral_mod_mode,
            spectral_mod_percentile,
            spectral_mod_multiplier,
            dynamic_enabled,
            dyn_cfg_augmentation,
        )

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            sharpness_enabled,
            sharpness_multiplier,
            sharpness_method,
            affect_uncond,
            tone_enabled,
            tonemap_multiplier,
            tonemap_method,
            tonemap_percentile,
            contrast_multiplier,
            noise_enabled,
            extra_noise_type,
            extra_noise_method,
            extra_noise_multiplier,
            extra_noise_lowpass,
            combat_method,
            combat_cfg_drift,
            rescale_cfg_phi,
            advanced_enabled,
            divisive_norm_size,
            divisive_norm_multiplier,
            spectral_mod_mode,
            spectral_mod_percentile,
            spectral_mod_multiplier,
            dynamic_enabled,
            dyn_cfg_augmentation,
        ) = script_args

        # Get XYZ values if present
        xyz = getattr(p, "_latent_modifier_xyz", {})

        # Override values with XYZ if present
        if "sharpness_enabled" in xyz:
            sharpness_enabled = xyz["sharpness_enabled"] == "True"
        if "sharpness_multiplier" in xyz:
            sharpness_multiplier = float(xyz["sharpness_multiplier"])
        if "sharpness_method" in xyz:
            sharpness_method = xyz["sharpness_method"]
        if "affect_uncond" in xyz:
            affect_uncond = xyz["affect_uncond"]

        if "tone_enabled" in xyz:
            tone_enabled = xyz["tone_enabled"] == "True"
        if "tonemap_multiplier" in xyz:
            tonemap_multiplier = float(xyz["tonemap_multiplier"])
        if "tonemap_method" in xyz:
            tonemap_method = xyz["tonemap_method"]
        if "tonemap_percentile" in xyz:
            tonemap_percentile = float(xyz["tonemap_percentile"])
        if "contrast_multiplier" in xyz:
            contrast_multiplier = float(xyz["contrast_multiplier"])

        if "noise_enabled" in xyz:
            noise_enabled = xyz["noise_enabled"] == "True"
        if "extra_noise_type" in xyz:
            extra_noise_type = xyz["extra_noise_type"]
        if "extra_noise_method" in xyz:
            extra_noise_method = xyz["extra_noise_method"]
        if "extra_noise_multiplier" in xyz:
            extra_noise_multiplier = float(xyz["extra_noise_multiplier"])
        if "extra_noise_lowpass" in xyz:
            extra_noise_lowpass = int(xyz["extra_noise_lowpass"])
        if "combat_method" in xyz:
            combat_method = xyz["combat_method"]
        if "combat_cfg_drift" in xyz:
            combat_cfg_drift = float(xyz["combat_cfg_drift"])
        if "rescale_cfg_phi" in xyz:
            rescale_cfg_phi = float(xyz["rescale_cfg_phi"])

        if "advanced_enabled" in xyz:
            advanced_enabled = xyz["advanced_enabled"] == "True"
        if "divisive_norm_size" in xyz:
            divisive_norm_size = int(xyz["divisive_norm_size"])
        if "divisive_norm_multiplier" in xyz:
            divisive_norm_multiplier = float(xyz["divisive_norm_multiplier"])
        if "spectral_mod_mode" in xyz:
            spectral_mod_mode = xyz["spectral_mod_mode"]
        if "spectral_mod_percentile" in xyz:
            spectral_mod_percentile = float(xyz["spectral_mod_percentile"])
        if "spectral_mod_multiplier" in xyz:
            spectral_mod_multiplier = float(xyz["spectral_mod_multiplier"])

        if "dynamic_enabled" in xyz:
            dynamic_enabled = xyz["dynamic_enabled"] == "True"
        if "dyn_cfg_augmentation" in xyz:
            dyn_cfg_augmentation = xyz["dyn_cfg_augmentation"]

        # If nothing is enabled, return early
        if not any(
            [
                sharpness_enabled,
                tone_enabled,
                noise_enabled,
                advanced_enabled,
                dynamic_enabled,
            ]
        ):
            return

        unet = p.sd_model.forge_objects.unet

        unet = opModelSamplerLatentMegaModifier(
            unet,
            sharpness_multiplier if sharpness_enabled else 0.0,
            sharpness_method,
            tonemap_multiplier if tone_enabled else 0.0,
            tonemap_method,
            tonemap_percentile if tone_enabled else 100.0,
            contrast_multiplier if tone_enabled else 0.0,
            combat_method,
            combat_cfg_drift if noise_enabled else 0.0,
            rescale_cfg_phi if noise_enabled else 0.0,
            extra_noise_type,
            extra_noise_method,
            extra_noise_multiplier if noise_enabled else 0.0,
            extra_noise_lowpass if noise_enabled else 100,
            divisive_norm_size if advanced_enabled else 127,
            divisive_norm_multiplier if advanced_enabled else 0.0,
            spectral_mod_mode,
            spectral_mod_percentile if advanced_enabled else 5.0,
            spectral_mod_multiplier if advanced_enabled else 0.0,
            affect_uncond if sharpness_enabled else "None",
            dyn_cfg_augmentation if dynamic_enabled else "None",
            seed=p.seeds[0],
        )[0]

        p.sd_model.forge_objects.unet = unet

        # Add parameters to generation info based on what's enabled
        extra_params = {}

        if sharpness_enabled:
            extra_params.update(
                {
                    "latent_modifier_sharpness_enabled": True,
                    "latent_modifier_sharpness_multiplier": sharpness_multiplier,
                    "latent_modifier_sharpness_method": sharpness_method,
                    "latent_modifier_affect_uncond": affect_uncond,
                }
            )

        if tone_enabled:
            extra_params.update(
                {
                    "latent_modifier_tone_enabled": True,
                    "latent_modifier_tonemap_multiplier": tonemap_multiplier,
                    "latent_modifier_tonemap_method": tonemap_method,
                    "latent_modifier_tonemap_percentile": tonemap_percentile,
                    "latent_modifier_contrast_multiplier": contrast_multiplier,
                }
            )

        if noise_enabled:
            extra_params.update(
                {
                    "latent_modifier_noise_enabled": True,
                    "latent_modifier_combat_method": combat_method,
                    "latent_modifier_combat_cfg_drift": combat_cfg_drift,
                    "latent_modifier_rescale_cfg_phi": rescale_cfg_phi,
                    "latent_modifier_extra_noise_type": extra_noise_type,
                    "latent_modifier_extra_noise_method": extra_noise_method,
                    "latent_modifier_extra_noise_multiplier": extra_noise_multiplier,
                    "latent_modifier_extra_noise_lowpass": extra_noise_lowpass,
                }
            )

        if advanced_enabled:
            extra_params.update(
                {
                    "latent_modifier_advanced_enabled": True,
                    "latent_modifier_divisive_norm_size": divisive_norm_size,
                    "latent_modifier_divisive_norm_multiplier": divisive_norm_multiplier,
                    "latent_modifier_spectral_mod_mode": spectral_mod_mode,
                    "latent_modifier_spectral_mod_percentile": spectral_mod_percentile,
                    "latent_modifier_spectral_mod_multiplier": spectral_mod_multiplier,
                }
            )

        if dynamic_enabled:
            extra_params.update(
                {
                    "latent_modifier_dynamic_enabled": True,
                    "latent_modifier_dyn_cfg_augmentation": dyn_cfg_augmentation,
                }
            )

        p.extra_generation_params.update(extra_params)
        return


def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_latent_modifier_xyz"):
        p._latent_modifier_xyz = {}
    p._latent_modifier_xyz[field] = x


def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        # Sharpness & Detail
        xyz_grid.AxisOption(
            "(Latent) Sharpness Enabled",
            str,
            partial(set_value, field="sharpness_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Sharpness Multiplier",
            float,
            partial(set_value, field="sharpness_multiplier"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Sharpness Method",
            str,
            partial(set_value, field="sharpness_method"),
            choices=lambda: ["anisotropic", "joint-anisotropic", "gaussian", "cas"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Affect Uncond",
            str,
            partial(set_value, field="affect_uncond"),
            choices=lambda: ["None", "Sharpness"],
        ),
        # Tone & Contrast
        xyz_grid.AxisOption(
            "(Latent) Tone Enabled",
            str,
            partial(set_value, field="tone_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Tonemap Multiplier",
            float,
            partial(set_value, field="tonemap_multiplier"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Tonemap Method",
            str,
            partial(set_value, field="tonemap_method"),
            choices=lambda: [
                "reinhard",
                "reinhard_perchannel",
                "arctan",
                "quantile",
                "gated",
                "cfg-mimic",
                "spatial-norm",
            ],
        ),
        xyz_grid.AxisOption(
            "(Latent) Tonemap Percentile",
            float,
            partial(set_value, field="tonemap_percentile"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Contrast Multiplier",
            float,
            partial(set_value, field="contrast_multiplier"),
        ),
        # Noise & Combat
        xyz_grid.AxisOption(
            "(Latent) Noise Enabled",
            str,
            partial(set_value, field="noise_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Extra Noise Type",
            str,
            partial(set_value, field="extra_noise_type"),
            choices=lambda: [
                "gaussian",
                "uniform",
                "perlin",
                "pink",
                "green",
                "pyramid",
            ],
        ),
        xyz_grid.AxisOption(
            "(Latent) Extra Noise Method",
            str,
            partial(set_value, field="extra_noise_method"),
            choices=lambda: [
                "add",
                "add_scaled",
                "speckle",
                "cads",
                "cads_rescaled",
                "cads_speckle",
                "cads_speckle_rescaled",
            ],
        ),
        xyz_grid.AxisOption(
            "(Latent) Extra Noise Multiplier",
            float,
            partial(set_value, field="extra_noise_multiplier"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Extra Noise Lowpass",
            int,
            partial(set_value, field="extra_noise_lowpass"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Combat Method",
            str,
            partial(set_value, field="combat_method"),
            choices=lambda: [
                "subtract",
                "subtract_channels",
                "subtract_median",
                "sharpen",
            ],
        ),
        xyz_grid.AxisOption(
            "(Latent) Combat CFG Drift",
            float,
            partial(set_value, field="combat_cfg_drift"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Rescale CFG Phi",
            float,
            partial(set_value, field="rescale_cfg_phi"),
        ),
        # Advanced Processing
        xyz_grid.AxisOption(
            "(Latent) Advanced Enabled",
            str,
            partial(set_value, field="advanced_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Divisive Norm Size",
            int,
            partial(set_value, field="divisive_norm_size"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Divisive Norm Multiplier",
            float,
            partial(set_value, field="divisive_norm_multiplier"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Spectral Mod Mode",
            str,
            partial(set_value, field="spectral_mod_mode"),
            choices=lambda: ["hard_clamp", "soft_clamp"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Spectral Mod Percentile",
            float,
            partial(set_value, field="spectral_mod_percentile"),
        ),
        xyz_grid.AxisOption(
            "(Latent) Spectral Mod Multiplier",
            float,
            partial(set_value, field="spectral_mod_multiplier"),
        ),
        # Dynamic CFG
        xyz_grid.AxisOption(
            "(Latent) Dynamic Enabled",
            str,
            partial(set_value, field="dynamic_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Latent) Dyn CFG Augmentation",
            str,
            partial(set_value, field="dyn_cfg_augmentation"),
            choices=lambda: ["None", "dyncfg-halfcosine", "dyncfg-halfcosine-mimic"],
        ),
    ]

    if not any(x.label.startswith("(Latent)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] Latent Modifier Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


script_callbacks.on_before_ui(on_before_ui)
