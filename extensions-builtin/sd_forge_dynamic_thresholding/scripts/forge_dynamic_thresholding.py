import gradio as gr

from modules import scripts
from lib_dynamic_thresholding.dynthres import (
    DynamicThresholdingComfyNode,
    DynamicThresholdingSimpleComfyNode,
)

opDynamicThresholdingNode = DynamicThresholdingComfyNode().patch
opDynamicThresholdingSimpleNode = DynamicThresholdingSimpleComfyNode().patch

# Mode descriptions for tooltips
MODE_DESCRIPTIONS = {
    "Constant": "No change over time",
    "Linear Down": "Linearly decreases from start to end",
    "Cosine Down": "Decreases following a cosine curve",
    "Half Cosine Down": "Similar to Cosine Down but with different curve shape",
    "Linear Up": "Linearly increases from start to end",
    "Cosine Up": "Increases following a cosine curve",
    "Half Cosine Up": "Similar to Cosine Up but with different curve shape",
    "Power Up": "Increases following a power curve (affected by Sched Val)",
    "Power Down": "Decreases following a power curve (affected by Sched Val)",
    "Linear Repeating": "Oscillates linearly (frequency controlled by Sched Val)",
    "Cosine Repeating": "Oscillates in a cosine pattern (frequency controlled by Sched Val)",
    "Sawtooth": "Creates a repeating sawtooth pattern (frequency controlled by Sched Val)",
}


class DynamicThresholdingForForge(scripts.Script):
    sorting_priority = 11

    def title(self):
        return "DynamicThresholding (CFG-Fix) Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enabled",
                    value=False,
                    info="Enable Dynamic Thresholding for the current generation",
                )
                simple_mode = gr.Checkbox(
                    label="Simple Mode",
                    value=False,
                    info="Use simplified settings with only essential controls. Recommended for beginners",
                )

            # Simple mode controls
            with gr.Group(visible=True) as simple_controls:
                mimic_scale = gr.Slider(
                    label="Mimic Scale",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.5,
                    value=7.0,
                    info="Base CFG scale that the algorithm tries to mimic. Higher values = stronger prompt adherence",
                )
                threshold_percentile = gr.Slider(
                    label="Threshold Percentile",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    info="Percentage of strongest signals to consider. Lower values = more aggressive thresholding",
                )

            # Advanced mode controls
            with gr.Group(visible=True) as advanced_controls:
                mimic_mode = gr.Radio(
                    label="Mimic Mode",
                    choices=list(MODE_DESCRIPTIONS.keys()),
                    value="Constant",
                    info="Controls how the mimic scale changes during generation",
                )
                mimic_scale_min = gr.Slider(
                    label="Mimic Scale Min",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.5,
                    value=0.0,
                    info="Minimum value for mimic scale when using non-constant modes",
                )
                cfg_mode = gr.Radio(
                    label="CFG Mode",
                    choices=list(MODE_DESCRIPTIONS.keys()),
                    value="Constant",
                    info="Controls how the CFG scale changes during generation",
                )
                cfg_scale_min = gr.Slider(
                    label="CFG Scale Min",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.5,
                    value=0.0,
                    info="Minimum value for CFG scale when using non-constant modes",
                )
                sched_val = gr.Slider(
                    label="Schedule Value",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.01,
                    value=1.0,
                    info="Controls power curve steepness or repetition frequency in applicable modes",
                )
                separate_feature_channels = gr.Radio(
                    label="Separate Feature Channels",
                    choices=["enable", "disable"],
                    value="enable",
                    info="Apply thresholding separately to each feature channel. Can preserve fine details but increases processing time",
                )
                scaling_startpoint = gr.Radio(
                    label="Scaling Startpoint",
                    choices=["MEAN", "ZERO"],
                    value="MEAN",
                    info="MEAN: Use mean value as center point (recommended)\nZERO: Use zero as center point",
                )
                variability_measure = gr.Radio(
                    label="Variability Measure",
                    choices=["AD", "STD"],
                    value="AD",
                    info="AD: Absolute Deviation (more robust to outliers)\nSTD: Standard Deviation (more sensitive to extremes)",
                )
                interpolate_phi = gr.Slider(
                    label="Interpolate Phi",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    info="Blend between dynamic thresholding (1.0) and standard CFG (0.0)",
                )

                # Add detailed mode descriptions that update when radio buttons change
                mode_description = gr.Markdown(value="", visible=True)

                def update_mode_description(mimic_mode_val, cfg_mode_val):
                    description = f"### Mode Descriptions\n\n**Mimic Mode**: {MODE_DESCRIPTIONS[mimic_mode_val]}\n\n"
                    description += f"**CFG Mode**: {MODE_DESCRIPTIONS[cfg_mode_val]}"
                    return description

                mimic_mode.change(
                    fn=update_mode_description,
                    inputs=[mimic_mode, cfg_mode],
                    outputs=[mode_description],
                )
                cfg_mode.change(
                    fn=update_mode_description,
                    inputs=[mimic_mode, cfg_mode],
                    outputs=[mode_description],
                )

            def update_visibility(simple):
                return gr.Group.update(visible=True), gr.Group.update(
                    visible=not simple
                )

            simple_mode.change(
                fn=update_visibility,
                inputs=[simple_mode],
                outputs=[simple_controls, advanced_controls],
            )

        return [
            enabled,
            simple_mode,
            mimic_scale,
            threshold_percentile,
            mimic_mode,
            mimic_scale_min,
            cfg_mode,
            cfg_scale_min,
            sched_val,
            separate_feature_channels,
            scaling_startpoint,
            variability_measure,
            interpolate_phi,
        ]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        (
            enabled,
            simple_mode,
            mimic_scale,
            threshold_percentile,
            mimic_mode,
            mimic_scale_min,
            cfg_mode,
            cfg_scale_min,
            sched_val,
            separate_feature_channels,
            scaling_startpoint,
            variability_measure,
            interpolate_phi,
        ) = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        if simple_mode:
            unet = opDynamicThresholdingSimpleNode(
                unet, mimic_scale, threshold_percentile
            )[0]
        else:
            unet = opDynamicThresholdingNode(
                unet,
                mimic_scale,
                threshold_percentile,
                mimic_mode,
                mimic_scale_min,
                cfg_mode,
                cfg_scale_min,
                sched_val,
                separate_feature_channels,
                scaling_startpoint,
                variability_measure,
                interpolate_phi,
            )[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(
            dict(
                dynthres_enabled=enabled,
                dynthres_simple_mode=simple_mode,
                dynthres_mimic_scale=mimic_scale,
                dynthres_threshold_percentile=threshold_percentile,
            )
        )

        if not simple_mode:
            p.extra_generation_params.update(
                dict(
                    dynthres_mimic_mode=mimic_mode,
                    dynthres_mimic_scale_min=mimic_scale_min,
                    dynthres_cfg_mode=cfg_mode,
                    dynthres_cfg_scale_min=cfg_scale_min,
                    dynthres_sched_val=sched_val,
                    dynthres_separate_feature_channels=separate_feature_channels,
                    dynthres_scaling_startpoint=scaling_startpoint,
                    dynthres_variability_measure=variability_measure,
                    dynthres_interpolate_phi=interpolate_phi,
                )
            )

        return
