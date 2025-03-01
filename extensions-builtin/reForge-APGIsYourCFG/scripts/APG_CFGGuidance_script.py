import logging
import sys
import traceback
from typing import Any
from functools import partial
import gradio as gr
from modules import script_callbacks, scripts
from APGIsYourCFG.nodes_APGImYourCFGNow import APG_ImYourCFGNow


class APGIsNowYourCFGScript(scripts.Script):
    def __init__(self):
        # APG parameters
        self.apg_enabled = False
        self.apg_moment = 0.5
        self.apg_adaptive_moment = 0.180
        self.apg_norm_thr = 15.0
        self.apg_eta = 1.0
        # Guidance limiter parameters
        self.guidance_limiter_enabled = False
        self.guidance_sigma_start = 5.42
        self.guidance_sigma_end = 0.28

    sorting_priority = 15

    def title(self):
        return '"APG\'s now your CFG" and Guidance Limiter for reForge'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            # APG Section
            gr.HTML("<p><b>APG Settings</b></p>")
            apg_enabled = gr.Checkbox(label="Enable APG", value=self.apg_enabled)
            with gr.Group(visible=True):
                apg_momentum = gr.Slider(
                    label="APG Momentum",
                    minimum=-2.5,
                    maximum=2.5,
                    step=0.01,
                    value=self.apg_moment,
                )
                apg_adaptive_momentum = gr.Slider(
                    label="APG Adaptive Momentum",
                    minimum=-2.5,
                    maximum=2.5,
                    step=0.01,
                    value=self.apg_adaptive_moment,
                )
                apg_norm_thr = gr.Slider(
                    label="APG Norm Threshold",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.5,
                    value=self.apg_norm_thr,
                )
                apg_eta = gr.Slider(
                    label="APG Eta",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=self.apg_eta,
                )

            # Guidance Limiter Section
            gr.HTML("<br><p><b>Guidance Limiter Settings</b></p>")
            guidance_limiter_enabled = gr.Checkbox(
                label="Enable Guidance Limiter", value=self.guidance_limiter_enabled
            )
            with gr.Group(visible=True):
                guidance_sigma_start = gr.Slider(
                    label="Guidance Sigma Start",
                    minimum=-1.0,
                    maximum=10000.0,
                    step=0.01,
                    value=self.guidance_sigma_start,
                )
                guidance_sigma_end = gr.Slider(
                    label="Guidance Sigma End",
                    minimum=-1.0,
                    maximum=10000.0,
                    step=0.01,
                    value=self.guidance_sigma_end,
                )

        return (
            apg_enabled,
            apg_momentum,
            apg_adaptive_momentum,
            apg_norm_thr,
            apg_eta,
            guidance_limiter_enabled,
            guidance_sigma_start,
            guidance_sigma_end,
        )

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 8:
            (
                self.apg_enabled,
                self.apg_moment,
                self.apg_adaptive_moment,
                self.apg_norm_thr,
                self.apg_eta,
                self.guidance_limiter_enabled,
                self.guidance_sigma_start,
                self.guidance_sigma_end,
            ) = args[:8]
        else:
            logging.warning(
                "Not enough arguments provided to process_before_every_sampling"
            )
            return

        # Retrieve values from XYZ plot if available
        xyz = getattr(p, "_apg_xyz", {})
        if "apg_enabled" in xyz:
            self.apg_enabled = xyz["apg_enabled"] == "True"
        if "apg_moment" in xyz:
            self.apg_moment = xyz["apg_moment"]
        if "apg_adaptive_moment" in xyz:
            self.apg_adaptive_moment = xyz["apg_adaptive_moment"]
        if "apg_norm_thr" in xyz:
            self.apg_norm_thr = xyz["apg_norm_thr"]
        if "apg_eta" in xyz:
            self.apg_eta = xyz["apg_eta"]
        if "guidance_limiter_enabled" in xyz:
            self.guidance_limiter_enabled = xyz["guidance_limiter_enabled"] == "True"
        if "guidance_sigma_start" in xyz:
            self.guidance_sigma_start = xyz["guidance_sigma_start"]
        if "guidance_sigma_end" in xyz:
            self.guidance_sigma_end = xyz["guidance_sigma_end"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        # If neither feature is enabled, return original unet
        if not self.apg_enabled and not self.guidance_limiter_enabled:
            p.sd_model.forge_objects.unet = unet
            return

        # Configure parameters based on what's enabled
        patch_params = {
            # APG parameters (only applied if APG is enabled)
            "momentum": self.apg_moment if self.apg_enabled else 0,
            "adaptive_momentum": self.apg_adaptive_moment if self.apg_enabled else 0,
            "norm_threshold": self.apg_norm_thr if self.apg_enabled else 0,
            "eta": self.apg_eta if self.apg_enabled else 1.0,
            # Guidance limiter parameters (only applied if limiter is enabled)
            "guidance_limiter": self.guidance_limiter_enabled,
            "guidance_sigma_start": self.guidance_sigma_start
            if self.guidance_limiter_enabled
            else -1,
            "guidance_sigma_end": self.guidance_sigma_end
            if self.guidance_limiter_enabled
            else -1,
        }

        unet = APG_ImYourCFGNow().patch(unet, **patch_params)[0]

        p.sd_model.forge_objects.unet = unet

        # Only include enabled features in generation params
        args = {}
        if self.apg_enabled:
            args.update(
                {
                    "apgisyourcfg_enabled": True,
                    "apgisyourcfg_momentum": self.apg_moment,
                    "apgisyourcfg_adaptive_momentum": self.apg_adaptive_moment,
                    "apgisyourcfg_norm_thr": self.apg_norm_thr,
                    "apgisyourcfg_eta": self.apg_eta,
                }
            )
        if self.guidance_limiter_enabled:
            args.update(
                {
                    "guidance_limiter_enabled": True,
                    "guidance_sigma_start": self.guidance_sigma_start,
                    "guidance_sigma_end": self.guidance_sigma_end,
                }
            )

        p.extra_generation_params.update(args)
        str_args: str = ", ".join([f'{k}:"{v}"' for k, v in args.items()])
        logging.debug('WOLOLO: "APG is now your CFG!"')
        logging.debug(str_args)

        return


# XYZ plot functionality
def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_apg_xyz"):
        p._apg_xyz = {}
    p._apg_xyz[field] = x


def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(APG) APG Enabled",
            str,
            partial(set_value, field="apg_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(APG) APG Momentum",
            float,
            partial(set_value, field="apg_moment"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Adaptive Momentum",
            float,
            partial(set_value, field="apg_adaptive_moment"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Norm Threshold",
            float,
            partial(set_value, field="apg_norm_thr"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Eta",
            float,
            partial(set_value, field="apg_eta"),
        ),
        xyz_grid.AxisOption(
            "(APG) Guidance Limiter Enabled",
            str,
            partial(set_value, field="guidance_limiter_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(APG) Guidance Sigma Start",
            float,
            partial(set_value, field="guidance_sigma_start"),
        ),
        xyz_grid.AxisOption(
            "(APG) Guidance Sigma End",
            float,
            partial(set_value, field="guidance_sigma_end"),
        ),
    ]

    if not any(x.label.startswith("(APG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] APG Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


script_callbacks.on_before_ui(on_before_ui)
