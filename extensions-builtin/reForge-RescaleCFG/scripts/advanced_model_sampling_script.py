import logging
import sys
import gradio as gr
from modules import scripts, script_callbacks
from RescaleCFG.nodes_RescaleCFG import RescaleCFG
from RescaleCFG.nodes_AltRescaleCFG import AltRescaleCFG
from functools import partial
from typing import Any

class RescaleCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.multiplier = 0.7
        self.version = "Normal"  # Add version parameter

    sorting_priority = 15

    def title(self):
        return "RescaleCFG for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for RescaleCFG.</i></p>")
            enabled = gr.Checkbox(label="Enable RescaleCFG", value=self.enabled)
            version = gr.Dropdown(
                label="RescaleCFG Version",
                choices=["Normal", "Alternative"],
                value=self.version
            )
            multiplier = gr.Slider(label="RescaleCFG Multiplier", minimum=0.0, maximum=1.0, step=0.01, value=self.multiplier)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled]
        )

        return (enabled, version, multiplier)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 3:
            self.enabled, self.version, self.multiplier = args[:3]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        xyz = getattr(p, "_rescale_xyz", {})
        if "enabled" in xyz:
            self.enabled = xyz["enabled"] == "True"
        if "version" in xyz:
            self.version = xyz["version"]
        if "multiplier" in xyz:
            self.multiplier = xyz["multiplier"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        # Choose between normal and alternative version
        if self.version == "Alternative":
            unet = AltRescaleCFG().patch(unet, self.multiplier)[0]
        else:
            unet = RescaleCFG().patch(unet, self.multiplier)[0]

        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update({
            "rescale_cfg_enabled": True,
            "rescale_cfg_version": self.version,
            "rescale_cfg_multiplier": self.multiplier,
        })

        logging.debug(f"RescaleCFG: Enabled: {self.enabled}, Version: {self.version}, Multiplier: {self.multiplier}")

        return

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_rescale_xyz"):
        p._rescale_xyz = {}
    p._rescale_xyz[field] = x

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
            "(RescaleCFG) Enabled",
            str,
            partial(set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(RescaleCFG) Version",
            str,
            partial(set_value, field="version"),
            choices=lambda: ["Normal", "Alternative"]
        ),
        xyz_grid.AxisOption(
            "(RescaleCFG) Multiplier",
            float,
            partial(set_value, field="multiplier"),
        ),
    ]

    if not any(x.label.startswith("(RescaleCFG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] RescaleCFG Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)
