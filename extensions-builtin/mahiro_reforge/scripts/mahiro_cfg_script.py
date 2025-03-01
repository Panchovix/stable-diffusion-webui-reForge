import logging
import sys
import gradio as gr
from modules import scripts, script_callbacks
from mahiro.nodes_mahiro import Mahiro
from functools import partial
from typing import Any


class MahiroCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False

    sorting_priority = 15.1

    def title(self):
        return "Mahiro CFG for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Toggle Mahiro CFG guidance function.</i></p>")
            enabled = gr.Checkbox(label="Enable Mahiro CFG", value=self.enabled)

        enabled.change(lambda x: self.update_enabled(x), inputs=[enabled])
        return [enabled]

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 1:
            self.enabled = args[0]
        else:
            logging.warning(
                "Not enough arguments provided to process_before_every_sampling"
            )
            return

        xyz = getattr(p, "_mahiro_xyz", {})
        if "enabled" in xyz:
            self.enabled = xyz["enabled"] == "True"

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        unet = Mahiro().patch(unet)[0]
        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update(
            {
                "mahiro_cfg_enabled": True,
            }
        )

        logging.debug(f"Mahiro CFG: Enabled: {self.enabled}")
        return


def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_mahiro_xyz"):
        p._mahiro_xyz = {}
    p._mahiro_xyz[field] = x


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
            "(Mahiro CFG) Enabled",
            str,
            partial(set_value, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
    ]

    if not any(x.label.startswith("(Mahiro CFG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] Mahiro CFG Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


script_callbacks.on_before_ui(on_before_ui)
