"""this module defines internal paths used by program and is safe to import before dependencies are installed in launch.py"""

import argparse
import os
import sys
import shlex
from pathlib import Path


def normalized_filepath(filepath: str | None):
    return Path(filepath).absolute()


commandline_args = os.environ.get("COMMANDLINE_ARGS", "")
sys.argv += shlex.split(commandline_args)

cwd = Path.cwd()
modules_path = Path(__file__).resolve().parent
script_path = Path(modules_path)

sd_configs_path = script_path / "configs"
sd_default_config = sd_configs_path / "v1-inference.yaml"
sd_model_file = script_path / "model.ckpt"
default_sd_model_file = sd_model_file

# Parse the --data-dir flag first so we can use it as a base for our other argument default values
parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument(
    "--data-dir",
    type=Path,
    default=modules_path.parent,
    help="base path where all user data is stored",
)
parser_pre.add_argument(
    "--models-dir",
    type=Path,
    default=None,
    help="base path where models are stored; overrides --data-dir",
)
cmd_opts_pre = parser_pre.parse_known_args()[0]

data_path = Path(cmd_opts_pre.data_dir)

models_path = (
    Path(cmd_opts_pre.models_dir)
    if Path(cmd_opts_pre.models_dir)
    else Path(data_path) / "models"
)
extensions_dir = Path(data_path) / "extensions"
extensions_builtin_dir = script_path / "extensions-builtin"
config_states_dir = script_path / "config_states"
default_output_dir = Path(data_path) / "outputs"

roboto_ttf_file = modules_path / "Roboto-Regular.ttf"
