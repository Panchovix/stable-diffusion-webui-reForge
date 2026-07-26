import os

import launch
from modules import cmd_args, script_loading
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401

parser = cmd_args.parser

script_loading.preload_extensions(extensions_dir, parser, extension_list=launch.list_extensions(launch.args.ui_settings_file))
script_loading.preload_extensions(extensions_builtin_dir, parser)

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()

cmd_opts.webui_is_non_local = any([cmd_opts.share, cmd_opts.listen, cmd_opts.ngrok, cmd_opts.server_name])
cmd_opts.disable_extension_access = cmd_opts.webui_is_non_local and not cmd_opts.enable_insecure_extension_access

# Block all huggingface_hub / diffusers / transformers network fetches unless
# the user explicitly passes --allow-download.  Both env-vars are checked at
# call time (not import time) by the respective libraries, so setting them here
# is sufficient even though the libraries may already be imported.
if not getattr(cmd_opts, 'allow_download', False):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
