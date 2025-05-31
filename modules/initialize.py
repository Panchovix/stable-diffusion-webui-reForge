import importlib
import logging
import sys
import warnings
import os
from modules.paths_internal import models_path

from modules.timer import startup_timer


def imports():
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    import torch  # noqa: F401
    torch.set_grad_enabled(False)

    startup_timer.record("import torch")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")

    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    import gradio  # noqa: F401
    startup_timer.record("import gradio")

    from modules import paths, timer, errors  # noqa: F401
    startup_timer.record("setup paths")

    from modules import shared_init
    shared_init.initialize()
    startup_timer.record("initialize shared")

    from modules import processing, gradio_extensions, ui  # noqa: F401
    startup_timer.record("other imports")


def check_versions():
    from modules.shared_cmd_options import cmd_opts

    if not cmd_opts.skip_version_check:
        from modules import errors
        errors.check_versions()


def initialize():
    from modules import initialize_util
    initialize_util.fix_torch_version()
    initialize_util.fix_asyncio_event_loop_policy()
    initialize_util.validate_tls_options()
    initialize_util.configure_sigint_handler()
    initialize_util.configure_opts_onchange()

    from modules import sd_models
    sd_models.setup_model()
    startup_timer.record("setup SD model")

    from modules.shared_cmd_options import cmd_opts

    from modules import codeformer_model
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    from modules import gfpgan_model
    gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    from modules import face_bopb2l_model
    face_bopb2l_model.setup_model()
    startup_timer.record("setup ms-bopb2l")

    from modules import face_restoreformer_model
    face_restoreformer_model.setup_model(os.path.join(models_path, 'RestoreFormer'))
    startup_timer.record("setup restoreformer")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules.shared_cmd_options import cmd_opts

    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
    extensions.list_extensions()
    startup_timer.record("list extensions")

    from modules import initialize_util
    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import shared, upscaler, scripts
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    from modules import localization
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    if reload_script_modules and shared.opts.enable_reloading_ui_scripts:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from modules import modelloader
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    from modules import shared_items
    shared_items.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    from modules import extra_networks
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")

    if not cmd_opts.skip_google_blockly:
        from modules_forge import google_blockly
        google_blockly.initialization()
        startup_timer.record("initialize google blockly")

    return
