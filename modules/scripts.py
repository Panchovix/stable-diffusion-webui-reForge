import os
import re
import sys
import inspect
from collections import namedtuple
from dataclasses import dataclass

import gradio as gr

from modules import (
    shared,
    paths,
    script_callbacks,
    extensions,
    script_loading,
    errors,
    timer,
    util,
)

from modules.scripts_postprocessing import ScriptPostprocessing, ScriptPostprocessingRunner

topological_sort = util.topological_sort

AlwaysVisible = object()


class MaskBlendArgs:
    def __init__(
        self,
        current_latent,
        nmask,
        init_latent,
        mask,
        blended_latent,
        denoiser=None,
        sigma=None,
    ):
        self.current_latent = current_latent
        self.nmask = nmask
        self.init_latent = init_latent
        self.mask = mask
        self.blended_latent = blended_latent

        self.denoiser = denoiser
        self.is_final_blend = denoiser is None
        self.sigma = sigma


class PostSampleArgs:
    def __init__(self, samples):
        self.samples = samples


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class PostProcessMaskOverlayArgs:
    def __init__(self, index, mask_for_overlay, overlay_image):
        self.index = index
        self.mask_for_overlay = mask_for_overlay
        self.overlay_image = overlay_image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


@dataclass
class OnComponent:
    component: gr.Block


class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False
    tabname = None

    group = None
    """A gr.Group component that has all script's UI inside it."""

    create_group = True
    """If False, for alwayson scripts, a group component will not be created."""

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only = False
    """If true, the script setup will only be run in Gradio UI, not in API"""

    controls = None
    """A list of controls returned by the ui()."""

    sorting_priority = 0
    """Larger number will appear downwards in the UI."""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """

        pass

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and False otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
        """

        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """

        pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass

    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_before_every_step(self, p, *args, **kwargs):
        """
        Called before every step within the sampler.
        **kwargs will have the following items:
         - d - the current generation data
        """
        pass

    def process_before_every_sampling(self, p, *args, **kwargs):
        """
        Similar to process(), called before every sampling.
        If you use high-res fix, this will be called two times.
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass

    def on_mask_blend(self, p, mba: MaskBlendArgs, *args):
        """
        Called in inpainting mode when the original content is blended with the inpainted content.
        This is called at every step in the denoising process and once at the end.
        If is_final_blend is true, this is called for the final blending stage.
        Otherwise, denoiser and sigma are defined and may be used to inform the procedure.
        """

        pass

    def post_sample(self, p, ps: PostSampleArgs, *args):
        """
        Called after the samples have been generated,
        but before they have been decoded by the VAE, if applicable.
        Check getattr(samples, 'already_decoded', False) to test if the images are decoded.
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess_maskoverlay(self, p, ppmo: PostProcessMaskOverlayArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess_image_after_composite(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        Same as postprocess_image but after inpaint_full_res composite
        So that it operates on the full image instead of the inpaint_full_res crop region.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def on_before_component(self, callback, *, elem_id):
        """
        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.

        May be called in show() or ui() - but it may be too late in latter as some components may already be created.

        This function is an alternative to before_component in that it also cllows to run before a component is created, but
        it doesn't require to be called for every created component - just for the one you need.
        """
        if self.on_before_component_elem_id is None:
            self.on_before_component_elem_id = []

        self.on_before_component_elem_id.append((elem_id, callback))

    def on_after_component(self, callback, *, elem_id):
        """
        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.
        """
        if self.on_after_component_elem_id is None:
            self.on_after_component_elem_id = []

        self.on_after_component_elem_id.append((elem_id, callback))

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabkind = "img2img" if self.is_img2img else "txt2img"
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r"[^a-z_0-9]", "", re.sub(r"\s", "_", self.title().lower()))

        return f"script_{tabname}{title}_{item_id}"

    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        pass


class ScriptBuiltinUI(Script):
    setup_for_ui_only = True

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabname = (
            ("img2img" if self.is_img2img else "txt2img") + "_" if need_tabname else ""
        )

        return f"{tabname}{item_id}"

    def show(self, is_img2img):
        return AlwaysVisible


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])
ScriptClassData = namedtuple(
    "ScriptClassData", ["script_class", "path", "basedir", "module"]
)

scripts_data:list[ScriptClassData] = []
postprocessing_scripts_data = []

@dataclass
class ScriptWithDependencies:
    script_canonical_name: str
    file: ScriptFile
    requires: list
    load_before: list
    load_after: list


def list_scripts(scriptdirname, extension, *, include_extensions=True):
    scripts = {}

    loaded_extensions = {ext.canonical_name: ext for ext in extensions.active()}
    loaded_extensions_scripts = {ext.canonical_name: [] for ext in extensions.active()}

    # build script dependency map
    root_script_basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(root_script_basedir):
        for filename in sorted(os.listdir(root_script_basedir)):
            if not os.path.isfile(os.path.join(root_script_basedir, filename)):
                continue

            if os.path.splitext(filename)[1].lower() != extension:
                continue

            script_file = ScriptFile(
                paths.script_path, filename, os.path.join(root_script_basedir, filename)
            )
            scripts[filename] = ScriptWithDependencies(
                filename, script_file, [], [], []
            )

    if include_extensions:
        for ext in extensions.active():
            extension_scripts_list = ext.list_files(scriptdirname, extension)
            for extension_script in extension_scripts_list:
                if not os.path.isfile(extension_script.path):
                    continue

                script_canonical_name = (
                    ("builtin/" if ext.is_builtin else "")
                    + ext.canonical_name
                    + "/"
                    + extension_script.filename
                )
                relative_path = scriptdirname + "/" + extension_script.filename

                script = ScriptWithDependencies(
                    script_canonical_name=script_canonical_name,
                    file=extension_script,
                    requires=ext.metadata.get_script_requirements(
                        "Requires", relative_path, scriptdirname
                    ),
                    load_before=ext.metadata.get_script_requirements(
                        "Before", relative_path, scriptdirname
                    ),
                    load_after=ext.metadata.get_script_requirements(
                        "After", relative_path, scriptdirname
                    ),
                )

                scripts[script_canonical_name] = script
                loaded_extensions_scripts[ext.canonical_name].append(script)

    for script_canonical_name, script in scripts.items():
        # load before requires inverse dependency
        # in this case, append the script name into the load_after list of the specified script
        for load_before in script.load_before:
            # if this requires an individual script to be loaded before
            other_script = scripts.get(load_before)
            if other_script:
                other_script.load_after.append(script_canonical_name)

            # if this requires an extension
            other_extension_scripts = loaded_extensions_scripts.get(load_before)
            if other_extension_scripts:
                for other_script in other_extension_scripts:
                    other_script.load_after.append(script_canonical_name)

        # if After mentions an extension, remove it and instead add all of its scripts
        for load_after in list(script.load_after):
            if load_after not in scripts and load_after in loaded_extensions_scripts:
                script.load_after.remove(load_after)

                for other_script in loaded_extensions_scripts.get(load_after, []):
                    script.load_after.append(other_script.script_canonical_name)

    dependencies = {}

    for script_canonical_name, script in scripts.items():
        for required_script in script.requires:
            if (
                required_script not in scripts
                and required_script not in loaded_extensions
            ):
                errors.report(
                    f'Script "{script_canonical_name}" requires "{required_script}" to be loaded, but it is not.',
                    exc_info=False,
                )

        dependencies[script_canonical_name] = script.load_after

    ordered_scripts = topological_sort(dependencies)
    scripts_list = [
        scripts[script_canonical_name].file for script_canonical_name in ordered_scripts
    ]

    return scripts_list


def list_files_with_name(filename):
    res = []

    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]

    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue

        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)

    return res


def load_scripts():
    global current_basedir
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py") + list_scripts(
        "modules/processing_scripts", ".py", include_extensions=False
    )

    for s in scripts_list:
        if s.basedir not in sys.path:
            sys.path = [s.basedir] + sys.path

    syspath = sys.path

    # print(f'Current System Paths = {syspath}')

    def register_scripts_from_module(module):
        for script_class in module.__dict__.values():
            if not inspect.isclass(script_class):
                continue

            if issubclass(script_class, Script):
                scripts_data.append(
                    ScriptClassData(
                        script_class, scriptfile.path, scriptfile.basedir, module
                    )
                )
            elif issubclass(script_class, ScriptPostprocessing):
                postprocessing_scripts_data.append(
                    ScriptClassData(
                        script_class, scriptfile.path, scriptfile.basedir, module
                    )
                )

    # here the scripts_list is already ordered
    # processing_script is not considered though
    for scriptfile in scripts_list:
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module)

        except Exception:
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

        finally:
            sys.path = syspath
            current_basedir = paths.script_path
            timer.startup_timer.record(scriptfile.filename)

    global scripts_txt2img, scripts_img2img, scripts_postproc

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = ScriptPostprocessingRunner()


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: ScriptPostprocessingRunner = None
scripts_current: ScriptRunner = None


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


reload_scripts = load_scripts  # compatibility alias
