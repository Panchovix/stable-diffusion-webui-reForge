import os

import gradio as gr

from modules import (
    shared,
    script_callbacks,
    script_loading,
    errors,
)

from modules.scripts_postprocessing import ScriptPostprocessing

def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default

class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}
        self.infotext_fields = []
        self.paste_field_names = []
        self.inputs = [None]

        self.callback_map = {}
        self.callback_names = [
            "before_process",
            "process",
            "before_process_batch",
            "after_extra_networks_activate",
            "process_batch",
            "postprocess",
            "postprocess_batch",
            "postprocess_batch_list",
            "post_sample",
            "on_mask_blend",
            "postprocess_image",
            "postprocess_maskoverlay",
            "postprocess_image_after_composite",
            "before_component",
            "after_component",
        ]

        self.on_before_component_elem_id = {}
        """dict of callbacks to be called before an element is created; key=elem_id, value=list of callbacks"""

        self.on_after_component_elem_id = {}
        """dict of callbacks to be called after an element is created; key=elem_id, value=list of callbacks"""

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        auto_processing_scripts = (
            scripts_auto_postprocessing.create_auto_preprocessing_script_data()
        )

        for script_data in auto_processing_scripts + scripts_data:
            try:
                script:Script|ScriptPostprocessing = script_data.script_class()
            except Exception:
                errors.report(
                    f"Error # failed to initialize Script {script_data.module}: ",
                    exc_info=True,
                )
                continue

            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"

            visibility = script.show(script.is_img2img)

            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True

            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)

        self.callback_map.clear()

        self.apply_on_before_component_callbacks()

    def apply_on_before_component_callbacks(self):
        for script in self.scripts:
            on_before = script.on_before_component_elem_id or []
            on_after = script.on_after_component_elem_id or []

            for elem_id, callback in on_before:
                if elem_id not in self.on_before_component_elem_id:
                    self.on_before_component_elem_id[elem_id] = []

                self.on_before_component_elem_id[elem_id].append((callback, script))

            for elem_id, callback in on_after:
                if elem_id not in self.on_after_component_elem_id:
                    self.on_after_component_elem_id[elem_id] = []

                self.on_after_component_elem_id[elem_id].append((callback, script))

            on_before.clear()
            on_after.clear()

    def create_script_ui(self, script):
        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)

        try:
            self.create_script_ui_inner(script)
        except Exception:
            errors.report(f"Error creating UI for {script.name}: ", exc_info=True)

    def create_script_ui_inner(self, script):
        import modules.api.models as api_models

        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)
        script.controls = controls

        if controls is None:
            return

        script.name = wrap_call(
            script.title, script.filename, "title", default=script.filename
        ).lower()

        api_args = []

        for control in controls:
            control.custom_script_source = os.path.basename(script.filename)

            arg_info = api_models.ScriptArg(label=control.label or "")

            for field in ("value", "minimum", "maximum", "step"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)

            choices = getattr(
                control, "choices", None
            )  # as of gradio 3.41, some items in choices are strings, and some are tuples where the first elem is the string
            if choices is not None:
                arg_info.choices = [
                    x[0] if isinstance(x, tuple) else x for x in choices
                ]

            api_args.append(arg_info)

        script.api_info = api_models.ScriptInfo(
            name=script.name,
            is_img2img=script.is_img2img,
            is_alwayson=script.alwayson,
            args=api_args,
        )

        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields

        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names

        self.inputs += controls
        script.args_to = len(self.inputs)

    def setup_ui_for_section(self, section, scriptlist=None):
        if scriptlist is None:
            scriptlist = self.alwayson_scripts

        scriptlist = sorted(scriptlist, key=lambda x: x.sorting_priority)

        for script in scriptlist:
            if script.alwayson and script.section != section:
                continue

            if script.create_group:
                with gr.Group(visible=script.alwayson) as group:
                    self.create_script_ui(script)

                script.group = group
            else:
                self.create_script_ui(script)

    def prepare_ui(self):
        self.inputs = [None]

    def setup_ui(self):
        all_titles:list[str] = [
            wrap_call(script.title, script.filename, "title") or script.filename
            for script in self.scripts
        ]
        self.title_map[str,] = {
            title.lower(): script for title, script in zip(all_titles, self.scripts)
        }
        self.titles = [
            wrap_call(script.title, script.filename, "title")
            or f"{script.filename} [error]"
            for script in self.selectable_scripts
        ]

        self.setup_ui_for_section(None)

        dropdown = gr.Dropdown(
            label="Script",
            elem_id="script_list",
            choices=["None"] + self.titles,
            value="None",
            type="index",
        )
        self.inputs[0] = dropdown

        self.setup_ui_for_section(None, self.selectable_scripts)

        def select_script(script_index):
            if script_index is None:
                script_index = 0
            selected_script = (
                self.selectable_scripts[script_index - 1] if script_index > 0 else None
            )

            return [
                gr.update(visible=selected_script == s) for s in self.selectable_scripts
            ]

        def init_field(title):
            """called when an initial value is set from ui-config.json to show script's UI components"""

            if title == "None":
                return

            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=[script.group for script in self.selectable_scripts],
        )

        self.script_load_ctr = 0

        def onload_script_visibility(params):
            title = params.get("Script", None)
            if title:
                try:
                    title_index = self.titles.index(title)
                    visibility = title_index == self.script_load_ctr
                    self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                    return gr.update(visible=visibility)
                except ValueError:
                    params["Script"] = None
                    massage = f'Cannot find Script: "{title}"'
                    print(massage)
                    gr.Warning(massage)
            return gr.update(visible=False)

        self.infotext_fields.append(
            (dropdown, lambda x: gr.update(value=x.get("Script", "None")))
        )
        self.infotext_fields.extend(
            [
                (script.group, onload_script_visibility)
                for script in self.selectable_scripts
            ]
        )

        self.apply_on_before_component_callbacks()

        return self.inputs

    def run(self, p, *args):
        script_index = args[0]

        if script_index == 0 or script_index is None:
            return None

        script = self.selectable_scripts[script_index - 1]

        if script is None:
            return None

        script_args = args[script.args_from : script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def list_scripts_for_method(self, method_name):
        if method_name in ("before_component", "after_component"):
            return self.scripts
        else:
            return self.alwayson_scripts

    def create_ordered_callbacks_list(self, method_name, *, enable_user_sort=True):
        script_list = self.list_scripts_for_method(method_name)
        category = f"script_{method_name}"
        callbacks = []

        for script in script_list:
            if getattr(script.__class__, method_name, None) == getattr(
                Script, method_name, None
            ):
                continue

            script_callbacks.add_callback(
                callbacks,
                script,
                category=category,
                name=script.__class__.__name__,
                filename=script.filename,
            )

        return script_callbacks.sort_callbacks(
            category, callbacks, enable_user_sort=enable_user_sort
        )

    def ordered_callbacks(self, method_name, *, enable_user_sort=True):
        script_list = self.list_scripts_for_method(method_name)
        category = f"script_{method_name}"

        scrpts_len, callbacks = self.callback_map.get(category, (-1, None))

        if callbacks is None or scrpts_len != len(script_list):
            callbacks = self.create_ordered_callbacks_list(
                method_name, enable_user_sort=enable_user_sort
            )
            self.callback_map[category] = len(script_list), callbacks

        return callbacks

    def ordered_scripts(self, method_name):
        return [x.callback for x in self.ordered_callbacks(method_name)]

    def before_process(self, p):
        for script in self.ordered_scripts("before_process"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process(p, *script_args)
            except Exception:
                errors.report(
                    f"Error running before_process: {script.filename}", exc_info=True
                )

    def process(self, p):
        for script in self.ordered_scripts("process"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process(p, *script_args)
            except Exception:
                errors.report(
                    f"Error running process: {script.filename}", exc_info=True
                )

    def process_before_every_step(self, p, **kwargs):
        for script in self.ordered_scripts("process_before_every_step"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process_before_every_step(p, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running process_before_every_step: {script.filename}",
                    exc_info=True,
                )

    def process_before_every_sampling(self, p, **kwargs):
        for script in self.ordered_scripts("process_before_every_sampling"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process_before_every_sampling(p, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running process_before_every_sampling: {script.filename}",
                    exc_info=True,
                )

    def before_process_batch(self, p, **kwargs):
        for script in self.ordered_scripts("before_process_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running before_process_batch: {script.filename}",
                    exc_info=True,
                )

    def before_process_init_images(self, p, pp, **kwargs):
        for script in self.ordered_scripts("before_process_init_images"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process_init_images(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running before_process_init_images: {script.filename}",
                    exc_info=True,
                )

    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.ordered_scripts("after_extra_networks_activate"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running after_extra_networks_activate: {script.filename}",
                    exc_info=True,
                )

    def process_batch(self, p, **kwargs):
        for script in self.ordered_scripts("process_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running process_batch: {script.filename}", exc_info=True
                )

    def postprocess(self, p, processed):
        for script in self.ordered_scripts("postprocess"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(
                    f"Error running postprocess: {script.filename}", exc_info=True
                )

    def postprocess_batch(self, p, images, **kwargs):
        for script in self.ordered_scripts("postprocess_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(
                    f"Error running postprocess_batch: {script.filename}", exc_info=True
                )

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.ordered_scripts("postprocess_batch_list"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(
                    f"Error running postprocess_batch_list: {script.filename}",
                    exc_info=True,
                )

    def post_sample(self, p, ps: PostSampleArgs):
        for script in self.ordered_scripts("post_sample"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.post_sample(p, ps, *script_args)
            except Exception:
                errors.report(
                    f"Error running post_sample: {script.filename}", exc_info=True
                )

    def on_mask_blend(self, p, mba: MaskBlendArgs):
        for script in self.ordered_scripts("on_mask_blend"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.on_mask_blend(p, mba, *script_args)
            except Exception:
                errors.report(
                    f"Error running post_sample: {script.filename}", exc_info=True
                )

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        for script in self.ordered_scripts("postprocess_image"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                errors.report(
                    f"Error running postprocess_image: {script.filename}", exc_info=True
                )

    def postprocess_maskoverlay(self, p, ppmo: PostProcessMaskOverlayArgs):
        for script in self.ordered_scripts("postprocess_maskoverlay"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_maskoverlay(p, ppmo, *script_args)
            except Exception:
                errors.report(
                    f"Error running postprocess_image: {script.filename}", exc_info=True
                )

    def postprocess_image_after_composite(self, p, pp: PostprocessImageArgs):
        for script in self.ordered_scripts("postprocess_image_after_composite"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_image_after_composite(p, pp, *script_args)
            except Exception:
                errors.report(
                    f"Error running postprocess_image_after_composite: {script.filename}",
                    exc_info=True,
                )

    def before_component(self, component, **kwargs):
        for callback, script in self.on_before_component_elem_id.get(
            kwargs.get("elem_id"), []
        ):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(
                    f"Error running on_before_component: {script.filename}",
                    exc_info=True,
                )

        for script in self.ordered_scripts("before_component"):
            try:
                script.before_component(component, **kwargs)
            except Exception:
                errors.report(
                    f"Error running before_component: {script.filename}", exc_info=True
                )

    def after_component(self, component, **kwargs):
        for callback, script in self.on_after_component_elem_id.get(
            component.elem_id, []
        ):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(
                    f"Error running on_after_component: {script.filename}",
                    exc_info=True,
                )

        for script in self.ordered_scripts("after_component"):
            try:
                script.after_component(component, **kwargs)
            except Exception:
                errors.report(
                    f"Error running after_component: {script.filename}", exc_info=True
                )

    def script(self, title: str):
        return self.title_map.get(title.lower())

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to

    def before_hr(self, p):
        for script in self.ordered_scripts("before_hr"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                errors.report(
                    f"Error running before_hr: {script.filename}", exc_info=True
                )

    def setup_scrips(self, p, *, is_ui=True):
        for script in self.ordered_scripts("setup"):
            if not is_ui and script.setup_for_ui_only:
                continue

            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.setup(p, *script_args)
            except Exception:
                errors.report(f"Error running setup: {script.filename}", exc_info=True)

    def set_named_arg(self, args, script_name, arg_elem_id, value, fuzzy=False):
        """Locate an arg of a specific script in script_args and set its value
        Args:
            args: all script args of process p, p.script_args
            script_name: the name target script name to
            arg_elem_id: the elem_id of the target arg
            value: the value to set
            fuzzy: if True, arg_elem_id can be a substring of the control.elem_id else exact match
        Returns:
            Updated script args
        when script_name in not found or arg_elem_id is not found in script controls, raise RuntimeError
        """
        script = next((x for x in self.scripts if x.name == script_name), None)
        if script is None:
            raise RuntimeError(f"script {script_name} not found")

        for i, control in enumerate(script.controls):
            if (
                arg_elem_id in control.elem_id
                if fuzzy
                else arg_elem_id == control.elem_id
            ):
                index = script.args_from + i

                if isinstance(args, tuple):
                    return args[:index] + (value,) + args[index + 1 :]
                elif isinstance(args, list):
                    args[index] = value
                    return args
                else:
                    raise RuntimeError(f"args is not a list or tuple, but {type(args)}")
        raise RuntimeError(
            f"arg_elem_id {arg_elem_id} not found in script {script_name}"
        )