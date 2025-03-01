from functools import reduce
import warnings


from modules import gradio_extensions
from modules import (
    sd_models,
    script_callbacks,
    extra_networks,
    ui_settings,
)

from modules.shared import opts

import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
import modules.processing_scripts.comments as comments

create_setting_component = ui_settings.create_setting_component

warnings.filterwarnings(
    "default" if opts.show_warnings else "ignore", category=UserWarning
)
warnings.filterwarnings(
    "default" if opts.show_gradio_deprecation_warnings else "ignore",
    category=gradio_extensions.GradioDeprecationWarning,
)


def update_token_counter(text, steps, styles, *, is_positive=True):
    params = script_callbacks.BeforeTokenCounterParams(
        text, steps, styles, is_positive=is_positive
    )
    script_callbacks.before_token_counter_callback(params)
    text = params.prompt
    steps = params.steps
    styles = params.styles
    is_positive = params.is_positive

    if shared.opts.include_styles_into_token_counters:
        apply_styles = (
            shared.prompt_styles.apply_styles_to_prompt
            if is_positive
            else shared.prompt_styles.apply_negative_styles_to_prompt
        )
        text = apply_styles(text, styles)
    else:
        text = comments.strip_comments(text).strip()

    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
            prompt_flat_list, steps
        )

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    try:
        cond_stage_model = sd_models.model_data.sd_model.cond_stage_model
        assert cond_stage_model is not None
    except Exception:
        return "<span class='gr-box gr-text-input'>?/?</span>"

    flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max(
        [
            model_hijack.get_prompt_lengths(prompt, cond_stage_model)
            for prompt in prompts
        ],
        key=lambda args: args[0],
    )
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def update_negative_prompt_token_counter(*args):
    return update_token_counter(*args, is_positive=False)
