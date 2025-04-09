from modules import scripts, shared, script_callbacks
import re


def strip_comments(text):
    text = re.sub(r'(^/\*.*?\*/(\n|$))', '\n', text, flags=re.MULTILINE|re.DOTALL)  # multiline comments (/* */)
    text = re.sub(r'(#.*(\n|$))', '\n', text)  # single line comment (#)
    text = re.sub(r'(//.*(\n|$))', '\n', text)  # single line comment (//)
    text = re.sub(r'(/\*.*(\n|$))|(\*/.*(\n|$))', '\n', text) # dangling multiline comment brackets (/* */)
    #text = re.sub(r'[\n]{3,}', '\n\n', text)  # remove multiple consecutive newlines
    #text = re.sub(r'[ ]{2,}', ' ', text)  # remove multiple consecutive spaces
    #text = text.rstrip() # finally, strip leading and trailing whitespace

    return text


class ScriptStripComments(scripts.Script):
    def title(self):
        return "Comments"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        p.all_prompts = [strip_comments(x) for x in p.all_prompts]
        p.all_negative_prompts = [strip_comments(x) for x in p.all_negative_prompts]

        p.main_prompt = strip_comments(p.main_prompt)
        p.main_negative_prompt = strip_comments(p.main_negative_prompt)

        if getattr(p, 'enable_hr', False):
            p.all_hr_prompts = [strip_comments(x) for x in p.all_hr_prompts]
            p.all_hr_negative_prompts = [strip_comments(x) for x in p.all_hr_negative_prompts]

            p.hr_prompt = strip_comments(p.hr_prompt)
            p.hr_negative_prompt = strip_comments(p.hr_negative_prompt)


def before_token_counter(params: script_callbacks.BeforeTokenCounterParams):
    params.prompt = strip_comments(params.prompt)


script_callbacks.on_before_token_counter(before_token_counter)


shared.options_templates.update(shared.options_section(('sd', "Stable Diffusion", "sd"), {
    "enable_prompt_comments_def": shared.OptionInfo(False, "Save comments").info("Toggles saving of comments in finished image files. Use # anywhere in the prompt to hide the text between # and the end of the line from the generation. For multiline comments, use /* to open and */ to close."),
}))
