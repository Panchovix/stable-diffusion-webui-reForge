import torch
import copy

from modules_forge.supported_preprocessor import PreprocessorClipVision, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


def revision_conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
    revision_condition = model_options['revision_conditions']

    weight = float(revision_condition["weight"])
    adm_inputs = []
    for c in revision_condition['cond']:
        adm_cond = c.image_embeds
        adm_inputs.append(adm_cond * weight)
    adm_out = torch.stack(adm_inputs).sum(0)

    new_y = adm_out[:, :1280]
    cond = copy.deepcopy(cond)
    uncond = copy.deepcopy(uncond)

    for c in cond:
        c['model_conds']['y'].cond[:, :1280] = new_y.clone()

    for c in uncond:
        c['model_conds']['y'].cond[:, :1280] = torch.zeros_like(new_y)

    if revision_condition["ignore_prompt"]:
        for c in cond + uncond:
            c['model_conds']['c_crossattn'].cond = torch.zeros_like(c['model_conds']['c_crossattn'].cond)

    return model, x, timestep, uncond, cond, cond_scale, model_options, seed


class PreprocessorClipVisionForRevision(PreprocessorClipVision):
    def __init__(self, name, url, filename, ignore_prompt=False):
        super().__init__(name, url, filename)
        self.tags = ['Revision [SDXL]']
        self.model_filename_filters = ['Revision']
        self.do_not_need_model = True
        self.ignore_prompt = ignore_prompt
        self.slider_resolution = PreprocessorParameter(label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=False)


    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        
        if 'revision_conditions' not in process.sd_model.forge_objects.unet.model_options:
            unit = kwargs['unit']
            weight = float(unit.weight)

            unet = process.sd_model.forge_objects.unet.clone()
            unet.model_options['revision_conditions'] = {}
            unet.model_options['revision_conditions']['cond'] = [cond]
            unet.model_options['revision_conditions']['weight'] = weight
            unet.model_options['revision_conditions']['ignore_prompt'] = self.ignore_prompt

            unet.add_conditioning_modifier(revision_conditioning_modifier, ensure_uniqueness=True)

            process.sd_model.forge_objects.unet = unet
        else:   # for multi-input (ControlNet > Batch Upload)
            unet.model_options['revision_conditions']['cond'].append([cond])

        return cond, mask


add_supported_preprocessor(PreprocessorClipVisionForRevision(
    name='CLIP-G (Revision)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors',
    ignore_prompt=False
))

add_supported_preprocessor(PreprocessorClipVisionForRevision(
    name='CLIP-G (Revision ignore prompt)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors',
    ignore_prompt=True
))
