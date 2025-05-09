from modules_forge.supported_preprocessor import PreprocessorClipVision, Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import numpy_to_pytorch
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApply, InsightFaceLoader
from pathlib import Path

cached_insightface = None

opIPAdapterApply = IPAdapterApply().apply_ipadapter
opInsightFaceLoader = InsightFaceLoader().load_insight_face


class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.slider_1 = PreprocessorParameter(label='Noise', minimum=0.0, maximum=1.0, value=0.23, step=0.01, visible=True)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.sorting_priority = 20
        self.do_tiled = (0, None)

    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=input_image,
            weight_type="original",
            noise=slider_1,
            embeds=None,
            unfold_batch=False,
            do_tiled=self.do_tiled,
        )
        return cond


# needs (simple enough) changes to backend.patcher.clipvision to preprocess image at greater size, but also needs retrained IPAdapters
# class PreprocessorClipVisionForIPAdapter_448(PreprocessorClipVisionForIPAdapter):
    # def __init__(self, name, url, filename):
        # super().__init__(name, url, filename)
        # self.do_tiled = (448, None)

# add_supported_preprocessor(PreprocessorClipVisionForIPAdapter_448(
    # name='CLIP-ViT-H-448 (Ostris) (IPAdapter)',
    # url='https://huggingface.co/ostris/CLIP-ViT-H-14-448/resolve/main/model.safetensors',
    # filename='CLIP-ViT-H-14-448.safetensors'
# ))

class PreprocessorClipVisionWithInsightFaceForIPAdapter(PreprocessorClipVisionForIPAdapter):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.do_tiled = (0, None)

    def load_insightface(self):
        global cached_insightface
        if cached_insightface is None:
            cached_insightface = opInsightFaceLoader()[0]
        return cached_insightface

    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=None if '(Portrait)' in self.name else self.load_clipvision(),
            insightface=self.load_insightface(),
            image=input_image,
            weight_type="original",
            noise=slider_1,
            embeds=None,
            unfold_batch=False,
            do_tiled=self.do_tiled,
        )
        return cond

class PreprocessorClipVisionWithInsightFaceForIPAdapter_Tiled(PreprocessorClipVisionWithInsightFaceForIPAdapter):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.do_tiled = (256, 'Interleaved')    # 256 seems better than 224, for Insightface, maybe


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-H-Face (Ostris) (IPAdapter)',
    url='https://huggingface.co/ostris/CLIP-H-Face-v3/resolve/main/model.safetensors',
    filename='CLIP-H-Face-v3.safetensors'
))


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-bigG (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter(
    name='InsightFace+CLIP-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter_Tiled(
    name='InsightFace+CLIP-H (IPAdapter) (Tiled)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))
add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter_Tiled(
    name='InsightFace+CLIP-H-Face (Ostris) (IPAdapter) (Tiled)',
    url='https://huggingface.co/ostris/CLIP-H-Face-v3/resolve/main/model.safetensors',
    filename='CLIP-H-Face-v3.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter_Tiled(
    name='InsightFace (IPAdapter) (Portrait) (Tiled)',
    url='',
    filename=''
))


add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter(
    name='InsightFace+CLIP-G (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))



class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        model = state_dict

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
            return None

        o = IPAdapterPatcher(model)

        model_filename = Path(ckpt_path).name.lower()
        if 'v2' in model_filename:
            o.faceid_v2 = True
            o.weight_v2 = True

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = False
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        if isinstance(cond, list):  # should always be True
            images = []
            for c in cond:
                tile_size = c['do_tiled'][0]
                tile_type = c['do_tiled'][1]

                image = c['image']
                if tile_size > 0:
                    r = min(image.shape[0] // tile_size, image.shape[1] // tile_size)
                    if tile_type == 'Interleaved' and r >= 2:   #interleaved split into r*r images
                        for i in range(r):
                            for j in range(r):
                                part = image[i::r, j::r]
                                images.append(numpy_to_pytorch(part))
                    elif tile_type == "Tiled" and r >= 1:
                        for i in range(0, image.shape[0], tile_size):
                            for j in range(0, image.shape[1], tile_size):
                                tile = image[i:i+tile_size, j:j+tile_size]
                                images.append(numpy_to_pytorch(tile))
                    else:
                        images.append(numpy_to_pytorch(image))
                else:
                    images.append(numpy_to_pytorch(image))

            pcond = cond[0].copy()
            pcond['image'] = images
        else:
            pcond = cond.copy()

        del pcond['do_tiled']

        unet = opIPAdapterApply(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=mask.squeeze(1) if mask is not None else None,
            **pcond,
        )[0]

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(IPAdapterPatcher)
