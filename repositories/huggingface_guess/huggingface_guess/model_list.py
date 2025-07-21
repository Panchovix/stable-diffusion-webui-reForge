import torch

from enum import Enum
from huggingface_guess import latent, utils, diffusers_convert


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8


class BASE:
    huggingface_repo = None
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = latent.LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    unet_target = 'unet'
    vae_target = 'vae'

    @classmethod
    def matches(s, unet_config, state_dict=None):
        for k in s.unet_config:
            if k not in unet_config or s.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in s.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict, prefix=""):
        return ModelType.EPS

    def clip_target(self, state_dict):
        return {}

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.state_dict_prefix_replace(state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True)
        return state_dict

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.text_encoder_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_clip_vision_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        if self.clip_vision_prefix is not None:
            replace_prefix[""] = self.clip_vision_prefix
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.vae_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)


class SD15(BASE):
    huggingface_repo = "runwayml/stable-diffusion-v1-5"

    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent.SD15
    memory_usage_factor = 1.0

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
            if ids.dtype == torch.float32:
                state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict:
                state_dict.pop(p)

        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self, state_dict={}):
        return {'clip_l': 'text_encoder'}


class SD20(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-2-1"

    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
        "attn_precision": torch.float32,
    }

    latent_format = latent.SD15
    memory_usage_factor = 1.0

    def model_type(self, state_dict, prefix=""):
        if self.unet_config["in_channels"] == 4:  # SD2.0 inpainting models are not v prediction
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict.get(k, None)
            if out is not None and torch.std(out, unbiased=False) > 0.09:  # not sure how well this will actually work. I guess we will find out.
                return ModelType.V_PREDICTION
        return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_h."  # SD2 in sgm format
        replace_prefix["cond_stage_model.model."] = "clip_h."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_h.", "clip_h.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self, state_dict={}):
        return {'clip_h': 'text_encoder'}


class SD21UnclipL(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 1536,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 768}


class SD21UnclipH(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2048,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1024}


class SDXLRefiner(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-refiner-1.0"

    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL
    memory_usage_factor = 1.0

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_g."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix["clip_g"] = "conditioner.embedders.0.model"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self, state_dict={}):
        return {'clip_g': 'text_encoder'}


class SDXL(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-base-1.0"

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL

    memory_usage_factor = 0.7

    def model_type(self, state_dict, prefix=""):
        if 'edm_mean' in state_dict and 'edm_std' in state_dict:  # Playground V2.5
            self.latent_format = latent.SDXL_Playground_2_5()
            self.sampling_settings["sigma_data"] = 0.5
            self.sampling_settings["sigma_max"] = 80.0
            self.sampling_settings["sigma_min"] = 0.002
            return ModelType.EDM
        elif "edm_vpred.sigma_max" in state_dict:
            self.sampling_settings["sigma_max"] = float(state_dict["edm_vpred.sigma_max"].item())
            if "edm_vpred.sigma_min" in state_dict:
                self.sampling_settings["sigma_min"] = float(state_dict["edm_vpred.sigma_min"].item())
            return ModelType.V_PREDICTION_EDM
        elif "v_pred" in state_dict:
            return ModelType.V_PREDICTION
        else:
            return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"
        replace_prefix["conditioner.embedders.1.model."] = "clip_g."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        keys_to_replace = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix["clip_g"] = "conditioner.embedders.1.model"
        replace_prefix["clip_l"] = "conditioner.embedders.0"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self, state_dict={}):
        return {'clip_l': 'text_encoder', 'clip_g': 'text_encoder_2'}


class SSD1B(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class Segmind_Vega(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class KOALA_700M(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 5],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class KOALA_1B(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 6],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class SVD_img2vid(BASE):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 768,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }

    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
        "attn_precision": torch.float32,
    }

    clip_vision_prefix = "conditioner.embedders.0.open_clip.model.visual."

    latent_format = latent.SD15

    sampling_settings = {"sigma_max": 700.0, "sigma_min": 0.002}


class SV3D_u(SVD_img2vid):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 256,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }

    vae_key_prefix = ["conditioner.embedders.1.encoder."]


class SV3D_p(SV3D_u):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 1280,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }


class Stable_Zero123(BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    required_keys = {
        "cc_projection.weight": None,
        "cc_projection.bias": None,
    }

    clip_vision_prefix = "cond_stage_model.model.visual."

    latent_format = latent.SD15


class SD_X4Upscaler(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 256,
        'in_channels': 7,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "disable_self_attentions": [True, True, True, False],
        "num_classes": 1000,
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent.SD_X4

    sampling_settings = {
        "linear_start": 0.0001,
        "linear_end": 0.02,
    }


class Stable_Cascade_C(BASE):
    unet_config = {
        "stable_cascade_stage": 'c',
    }

    unet_extra_config = {}

    latent_format = latent.SC_Prior
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 2.0,
    }

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoder."]
    clip_vision_prefix = "clip_l_vision."

    def process_unet_state_dict(self, state_dict):
        key_list = list(state_dict.keys())
        for y in ["weight", "bias"]:
            suffix = "in_proj_{}".format(y)
            keys = filter(lambda a: a.endswith(suffix), key_list)
            for k_from in keys:
                weights = state_dict.pop(k_from)
                prefix = k_from[:-(len(suffix) + 1)]
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["to_q", "to_k", "to_v"]
                    k_to = "{}.{}.{}".format(prefix, p[x], y)
                    state_dict[k_to] = weights[shape_from * x:shape_from * (x + 1)]
        return state_dict

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.state_dict_prefix_replace(state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True)
        if "clip_g.text_projection" in state_dict:
            state_dict["clip_g.transformer.text_projection.weight"] = state_dict.pop("clip_g.text_projection").transpose(0, 1)
        return state_dict

    def clip_target(self, state_dict={}):
        return {'clip_g': 'text_encoder'}


class Stable_Cascade_B(Stable_Cascade_C):
    unet_config = {
        "stable_cascade_stage": 'b',
    }

    unet_extra_config = {}

    latent_format = latent.SC_B
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 1.0,
    }

    clip_vision_prefix = None


class SD15_instructpix2pix(SD15):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }


class SDXL_instructpix2pix(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
        "in_channels": 8,
    }


class SD3(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-3-medium-diffusers"
    
    unet_config = {
        "in_channels": 16,
        "pos_embed_scaling_factor": None,
    }

    sampling_settings = {
        "shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent.SD3

    memory_usage_factor = 1.2

    unet_target = 'transformer'

    text_encoder_key_prefix = ["text_encoders."]

    def clip_target(self, state_dict={}):
        return {'clip_l': 'text_encoder', 'clip_g': 'text_encoder_2', 't5xxl': 'text_encoder_3'}
        

class StableAudio(BASE):
    unet_config = {
        "audio_model": "dit1.0",
    }

    sampling_settings = {"sigma_max": 500.0, "sigma_min": 0.03}

    unet_extra_config = {}
    latent_format = latent.StableAudio1

    text_encoder_key_prefix = ["text_encoders."]
    vae_key_prefix = ["pretransform.model."]

    def process_unet_state_dict(self, state_dict):
        for k in list(state_dict.keys()):
            if k.endswith(".cross_attend_norm.beta") or k.endswith(".ff_norm.beta") or k.endswith(".pre_norm.beta"):  # These weights are all zero
                state_dict.pop(k)
        return state_dict

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)


class AuraFlow(BASE):
    unet_config = {
        "cond_seq_dim": 2048,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.73,
    }

    unet_extra_config = {}
    latent_format = latent.SDXL

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]


class HunyuanDiT(BASE):
    unet_config = {
        "image_model": "hydit",
    }

    unet_extra_config = {
        "attn_precision": torch.float32,
    }

    sampling_settings = {
        "linear_start": 0.00085,
        "linear_end": 0.018,
    }

    latent_format = latent.SDXL

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]


class HunyuanDiT1(HunyuanDiT):
    unet_config = {
        "image_model": "hydit1",
    }

    unet_extra_config = {}

    sampling_settings = {
        "linear_start": 0.00085,
        "linear_end": 0.03,
    }


class Flux(BASE):
    huggingface_repo = 'black-forest-labs/FLUX.1-dev'

    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {
    }

    unet_extra_config = {}
    latent_format = latent.Flux

    memory_usage_factor = 2.6

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = 'transformer'

    def clip_target(self, state_dict={}):
        result = {}
        pref = self.text_encoder_key_prefix[0]

        if "{}clip_l.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:
            result['clip_l'] = 'text_encoder'

        if "{}t5xxl.transformer.encoder.final_layer_norm.weight".format(pref) in state_dict:
            result['t5xxl'] = 'text_encoder_2'

        return result


class FluxSchnell(Flux):
    huggingface_repo = 'black-forest-labs/FLUX.1-schnell'

    unet_config = {
        "image_model": "flux",
        "guidance_embed": False,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.0,
    }


class Chroma(FluxSchnell):
    huggingface_repo = 'Chroma'

    unet_config = {
        "image_model": "chroma",
        "guidance_embed": False,
    }

    def clip_target(self, state_dict={}):
        result = {}
        pref = self.text_encoder_key_prefix[0]

        if "{}t5xxl.transformer.encoder.final_layer_norm.weight".format(pref) in state_dict:
            result['t5xxl'] = 'text_encoder'

        return result


class CosmosT2IPredict2(BASE):
    huggingface_repo = 'Cosmos'
    unet_config = {
        "image_model": "cosmos_predict2",
        "in_channels": 16,
    }

    sampling_settings = {
        "sigma_data": 1.0,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
    }

    unet_extra_config = {}
    latent_format = latent.Wan21

    memory_usage_factor = 1.6

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    unet_target = 'transformer'
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def clip_target(self, state_dict={}):
        result = {}
        pref = self.text_encoder_key_prefix[0]

        if "{}t5xxl.transformer.encoder.final_layer_norm.weight".format(pref) in state_dict:
            result['t5xxl'] = 'text_encoder'

        return result


models = [SD15_instructpix2pix, SD15, SD20, SD21UnclipL, SD21UnclipH, SDXL_instructpix2pix, SDXLRefiner, SDXL, SSD1B, SD3, Flux, FluxSchnell, Chroma, CosmosT2IPredict2]
