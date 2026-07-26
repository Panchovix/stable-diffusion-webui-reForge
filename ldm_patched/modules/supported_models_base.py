# Original code from Comfy, https://github.com/comfyanonymous/ComfyUI


import logging
import torch
from . import model_base
from . import utils
from . import latent_formats

_log = logging.getLogger(__name__)

# Models that do NOT embed alphas_cumprod (schedule defaults apply):
#   - Standard SD1.5 fine-tunes
#   - Standard SDXL base / SDXL refiner
#   - FLUX.1, SD3 (use flow matching — no alphas_cumprod at all)
#
# Models that DO embed alphas_cumprod (schedule from checkpoint):
#   - CompVis full-model saves (ldm/ddpm.py training)
#   - Illustrious-XL v0–v2 (epsilon, standard SDXL schedule baked in)
#   - Illustrious-XL v3.5-vpred (v-prediction + ZTSNR baked in)
#   - NoobAI-XL v-pred (v-prediction + ZTSNR)
#   - Hassaku XL (epsilon, standard schedule baked in)


def apply_checkpoint_sampling_params(out, state_dict):
    """Read noise-schedule information that a checkpoint may have embedded as
    training buffers (register_buffer keys saved into the state_dict) and apply
    it to the model object so that the sigma schedule at inference matches the
    one used during training.

    Parameters
    ----------
    out        : BaseModel instance — must expose out.model_sampling.set_sigmas()
    state_dict : flat state dict (top-level keys, not UNet-prefixed)

    Returns
    -------
    ztsnr_detected : bool — True if the stored schedule encodes zero terminal SNR
    """
    model_name = type(out).__name__

    if "alphas_cumprod" in state_dict:
        alphas = state_dict["alphas_cumprod"].float()

        # BF16 rounding trap (MODEL_META.md §Precision / A1111 #14071):
        # alphas_cumprod[0] is stored as BF16 ≈ 0.99915, which rounds to exactly
        # 1.0.  After .float() it is still 1.0 → sigma_min = 0, log_sigmas[0] = -inf,
        # corrupting the timestep lookup for all low-noise steps.
        # Fix: recompute alphas_cumprod from the embedded betas (also BF16, but
        # betas[0] ≈ 0.00085 survives BF16 with <0.1% error) via f32 cumprod.
        # The result matches the diffusers scheduler reference to within 1e-5.
        bf16_saturated = bool((alphas[0].item() == 1.0) and (alphas[-1].item() < 1.0))
        if bf16_saturated:
            if "betas" in state_dict:
                betas_f32 = state_dict["betas"].float()
                alphas = torch.cumprod(1.0 - betas_f32, dim=0)
                _log.info(
                    "%s: alphas_cumprod[0] saturated to 1.0 (BF16) — "
                    "recomputed from checkpoint betas in f32",
                    model_name,
                )
            else:
                # No betas fallback: clamp to avoid -inf in log_sigmas.
                # sigma_min will be approximate but finite.
                alphas = alphas.clamp(max=1.0 - 1e-5)
                _log.warning(
                    "%s: alphas_cumprod[0] saturated (BF16) and no betas key found — "
                    "clamped to 1-1e-5; sigma_min will be approximate",
                    model_name,
                )

        sigmas = ((1.0 - alphas) / alphas.clamp(min=1e-10)) ** 0.5

        # ZTSNR heuristic: if the last alpha is essentially 0 the training used
        # zero-terminal SNR — the rescaling is already baked into the tensor.
        ztsnr_detected = bool(alphas[-1].item() < 1e-5)

        out.model_sampling.set_sigmas(sigmas)
        _log.info(
            "%s: using checkpoint-embedded alphas_cumprod "
            "(sigma_min=%.4f  sigma_max=%.4f%s)",
            model_name,
            sigmas[0].item(),
            sigmas[-1].item(),
            "  ZTSNR detected" if ztsnr_detected else "",
        )
        return ztsnr_detected

    elif "betas" in state_dict:
        betas = state_dict["betas"].float()
        alphas = torch.cumprod(1.0 - betas, dim=0)
        sigmas = ((1.0 - alphas) / alphas.clamp(min=1e-10)) ** 0.5
        out.model_sampling.set_sigmas(sigmas)
        _log.info(
            "%s: using checkpoint-embedded betas → sigma schedule "
            "(sigma_min=%.4f  sigma_max=%.4f)",
            model_name, sigmas[0].item(), sigmas[-1].item(),
        )
        return False

    else:
        ms = out.model_sampling
        _log.info(
            "%s: no alphas_cumprod/betas in checkpoint — "
            "using default %s schedule (linear_start=%.5f  linear_end=%.5f)",
            model_name,
            getattr(ms, "beta_schedule", "linear"),
            getattr(ms, "linear_start", 0.00085),
            getattr(ms, "linear_end", 0.012),
        )
        return False

class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}

class BASE:
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
    latent_format = latent_formats.LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    custom_operations = None
    scaled_fp8 = None
    optimizations = {"fp8": False}

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
        return model_base.ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        self.optimizations = self.optimizations.copy()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        if self.noise_aug_config is not None:
            out = model_base.SD21UNCLIP(self, self.noise_aug_config, model_type=self.model_type(state_dict, prefix), device=device)
        else:
            out = model_base.BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        apply_checkpoint_sampling_params(out, state_dict)
        return out

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.state_dict_prefix_replace(state_dict, dict.fromkeys(self.text_encoder_key_prefix, ""), filter_keys=True)
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

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config['dtype'] = dtype
        self.manual_cast_dtype = manual_cast_dtype
