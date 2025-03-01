import ldm_patched.modules.sd
import ldm_patched.modules.model_sampling
import torch


class LCM(ldm_patched.modules.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(
            sigma.shape[:1] + (1,) * (model_output.ndim - 1)
        )
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0  # timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input


class ModelSamplingDiscreteDistilled(
    ldm_patched.modules.model_sampling.ModelSamplingDiscrete
):
    original_timesteps = 50

    def __init__(self, model_config=None):
        super().__init__(model_config)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[
                self.num_timesteps - 1 - x * self.skip_steps
            ]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return (
            dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps
            + (self.skip_steps - 1)
        ).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(
            (
                (timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1))
                / self.skip_steps
            ).float(),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def to(self, device):
        self.sigmas = self.sigmas.to(device)
        self.log_sigmas = self.log_sigmas.to(device)
        return self

    def detach(self):
        self.sigmas = self.sigmas.detach()
        self.log_sigmas = self.log_sigmas.detach()
        return self


def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5


class ModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling": (["eps", "v_prediction", "lcm"],),
                "zsnr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, zsnr):
        m = model.clone()

        sampling_base = ldm_patched.modules.model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION
        elif sampling == "lcm":
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        try:
            # Get the current model_sampling object
            current_sampling = getattr(m.model, "model_sampling", None)

            # Create new sampling object
            model_sampling = ModelSamplingAdvanced(model.model.model_config)
            if zsnr:
                model_sampling.set_sigmas(
                    rescale_zero_terminal_snr_sigmas(model_sampling.sigmas)
                )

            # Directly set the attribute instead of using add_object_patch
            setattr(m.model, "model_sampling", model_sampling)

            # Store the original in the backup if needed
            if current_sampling is not None:
                if not hasattr(m, "_sampling_backup"):
                    m._sampling_backup = {}
                m._sampling_backup["model_sampling"] = current_sampling

        except Exception as e:
            print(f"Error while patching model sampling: {str(e)}")
            raise e

        return (m,)

    def restore(self, model):
        """Restore original sampling if needed"""
        if hasattr(model, "_sampling_backup"):
            if "model_sampling" in model._sampling_backup:
                setattr(
                    model.model,
                    "model_sampling",
                    model._sampling_backup["model_sampling"],
                )
                del model._sampling_backup["model_sampling"]


class ModelSamplingContinuousEDM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling": (["v_prediction", "eps"],),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        if sampling == "eps":
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(
            ldm_patched.modules.model_sampling.ModelSamplingContinuousEDM, sampling_type
        ):
            pass

        try:
            # Get the current model_sampling object
            current_sampling = getattr(m.model, "model_sampling", None)

            # Create new sampling object
            model_sampling = ModelSamplingAdvanced(model.model.model_config)
            model_sampling.set_sigma_range(sigma_min, sigma_max)

            # Directly set the attribute instead of using add_object_patch
            setattr(m.model, "model_sampling", model_sampling)

            # Store the original in the backup if needed
            if current_sampling is not None:
                if not hasattr(m, "_sampling_backup"):
                    m._sampling_backup = {}
                m._sampling_backup["model_sampling"] = current_sampling

        except Exception as e:
            print(f"Error while patching model sampling: {str(e)}")
            raise e

        return (m,)

    def restore(self, model):
        """Restore original sampling if needed"""
        if hasattr(model, "_sampling_backup"):
            if "model_sampling" in model._sampling_backup:
                setattr(
                    model.model,
                    "model_sampling",
                    model._sampling_backup["model_sampling"],
                )
                del model._sampling_backup["model_sampling"]


NODE_CLASS_MAPPINGS = {
    "ModelSamplingDiscrete": ModelSamplingDiscrete,
    "ModelSamplingContinuousEDM": ModelSamplingContinuousEDM,
}
