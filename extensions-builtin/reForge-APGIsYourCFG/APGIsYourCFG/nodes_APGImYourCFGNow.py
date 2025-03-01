from ldm_patched.modules.model_patcher import ModelPatcher
import torch


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
):
    dtype = v0.dtype
    # v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update

    return pred_guided


class APG_ImYourCFGNow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "momentum": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": -1.5,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                    },
                ),
                "adaptive_momentum": (
                    "FLOAT",
                    {
                        "default": 0.180,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "norm_threshold": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "guidance_limiter": ("BOOLEAN", {"default": False}),
                "guidance_sigma_start": (
                    "FLOAT",
                    {
                        "default": 5.42,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "guidance_sigma_end": (
                    "FLOAT",
                    {
                        "default": 0.28,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "print_data": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        momentum: float = 0.5,
        adaptive_momentum: float = 0.180,
        norm_threshold: float = 15.0,
        eta: float = 1.0,
        guidance_limiter: bool = False,
        guidance_sigma_start: float = 5.42,
        guidance_sigma_end: float = 0.28,
        print_data=False,
        extras=[],
    ):
        momentum_buffer = MomentumBuffer(momentum)
        extras = [momentum_buffer, momentum, adaptive_momentum]

        def apg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model = args["model"]
            cond_scale = args["cond_scale"]

            if guidance_limiter:
                if (guidance_sigma_start >= 0 and sigma[0] > guidance_sigma_start) or (
                    guidance_sigma_end >= 0 and sigma[0] <= guidance_sigma_end
                ):
                    if print_data:
                        print(f" guidance limiter active (sigma: {sigma[0]})")
                    return uncond + (cond - uncond)

            momentum_buffer = extras[0]
            momentum = extras[1]
            adaptive_momentum = extras[2]

            t = model.model_sampling.timestep(sigma)[0].item()

            if (
                torch.is_tensor(momentum_buffer.running_average)
                and (cond.shape[3] != momentum_buffer.running_average.shape[3])
            ) or t == 999:
                momentum_buffer = MomentumBuffer(momentum)
                extras[0] = momentum_buffer
            else:
                signal_scale = momentum
                if adaptive_momentum > 0:
                    if momentum < 0:
                        signal_scale += -momentum * (adaptive_momentum**4) * (1000 - t)
                        if signal_scale > 0:
                            signal_scale = 0
                    else:
                        signal_scale -= momentum * (adaptive_momentum**4) * (1000 - t)
                        if signal_scale < 0:
                            signal_scale = 0

                momentum_buffer.momentum = signal_scale

            if print_data:
                print(" momentum: ", momentum_buffer.momentum, " t: ", t)

            return normalized_guidance(
                cond, uncond, cond_scale, momentum_buffer, eta, norm_threshold
            )

        m = model.clone()
        m.set_model_sampler_cfg_function(apg_function, extras == extras)
        m.model_options["disable_cfg1_optimization"] = False

        return (m,)


NODE_CLASS_MAPPINGS = {
    "APG_ImYourCFGNow": APG_ImYourCFGNow,
}
