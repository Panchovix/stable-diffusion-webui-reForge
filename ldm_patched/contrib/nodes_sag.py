import torch
from torch import einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from ldm_patched.ldm.modules.attention import optimized_attention
import ldm_patched.modules.samplers

def attention_basic_with_sim(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold

    total = mask.shape[-1]
    x = round(math.sqrt((lh / lw) * total))
    xx = None
    for i in range(0, math.floor(math.sqrt(total) / 2)):
        for j in [(x + i), max(1, x - i)]:
            if total % j == 0:
                xx = j
                break
        if xx is not None:
            break

    x = xx
    y = total // x

    mask = (
        mask.reshape(b, x, y)
        .unsqueeze(1)
        .type(attn.dtype)
    )
    mask = F.interpolate(mask, (lh, lw))

    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img

class SelfAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.01}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale, blur_sigma):
        m = model.clone()

        attn_scores = None

        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                return optimized_attention(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])

        def post_cfg_function(args):
            nonlocal attn_scores
            uncond_attn = attn_scores

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = 1.0
            model = args["model"]
            uncond_pred = args["uncond_denoised"]
            uncond = args["uncond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]

            if min(cfg_result.shape[2:]) <= 4:
                return cfg_result

            if uncond_pred is None or uncond_attn is None:
                return cfg_result

            b = cfg_result.shape[0]
            uncond_pred = uncond_pred[:b]
            if x.shape[0] > b:
                x = x[-b:]

            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold)
            degraded_noised = degraded + x - uncond_pred

            # Strip mask from uncond to avoid batch size mismatch in mult computation
            uncond_sag = []
            for u in uncond:
                u_copy = u.copy()
                u_copy.pop('mask', None)
                uncond_sag.append(u_copy)

            # Strip model_function_wrapper from model_options.
            # The wrapper (used by cfg_denoiser for batched cond+uncond) may force
            # batch=2 output even when only a single sample is passed, causing a
            # shape mismatch in _calc_cond_batch when accumulating results.
            sag_model_options = {k: v for k, v in model_options.items()
                                 if k != 'model_function_wrapper'}

            (sag,) = ldm_patched.modules.samplers.calc_cond_batch(model, [uncond_sag], degraded_noised, sigma, sag_model_options)
            return cfg_result + (degraded - sag) * sag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)

        return (m, )

# Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
NODE_CLASS_MAPPINGS = {
    "SelfAttentionGuidance": SelfAttentionGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfAttentionGuidance": "Self-Attention Guidance",
}
