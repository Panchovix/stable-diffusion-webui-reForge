import torch
import tqdm
from modules.shared import opts

if opts.sd_sampling == "A1111":
    from k_diffusion import sampling
elif opts.sd_sampling == "ldm patched (Comfy)":
    from ldm_patched.k_diffusion import sampling as sampling


@torch.no_grad()
def restart_sampler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    restart_list=None,
):
    """Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0
    if opts.sd_sampling == "A1111":
        from k_diffusion.sampling import to_d, get_sigmas_karras
    elif opts.sd_sampling == "ldm patched (Comfy)":
        from ldm_patched.k_diffusion.sampling import to_d, get_sigmas_karras

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": step_id,
                    "sigma": new_sigma,
                    "sigma_hat": old_sigma,
                    "denoised": denoised,
                }
            )
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(
                steps - restart_steps * restart_times,
                sigmas[-2].item(),
                sigmas[0].item(),
                device=sigmas.device,
            )
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {
        int(torch.argmin(abs(sigmas - key), dim=0)): value
        for key, value in restart_list.items()
    }

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(
                    restart_steps,
                    sigmas[min_idx].item(),
                    sigmas[max_idx].item(),
                    device=sigmas.device,
                )[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = (
                x
                + sampling.torch.randn_like(x)
                * s_noise
                * (old_sigma**2 - last_sigma**2) ** 0.5
            )
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


@torch.no_grad()
def sample_dpmpp_2m_cfgpp(
    model, x, sigmas, extra_args=None, callback=None, disable=None
):
    """DPM-Solver++(2M) CFG++.
    From https://github.com/crowsonkb/k-diffusion/blob/master/ldm_patched.k_diffusion/sampling.py
    """
    model.cond_scale_miltiplier = 1 / 12.5
    model.need_last_noise_uncond = True

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    old_noise_uncond = None

    for i in tqdm.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        noise_uncond = model.last_noise_uncond
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or old_noise_uncond is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_noise_uncond
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        old_noise_uncond = noise_uncond
    return x
