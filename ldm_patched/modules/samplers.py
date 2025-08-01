# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official

from __future__ import annotations
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.unipc import uni_pc
from typing import TYPE_CHECKING, Callable, NamedTuple
if TYPE_CHECKING:
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules.model_base import BaseModel
    from ldm_patched.modules.controlnet import ControlBase
import torch
from functools import partial
import collections
from ldm_patched.modules import model_management
import math
import logging
from modules import shared
import numpy as np
import ldm_patched.modules.sampler_helpers
import ldm_patched.modules.model_patcher
import ldm_patched.modules.patcher_extension
import ldm_patched.hooks
import scipy.stats
import numpy


def add_area_dims(area, num_dims):
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area

def get_area_and_mult(conds, x_in, timestep_in):
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = list(conds['area'])
        area = add_area_dims(area, len(dims))
        if (len(area) // 2) > len(dims):
            area = area[:len(dims)] + area[len(area) // 2:(len(area) // 2) + len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert (mask.shape[1:] == x_in.shape[2:])

        mask = mask[:input_x.shape[0]]
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds and area is not None:
        fuzz = 8
        for i in range(len(dims)):
            rr = min(fuzz, mult.shape[2 + i] // 4)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0 / rr) * (t + 1))
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    hooks = conds.get('hooks', None)
    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches', 'uuid', 'hooks'])
    return cond_obj(input_x, mult, conditioning, area, control, patches, conds.get('uuid', 0), hooks)

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def finalize_default_conds(model: 'BaseModel', hooked_to_run: dict[ldm_patched.hooks.HookGroup,list[tuple[tuple,int]]], default_conds: list[list[dict]], x_in, timestep, model_options):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for _ in default_conds:
        default_mults.append(torch.ones_like(x_in))
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            if area is not None:
                curr_default_mult: torch.Tensor = default_mults[i]
                dims = len(area) // 2
                for i in range(dims):
                    curr_default_mult = curr_default_mult.narrow(i + 2, area[i + dims], area[i])
                curr_default_mult -= cond_obj.mult
            else:
                default_mults[i] -= cond_obj.mult
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # do get_area_and_mult to get all the expected values
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue
            # replace p's mult with calculated mult
            p = p._replace(mult=mult)
            if p.hooks is not None:
                model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            hooked_to_run.setdefault(p.hooks, list())
            hooked_to_run[p.hooks] += [(p, i)]

def calc_cond_batch(model: 'BaseModel', conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = ldm_patched.modules.patcher_extension.WrapperExecutor.new_executor(
        _calc_cond_batch,
        ldm_patched.modules.patcher_extension.get_all_wrappers(ldm_patched.modules.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)

def _calc_cond_batch(model: 'BaseModel', conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    out_conds = []
    out_counts = []
    # separate conds by matching hooks
    hooked_to_run: dict[ldm_patched.hooks.HookGroup,list[tuple[tuple,int]]] = {}
    default_conds = []
    has_default_conds = False

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        default_c = []
        if cond is not None:
            for x in cond:
                if 'default' in x:
                    default_c.append(x)
                    has_default_conds = True
                    continue
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                if p.hooks is not None:
                    model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
                hooked_to_run.setdefault(p.hooks, list())
                hooked_to_run[p.hooks] += [(p, i)]
        default_conds.append(default_c)

    if has_default_conds:
        finalize_default_conds(model, hooked_to_run, default_conds, x_in, timestep, model_options)

    # We check if the model patcher exists before calling prepare_state
    # Will have probably to re-visit in case of issues
    if hasattr(model, 'current_patcher') and model.current_patcher is not None:
        model.current_patcher.prepare_state(timestep)
    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = model_management.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                cond_shapes = collections.defaultdict(list)
                for tt in batch_amount:
                    cond = {k: v.size() for k, v in to_run[tt][0].conditioning.items()}
                    for k, v in to_run[tt][0].conditioning.items():
                        cond_shapes[k].append(v.size())

                if model.memory_required(input_shape, cond_shapes=cond_shapes) * 1.5 < free_memory:
                    to_batch = batch_amount
                    break

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            uuids = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                mult.append(p.mult)
                c.append(p.conditioning)
                area.append(p.area)
                cond_or_uncond.append(o[1])
                uuids.append(p.uuid)
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            transformer_options = {} 
            if hasattr(model, 'current_patcher') and model.current_patcher is not None:
                transformer_options = model.current_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                transformer_options = ldm_patched.modules.patcher_extension.merge_nested_dicts(transformer_options,
                                                                                 model_options['transformer_options'],
                                                                                 copy_dict1=False)

            if patches is not None:
                # TODO: replace with merge_nested_dicts function
                if "patches" in transformer_options:
                    cur_patches = transformer_options["patches"].copy()
                    for p in patches:
                        if p in cur_patches:
                            cur_patches[p] = cur_patches[p] + patches[p]
                        else:
                            cur_patches[p] = patches[p]
                    transformer_options["patches"] = cur_patches
                else:
                    transformer_options["patches"] = patches

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            transformer_options["uuids"] = uuids[:]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond), transformer_options)

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
            else:
                output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

            for o in range(batch_chunks):
                cond_index = cond_or_uncond[o]
                a = area[o]
                if a is None:
                    out_conds[cond_index] += output[o] * mult[o]
                    out_counts[cond_index] += mult[o]
                else:
                    out_c = out_conds[cond_index]
                    out_cts = out_counts[cond_index]
                    dims = len(a) // 2
                    for i in range(dims):
                        out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                        out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                    out_c += output[o] * mult[o]
                    out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds

def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options): #TODO: remove
    # logging.warning("WARNING: The ldm_patched.modules.samplers.calc_cond_uncond_batch function is deprecated please use the calc_cond_batch one instead.")
    return tuple(calc_cond_batch(model, [cond, uncond], x_in, timestep, model_options))

import math

def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    edit_strength = sum((item.get('strength', 1) for item in cond))

    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred if uncond_pred is not None else None,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif uncond_pred is None:
        cfg_result = cond_pred
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result

#The main sampling function shared by all the samplers
#Returns denoised
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)
        x = args["input"]

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image) * latent_mask
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)

def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(float(s.sigma(ts)))

    if append_zero:
        sigs += [0.0]

    return torch.FloatTensor(sigs)

def get_sigmas_kl_optimal(model_sampling, steps, device='cpu'):
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)

    alpha_min = torch.arctan(torch.tensor(sigma_min))
    alpha_max = torch.arctan(torch.tensor(sigma_max))
    step_indices = torch.arange(steps + 1)
    sigmas = torch.tan(step_indices / steps * alpha_min + (1.0 - step_indices / steps) * alpha_max)
    return sigmas.to(device)

def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(model_sampling.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)

# from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule(model_sampling, steps, threshold_noise=0.025, linear_steps=None):
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * model_sampling.sigma_max.cpu()

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas

def get_sigmas_karras(model_sampling, steps, device='cpu'):
    rho = shared.opts.reforge_karras_rho
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    return k_diffusion_sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)

def get_sigmas_exponential(model_sampling, steps, device='cpu'):
    shrink_factor = shared.opts.reforge_exponential_shrink_factor
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device).exp()
    sigmas = sigmas * torch.exp(shrink_factor * torch.linspace(0, 1, steps, device=device))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_polyexponential(model_sampling, steps, device='cpu'):
    rho = shared.opts.reforge_polyexponential_rho
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    return k_diffusion_sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device)

def get_sigmas_ays_custom(model_sampling, steps, device='cpu'):
    try:
        sigmas_str = shared.opts.reforge_ays_custom_sigmas
        sigmas_values = sigmas_str.strip('[]').split(',')
        sigmas = np.array([float(x.strip()) for x in sigmas_values])
        
        if steps != len(sigmas):
            sigmas = np.interp(np.linspace(0, 1, steps), np.linspace(0, 1, len(sigmas)), sigmas)
        sigmas = np.append(sigmas, [0.0])
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        print(f"Error parsing custom sigmas: {e}")
        print("Falling back to default normal scheduler sigmas")
        return normal_scheduler(model_sampling, steps)

def cosine_scheduler(model_sampling, steps, device='cpu'):
    sf = shared.opts.reforge_cosine_sf_factor
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    sigmas = torch.zeros(steps, device=device)
    if steps == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(steps):
            p = x / (steps-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C * sf
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler(model_sampling, steps, device='cpu'):
    decay = shared.opts.reforge_cosexpblend_exp_decay
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    sigmas = []
    if steps == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = decay ** (1/(steps-1))
        E = sigma_max
        for x in range(steps):
            p = x / (steps-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

def phi_scheduler(model_sampling, steps, device='cpu'):
    power = shared.opts.reforge_phi_power
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    sigmas = torch.zeros(steps, device=device)
    if steps == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(steps):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(steps-1))**(phi**power))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_laplace(model_sampling, steps, device='cpu'):
    mu = shared.opts.reforge_laplace_mu
    beta = shared.opts.reforge_laplace_beta
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, steps, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_karras_dynamic(model_sampling, steps, device='cpu'):
    rho = shared.opts.reforge_karras_dynamic_rho
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(steps):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/steps)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_sinusoidal_sf(model_sampling, steps, device='cpu'):
    sf = shared.opts.reforge_sinusoidal_sf_factor
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_invcosinusoidal_sf(model_sampling, steps, device='cpu'):
    sf = shared.opts.reforge_invcosinusoidal_sf_factor
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_react_cosinusoidal_dynsf(model_sampling, steps, device='cpu'):
    sf = shared.opts.reforge_react_cosinusoidal_dynsf_factor
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    x = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(steps*x/steps))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False): #TODO: handle dim != 2
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified

def resolve_areas_and_cond_masks(conditions, h, w, device):
    logging.warning("WARNING: The ldm_patched.modules.samplers.resolve_areas_and_cond_masks function is deprecated please use the resolve_areas_and_cond_masks_multidim one instead.")
    return resolve_areas_and_cond_masks_multidim(conditions, [h, w], device)

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    def area_inside(a, area_cmp):
        a = add_area_dims(a, len(area_cmp) // 2)
        area_cmp = add_area_dims(area_cmp, len(a) // 2)

        a_l = len(a) // 2
        area_cmp_l = len(area_cmp) // 2
        for i in range(min(a_l, area_cmp_l)):
            if a[a_l + i] < area_cmp[area_cmp_l + i]:
                return False
        for i in range(min(a_l, area_cmp_l)):
            if (a[i] + a[a_l + i]) > (area_cmp[i] + area_cmp[area_cmp_l + i]):
                return False
        return True

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if area_inside(c_area, a):
                if smallest is None:
                    smallest = x
                elif 'area' not in smallest:
                    smallest = x
                else:
                    if math.prod(smallest['area'][:len(smallest['area']) // 2]) > math.prod(a[:len(a) // 2]):
                        smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]

def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        # handle clip hook schedule, if needed
        if 'clip_start_percent' in x:
            timestep_start = s.percent_to_sigma(max(x['clip_start_percent'], x.get('start_percent', 0.0)))
            timestep_end = s.percent_to_sigma(min(x['clip_end_percent'], x.get('end_percent', 1.0)))
        else:
            if 'start_percent' in x:
                timestep_start = s.percent_to_sigma(x['start_percent'])
            if 'end_percent' in x:
                timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n

def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n

def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4: #TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
                  "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"]

class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        for c in conds[k]:
            if 'hooks' in c:
                for hook in c['hooks'].hooks:
                    hook.initialize_timesteps(model)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), conds[k], 'control', lambda cond_cnets, x: cond_cnets[x])
                apply_empty_x_to_equal_area(positive, conds[k], 'gligen', lambda cond_cnets, x: cond_cnets[x])

    return conds

def preprocess_conds_hooks(conds: dict[str, list[dict[str]]]):
    # determine which ControlNets have extra_hooks that should be combined with normal hooks
    hook_replacement: dict[tuple[ControlBase, ldm_patched.hooks.HookGroup], list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            if 'control' in kk:
                control: 'ControlBase' = kk['control']
                extra_hooks = control.get_extra_hooks()
                if len(extra_hooks) > 0:
                    hooks: ldm_patched.hooks.HookGroup = kk.get('hooks', None)
                    to_replace = hook_replacement.setdefault((control, hooks), [])
                    to_replace.append(kk)
    # if nothing to replace, do nothing
    if len(hook_replacement) == 0:
        return

    # for optimal sampling performance, common ControlNets + hook combos should have identical hooks
    # on the cond dicts
    for key, conds_to_modify in hook_replacement.items():
        control = key[0]
        hooks = key[1]
        hooks = ldm_patched.hooks.HookGroup.combine_all_hooks(control.get_extra_hooks() + [hooks])
        # if combined hooks are not None, set as new hooks for all relevant conds
        if hooks is not None:
            for cond in conds_to_modify:
                cond['hooks'] = hooks

def filter_registered_hooks_on_conds(conds: dict[str, list[dict[str]]], model_options: dict[str]):
    '''Modify 'hooks' on conds so that only hooks that were registered remain. Properly accounts for
    HookGroups that have the same reference.'''
    registered: ldm_patched.hooks.HookGroup = model_options.get('registered_hooks', None)
    # if None were registered, make sure all hooks are cleaned from conds
    if registered is None:
        for k in conds:
            for kk in conds[k]:
                kk.pop('hooks', None)
        return
    # find conds that contain hooks to be replaced - group by common HookGroup refs
    hook_replacement: dict[ldm_patched.hooks.HookGroup, list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            hooks: ldm_patched.hooks.HookGroup = kk.get('hooks', None)
            if hooks is not None:
                if not hooks.is_subset_of(registered):
                    to_replace = hook_replacement.setdefault(hooks, [])
                    to_replace.append(kk)
    # for each hook to replace, create a new proper HookGroup and assign to all common conds
    for hooks, conds_to_modify in hook_replacement.items():
        new_hooks = hooks.new_with_common_hooks(registered)
        if len(new_hooks) == 0:
            new_hooks = None
        for kk in conds_to_modify:
            kk['hooks'] = new_hooks


def get_total_hook_groups_in_conds(conds: dict[str, list[dict[str]]]):
    hooks_set = set()
    for k in conds:
        for kk in conds[k]:
            hooks_set.add(kk.get('hooks', None))
    return len(hooks_set)

def cast_to_load_options(model_options: dict[str], device=None, dtype=None):
    '''
    If any patches from hooks, wrappers, or callbacks have .to to be called, call it.
    '''
    if model_options is None:
        return
    to_load_options = model_options.get("to_load_options", None)
    if to_load_options is None:
        return

    casts = []
    if device is not None:
        casts.append(device)
    if dtype is not None:
        casts.append(dtype)
    # if nothing to apply, do nothing
    if len(casts) == 0:
        return

    # try to call .to on patches
    if "patches" in to_load_options:
        patches = to_load_options["patches"]
        for name in patches:
            patch_list = patches[name]
            for i in range(len(patch_list)):
                if hasattr(patch_list[i], "to"):
                    for cast in casts:
                        patch_list[i] = patch_list[i].to(cast)
    if "patches_replace" in to_load_options:
        patches = to_load_options["patches_replace"]
        for name in patches:
            patch_list = patches[name]
            for k in patch_list:
                if hasattr(patch_list[k], "to"):
                    for cast in casts:
                        patch_list[k] = patch_list[k].to(cast)
    # try to call .to on any wrappers/callbacks
    wrappers_and_callbacks = ["wrappers", "callbacks"]
    for wc_name in wrappers_and_callbacks:
        if wc_name in to_load_options:
            wc: dict[str, list] = to_load_options[wc_name]
            for wc_dict in wc.values():
                for wc_list in wc_dict.values():
                    for i in range(len(wc_list)):
                        if hasattr(wc_list[i], "to"):
                            for cast in casts:
                                wc_list[i] = wc_list[i].to(cast)

class CFGGuider:
    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = ldm_patched.modules.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = ldm_patched.modules.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = ldm_patched.modules.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            ldm_patched.modules.patcher_extension.get_all_wrappers(ldm_patched.modules.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.inner_model, self.conds, self.loaded_models = ldm_patched.modules.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = ldm_patched.modules.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            self.model_patcher.cleanup()

        ldm_patched.modules.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = ldm_patched.modules.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = ldm_patched.hooks.EnumHookMode.MinVram
            ldm_patched.modules.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = ldm_patched.modules.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                ldm_patched.modules.patcher_extension.get_all_wrappers(ldm_patched.modules.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds
        return output


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

class SchedulerHandler(NamedTuple):
    handler: Callable[..., torch.Tensor]
    use_ms: bool = True  # If True: handler(model_sampling, steps), else: handler(n, sigma_min, sigma_max)
    needs_sdxl: bool = False  # If True: handler needs is_sdxl parameter

SCHEDULER_HANDLERS = {
    "normal": SchedulerHandler(normal_scheduler),
    "karras": SchedulerHandler(get_sigmas_karras),
    "exponential": SchedulerHandler(get_sigmas_exponential),
    "polyexponential": SchedulerHandler(get_sigmas_polyexponential),
    "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
    "simple": SchedulerHandler(simple_scheduler),
    "ddim_uniform": SchedulerHandler(ddim_scheduler),
    "kl_optimal": SchedulerHandler(get_sigmas_kl_optimal),
    "beta": SchedulerHandler(beta_scheduler),
    "cosine": SchedulerHandler(cosine_scheduler),
    "cosexpblend": SchedulerHandler(cosexpblend_scheduler),
    "phi": SchedulerHandler(phi_scheduler),
    "laplace": SchedulerHandler(get_sigmas_laplace),
    "karras_dynamic": SchedulerHandler(get_sigmas_karras_dynamic),
    "sinusoidal_sf": SchedulerHandler(get_sigmas_sinusoidal_sf),
    "invcosinusoidal_sf": SchedulerHandler(get_sigmas_invcosinusoidal_sf),
    "react_cosinusoidal_dynsf": SchedulerHandler(get_sigmas_react_cosinusoidal_dynsf),
    "ays_custom": SchedulerHandler(get_sigmas_ays_custom),
    "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
    # AYS schedulers that need SDXL parameter
    "ays": SchedulerHandler(k_diffusion_sampling.get_sigmas_ays, use_ms=False, needs_sdxl=True),
    "ays_gits": SchedulerHandler(k_diffusion_sampling.get_sigmas_ays_gits, use_ms=False, needs_sdxl=True),
    "ays_11steps": SchedulerHandler(k_diffusion_sampling.get_sigmas_ays_11steps, use_ms=False, needs_sdxl=True),
    "ays_32steps": SchedulerHandler(k_diffusion_sampling.get_sigmas_ays_32steps, use_ms=False, needs_sdxl=True),
}

SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

def calculate_sigmas(model_sampling: object, scheduler_name: str, steps: int, is_sdxl: bool = False) -> torch.Tensor:
    handler = SCHEDULER_HANDLERS.get(scheduler_name)
    if handler is None:
        err = f"error invalid scheduler {scheduler_name}"
        logging.error(err)
        raise ValueError(err)

    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)

    if handler.use_ms:
        return handler.handler(model_sampling, steps)
    elif handler.needs_sdxl:
        return handler.handler(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, is_sdxl=is_sdxl)
    else:
        return handler.handler(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)

def sampler_object(name):
    if name == "uni_pc":
        sampler = KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
