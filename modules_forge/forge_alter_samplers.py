import torch
from modules import sd_samplers_kdiffusion, sd_samplers_common

from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import calculate_sigmas
from modules import shared


ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}
ALL_SOLVERS = list(ADAPTIVE_SOLVERS | FIXED_SOLVERS)
ALL_SOLVERS.sort()

class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, solver=None, rtol=None, atol=None):
        self.sampler_name = sampler_name
        self.scheduler_name = None
        self.unet = sd_model.forge_objects.unet
        self.model = sd_model
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        sampler_functions = {
            'euler_comfy': k_diffusion_sampling.sample_euler,
            'euler_ancestral_comfy': k_diffusion_sampling.sample_euler_ancestral,
            'heun_comfy': k_diffusion_sampling.sample_heun,
            'dpmpp_2s_ancestral_comfy': k_diffusion_sampling.sample_dpmpp_2s_ancestral,
            'dpmpp_sde_comfy': k_diffusion_sampling.sample_dpmpp_sde,
            'dpmpp_2m_comfy': k_diffusion_sampling.sample_dpmpp_2m,
            'dpmpp_2m_sde_comfy': k_diffusion_sampling.sample_dpmpp_2m_sde,
            'dpmpp_3m_sde_comfy': k_diffusion_sampling.sample_dpmpp_3m_sde,
            'euler_ancestral_turbo': k_diffusion_sampling.sample_euler_ancestral,
            'dpmpp_2m_turbo': k_diffusion_sampling.sample_dpmpp_2m,
            'dpmpp_2m_sde_turbo': k_diffusion_sampling.sample_dpmpp_2m_sde,
            'ddpm': k_diffusion_sampling.sample_ddpm,
            'heunpp2': k_diffusion_sampling.sample_heunpp2,
            'ipndm': k_diffusion_sampling.sample_ipndm,
            'ipndm_v': k_diffusion_sampling.sample_ipndm_v,
            'deis': k_diffusion_sampling.sample_deis,
            'euler_cfg_pp': k_diffusion_sampling.sample_euler_cfg_pp,
            'euler_ancestral_cfg_pp': k_diffusion_sampling.sample_euler_ancestral_cfg_pp,
            'sample_euler_ancestral_RF': k_diffusion_sampling.sample_euler_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp': k_diffusion_sampling.sample_dpmpp_2s_ancestral_cfg_pp,
            'sample_dpmpp_2s_ancestral_RF': k_diffusion_sampling.sample_dpmpp_2s_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp_dyn': k_diffusion_sampling.sample_dpmpp_2s_ancestral_cfg_pp_dyn,
            'dpmpp_2s_ancestral_cfg_pp_intern': k_diffusion_sampling.sample_dpmpp_2s_ancestral_cfg_pp_intern,
            'dpmpp_sde_cfg_pp': k_diffusion_sampling.sample_dpmpp_sde_cfg_pp,
            'dpmpp_2m_cfg_pp': k_diffusion_sampling.sample_dpmpp_2m_cfg_pp,
            'ode_bosh3': self.sample_ode_bosh3,
            'ode_fehlberg2': self.sample_ode_fehlberg2,
            'ode_adaptive_heun': self.sample_ode_adaptive_heun,
            'ode_dopri5': self.sample_ode_dopri5,
            'ode_custom':self.sample_ode_custom,
            'dpmpp_3m_sde_cfg_pp': k_diffusion_sampling.sample_dpmpp_3m_sde_cfg_pp,
            'dpmpp_2m_dy': k_diffusion_sampling.sample_dpmpp_2m_dy,
            'dpmpp_3m_dy': k_diffusion_sampling.sample_dpmpp_3m_dy,
            'dpmpp_3m_sde_dy': k_diffusion_sampling.sample_dpmpp_3m_sde_dy,
            'euler_dy_cfg_pp': k_diffusion_sampling.sample_euler_dy_cfg_pp,
            'euler_smea_dy_cfg_pp': k_diffusion_sampling.sample_euler_smea_dy_cfg_pp,
            'euler_ancestral_dy_cfg_pp': k_diffusion_sampling.sample_euler_ancestral_dy_cfg_pp,
            'dpmpp_2m_dy_cfg_pp': k_diffusion_sampling.sample_dpmpp_2m_dy_cfg_pp,
            'clyb_4m_sde_momentumized': k_diffusion_sampling.sample_clyb_4m_sde_momentumized,
            'res_solver': k_diffusion_sampling.sample_res_solver,
            'kohaku_lonyu_yog_cfg_pp': k_diffusion_sampling.sample_kohaku_lonyu_yog_cfg_pp,
            'custom_sampler': k_diffusion_sampling.sample_custom,
            'res_multistep' : k_diffusion_sampling.sample_res_multistep,
            'res_multistep_cfg_pp' : k_diffusion_sampling.sample_res_multistep_cfg_pp,
            'gradient_estimation' : k_diffusion_sampling.sample_gradient_estimation,
            'Kohaku_LoNyu_Yog' : k_diffusion_sampling.sample_Kohaku_LoNyu_Yog,
            'ER SDE': k_diffusion_sampling.sample_er_sde,
        }
        
        sampler_function = sampler_functions.get(sampler_name)
        if sampler_function is None:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        super().__init__(sampler_function, sd_model, None)

    def sample_func(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        if self.sampler_name == 'ode_bosh3':
            return self.sample_ode_bosh3(model, x, sigmas, extra_args, callback, disable)
        elif self.sampler_name == 'ode_fehlberg2':
            return self.sample_ode_fehlberg2(model, x, sigmas, extra_args, callback, disable)
        elif self.sampler_name == 'ode_adaptive_heun':
            return self.sample_ode_adaptive_heun(model, x, sigmas, extra_args, callback, disable)
        elif self.sampler_name == 'ode_dopri5':
            return self.sample_ode_dopri5(model, x, sigmas, extra_args, callback, disable)
        elif self.sampler_name == 'ode_custom':
            return self.sample_ode_custom(model, x, sigmas, extra_args, callback, disable)
        else:
            # For non-ODE samplers, use the original sampler function
            return super().sample_func(model, x, sigmas, extra_args, callback, disable)

    def sample_ode_bosh3(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="bosh3", 
                                               rtol=10**shared.opts.ode_bosh3_rtol, 
                                               atol=10**shared.opts.ode_bosh3_atol, 
                                               max_steps=shared.opts.ode_bosh3_max_steps)

    def sample_ode_fehlberg2(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="fehlberg2", 
                                               rtol=10**shared.opts.ode_fehlberg2_rtol, 
                                               atol=10**shared.opts.ode_fehlberg2_atol, 
                                               max_steps=shared.opts.ode_fehlberg2_max_steps)

    def sample_ode_adaptive_heun(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="adaptive_heun", 
                                               rtol=10**shared.opts.ode_adaptive_heun_rtol, 
                                               atol=10**shared.opts.ode_adaptive_heun_atol, 
                                               max_steps=shared.opts.ode_adaptive_heun_max_steps)

    def sample_ode_dopri5(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="dopri5", 
                                               rtol=10**shared.opts.ode_dopri5_rtol, 
                                               atol=10**shared.opts.ode_dopri5_atol, 
                                               max_steps=shared.opts.ode_dopri5_max_steps)

    def sample_ode_custom(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        solver = shared.opts.ode_custom_solver
        rtol = 10**shared.opts.ode_custom_rtol if solver in ADAPTIVE_SOLVERS else None
        atol = 10**shared.opts.ode_custom_atol if solver in ADAPTIVE_SOLVERS else None
        max_steps = shared.opts.ode_custom_max_steps
        
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver=solver, rtol=rtol, atol=atol, max_steps=max_steps)
    
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        self.scheduler_name = p.scheduler
        return super().sample(p, x, conditioning, unconditional_conditioning, steps, image_conditioning)

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        self.scheduler_name = p.scheduler
        return super().sample_img2img(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)

    def get_sigmas(self, p, steps):
        if self.scheduler_name is None:
            self.scheduler_name = 'Normal'  # Default to 'Normal' if not set

        forge_schedulers = {
            "Normal": "normal",
            "Karras": "karras",
            "Exponential": "exponential",
            "SGM Uniform": "sgm_uniform",
            "Simple": "simple",
            "DDIM": "ddim_uniform",
            "Align Your Steps": "ays",
            "Align Your Steps GITS": "ays_gits",
            "Align Your Steps 11": "ays_11steps",
            "Align Your Steps 32": "ays_32steps",
            "KL Optimal": "kl_optimal",
            "Beta": "beta",
            "Sinusoidal SF": "sinusoidal_sf",
            "Invcosinusoidal SF": "invcosinusoidal_sf",
            "React Cosinusoidal DynSF": "react_cosinusoidal_dynsf",
            "Cosine": "cosine",
            "Cosine-exponential Blend": "cosexpblend",
            "Phi": "phi",
            "Laplace": "laplace",
            "Karras Dynamic": "karras_dynamic",
            "Uniform": "uniform",
            "Polyexponential": "polyexponential",
            "Turbo": "turbo",
            "Align Your Steps Custom": "ays_custom",
        }
        
        use_turbo = self.sampler_name.endswith('_turbo') or self.scheduler_name == "Turbo"
        
        if self.scheduler_name in forge_schedulers:
            matched_scheduler = forge_schedulers[self.scheduler_name]
        else:
            # Default to 'normal' if the selected scheduler is not available in forge_alter
            matched_scheduler = 'normal'

        try:
            if use_turbo:
                # Use Turbo scheduler
                timesteps = torch.flip(torch.arange(1, steps + 1) * float(1000.0 / steps) - 1, (0,)).round().long().clip(0, 999)
                sigmas = self.unet.model.model_sampling.sigma(timesteps)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
            else:
                sigmas = calculate_sigmas(self.unet.model.model_sampling, matched_scheduler, steps, is_sdxl=getattr(self.model, "is_sdxl", False))
        except Exception as e:
            print(f"Error calculating sigmas for scheduler {matched_scheduler}: {str(e)}")
            print("Falling back to normal scheduler")
            sigmas = calculate_sigmas(self.unet.model.model_sampling, "normal", steps, is_sdxl=getattr(self.model, "is_sdxl", False))

        if sigmas is None:
            raise ValueError(f"Invalid scheduler: {self.scheduler_name}")

        return sigmas.to(self.unet.load_device)


def build_constructor(sampler_name):
    def constructor(model):
        return AlterSampler(model, sampler_name)
    return constructor

samplers_data_alter = [
    sd_samplers_common.SamplerData('ER SDE', build_constructor(sampler_name='ER SDE'), ['ER SDE'], {}),
    sd_samplers_common.SamplerData('Kohaku_LoNyu_Yog', build_constructor(sampler_name='Kohaku_LoNyu_Yog'), ['Kohaku_LoNyu_Yog'], {}),
    sd_samplers_common.SamplerData('Euler CFG++', build_constructor(sampler_name='euler_cfg_pp'), ['euler_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Euler Ancestral CFG++', build_constructor(sampler_name='euler_ancestral_cfg_pp'), ['euler_ancestral_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral CFG++', build_constructor(sampler_name='dpmpp_2s_ancestral_cfg_pp'), ['dpmpp_2s_ancestral_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M CFG++', build_constructor(sampler_name='dpmpp_2m_cfg_pp'), ['dpmpp_2m_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ SDE CFG++', build_constructor(sampler_name='dpmpp_sde_cfg_pp'), ['dpmpp_sde_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 3M SDE CFG++', build_constructor(sampler_name='dpmpp_3m_sde_cfg_pp'), ['dpmpp_3m_sde_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Euler DY CFG++', build_constructor(sampler_name='euler_dy_cfg_pp'), ['euler_dy_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Euler SMEA DY CFG++', build_constructor(sampler_name='euler_smea_dy_cfg_pp'), ['euler_smea_dy_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Euler A DY CFG++', build_constructor(sampler_name='euler_ancestral_dy_cfg_pp'), ['euler_ancestral_dy_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M DY CFG++', build_constructor(sampler_name='dpmpp_2m_dy_cfg_pp'), ['dpmpp_2m_dy_cfg_pp'], {}),
    sd_samplers_common.SamplerData('RES Multistep', build_constructor(sampler_name='res_multistep'), ['res_multistep'], {}),
    sd_samplers_common.SamplerData('RES Multistep CFG++', build_constructor(sampler_name='res_multistep_cfg_pp'), ['res_multistep_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Gradient Estimation', build_constructor(sampler_name='gradient_estimation'), ['gradient_estimation'], {}),
    # sd_samplers_common.SamplerData('Kohaku_LoNyu_Yog CFG++', build_constructor(sampler_name='kohaku_lonyu_yog_cfg_pp'), ['kohaku_lonyu_yog_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M DY', build_constructor(sampler_name='dpmpp_2m_dy'), ['dpmpp_2m_dy'], {}),
    sd_samplers_common.SamplerData('DPM++ 3M DY', build_constructor(sampler_name='dpmpp_3m_dy'), ['dpmpp_3m_dy'], {}),
    sd_samplers_common.SamplerData('DPM++ 3M SDE DY', build_constructor(sampler_name='dpmpp_3m_sde_dy'), ['dpmpp_3m_sde_dy'], {}),
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm'), ['ddpm'], {}),
    sd_samplers_common.SamplerData('HeunPP2', build_constructor(sampler_name='heunpp2'), ['heunpp2'], {}),
    sd_samplers_common.SamplerData('IPNDM', build_constructor(sampler_name='ipndm'), ['ipndm'], {}),
    sd_samplers_common.SamplerData('IPNDM_V', build_constructor(sampler_name='ipndm_v'), ['ipndm_v'], {}),
    sd_samplers_common.SamplerData('DEIS', build_constructor(sampler_name='deis'), ['deis'], {}),
    sd_samplers_common.SamplerData('RES Solver', build_constructor(sampler_name='res_solver'), ['res_solver'], {}),
    sd_samplers_common.SamplerData('ODE (Bosh3)', build_constructor(sampler_name='ode_bosh3'), ['ode_bosh3'], {}),
    sd_samplers_common.SamplerData('ODE (Fehlberg2)', build_constructor(sampler_name='ode_fehlberg2'), ['ode_fehlberg2'], {}),
    sd_samplers_common.SamplerData('ODE (Adaptive Heun)', build_constructor(sampler_name='ode_adaptive_heun'), ['ode_adaptive_heun'], {}),
    sd_samplers_common.SamplerData('ODE (Dopri5)', build_constructor(sampler_name='ode_dopri5'), ['ode_dopri5'], {}),
    sd_samplers_common.SamplerData('ODE Custom', build_constructor(sampler_name='ode_custom'), ['ode_custom'], {}),
    sd_samplers_common.SamplerData('Euler A Turbo', build_constructor(sampler_name='euler_ancestral_turbo'), ['euler_ancestral_turbo'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M Turbo', build_constructor(sampler_name='dpmpp_2m_turbo'), ['dpmpp_2m_turbo'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE Turbo', build_constructor(sampler_name='dpmpp_2m_sde_turbo'), ['dpmpp_2m_sde_turbo'], {}),
    sd_samplers_common.SamplerData('Euler Comfy', build_constructor(sampler_name='euler_comfy'), ['euler_comfy'], {}),
    sd_samplers_common.SamplerData('Euler A Comfy', build_constructor(sampler_name='euler_ancestral_comfy'), ['euler_ancestral_comfy'], {}),
    sd_samplers_common.SamplerData('Heun Comfy', build_constructor(sampler_name='heun_comfy'), ['heun_comfy'], {}),
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral Comfy', build_constructor(sampler_name='dpmpp_2s_ancestral_comfy'), ['dpmpp_2s_ancestral_comfy'], {}),
    sd_samplers_common.SamplerData('DPM++ SDE Comfy', build_constructor(sampler_name='dpmpp_sde_comfy'), ['dpmpp_sde_comfy'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M Comfy', build_constructor(sampler_name='dpmpp_2m_comfy'), ['dpmpp_2m_comfy'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE Comfy', build_constructor(sampler_name='dpmpp_2m_sde_comfy'), ['dpmpp_2m_sde_comfy'], {}),
    sd_samplers_common.SamplerData('DPM++ 3M SDE Comfy', build_constructor(sampler_name='dpmpp_3m_sde_comfy'), ['dpmpp_3m_sde_comfy'], {}),
    # sd_samplers_common.SamplerData('CLYB 4M SDE Momentumized', build_constructor(sampler_name='clyb_4m_sde_momentumized'), ['clyb_4m_sde_momentumized'], {}), 
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral CFG++ Dyn', build_constructor(sampler_name='dpmpp_2s_ancestral_cfg_pp_dyn'), ['dpmpp_2s_ancestral_cfg_pp_dyn'], {}),
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral CFG++ Intern', build_constructor(sampler_name='dpmpp_2s_ancestral_cfg_pp_intern'), ['dpmpp_2s_ancestral_cfg_pp_intern'], {}),
    sd_samplers_common.SamplerData('Custom Sampler', build_constructor(sampler_name='custom_sampler'), ['custom_sampler'], {}),
    #sd_samplers_common.SamplerData('Euler A RF', build_constructor(sampler_name='sample_euler_ancestral_RF'), ['sample_euler_ancestral_RF'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2S Ancestral RF', build_constructor(sampler_name='sample_dpmpp_2s_ancestral_RF'), ['sample_dpmpp_2s_ancestral_RF'], {}),
]
