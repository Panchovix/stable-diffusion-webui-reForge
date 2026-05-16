# Original code from Comfy, https://github.com/comfyanonymous/ComfyUI



import functools
import math
from functools import partial

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm

from ldm_patched.modules import utils
from ldm_patched.k_diffusion import deis
from ldm_patched.k_diffusion import sa_solver
import ldm_patched.modules.model_patcher
import ldm_patched.modules.model_sampling
import torchdiffeq
import modules.shared
from torch import no_grad, FloatTensor
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple, List
from itertools import pairwise
from ldm_patched.modules.model_sampling import CONST
import modules.shared as shared
import numpy as np

from modules.sd_samplers_kdiffusion_smea import Rescaler

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)

# align your steps
def get_sigmas_ays(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))

        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys

    if is_sdxl:
        sigmas = [sigma_max, sigma_max/2.314, sigma_max/3.875, sigma_max/6.701, sigma_max/10.89, sigma_max/16.954, sigma_max/26.333, sigma_max/38.46, sigma_max/62.457, sigma_max/129.336, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [sigma_max, sigma_max/2.257, sigma_max/3.785, sigma_max/5.418, sigma_max/7.749, sigma_max/10.469, sigma_max/15.176, sigma_max/22.415, sigma_max/36.629, sigma_max/96.151, 0.029]


    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def get_sigmas_ays_gits(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))
        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys

    if is_sdxl:
        sigmas = [sigma_max, sigma_max/3.087, sigma_max/5.693, sigma_max/9.558, sigma_max/14.807, sigma_max/22.415, sigma_max/34.964, sigma_max/54.533, sigma_max/81.648, sigma_max/115.078, 0.029]

    else:
        sigmas = [sigma_max, sigma_max/3.165, sigma_max/5.829, sigma_max/11.824, sigma_max/20.819, sigma_max/36.355, sigma_max/60.895, sigma_max/93.685, sigma_max/140.528, sigma_max/155.478, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def get_sigmas_ays_11steps(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    # This is the same as the original AYS
    return get_sigmas_ays(n, sigma_min, sigma_max, is_sdxl, device)

def get_sigmas_ays_32steps(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))
        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys
    
    if is_sdxl:
        sigmas = [sigma_max, sigma_max/1.310860875657935, sigma_max/1.718356235075352, sigma_max/2.252525958180810, sigma_max/2.688026675053433, sigma_max/3.174423075322040, sigma_max/3.748832539417044, sigma_max/4.463856789920335, sigma_max/5.326233593328242, sigma_max/6.355213820679800, sigma_max/7.477672611007930, sigma_max/8.745803592589411, sigma_max/10.228995682978878, sigma_max/11.864653584709637, sigma_max/13.685783347784952, sigma_max/15.786441921021279, sigma_max/18.202564111697559, sigma_max/20.980440157432400, sigma_max/24.182245076323649, sigma_max/27.652401723193991, sigma_max/31.246429590323925, sigma_max/35.307579021272943, sigma_max/40.308138967569972, sigma_max/47.132212095147923, sigma_max/55.111585405517003, sigma_max/65.460441760115945, sigma_max/82.786347724072168, sigma_max/104.698036963744033, sigma_max/138.041693219503482, sigma_max/264.794761864988552, sigma_max/507.935470821253285, 0.015000000000000000]
    else:
        sigmas = [sigma_max, sigma_max/1.300323183382763, sigma_max/1.690840379611262, sigma_max/2.198638945761486, sigma_max/2.622696705671493, sigma_max/3.098705619671305, sigma_max/3.661108232617473, sigma_max/4.152506637972936, sigma_max/4.662023756728857, sigma_max/5.234059175875519, sigma_max/5.874818853387466, sigma_max/6.593316416277412, sigma_max/7.399687115002039, sigma_max/8.213824943635682, sigma_max/9.050917900247738, sigma_max/9.973321246245751, sigma_max/11.115344803852001, sigma_max/12.529738625194212, sigma_max/14.124109921351757, sigma_max/15.959814856974724, sigma_max/18.099481611774999, sigma_max/20.526004748634670, sigma_max/23.506648288108032, sigma_max/27.541589307433523, sigma_max/32.269132736422456, sigma_max/38.982216080970984, sigma_max/53.219344283057142, sigma_max/72.656173487928834, sigma_max/103.609326413189740, sigma_max/218.693105563304210, sigma_max/461.605857767280530, 0.015000000000000000]
        
    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)
    
    return torch.FloatTensor(sigmas).to(device)

def cosine_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = (sigma_min / sigma_max)**(1/(n-1))
        E = sigma_max
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

def phi_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(n-1))**(phi*phi))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas

def get_sigmas_karras_dynamic(n, sigma_min, sigma_max, device='cpu'):
    rho = 7.
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/n)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_sinusoidal_sf(n, sigma_min, sigma_max, sf=3.5, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_invcosinusoidal_sf(n, sigma_min, sigma_max, sf=3.5, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_react_cosinusoidal_dynsf(n, sigma_min, sigma_max, sf=2.15, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(n*x/n))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)

def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=None):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    eta = eta if eta is not None else shared.opts.ancestral_eta
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x, seed=None):
    if seed is not None:
        # ComfyUI path (see ldm_patched/modules/samplers.py:1163): an explicit
        # seed is supplied via extra_args. Build a fresh torch.Generator and
        # use torch.randn with it. This branch is byte-identical to the
        # previous implementation so the Comfy sampling path is untouched.
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

    # WebUI path: no explicit seed. Route through torch.randn_like so that
    # modules/sd_samplers_common.py::TorchHijack (installed on this module's
    # `torch` attribute in Sampler.initialize) can intercept the call and
    # return seeded noise from p.rng.next(). Regular WebUI k-diffusion
    # samplers have used this hijack-based reproducibility mechanism on the
    # A1111 backend for a long time via k_diff.k_diffusion.sampling, which
    # already uses torch.randn_like. Switching this branch to torch.randn_like
    # extends the same reproducibility path to AlterSamplers, which always
    # route through ldm_patched.k_diffusion.sampling regardless of backend.
    return lambda sigma, sigma_next: torch.randn_like(x)

def ei_h_phi_1(h: torch.Tensor) -> torch.Tensor:
    """Compute the result of h*phi_1(h) in exponential integrator methods."""
    return torch.expm1(h)


def ei_h_phi_2(h: torch.Tensor) -> torch.Tensor:
    """Compute the result of h*phi_2(h) in exponential integrator methods."""
    return (torch.expm1(h) - h) / h

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}
ALL_SOLVERS = list(ADAPTIVE_SOLVERS | FIXED_SOLVERS)
ALL_SOLVERS.sort()
class ODEFunction:
    def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.t_min = t_min.item()
        self.t_max = t_max.item()
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.step = 0

        if is_adaptive:
            self.pbar = tqdm(
                total=100,
                desc="solve",
                unit="%",
                leave=False,
                position=1
            )
        else:
            self.pbar = tqdm(
                total=n_steps,
                desc="solve",
                leave=False,
                position=1
            )

    def __call__(self, t, y):
        if t <= 1e-5:
            return torch.zeros_like(y)

        denoised = self.model(y.unsqueeze(0), t.unsqueeze(0), **self.extra_args)
        return (y - denoised.squeeze(0)) / t

    def _callback(self, t0, y0, step):
        if self.callback is not None:
            y0 = y0.unsqueeze(0)

            self.callback({
                "x": y0,
                "i": step,
                "sigma": t0,
                "sigma_hat": t0,
                "denoised": y0, # for a bad latent preview
            })

    def callback_step(self, t0, y0, dt):
        if self.is_adaptive:
            return

        self._callback(t0, y0, self.step)

        self.pbar.update(1)
        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        if not self.is_adaptive:
            return

        progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)

        self._callback(t0, y0, round((self.n_steps - 1) * progress))

        new_step = round(100 * progress)
        self.pbar.update(new_step - self.step)
        self.step = new_step

    def reset(self):
        self.step = 0
        self.pbar.reset()

class ODESampler:
    def __init__(self, solver, rtol, atol, max_steps):
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        t_max = sigmas.max()
        t_min = sigmas.min()
        n_steps = len(sigmas)

        if self.solver in FIXED_SOLVERS:
            t = sigmas
            is_adaptive = False
        else:
            t = torch.stack([t_max, t_min])
            is_adaptive = True

        ode = ODEFunction(model, t_min, t_max, n_steps, is_adaptive=is_adaptive, callback=callback, extra_args=extra_args)

        samples = torch.empty_like(x)
        for i in trange(x.shape[0], desc=self.solver, disable=disable):
            ode.reset()

            samples[i] = torchdiffeq.odeint(
                ode,
                x[i],
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
                options={
                    "min_step": 1e-5,
                    "max_num_steps": self.max_steps,
                    "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float64
                }
            )[-1]

        if callback is not None:
            callback({
                "x": samples,
                "i": n_steps - 1,
                "sigma": t_min,
                "sigma_hat": t_min,
                "denoised": samples, # only accurate if t_min = 0, for now
            })

        return samples


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
    
def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)."""
    if isinstance(model_sampling, ldm_patched.modules.model_sampling.CONST):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()


def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma."""
    if isinstance(model_sampling, ldm_patched.modules.model_sampling.CONST):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR."""
    if len(sigmas) <= 1:
        return sigmas
    if isinstance(model_sampling, ldm_patched.modules.model_sampling.CONST):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    s_churn = modules.shared.opts.euler_og_s_churn
    s_tmin = modules.shared.opts.euler_og_s_tmin
    s_noise = modules.shared.opts.euler_og_s_noise
    s_tmax = float('inf')

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    if hasattr(model, 'model_sampling') and isinstance(model.model_sampling, CONST):
        return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with Euler method steps."""
    eta = modules.shared.opts.euler_ancestral_og_eta
    s_noise = modules.shared.opts.euler_ancestral_og_s_noise

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5
            # Euler method
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x

@torch.no_grad()
def sample_euler_a2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Euler ancestral sampler that averages two noise paths and extrapolates along their mean direction.
    Assumes Rectified Flow parameterization (alpha=1-sigma, sigma in [0,1]). For FLUX/SD3 (CONST model type)."""
    extra_args = {} if extra_args is None else extra_args
    eta = extra_args.get("eta", modules.shared.opts.euler_a2_eta)
    s_noise = extra_args.get("s_noise", modules.shared.opts.euler_a2_s_noise)
    extrapolation = extra_args.get("extrapolation", modules.shared.opts.euler_a2_extrapolation)

    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down

        sigma_down_i_ratio = sigma_down / sigmas[i]
        deterministic_path = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised

        if eta > 0 and s_noise != 0:
            base = (alpha_ip1 / alpha_down) * deterministic_path
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2).clamp_min(0).sqrt()
            noise_scale = s_noise * renoise_coeff

            noise_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            noise_2 = noise_sampler(sigmas[i], sigmas[i + 1])

            path_1 = base + noise_1 * noise_scale
            path_2 = base + noise_2 * noise_scale
            merged = 0.5 * (path_1 + path_2)
            direction = merged - base
            x = merged + extrapolation * direction
        else:
            x = deterministic_path
    return x

@torch.no_grad()
def sample_euler_a2_edm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Euler A2 for EDM/Karras models (SDXL, SD1.5): x_t = x0 + sigma*noise, no alpha term.
    Uses get_ancestral_step for correct sigma_down/sigma_up budget, then applies dual-path averaging + extrapolation."""
    extra_args = {} if extra_args is None else extra_args
    eta = extra_args.get("eta", modules.shared.opts.euler_a2_eta)
    s_noise = extra_args.get("s_noise", modules.shared.opts.euler_a2_s_noise)
    extrapolation = extra_args.get("extrapolation", modules.shared.opts.euler_a2_extrapolation)

    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if sigma_down == 0:
            x = denoised
            continue

        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        deterministic_path = x + d * dt

        if eta > 0 and s_noise != 0 and sigma_up > 0:
            noise_scale = s_noise * sigma_up
            noise_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            noise_2 = noise_sampler(sigmas[i], sigmas[i + 1])

            path_1 = deterministic_path + noise_1 * noise_scale
            path_2 = deterministic_path + noise_2 * noise_scale
            merged = 0.5 * (path_1 + path_2)
            direction = merged - deterministic_path
            x = merged + extrapolation * direction
        else:
            x = deterministic_path
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigma_down - sigmas[i]
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_churn = modules.shared.opts.heun_og_s_churn
    s_tmin = modules.shared.opts.heun_og_s_tmin
    s_noise = modules.shared.opts.heun_og_s_noise
    s_tmax = float('inf')

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    s_noise = modules.shared.opts.dpm2_ancestral_s_noise if s_noise is None else s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpm_2_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
        sigma_down = sigmas[i+1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i+1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=None, noise_sampler=None):
        s_noise = modules.shared.opts.dpm_fast_s_noise if s_noise is None else s_noise
        noise_sampler = default_noise_sampler(x, seed=self.extra_args.get("seed", None)) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, 
                       pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., 
                       s_noise=None, noise_sampler=None):
        s_noise = modules.shared.opts.dpm_adaptive_s_noise if s_noise is None else s_noise
        noise_sampler = default_noise_sampler(x, seed=self.extra_args.get("seed", None)) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    eta = modules.shared.opts.dpm_2s_ancestral_og_eta
    s_noise = modules.shared.opts.dpm_2s_ancestral_og_s_noise
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

def sample_dpmpp_2s_a_sure(model, x, sigmas, extra_args=None, callback=None, disable=None,
                             noise_sampler=None, eta=1., s_noise=1.,
                             sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                             sure_preheat_steps=-1, sure_jac_interval=-1,
                             sure_adam_mode='none', sure_adam_beta1=0.9,
                             sure_adam_beta2=0.999, sure_adam_wd=0.01,
                             sure_grad_mode='vjp',
                             sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(2S) Ancestral with SURE trajectory correction.

    SURE corrects x̂₀ on the first (main) model call at each step.
    The midpoint call (second evaluation within the 2S step) uses the plain
    model — it operates at an intermediate sigma where SURE would be unreliable.
    Ancestral noise is added after each step via get_ancestral_step, matching
    the paper's stochastic reverse process.

    sure_preheat_steps >= 0 : fixed plain-denoise steps before SURE
    sure_preheat_steps = -1  : auto (15% of n_steps, min 2)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    n_steps = len(sigmas) - 1
    preheat = sure_preheat_steps if sure_preheat_steps >= 0 \
              else max(2, math.ceil(0.15 * n_steps))

    _dyn_jac_interval: int        = 2 if sure_jac_interval < 1 else sure_jac_interval
    _jac_ratio_ema:   float | None = None
    _corr_count: int               = 0
    _EMA_A = 0.35
    _adam_state      = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state  = {} if sure_alpha_bo_trials > 0 else None

    _sure_logger.info(
        "DPM++2Sa-SURE: %d steps  preheat=%d  eta=%.2f  alpha=%.4f  adam=%s  alpha_bo=%d",
        n_steps, preheat, eta, sure_alpha, sure_adam_mode, sure_alpha_bo_trials,
    )

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        _tag       = f" step={i+1}/{n_steps} sigma={float(sigma):.4f}"

        sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=eta)

        # ── x̂₀ at current state ──────────────────────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        if i < preheat:
            denoised = x0_hat
            _sure_logger.info("[preheat%s]", _tag)
        else:
            # Algorithm 1: correct x̂₀ in x0-space at estimated residual noise σ̂₀
            sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
            _use_jac = (_dyn_jac_interval <= 1) or (_corr_count % _dyn_jac_interval == 0)
            denoised, _stats = _sure_correct_x0(
                model, x0_hat, sigma_hat_0, s_in, extra_args,
                alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                use_jac=_use_jac, sigma_t=sigma,
                adam_state=_adam_state, adam_mode=sure_adam_mode,
                adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                alpha_bo_trials=sure_alpha_bo_trials,
                alpha_bo_patience=sure_alpha_bo_patience,
                alpha_bo_state=_alpha_bo_state,
            )
            _corr_count += 1
            _jac_ratio_new = _stats.get('jac_ratio')
            if _jac_ratio_new is not None:
                _jac_ratio_ema = _jac_ratio_new if _jac_ratio_ema is None else \
                    (1.0 - _EMA_A) * _jac_ratio_ema + _EMA_A * _jac_ratio_new
            if _jac_ratio_ema is not None and _corr_count >= 3:
                if _jac_ratio_ema < 0.05 and _dyn_jac_interval < 8:
                    _dyn_jac_interval += 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f < 0.05)",
                                      _dyn_jac_interval, _jac_ratio_ema)
                elif _jac_ratio_ema > 0.25 and _dyn_jac_interval > 1:
                    _dyn_jac_interval -= 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f > 0.25)",
                                      _dyn_jac_interval, _jac_ratio_ema)
            _sure_logger.info("[adapt] jac_ratio_ema=%s  dyn_jac_interval=%d",
                              f"{_jac_ratio_ema:.3f}" if _jac_ratio_ema is not None else "n/a",
                              _dyn_jac_interval)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        # ── DPM-Solver++(2S) update ───────────────────────────────────────
        if float(sigma_down) == 0:
            # Final step: plain Euler to zero
            d = to_d(x, sigma, denoised)
            x = x + d * (sigma_down - sigma)
        else:
            t, t_next = t_fn(sigma), t_fn(sigma_down)
            r = 0.5
            h = t_next - t
            s_mid = t + r * h
            # Predictor: step to midpoint using SURE-corrected denoised
            x_2 = sigma_fn(s_mid) / sigma_fn(t) * x - (-h * r).expm1() * denoised
            # Corrector: plain model at midpoint sigma (intermediate point, not corrected)
            with torch.no_grad():
                denoised_2 = model(x_2, sigma_fn(s_mid) * s_in, **extra_args).detach()
            # Full step using midpoint denoised
            x = sigma_fn(t_next) / sigma_fn(t) * x - (-h).expm1() * denoised_2

        # ── Ancestral noise ───────────────────────────────────────────────
        if float(sigma_next) > 0:
            x = x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def sample_dpmpp_2s_a_sure_adaptive(model, x, sigma_min, sigma_max,
                                     extra_args=None, callback=None, disable=None,
                                     noise_sampler=None,
                                     rtol=0.05, atol=0.0078, h_init=0.05,
                                     pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81,
                                     eta=1., s_noise=1.,
                                     sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                                     sure_preheat_frac=0.3, sure_jac_interval=2,
                                     sure_adam_mode='none', sure_adam_beta1=0.9,
                                     sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                                     sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(2S) Ancestral + SURE with adaptive step size (PID).

    Error estimate: compare 1st-order Euler (x_low) vs 2S midpoint (x_high),
    both on the ODE part to sigma_down. Ancestral noise is added AFTER
    acceptance so the PID reacts to curvature, not noise variance.

    eta=1.0 — full ancestral SDE (paper intent)
    eta=0.0 — collapse to deterministic ODE
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    t_start   = t_fn(torch.tensor(sigma_max, dtype=x.dtype, device=x.device))
    t_end     = t_fn(torch.tensor(sigma_min, dtype=x.dtype, device=x.device))
    t_preheat = t_start + sure_preheat_frac * (t_end - t_start)

    pid    = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, order=2,
                                    accept_safety=accept_safety)
    atol_t = torch.tensor(atol, dtype=x.dtype, device=x.device)
    rtol_t = torch.tensor(rtol, dtype=x.dtype, device=x.device)

    _corr_count = 0
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None
    info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}
    x_prev = x.clone()
    s = t_start.clone()

    _sure_logger.info(
        "DPM++2Sa-SURE-Adaptive: sigma [%.4f → %.4f]  preheat_frac=%.2f"
        "  eta=%.2f  alpha=%.4f  jac_interval=%d  adam=%s",
        sigma_max, sigma_min, sure_preheat_frac, eta, sure_alpha, sure_jac_interval,
        sure_adam_mode,
    )

    with tqdm(disable=disable) as pbar:
        while s < t_end - 1e-5:
            t_next     = torch.minimum(t_end, s + pid.h)
            sigma_s    = sigma_fn(s)
            sigma_next = sigma_fn(t_next)

            # Split proposed step into ODE part (sigma_down) + noise amplitude (sigma_up)
            sigma_down, sigma_up = get_ancestral_step(sigma_s, sigma_next, eta=eta)
            t_down = t_fn(sigma_down)
            h      = t_down - s          # ODE step size in t-space

            in_preheat = float(s) < float(t_preheat)
            _tag = f" [adap-2sa] sigma={float(sigma_s):.4f}"

            # ── x̂₀ (SURE after preheat) ──────────────────────────────────
            if in_preheat:
                with torch.no_grad():
                    x0_s = model(x, sigma_s * s_in, **extra_args).detach()
            else:
                with torch.no_grad():
                    x0_hat = model(x, sigma_s * s_in, **extra_args).detach()
                sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
                _use_jac = (sure_jac_interval <= 1) or (_corr_count % sure_jac_interval == 0)
                x0_s, _ = _sure_correct_x0(
                    model, x0_hat, sigma_hat_0, s_in, extra_args,
                    alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                    use_jac=_use_jac, sigma_t=sigma_s,
                    adam_state=_adam_state, adam_mode=sure_adam_mode,
                    adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                    adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                    alpha_bo_trials=sure_alpha_bo_trials,
                    alpha_bo_patience=sure_alpha_bo_patience,
                    alpha_bo_state=_alpha_bo_state,
                )
                _corr_count += 1

            # ── 1st-order ODE step to sigma_down (x_low, error estimate) ──
            eps_s = (x - x0_s) / sigma_s
            x_low = x - sigma_down * h.expm1() * eps_s

            # ── 2S midpoint step to sigma_down (x_high, error estimate) ───
            r     = 0.5
            s_mid = s + r * h
            sigma_mid = sigma_fn(s_mid)
            x_2   = sigma_mid / sigma_s * x - (-h * r).expm1() * x0_s
            with torch.no_grad():
                x0_mid = model(x_2, sigma_mid * s_in, **extra_args).detach()
            x_high = sigma_down / sigma_s * x - (-h).expm1() * x0_mid

            # ── PID error (ODE part only, noise excluded) ──────────────────
            delta  = torch.maximum(atol_t, rtol_t * torch.maximum(x_low.abs(), x_prev.abs()))
            error  = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)

            if accept:
                x_prev = x_low
                x = x_high
                # Ancestral noise after acceptance
                if eta > 0 and s_noise > 0 and float(sigma_next) > 0:
                    x = x + noise_sampler(sigma_s, sigma_next) * s_noise * sigma_up
                s = t_next
                info['n_accept'] += 1
                pbar.update()
                if callback is not None:
                    callback({'x': x, 'i': info['steps'], 'sigma': sigma_s,
                              'sigma_hat': sigma_s, 'denoised': x0_s,
                              'error': error, 'h': pid.h, **info})
            else:
                info['n_reject'] += 1

            info['nfe']   += 2
            info['steps'] += 1

            _sure_logger.info(
                "[adap-2sa] step=%d  sigma=%.4f→%.4f  sigma_down=%.4f  error=%.4f"
                "  h=%.4f  accept=%s  preheat=%s",
                info['steps'], float(sigma_s), float(sigma_next),
                float(sigma_down), float(error), float(pid.h), accept, in_preheat,
            )

            if info['steps'] > 10000:
                _sure_logger.warning("DPM++2Sa-SURE-Adaptive: step limit reached")
                break

    _sure_logger.info(
        "DPM++2Sa-SURE-Adaptive done: %d accepted / %d rejected  nfe=%d",
        info['n_accept'], info['n_reject'], info['nfe'],
    )

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda lbda: (lbda.exp() + 1) ** -1
    lambda_fn = lambda sigma: ((1-sigma)/sigma).log()

    # logged_x = x.unsqueeze(0)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
        sigma_down = sigmas[i+1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i+1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            if sigmas[i] == 1.0:
                sigma_s = 0.9999
            else:
                t_i, t_down = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
                r = 1 / 2
                h = t_down - t_i
                s = t_i + r * h
                sigma_s = sigma_fn(s)
            # sigma_s = sigmas[i+1]
            sigma_s_i_ratio = sigma_s / sigmas[i]
            u = sigma_s_i_ratio * x + (1 - sigma_s_i_ratio) * denoised
            D_i = model(u, sigma_s * s_in, **extra_args)
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * D_i
            # print("sigma_i", sigmas[i], "sigma_ip1", sigmas[i+1],"sigma_down", sigma_down, "sigma_down_i_ratio", sigma_down_i_ratio, "sigma_s_i_ratio", sigma_s_i_ratio, "renoise_coeff", renoise_coeff)
        # Noise addition
        if sigmas[i + 1] > 0 and eta > 0:
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        # logged_x = torch.cat((logged_x, x.unsqueeze(0)), dim=0)
    return x

@torch.no_grad()
def sample_dpmpp_sde_classic(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic)."""
    # Older and faster DPM++ SDE version.
    eta = modules.shared.opts.dpmpp_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_sde_og_s_noise
    r = modules.shared.opts.dpmpp_sde_og_r
    
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)
            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x

@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic)."""
    eta = modules.shared.opts.dpmpp_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_sde_og_s_noise
    r = modules.shared.opts.dpmpp_sde_og_r
    
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            lambda_s_1 = lambda_s + r * h
            fac = 1 / (2 * r)

            sigma_s_1 = sigma_fn(lambda_s_1)

            alpha_s = sigmas[i] * lambda_s.exp()
            alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            # Step 1
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
            lambda_s_1_ = sd.log().neg()
            h_ = lambda_s_1_ - lambda_s
            x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s_1 * noise_sampler(sigmas[i], sigma_s_1) * s_noise * su
            denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
            lambda_t_ = sd.log().neg()
            h_ = lambda_t_ - lambda_s
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

def sample_dpmpp_2m_sure(model, x, sigmas, extra_args=None, callback=None, disable=None,
                          sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                          sure_preheat_steps=-1, sure_jac_interval=-1,
                          sure_adam_mode='none', sure_adam_beta1=0.9,
                          sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                          sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(2M) with SURE trajectory correction.

    Fully deterministic ODE — zero noise injection at any step. SURE correction
    replaces the denoiser x̂₀ after the preheat phase; the DPM++(2M) multistep
    extrapolation then smooths per-step gradient variance across steps for free.

    Preheat:
      sure_preheat_steps >= 0  — fixed count of plain-denoise steps before SURE.
      sure_preheat_steps = -1  — automatic: ceil(15% of n_steps), minimum 2.

    sure_alpha:        SURE gradient step size (default 0.05)
    sure_n_mc:         Monte Carlo samples for Hutchinson trace (default 1)
    sure_eps:          finite-difference epsilon (default 1e-3)
    sure_jac_interval: full Jacobian every N correction steps; -1 = adaptive
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn    = lambda sigma: sigma.log().neg()

    n_steps = len(sigmas) - 1
    preheat = sure_preheat_steps if sure_preheat_steps >= 0 \
              else max(2, math.ceil(0.15 * n_steps))

    _dyn_jac_interval: int        = 2 if sure_jac_interval < 1 else sure_jac_interval
    _jac_ratio_ema:   float | None = None
    _corr_count: int               = 0
    _EMA_A = 0.35
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None

    _sure_logger.info(
        "DPM++2M-SURE: %d steps  preheat=%d  alpha=%.4f  jac_interval=%s  adam=%s",
        n_steps, preheat, sure_alpha,
        "adaptive" if sure_jac_interval < 1 else str(sure_jac_interval),
        sure_adam_mode,
    )

    old_denoised = None

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        _tag       = f" step={i+1}/{n_steps} sigma={float(sigma):.4f}"

        if i < preheat:
            # ── Preheat: plain denoiser, build history for multistep ─────────
            with torch.no_grad():
                denoised = model(x, sigma * s_in, **extra_args).detach()
            _sure_logger.info("[preheat%s]", _tag)
        else:
            # ── Correction: SURE-guided denoiser (x0-space, per paper Alg. 1) ─
            with torch.no_grad():
                x0_hat = model(x, sigma * s_in, **extra_args).detach()
            sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
            _use_jac = (_dyn_jac_interval <= 1) or (_corr_count % _dyn_jac_interval == 0)
            denoised, _stats = _sure_correct_x0(
                model, x0_hat, sigma_hat_0, s_in, extra_args,
                alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                use_jac=_use_jac, sigma_t=sigma,
                adam_state=_adam_state, adam_mode=sure_adam_mode,
                adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                alpha_bo_trials=sure_alpha_bo_trials,
                alpha_bo_patience=sure_alpha_bo_patience,
                alpha_bo_state=_alpha_bo_state,
            )
            _corr_count += 1

            _jac_ratio_new = _stats['jac_ratio']
            if _jac_ratio_new is not None:
                _jac_ratio_ema = _jac_ratio_new if _jac_ratio_ema is None else \
                    (1.0 - _EMA_A) * _jac_ratio_ema + _EMA_A * _jac_ratio_new
            if _jac_ratio_ema is not None and _corr_count >= 3:
                if _jac_ratio_ema < 0.05 and _dyn_jac_interval < 8:
                    _dyn_jac_interval += 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f < 0.05)",
                                      _dyn_jac_interval, _jac_ratio_ema)
                elif _jac_ratio_ema > 0.25 and _dyn_jac_interval > 1:
                    _dyn_jac_interval -= 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f > 0.25)",
                                      _dyn_jac_interval, _jac_ratio_ema)
            _sure_logger.info("[adapt] jac_ratio_ema=%s  dyn_jac_interval=%d",
                              f"{_jac_ratio_ema:.3f}" if _jac_ratio_ema is not None else "n/a",
                              _dyn_jac_interval)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        # ── DPM-Solver++(2M) update ───────────────────────────────────────────
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        if old_denoised is None:
            # First step: no history yet — 1st-order fallback
            x = sigma_fn(t_next) / sigma_fn(t) * x - (-h).expm1() * denoised
        elif float(sigma_next) == 0:
            # Final step: 1st-order to reach zero
            x = sigma_fn(t_next) / sigma_fn(t) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = sigma_fn(t_next) / sigma_fn(t) * x - (-h).expm1() * denoised_d

        old_denoised = denoised

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def sample_dpmpp_2m_sde_sure(model, x, sigmas, extra_args=None, callback=None, disable=None,
                               noise_sampler=None, eta=1., s_noise=1., solver_type='midpoint',
                               sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                               sure_preheat_steps=-1, sure_jac_interval=-1,
                               sure_adam_mode='none', sure_adam_beta1=0.9,
                               sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                               sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(2M) SDE with SURE trajectory correction.

    Matches the SURE paper (arxiv 2512.23232) Algorithm 1 which uses a stochastic
    reverse process: SURE corrects x̂₀, then SDE noise is injected as
      x_{t-1} ~ N(x̂*_{0|t}, σ²_{t-1}·I)
    The DPM++2M SDE noise term is the proper Brownian-tree realisation of that
    Gaussian, more principled than plain sigma_next·ε.

    Parameters:
      eta:               SDE noise weight (1.0 = full SDE, 0.0 = ODE)
      s_noise:           global noise scale multiplier
      solver_type:       '2nd-order multistep corrector: 'midpoint' or 'heun'
      sure_alpha:        SURE gradient step size
      sure_n_mc:         Monte Carlo samples for Hutchinson trace
      sure_eps:          finite-difference epsilon
      sure_preheat_steps: plain-denoise steps before SURE (-1 = auto 15%)
      sure_jac_interval: Jacobian every N correction steps (-1 = adaptive)
    """
    if len(sigmas) <= 1:
        return x
    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) \
                    if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    n_steps = len(sigmas) - 1
    preheat = sure_preheat_steps if sure_preheat_steps >= 0 \
              else max(2, math.ceil(0.15 * n_steps))

    _dyn_jac_interval: int        = 2 if sure_jac_interval < 1 else sure_jac_interval
    _jac_ratio_ema:   float | None = None
    _corr_count: int               = 0
    _EMA_A = 0.35
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None

    _sure_logger.info(
        "DPM++2M-SDE-SURE: %d steps  preheat=%d  eta=%.2f  alpha=%.4f  solver=%s  adam=%s",
        n_steps, preheat, eta, sure_alpha, solver_type, sure_adam_mode,
    )

    old_denoised = None
    h, h_last = None, None

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        _tag       = f" step={i+1}/{n_steps} sigma={float(sigma):.4f}"

        # ── x̂₀ at current state ──────────────────────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        if i < preheat:
            # ── Preheat: plain denoiser ───────────────────────────────────────
            denoised = x0_hat
            _sure_logger.info("[preheat%s]", _tag)
        else:
            # ── Correction: Algorithm 1 — correct x̂₀ in x0-space at σ̂₀ ──────
            sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
            _use_jac = (_dyn_jac_interval <= 1) or (_corr_count % _dyn_jac_interval == 0)
            denoised, _stats = _sure_correct_x0(
                model, x0_hat, sigma_hat_0, s_in, extra_args,
                alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                use_jac=_use_jac, sigma_t=sigma,
                adam_state=_adam_state, adam_mode=sure_adam_mode,
                adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                alpha_bo_trials=sure_alpha_bo_trials,
                alpha_bo_patience=sure_alpha_bo_patience,
                alpha_bo_state=_alpha_bo_state,
            )
            _corr_count += 1

            _jac_ratio_new = _stats.get('jac_ratio')
            if _jac_ratio_new is not None:
                _jac_ratio_ema = _jac_ratio_new if _jac_ratio_ema is None else \
                    (1.0 - _EMA_A) * _jac_ratio_ema + _EMA_A * _jac_ratio_new
            if _jac_ratio_ema is not None and _corr_count >= 3:
                if _jac_ratio_ema < 0.05 and _dyn_jac_interval < 8:
                    _dyn_jac_interval += 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f < 0.05)",
                                      _dyn_jac_interval, _jac_ratio_ema)
                elif _jac_ratio_ema > 0.25 and _dyn_jac_interval > 1:
                    _dyn_jac_interval -= 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f > 0.25)",
                                      _dyn_jac_interval, _jac_ratio_ema)
            _sure_logger.info("[adapt] jac_ratio_ema=%s  dyn_jac_interval=%d",
                              f"{_jac_ratio_ema:.3f}" if _jac_ratio_ema is not None else "n/a",
                              _dyn_jac_interval)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        # ── DPM-Solver++(2M) SDE update ──────────────────────────────────────
        if float(sigma_next) == 0:
            x = denoised
        else:
            lambda_s = lambda_fn(sigma)
            lambda_t = lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()

            # 1st-order SDE update from SURE-corrected x̂₀
            x = sigma_next / sigma * (-h * eta).exp() * x \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            # 2nd-order multistep correction (blends current + previous x̂₀)
            if old_denoised is not None:
                if h_last is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) \
                                  * (1 / r) * (denoised - old_denoised)
                    else:  # midpoint
                        x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() \
                                  * (1 / r) * (denoised - old_denoised)

            # Brownian SDE noise — the proper realisation of N(x̂*₀, σ²_{t-1}·I)
            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigma, sigma_next) \
                          * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++(2M) SDE."""
    eta = modules.shared.opts.dpmpp_2m_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_2m_sde_og_s_noise
    solver_type = modules.shared.opts.dpmpp_2m_sde_og_solver_type
    
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    old_denoised = None
    h, h_last = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    eta = modules.shared.opts.dpmpp_3m_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_og_s_noise
    
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # DPM-Solver++(3M) SDE
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                # DPM-Solver++(2M) SDE
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


def sample_dpmpp_3m_sde_sure(model, x, sigmas, extra_args=None, callback=None, disable=None,
                               noise_sampler=None, eta=1., s_noise=1.,
                               sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                               sure_preheat_steps=-1, sure_jac_interval=-1,
                               sure_adam_mode='none', sure_adam_beta1=0.9,
                               sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                               sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(3M) SDE with SURE trajectory correction.

    Implements Algorithm 1 from arXiv:2512.23232 on top of the 3rd-order multistep
    SDE solver.  Every step:

      1. x̂₀   = Dθ(xₜ, σₜ)                     — plain denoiser (no_grad)
      2. σ̂₀   = PCA residual noise estimate       — residual noise in x̂₀
      3. x̂*₀  = x̂₀ − α·∇SURE(x̂₀, σ̂₀)           — x0-space SURE correction
      4. ODE update (1st / 2nd / 3rd order)        — deterministic, uses x̂*₀
      5. Brownian noise injection                   — stochastic reverse process

    The 3rd-order multistep correction blends differences of the last three
    corrected x̂*₀ estimates (denoised, denoised_1, denoised_2), so the
    correction propagates naturally through the higher-order history.

    Parameters:
      eta:               SDE noise weight (1.0 = full SDE, 0.0 = ODE)
      s_noise:           global noise scale multiplier
      sure_alpha:        SURE gradient step size
      sure_n_mc:         Monte Carlo samples for Hutchinson trace
      sure_eps:          finite-difference epsilon
      sure_preheat_steps: plain-denoise steps before SURE (-1 = auto 15%)
      sure_jac_interval: Jacobian every N correction steps (-1 = adaptive)
    """
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) \
                    if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    n_steps = len(sigmas) - 1
    preheat = sure_preheat_steps if sure_preheat_steps >= 0 \
              else max(2, math.ceil(0.15 * n_steps))

    _dyn_jac_interval: int        = 2 if sure_jac_interval < 1 else sure_jac_interval
    _jac_ratio_ema:   float | None = None
    _corr_count: int               = 0
    _EMA_A = 0.35
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None

    _sure_logger.info(
        "DPM++3M-SDE-SURE: %d steps  preheat=%d  eta=%.2f  alpha=%.4f  adam=%s",
        n_steps, preheat, eta, sure_alpha, sure_adam_mode,
    )

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        _tag       = f" step={i+1}/{n_steps} sigma={float(sigma):.4f}"

        # ── Step 1: plain denoiser call ───────────────────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        if i < preheat:
            # ── Preheat: use plain x̂₀ directly ──────────────────────────────
            denoised = x0_hat
            _sure_logger.info("[preheat%s]", _tag)
        else:
            # ── Steps 2–3: SURE correction in x0-space at σ̂₀ ────────────────
            sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
            _use_jac = (_dyn_jac_interval <= 1) or (_corr_count % _dyn_jac_interval == 0)
            denoised, _stats = _sure_correct_x0(
                model, x0_hat, sigma_hat_0, s_in, extra_args,
                alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                use_jac=_use_jac, sigma_t=sigma,
                adam_state=_adam_state, adam_mode=sure_adam_mode,
                adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                alpha_bo_trials=sure_alpha_bo_trials,
                alpha_bo_patience=sure_alpha_bo_patience,
                alpha_bo_state=_alpha_bo_state,
            )
            _corr_count += 1

            _jac_ratio_new = _stats.get('jac_ratio')
            if _jac_ratio_new is not None:
                _jac_ratio_ema = _jac_ratio_new if _jac_ratio_ema is None else \
                    (1.0 - _EMA_A) * _jac_ratio_ema + _EMA_A * _jac_ratio_new
            if _jac_ratio_ema is not None and _corr_count >= 3:
                if _jac_ratio_ema < 0.05 and _dyn_jac_interval < 8:
                    _dyn_jac_interval += 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f < 0.05)",
                                      _dyn_jac_interval, _jac_ratio_ema)
                elif _jac_ratio_ema > 0.25 and _dyn_jac_interval > 1:
                    _dyn_jac_interval -= 1
                    _sure_logger.info("[adapt] jac_interval -> %d  (ratio=%.3f > 0.25)",
                                      _dyn_jac_interval, _jac_ratio_ema)
            _sure_logger.info("[adapt] jac_ratio_ema=%s  dyn_jac_interval=%d",
                              f"{_jac_ratio_ema:.3f}" if _jac_ratio_ema is not None else "n/a",
                              _dyn_jac_interval)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        # ── Step 4: DPM-Solver++(3M) SDE update ──────────────────────────────
        if float(sigma_next) == 0:
            x = denoised
        else:
            lambda_s = lambda_fn(sigma)
            lambda_t = lambda_fn(sigma_next)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()

            # 1st-order SDE step from SURE-corrected x̂*₀
            x = sigma_next / sigma * (-h * eta).exp() * x \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # 3rd-order multistep correction
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1   = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2   = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + alpha_t * phi_2 * d1 - alpha_t * phi_3 * d2
            elif h_1 is not None:
                # 2nd-order multistep correction (warm-up)
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + alpha_t * phi_2 * d

            # ── Step 5: Brownian SDE noise ────────────────────────────────────
            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigma, sigma_next) \
                          * sigma_next * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def sample_dpmpp_2m_sde_sure_adaptive(model, x, sigma_min, sigma_max,
                                       extra_args=None, callback=None, disable=None,
                                       noise_sampler=None,
                                       rtol=0.05, atol=0.0078, h_init=0.05,
                                       pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81,
                                       eta=1., s_noise=1.,
                                       sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                                       sure_preheat_frac=0.3, sure_jac_interval=2,
                                       sure_adam_mode='none', sure_adam_beta1=0.9,
                                       sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                                       sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """DPM-Solver++(2M) SDE with SURE correction and adaptive step size (PID).

    Combines three ideas from the literature:
      - SURE trajectory correction (arxiv 2512.23232): corrects x̂₀ via gradient of
        the Stein Unbiased Risk Estimate before each reverse step.
      - DPM-Solver-2 single-step error estimate: compare 1st vs 2nd order ODE steps
        (noise excluded) to drive a PID step-size controller.
      - SDE noise injection (Brownian tree): added AFTER a step is accepted, matching
        the paper's stochastic reverse process N(x̂*₀, σ²_{t-1}·I).

    The error estimate is computed on the deterministic ODE part only — stochastic
    noise is NOT included — so the PID reacts to trajectory curvature, not noise.
    Noise is then added on each accepted step at the proper Brownian variance.

    Parameters:
      rtol, atol:         PID error tolerances
      h_init:             initial step size in log-sigma space
      pcoeff/icoeff/dcoeff: PID gains (default: I-only)
      accept_safety:      step accepted when PID factor >= this
      eta:                SDE noise weight (1.0 = full SDE / paper intent; 0.0 = ODE)
      s_noise:            global noise scale multiplier
      sure_alpha:         SURE gradient step size
      sure_n_mc:          Monte Carlo samples for Hutchinson trace
      sure_eps:           finite-difference epsilon
      sure_preheat_frac:  fraction of log-sigma range to run without correction
      sure_jac_interval:  Jacobian every N correction steps (default 2)
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) \
                    if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    t_start   = t_fn(torch.tensor(sigma_max, dtype=x.dtype, device=x.device))
    t_end     = t_fn(torch.tensor(sigma_min, dtype=x.dtype, device=x.device))
    t_preheat = t_start + sure_preheat_frac * (t_end - t_start)

    pid    = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, order=2,
                                    accept_safety=accept_safety)
    atol_t = torch.tensor(atol, dtype=x.dtype, device=x.device)
    rtol_t = torch.tensor(rtol, dtype=x.dtype, device=x.device)

    _corr_count = 0
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None
    info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}
    x_prev = x.clone()
    s = t_start.clone()

    _sure_logger.info(
        "DPM++2M-SDE-SURE-Adaptive: sigma [%.4f → %.4f]  preheat_frac=%.2f"
        "  eta=%.2f  alpha=%.4f  jac_interval=%d  adam=%s",
        sigma_max, sigma_min, sure_preheat_frac, eta, sure_alpha, sure_jac_interval,
        sure_adam_mode,
    )

    with tqdm(disable=disable) as pbar:
        while s < t_end - 1e-5:
            t_next     = torch.minimum(t_end, s + pid.h)
            sigma_s    = sigma_fn(s)
            sigma_next = sigma_fn(t_next)
            h          = t_next - s

            in_preheat = float(s) < float(t_preheat)
            _tag = f" [adap-sde] sigma={float(sigma_s):.4f}"

            # ── x̂₀ at current state (SURE after preheat) ──────────────────
            if in_preheat:
                with torch.no_grad():
                    x0_s = model(x, sigma_s * s_in, **extra_args).detach()
            else:
                with torch.no_grad():
                    x0_hat = model(x, sigma_s * s_in, **extra_args).detach()
                sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
                _use_jac = (sure_jac_interval <= 1) or (_corr_count % sure_jac_interval == 0)
                x0_s, _ = _sure_correct_x0(
                    model, x0_hat, sigma_hat_0, s_in, extra_args,
                    alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                    use_jac=_use_jac, sigma_t=sigma_s,
                    adam_state=_adam_state, adam_mode=sure_adam_mode,
                    adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                    adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                    alpha_bo_trials=sure_alpha_bo_trials,
                    alpha_bo_patience=sure_alpha_bo_patience,
                    alpha_bo_state=_alpha_bo_state,
                )
                _corr_count += 1

            eps_s = (x - x0_s) / sigma_s

            # ── 1st-order ODE step (x_low, for error estimate) ────────────
            x_low = x - sigma_next * h.expm1() * eps_s

            # ── 2nd-order ODE midpoint step (x_high, for error estimate) ──
            r         = 0.5
            s1        = s + r * h
            sigma_mid = sigma_fn(s1)
            u1        = x - sigma_mid * (r * h).expm1() * eps_s
            with torch.no_grad():
                x0_mid = model(u1, sigma_mid * s_in, **extra_args).detach()
            eps_mid = (u1 - x0_mid) / sigma_mid
            x_high  = x - sigma_next * h.expm1() * eps_s \
                        - sigma_next / (2 * r) * h.expm1() * (eps_mid - eps_s)

            # ── PID error on ODE part only (noise excluded intentionally) ──
            delta  = torch.maximum(atol_t, rtol_t * torch.maximum(x_low.abs(), x_prev.abs()))
            error  = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)

            if accept:
                x_prev = x_low
                x = x_high
                # Inject Brownian SDE noise — proper variance for reverse SDE path
                if eta > 0 and s_noise > 0:
                    noise_std = sigma_next * (-2 * h * eta).expm1().neg().sqrt()
                    x = x + noise_sampler(sigma_s, sigma_next) * noise_std * s_noise
                s = t_next
                info['n_accept'] += 1
                pbar.update()
                if callback is not None:
                    callback({'x': x, 'i': info['steps'], 'sigma': sigma_s,
                              'sigma_hat': sigma_s, 'denoised': x0_s,
                              'error': error, 'h': pid.h, **info})
            else:
                info['n_reject'] += 1

            info['nfe']   += 2
            info['steps'] += 1

            _sure_logger.info(
                "[adap-sde] step=%d  sigma=%.4f→%.4f  error=%.4f  h=%.4f"
                "  accept=%s  preheat=%s  eta=%.2f",
                info['steps'], float(sigma_s), float(sigma_next),
                float(error), float(pid.h), accept, in_preheat, eta,
            )

            if info['steps'] > 10000:
                _sure_logger.warning("DPM++2M-SDE-SURE-Adaptive: step limit reached")
                break

    _sure_logger.info(
        "DPM++2M-SDE-SURE-Adaptive done: %d accepted / %d rejected  nfe=%d",
        info['n_accept'], info['n_reject'], info['nfe'],
    )

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_3m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

@torch.no_grad()
def sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type=solver_type)

@torch.no_grad()
def sample_dpmpp_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=r)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu

def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)

@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x



@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None):
    s_churn = modules.shared.opts.heunpp2_s_churn
    s_tmin = modules.shared.opts.heunpp2_s_tmin
    s_noise = modules.shared.opts.heunpp2_s_noise
    s_tmax = float('inf')
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == s_end:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            w = 2 * sigmas[0]
            w2 = sigmas[i+1]/w
            w1 = 1 - w2
            d_prime = d * w1 + d_2 * w2
            x = x + d_prime * dt
        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]
            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)
            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3
            d_prime = w1 * d + w2 * d_2 + w3 * d_3
            x = x + d_prime * dt
    return x

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.ipndm_max_order
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if t_next == 0:     # Denoising step
            x_next = denoised
        elif order == 1:    # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
    return x_next

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.ipndm_v_max_order
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if t_next == 0:     # Denoising step
            x_next = denoised
        elif order == 1:    # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.deis_max_order
    deis_mode = modules.shared.opts.deis_mode
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    coeff_list = deis.get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if t_next <= 0:
            order = 1
        if order == 1:          # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next

@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = denoised + d * sigmas[i + 1]
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    eta = modules.shared.opts.euler_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.euler_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
            alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp()
            d = to_d(x, sigmas[i], alpha_s * uncond_denoised)   # to noise

            # DDIM stochastic sampling
            sigma_down, sigma_up = get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down

            # Euler method
            x = alpha_t * denoised + sigma_down * d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps and CFG++."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp_dyn(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_dyn_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_dyn_s_noise if s_noise is None else s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigma_down - sigmas[i]
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up)))
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp_intern(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    if hasattr(model, 'model_sampling') and isinstance(model.model_sampling, CONST):
        return sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_intern_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_intern_s_noise if s_noise is None else s_noise
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    s = sigmas[0]
    small_x = nn.functional.interpolate(x, scale_factor=0.5, mode='area')
    den = model(small_x, s * s_in, **extra_args)
    den = nn.functional.interpolate(den, scale_factor=2, mode='area')
    ups_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
    sigma_down, sigma_up = get_ancestral_step(s, sigmas[1], eta=eta)
    t, t_next = t_fn(s), t_fn(sigma_down)
    r = 1 / 2
    h = t_next - t
    s_ = t + r * h
    x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + (den - ups_temp)) - (-h * r).expm1() * den
    denoised_2 = model(x_2, sigma_fn(s_) * s_in, **extra_args)
    x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (den - temp[0])) - (-h).expm1() * denoised_2
    large_denoised = x
    x = x + noise_sampler(sigmas[0], sigmas[1]) * s_noise * sigma_up
    sigmas = sigmas[1:] # remove the first sigma we used
    for i in trange(len(sigmas) - 2, disable=disable):
        if sigma_down != 0:
            down_x = nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            denoised = model(down_x, sigmas[i] * s_in, **extra_args)
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)
        # denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            mergefactor = min(math.sqrt(i/(len(sigmas) - 2)), 1) 
            print(mergefactor)
            #merge up_den with x
            if mergefactor == 1:
                up_den = large_denoised
                up_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
                x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (up_den - up_temp)) - (-h * r).expm1() * up_den
            else:
                up_den = nn.functional.interpolate(denoised, scale_factor=2, mode='area')
                print(up_den.max(), large_denoised.max())
                up_den = (up_den * (1-mergefactor)) + (large_denoised * mergefactor)
                print(up_den.max(), large_denoised.max())
                up_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
                x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (up_den - up_temp)) - (-h * r).expm1() * up_den
            
            
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (up_den - temp[0])) - (-h).expm1() * denoised_2
            large_denoised = denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()
    old_uncond_denoised = None
    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x

@torch.no_grad()
def sample_dpmpp_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic) with CFG++."""
    eta = modules.shared.opts.dpmpp_sde_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_sde_cfg_pp_s_noise
    r = modules.shared.opts.dpmpp_sde_cfg_pp_r
    
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigmas[i + 1] - sigmas[i]
            x = denoised + d * sigmas[i + 1]
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * temp[0] + fac * temp[0]  # Use temp[0] instead of denoised
            x = denoised_2 + to_d(x, sigmas[i], denoised_d) * sd
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x

@torch.no_grad()
def sample_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, solver="dopri5", rtol=1e-3, atol=1e-4, max_steps=250):
    """Implements ODE-based sampling."""
    sampler = ODESampler(solver, rtol, atol, max_steps)
    return sampler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

@torch.no_grad()
def sample_dpmpp_3m_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    eta = modules.shared.opts.dpmpp_3m_sde_cfg_pp_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_cfg_pp_s_noise if s_noise is None else s_noise

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - temp[0])) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_dpmpp_2m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) with dynamic thresholding."""
    s_noise = modules.shared.opts.dpmpp_2m_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_2m_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        t, t_next = t_fn(sigma_hat), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    solver_type=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) SDE with dynamic thresholding."""
    eta = modules.shared.opts.dpmpp_2m_sde_dy_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2m_sde_dy_s_noise if s_noise is None else s_noise
    solver_type = modules.shared.opts.dpmpp_2m_sde_dy_solver_type if solver_type is None else solver_type
    s_dy_pow = modules.shared.opts.dpmpp_2m_sde_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_sde_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    gamma = 2**0.5 - 1

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max() * (gamma + 1)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigma_hat * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigma_hat, sigmas[i + 1] * (gamma + 1)) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(3M) SDE with dynamic thresholding."""
    eta = modules.shared.opts.dpmpp_3m_sde_dy_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_3m_sde_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_3m_sde_dy_s_extra_steps if s_extra_steps is None else s_extra_steps

    if len(sigmas) <= 1:
        return x

    gamma = 2**0.5 - 1

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max() * (gamma + 1)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1] * (gamma + 1)) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_dpmpp_3m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    s_noise = modules.shared.opts.dpmpp_3m_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_3m_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_3m_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    return sample_dpmpp_3m_sde_dy(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        0.0,
        s_noise,
        noise_sampler,
        s_dy_pow,
        s_extra_steps,
    )

@torch.no_grad()
def dy_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": c, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(c, sigma_hat, temp[0])
    c = denoised + d * sigma_next

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x


@torch.no_grad()
def sample_euler_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=None,
    s_tmin=None,
    s_tmax=float("inf"),
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler with dynamic thresholding and CFG++."""
    s_churn = modules.shared.opts.euler_dy_cfg_pp_s_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.euler_dy_cfg_pp_s_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.euler_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def smea_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode="nearest-exact")

    with Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(x, sigma_hat, temp[0])
    x = denoised + d * sigma_next
    x = torch.nn.functional.interpolate(input=x, size=(m, n), mode="nearest-exact")
    return x


@torch.no_grad()
def sample_euler_smea_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=None,
    s_tmin=None,
    s_tmax=float("inf"),
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler with SMEA, dynamic thresholding and CFG++."""
    s_churn = modules.shared.opts.euler_smea_dy_cfg_pp_s_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.euler_smea_dy_cfg_pp_s_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.euler_smea_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_smea_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_smea_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i + 1 // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
            if i + 1 // 2 == 0:
                x = smea_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def sample_euler_ancestral_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler ancestral with dynamic thresholding and CFG++."""
    eta = modules.shared.opts.euler_ancestral_dy_cfg_pp_eta if eta is None else eta
    s_noise = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigma_hat, sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        dt = sigma_down - sigma_hat
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma_hat, sigmas[i + 1] * (gamma + 1)) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2m_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) with dynamic thresholding and CFG++."""
    s_noise = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps    
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None
    h_last = None
    h = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        t, t_next = t_fn(sigma_hat), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
        h_last = h
    return x

@torch.no_grad()
def sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None, momentum=0.0):
    """DPM-Solver++(3M) SDE, modified with an extra SDE, and momentumized in both the SDE and ODE(?). 'its a first' - Clybius 2023
    The expression for d1 is derived from the extrapolation formula given in the paper “Diffusion Monte Carlo with stochastic Hamiltonians” by M. Foulkes, L. Mitas, R. Needs, and G. Rajagopal. The formula is given as follows:
    d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
    (if this is an incorrect citing, we blame Google's Bard and OpenAI's ChatGPT for this and NOT me :^) )

    where d1_0, d1_1, and d1_2 are defined as follows:
    d1_0 = (denoised - denoised_1) / r2
    d1_1 = (denoised_1 - denoised_2) / r1
    d1_2 = (denoised_2 - denoised_3) / r0

    The variables r0, r1, and r2 are defined as follows:
    r0 = h_3 / h_2
    r1 = h_2 / h
    r2 = h / h_1
    """

    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    vel, vel_sde = None, None
    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x_diff = momentum_func((-h_eta).expm1().neg() * denoised, vel, time)
            vel = x_diff
            x = torch.exp(-h_eta) * x + vel

            if h_3 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h
                d1_0 = (denoised   - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r2
                # d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r1 + r2) * (r0 + r1))
                # d2 = (d1_0 - d1_1) / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r1 + r2) * (r0 + r1))

                # r0 = h_3 / h_2
                # r1 = h_2 / h
                # r2 = h / h_1
                # d1_0 = (denoised - denoised_1) / r2
                # d1_1 = (denoised_1 - denoised_2) / r1
                # d1_2 = (denoised_2 - denoised_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                sde_diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                sde_diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                sde_diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
            h_1, h_2, h_3 = h, h_1, h_2

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    return 

class DenoiserModel(Protocol):
  def __call__(self, x: FloatTensor, t: FloatTensor, *args, **kwargs) -> FloatTensor: ...

class RefinedExpCallbackPayload(TypedDict):
  x: FloatTensor
  i: int
  sigma: FloatTensor
  sigma_hat: FloatTensor

class RefinedExpCallback(Protocol):
  def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...

class NoiseSampler(Protocol):
  def __call__(self, x: FloatTensor) -> FloatTensor: ...

class StepOutput(NamedTuple):
  x_next: FloatTensor
  denoised: FloatTensor
  denoised2: FloatTensor
  vel: FloatTensor
  vel_2: FloatTensor

def _gamma(
  n: int,
) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(
  s: int,
  x: float,
  gamma_s: Optional[int] = None
) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_

# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)

# by Katherine Crowson
def _phi_3(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6)

def _phi(
  neg_h: float,
  j: int,
):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)

  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)

  return phi_

class RESDECoeffsSecondOrder(NamedTuple):
  a2_1: float
  b1: float
  b2: float

def _de_second_order(
  h: float,
  c2: float,
  simple_phi_calc = False,
) -> RESDECoeffsSecondOrder:
  """
  Table 3
  https://arxiv.org/abs/2308.02157
  ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
  a2_1 = c2ϕ1,2
       = c2ϕ1(-c2*h)
  b1 = ϕ1 - ϕ2/c2
  """
  if simple_phi_calc:
    # Kat computed simpler expressions for phi for cases j={1,2,3}
    a2_1: float = c2 * _phi_1(-c2*h)
    phi1: float = _phi_1(-h)
    phi2: float = _phi_2(-h)
  else:
    # I computed general solution instead.
    # they're close, but there are slight differences. not sure which would be more prone to numerical error.
    a2_1: float = c2 * _phi(j=1, neg_h=-c2*h)
    phi1: float = _phi(j=1, neg_h=-h)
    phi2: float = _phi(j=2, neg_h=-h)
  phi2_c2: float = phi2/c2
  b1: float = phi1 - phi2_c2
  b2: float = phi2_c2
  return RESDECoeffsSecondOrder(
    a2_1=a2_1,
    b1=b1,
    b2=b2,
  )  

def _refined_exp_sosu_step(
  model: DenoiserModel,
  x: FloatTensor,
  sigma: FloatTensor,
  sigma_next: FloatTensor,
  c2 = 0.5,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0,
  vel = None,
  vel_2 = None,
  time = None
) -> StepOutput:
  """
  Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  lam_next, lam = (s.log().neg() for s in (sigma_next, sigma))

  # type hints aren't strictly true regarding float vs FloatTensor.
  # everything gets promoted to `FloatTensor` after interacting with `sigma: FloatTensor`.
  # I will use float to indicate any variables which are scalars.
  h: float = lam_next - lam
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  denoised: FloatTensor = model(x, sigma.repeat(x.size(0)), **extra_args)
  # if pbar is not None:
    # pbar.update(0.5)

  c2_h: float = c2*h

  diff_2 = momentum_func(a2_1*h*denoised, vel_2, time)
  vel_2 = diff_2
  x_2: FloatTensor = math.exp(-c2_h)*x + diff_2
  lam_2: float = lam + c2_h
  sigma_2: float = lam_2.neg().exp()

  denoised2: FloatTensor = model(x_2, sigma_2.repeat(x_2.size(0)), **extra_args)
  if pbar is not None:
    pbar.update()

  diff = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  vel = diff

  x_next: FloatTensor = math.exp(-h)*x + diff
  
  return StepOutput(
    x_next=x_next,
    denoised=denoised,
    denoised2=denoised2,
    vel=vel,
    vel_2=vel_2,
  )
  

@no_grad()
def sample_refined_exp_s(
  model: FloatTensor,
  x: FloatTensor,
  sigmas: FloatTensor,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  ita: FloatTensor = torch.zeros((1,)),
  c2 = .5,
  noise_sampler: NoiseSampler = torch.randn_like,
  simple_phi_calc = False,
  momentum = 0.0,
):
  """
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler" with Algorithm 1 second-order step
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigmas (`FloatTensor`): sigmas (ideally an exponential schedule!) e.g. get_sigmas_exponential(n=25, sigma_min=model.sigma_min, sigma_max=model.sigma_max)
    denoise_to_zero (`bool`, *optional*, defaults to `True`): whether to finish with a first-order step down to 0 (rather than stopping at sigma_min). True = fully denoise image. False = match Algorithm 2 in paper
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    callback (`RefinedExpCallback`, *optional*, defaults to `None`): you can supply this callback to see the intermediate denoising results, e.g. to preview each step of the denoising process
    disable (`bool`, *optional*, defaults to `False`): whether to hide `tqdm`'s progress bar animation from being printed
    ita (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
         - if used: batch noise doesn't match non-batch
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """
  #assert sigmas[-1] == 0
  device = x.device
  ita = ita.to(device)
  sigmas = sigmas.to(device)

  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

  vel, vel_2 = None, None
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
      time = sigmas[i] / sigma_max
      if 'sigma' not in locals():
        sigma = sigmas[i]
      eps = torch.randn_like(x).float()
      sigma_hat = sigma * (1 + ita)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * eps
      x_next, denoised, denoised2, vel, vel_2 = _refined_exp_sosu_step(
        model,
        x_hat,
        sigma_hat,
        sigma_next,
        c2=c2,
        extra_args=extra_args,
        pbar=pbar,
        simple_phi_calc=simple_phi_calc,
        momentum = momentum,
        vel = vel,
        vel_2 = vel_2,
        time = time
      )
      if callback is not None:
        payload = RefinedExpCallbackPayload(
          x=x,
          i=i,
          sigma=sigma,
          sigma_hat=sigma_hat,
          denoised=denoised,
          denoised2=denoised2,
        )
        callback(payload)
      x = x_next
    if denoise_to_zero:
      eps = torch.randn_like(x).float()
      sigma_hat = sigma * (1 + ita)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * eps
      x_next: FloatTensor = model(x_hat, sigma.to(x_hat.device).repeat(x_hat.size(0)), **extra_args)
      pbar.update()

      if callback is not None:
        payload = RefinedExpCallbackPayload(
          x=x,
          i=i,
          sigma=sigma,
          sigma_hat=sigma_hat,
          denoised=denoised,
          denoised2=denoised2,
        )
        callback(payload)


      x = x_next
  return x

# Many thanks to Kat + Birch-San for this wonderful sampler implementation! https://github.com/Birch-san/sdxl-play/commits/res/
def sample_res_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, ita=torch.Tensor((0.0,)), momentum=0.0):
    return sample_refined_exp_s(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler, denoise_to_zero=denoise_to_zero, simple_phi_calc=simple_phi_calc, c2=c2, ita=ita, momentum=momentum)

@torch.no_grad()
def sample_Kohaku_LoNyu_Yog(
    model, 
    x, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    s_churn=None, 
    s_tmin=None,
    s_tmax=float('inf'), 
    s_noise=None, 
    noise_sampler=None, 
    eta=None
):
    """Kohaku_LoNyu_Yog sampler with configurable parameters"""
    # Get values from shared options if not provided
    s_churn = modules.shared.opts.kohaku_lonyu_yog_s_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.kohaku_lonyu_yog_s_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.kohaku_lonyu_yog_s_noise if s_noise is None else s_noise
    eta = modules.shared.opts.kohaku_lonyu_yog_eta if eta is None else eta

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigma_down - sigmas[i]
        if i <= (len(sigmas) - 1) / 2:
            x2 = - x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, denoised2)
            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, denoised3)
            real_d = (d + d3) / 2
            x = x + real_d * dt
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
    return x

@torch.no_grad()
def sample_kohaku_lonyu_yog_cfg_pp(
    model, 
    x, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    s_churn=None, 
    s_tmin=None,
    s_tmax=float('inf'), 
    s_noise=None, 
    noise_sampler=None, 
    eta=None
):
    """Kohaku_LoNyu_Yog sampler with CFG++ implementation"""
    # Get values from shared options if not provided
    s_churn = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_noise if s_noise is None else s_noise
    eta = modules.shared.opts.kohaku_lonyu_yog_cfgpp_eta if eta is None else eta

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    # Add CFG++ handling
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )
    
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
        
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            
        dt = sigma_down - sigmas[i]
        
        if i <= (len(sigmas) - 1) / 2:
            x2 = -x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
            real_d = (d + d3) / 2
            x = x + real_d * dt
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
            
    return x

def sample_custom(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Custom sampler that uses configurations from shared options"""
    
    # Get sampler parameters from shared options
    sampler_name = modules.shared.opts.custom_sampler_name
    eta = modules.shared.opts.custom_sampler_eta
    s_noise = modules.shared.opts.custom_sampler_s_noise
    solver_type = modules.shared.opts.custom_sampler_solver_type
    r = modules.shared.opts.custom_sampler_r
    cfg_scale = modules.shared.opts.custom_cfg_conds
    cfg_scale2 = modules.shared.opts.custom_cfg_cond2_negative

    # Get the appropriate sampler function
    sampler_functions = {
            'euler_comfy': sample_euler,
            'euler_ancestral_comfy': sample_euler_ancestral,
            'euler_a2': sample_euler_a2,
            'euler_a2_edm': sample_euler_a2_edm,
            'heun_comfy': sample_heun,
            'dpmpp_2s_ancestral_comfy': sample_dpmpp_2s_ancestral,
            'dpmpp_sde_comfy': sample_dpmpp_sde,
            'dpmpp_2m_comfy': sample_dpmpp_2m,
            'dpmpp_2m_sde_comfy': sample_dpmpp_2m_sde,
            'dpmpp_3m_sde_comfy': sample_dpmpp_3m_sde,
            'euler_ancestral_turbo': sample_euler_ancestral,
            'dpmpp_2m_turbo': sample_dpmpp_2m,
            'dpmpp_2m_sde_turbo': sample_dpmpp_2m_sde,
            'ddpm': sample_ddpm,
            'heunpp2': sample_heunpp2,
            'ipndm': sample_ipndm,
            'ipndm_v': sample_ipndm_v,
            'deis': sample_deis,
            'euler_cfg_pp': sample_euler_cfg_pp,
            'euler_ancestral_cfg_pp': sample_euler_ancestral_cfg_pp,
            'sample_euler_ancestral_RF': sample_euler_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp': sample_dpmpp_2s_ancestral_cfg_pp,
            'sample_dpmpp_2s_ancestral_RF': sample_dpmpp_2s_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp_dyn': sample_dpmpp_2s_ancestral_cfg_pp_dyn,
            'dpmpp_2s_ancestral_cfg_pp_intern': sample_dpmpp_2s_ancestral_cfg_pp_intern,
            'dpmpp_sde_cfg_pp': sample_dpmpp_sde_cfg_pp,
            'dpmpp_2m_cfg_pp': sample_dpmpp_2m_cfg_pp,
            'dpmpp_3m_sde_cfg_pp': sample_dpmpp_3m_sde_cfg_pp,
            'dpmpp_2m_dy': sample_dpmpp_2m_dy,
            'dpmpp_3m_dy': sample_dpmpp_3m_dy,
            'dpmpp_3m_sde_dy': sample_dpmpp_3m_sde_dy,
            'euler_dy_cfg_pp': sample_euler_dy_cfg_pp,
            'euler_smea_dy_cfg_pp': sample_euler_smea_dy_cfg_pp,
            'euler_ancestral_dy_cfg_pp': sample_euler_ancestral_dy_cfg_pp,
            'dpmpp_2m_dy_cfg_pp': sample_dpmpp_2m_dy_cfg_pp,
            'clyb_4m_sde_momentumized': sample_clyb_4m_sde_momentumized,
            'res_solver': sample_res_solver,
            'kohaku_lonyu_yog_cfg_pp': sample_kohaku_lonyu_yog_cfg_pp,
        }

    sampler_function = sampler_functions.get(sampler_name)
    if sampler_function is None:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Prepare sampler kwargs based on which sampler is selected
    kwargs = {
        "model": model,
        "x": x,
        "sigmas": sigmas,
        "extra_args": extra_args,
        "callback": callback,
        "disable": disable,
    }

    # Add additional parameters based on sampler type
    if "cfg" in sampler_name:
        kwargs["cfg_scale"] = cfg_scale
    if "sde" in sampler_name:
        kwargs.update({
            "eta": eta,
            "s_noise": s_noise,
        })
    if "2m_sde" in sampler_name:
        kwargs["solver_type"] = solver_type
    if any(x in sampler_name for x in ["sde", "dpmpp"]):
        kwargs["r"] = r

    # Call the sampler
    return sampler_function(**kwargs)

@torch.no_grad()
def res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None, eta=1., cfg_pp=False):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_fn(h) * x + h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if cfg_pp:
            old_denoised = uncond_denoised
        else:
            old_denoised = denoised
        old_sigma_down = sigma_down
    return x

@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=True)

@torch.no_grad()
def sample_res_multistep_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=True)

@torch.no_grad()
def sample_gradient_estimation(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2., cfg_pp=False):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if cfg_pp:
            d = to_d(x, sigmas[i], uncond_denoised)
        else:
            d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # Euler method
            if cfg_pp:
                x = denoised + d * sigmas[i + 1]
            else:
                x = x + d * dt

            if i >= 1:
                # Gradient estimation
                d_bar = (ge_gamma - 1) * (d - old_d)
                x = x + d_bar * dt
        old_d = d
    return x

@torch.no_grad()
def sample_gradient_estimation_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    return sample_gradient_estimation(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, ge_gamma=ge_gamma, cfg_pp=True)

def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, noise_scaler=None, max_stage=3):
    """Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3). arXiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(x):
        return x * ((x ** 0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_seeds_2(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=0.5, solver_type="phi_1"):
    """SEEDS-2 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 2.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)
        sigma_s_1 = sigma_fn(lambda_s_1)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = ei_h_phi_2(-h_eta) / r
            b1 = ei_h_phi_1(-h_eta) - b2
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b2 * denoised_2)

        if inject_noise:
            segment_factor = (r - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x


@torch.no_grad()
def sample_seeds_3(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r_1=1./3, r_2=2./3):
    """SEEDS-3 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 3.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r_1)
        lambda_s_2 = torch.lerp(lambda_s, lambda_t, r_2)
        sigma_s_1, sigma_s_2 = sigma_fn(lambda_s_1), sigma_fn(lambda_s_2)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_s_2 = sigma_s_2 * lambda_s_2.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r_1 * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r_1 * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r_1 * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        a3_2 = r_2 / r_1 * ei_h_phi_2(-r_2 * h_eta)
        a3_1 = ei_h_phi_1(-r_2 * h_eta) - a3_2
        x_3 = sigma_s_2 / sigmas[i] * (-r_2 * h * eta).exp() * x - alpha_s_2 * (a3_1 * denoised + a3_2 * denoised_2)
        if inject_noise:
            segment_factor = (r_1 - r_2) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigma_s_2)
            x_3 = x_3 + sde_noise * sigma_s_2 * s_noise
        denoised_3 = model(x_3, sigma_s_2 * s_in, **extra_args)

        # Step 3
        b3 = ei_h_phi_2(-h_eta) / r_2
        b1 = ei_h_phi_1(-h_eta) - b3
        x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b3 * denoised_3)
        if inject_noise:
            segment_factor = (r_2 - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_2, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x

@torch.no_grad()
def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, use_pece=False, simple_order_2=False):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    if tau_func is None:
        # Use default interval for stochastic sampling
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x  # x: current state, x_pred: predicted next state

    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    # Lower order near the end to improve stability
    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        # Evaluation
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            # Update by the predicted state
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)    # (B, K, ...)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                # The noise from the previous predictor step
                x = x + noise

            if use_pece:
                # Evaluate the corrected state
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                pred_list[-1] = denoised

        # Predictor
        if sigmas[i + 1] == 0:
            # Denoising step
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)    # (B, K, ...)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x_pred


@torch.no_grad()
def sample_sa_solver_pece(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, simple_order_2=False):
    """Stochastic Adams Solver with PECE (Predict–Evaluate–Correct–Evaluate) mode (NeurIPS 2023)."""
    return sample_sa_solver(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, tau_func=tau_func, s_noise=s_noise, noise_sampler=noise_sampler, predictor_order=predictor_order, corrector_order=corrector_order, use_pece=True, simple_order_2=simple_order_2)

@torch.no_grad()
def sample_exp_heun_2_x0(model, x, sigmas, extra_args=None, callback=None, disable=None, solver_type="phi_2"):
    """Deterministic exponential Heun second order method in data prediction (x0) and logSNR time."""
    return sample_seeds_2(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=0.0, s_noise=0.0, noise_sampler=None, r=1.0, solver_type=solver_type)


@torch.no_grad()
def sample_exp_heun_2_x0_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type="phi_2"):
    """Stochastic exponential Heun second order method in data prediction (x0) and logSNR time."""
    return sample_seeds_2(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=1.0, solver_type=solver_type)


# ---------------------------------------------------------------------------
# DC-Solver (ECCV 2024) — Predictor-Corrector with Dynamic Compensation
# Ported to EDM sigma-space from https://github.com/wl-zhao/DC-Solver
# DC-Solver corrects predictor-corrector misalignment by interpolating previous
# model outputs to a fractional time point before each corrector step.
# ---------------------------------------------------------------------------

def _dc_dynamic_compensation(model_prev_list, sigma_prev_list, ratio, dc_order):
    """Lagrange polynomial interpolation of past model outputs at fractional sigma."""
    sigma_prev = sigma_prev_list[-2]
    sigma_cur  = sigma_prev_list[-1]
    # interpolate in log-sigma space
    log_target = (1 - ratio) * sigma_prev.log() + ratio * sigma_cur.log()
    sigma_target = log_target.exp()

    result = torch.zeros_like(model_prev_list[-1])
    n = min(dc_order + 1, len(model_prev_list))
    for i in range(n):
        term = model_prev_list[-(i + 1)]
        for j in range(n):
            if i != j:
                si = sigma_prev_list[-(i + 1)].log()
                sj = sigma_prev_list[-(j + 1)].log()
                coeff = (log_target - sj) / (si - sj)
                term = term * coeff
        result = result + term
    return result


@torch.no_grad()
def sample_dc_solver(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     order=2, dc_ratios=None):
    """DC-Solver: multistep predictor-corrector with dynamic compensation in EDM sigma-space.

    dc_ratios: list of floats in [0,1] controlling compensation per step.
               1.0 = no compensation (pure predictor-corrector).
               If None, defaults to all 1.0 (equivalent to DPM-Solver++(2M)).
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    steps = len(sigmas) - 1
    if dc_ratios is None:
        dc_ratios = [1.0] * steps
    dc_ratios = list(dc_ratios)

    model_prev_list  = []
    sigma_prev_list  = []

    for i in trange(steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # --- dynamic compensation: warp last model output before predictor ---
        ratio = dc_ratios[i] if i < len(dc_ratios) else 1.0
        if ratio != 1.0 and len(model_prev_list) >= 2:
            model_prev_list[-1] = _dc_dynamic_compensation(
                model_prev_list, sigma_prev_list, ratio, dc_order=order
            )

        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        model_prev_list.append(denoised)
        sigma_prev_list.append(sigma)

        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t

        if sigma_next == 0:
            x = denoised
        elif len(model_prev_list) == 1 or order == 1:
            # first-order: Euler in log-sigma space (DPM-Solver++(1))
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            # second-order multistep corrector (DPM-Solver++(2M) style)
            t_prev = t_fn(sigma_prev_list[-2])
            h_last = t - t_prev
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * model_prev_list[-2]
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d

        # keep history window at size = order
        if len(model_prev_list) > order:
            model_prev_list.pop(0)
            sigma_prev_list.pop(0)

    return x


# ---------------------------------------------------------------------------
# SURE Guided Posterior Sampling (Dec 2024) — trajectory correction via SURE
# Based on https://arxiv.org/html/2512.23232v1
# At each step: denoise → compute SURE gradient on x̂₀ → correct → re-noise.
# NOTE: requires torch.enable_grad; @torch.no_grad is intentionally omitted.
# ---------------------------------------------------------------------------

import time as _time
import logging as _logging

_sure_logger = _logging.getLogger("sure_sampler")
_sure_logger.setLevel(_logging.INFO)  # TEMP: burn-out debug — remove after diagnosis


def _sure_timer(device):
    """Returns a callable that gives elapsed ms since construction, GPU-accurate on CUDA."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        def elapsed():
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)  # ms
    else:
        t0 = _time.perf_counter()
        def elapsed():
            return (_time.perf_counter() - t0) * 1000.0  # ms
    return elapsed


def _repeat_extra_args(extra_args, n):
    """Repeat batch-dimension of tensor values in extra_args by n times."""
    if n == 1:
        return extra_args
    repeated = {}
    for k, v in extra_args.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            repeated[k] = v.repeat(n, *([1] * (v.dim() - 1)))
        else:
            repeated[k] = v
    return repeated

@functools.lru_cache
def _mc_jac_trace_grad(model, x0_hat, sigma_hat, s_in, extra_args, n_mc, eps_mc):
    """MC Jacobian trace + gradient w.r.t. x0_hat in a single eager forward+backward.

    Strategy
    --------
    * Calls the *unwrapped* eager model (model._orig_mod) to stay outside the
      CUDA-graph pool — no "untracked tensors" error.
    * Uses gradient checkpointing on the eager forward so activations are
      recomputed during backward instead of stored.  Since this is eager code,
      checkpointing is safe and halves the peak activation VRAM.
    * Differentiates w.r.t. x_noisy_p (the perturbed input) rather than x0_hat
      to keep the autograd graph short.  The two are related by
          x_noisy_p = x0_hat + ε·b + σ·noise   (∂x_noisy_p/∂x0_hat = I)
      so the VJP result is the same, but we avoid repeating x0_hat in the graph.

    Returns (jac_trace_value: float, jac_trace_grad: Tensor on CPU)
    where jac_trace_grad ≈ (J_D^T b - b) / (ε · n_mc)  as a tensor shaped like x0_hat.
    """
    eager_model = getattr(model, '_orig_mod', model)

    spatial   = [1] * (x0_hat.dim() - 1)
    x0_detach = x0_hat.detach()
    b         = torch.randn_like(x0_detach.repeat(n_mc, *spatial))   # (n_mc·B, C, H, W)
    noise     = torch.randn_like(b)
    x0_rep    = x0_detach.repeat(n_mc, *spatial)
    x_noisy_p = (x0_rep + eps_mc * b + sigma_hat * noise).detach()

    s_in_rep  = s_in.repeat(n_mc)
    extra_rep = _repeat_extra_args(extra_args, n_mc)
    sigma_in  = (sigma_hat * s_in_rep).detach()

    # Free reserved-but-unused CUDA blocks before the expensive backward pass so
    # the activation recompute during checkpoint backward has maximum headroom.
    if x0_hat.device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.enable_grad():
        x_noisy_p = x_noisy_p.requires_grad_(True)

        # Checkpointed forward through the eager model — activations are NOT stored;
        # they are recomputed during backward.  Peak VRAM ≈ one forward, not two.
        d_perturbed = torch.utils.checkpoint.checkpoint(
            lambda xp: eager_model(xp, sigma_in, **extra_rep),
            x_noisy_p,
            use_reentrant=False,
        )

        # jac_trace value (scalar) — x0_rep already detached, no grad flows there
        jac_trace_val = float(
            (b * (d_perturbed.detach() - x0_rep)).sum() / (eps_mc * n_mc)
        )

        # VJP: ∂(b·d_perturbed)/∂x_noisy_p = J_D^T b
        # upstream gradient for d_perturbed is b / (eps_mc * n_mc)
        (jD_T_b,) = torch.autograd.grad(
            d_perturbed, x_noisy_p,
            grad_outputs=b / (eps_mc * n_mc),
        )

    # ∂jac_trace/∂x0 = J_D^T b / (ε·n) − b / (ε·n)
    # (the second term comes from d/dx0 of the −x0_rep subtraction)
    # For n_mc > 1, sum over the MC batch dimension before returning.
    grad_raw = jD_T_b - b / (eps_mc * n_mc)              # shape (n_mc·B, C, H, W)
    grad_x0  = grad_raw.view(n_mc, *x0_hat.shape).sum(0) # sum MC samples → (B, C, H, W)
    result   = jac_trace_val, grad_x0.detach().cpu()

    # Explicitly release the backward graph and large intermediates so the CUDA
    # caching allocator reclaims activation VRAM before the next denoiser call.
    del d_perturbed, jD_T_b, grad_raw, grad_x0, b, noise, x_noisy_p, x0_rep
    if x0_hat.device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _mc_jac_trace(model, x0_hat, sigma_hat, s_in, extra_args, n_mc, eps_mc):
    """Batched Monte Carlo Jacobian trace: one model call for all n_mc samples.

    Repeats the batch along dim-0 by n_mc, runs a single forward pass, then
    reduces — replacing n_mc sequential model calls with one larger call.
    tr(J) ≈ mean_k [ b_k^T (D(x̂₀ + ε·b_k) - x̂₀) / ε ]
    """
    spatial = [1] * (x0_hat.dim() - 1)

    t_prep = _sure_timer(x0_hat.device)
    x0_rep = x0_hat.detach().repeat(n_mc, *spatial)
    b = torch.randn_like(x0_rep)
    # Fuse perturb + re-noise into one expression; free x0_rep early to avoid
    # holding it alongside the model activations during the forward pass.
    x_noisy_p = x0_rep + eps_mc * b + sigma_hat * torch.randn_like(x0_rep)
    del x0_rep  # free before the expensive model call
    s_in_rep = s_in.repeat(n_mc)
    extra_rep = _repeat_extra_args(extra_args, n_mc)
    ms_prep = t_prep()

    t_fwd = _sure_timer(x0_hat.device)
    d_perturbed = model(x_noisy_p, sigma_hat * s_in_rep, **extra_rep).detach()
    del x_noisy_p  # no longer needed
    ms_fwd = t_fwd()

    t_reduce = _sure_timer(x0_hat.device)
    # Reuse b for the reduce instead of repeating x0_hat again (saves one full copy).
    x0_ref = x0_hat.detach().repeat(n_mc, *spatial)
    jac_trace = (b * (d_perturbed - x0_ref)).sum() / (eps_mc * n_mc)
    ms_reduce = t_reduce()

    _sure_logger.debug(
        "[MC-jac]  prep=%.1fms  batched-fwd(n_mc=%d)=%.1fms  reduce=%.1fms",
        ms_prep, n_mc, ms_fwd, ms_reduce,
    )
    return jac_trace


def _sure_correct(model, x, sigma, s_in, extra_args, alpha=0.1, n_mc=1, eps_mc=1e-3, use_jac=True, _step_tag=""):
    """Single SURE gradient correction step in EDM sigma-space.

    Computes an unbiased MSE estimate (SURE) of the denoising error and takes
    one gradient step on x̂₀ to reduce it, then returns the corrected estimate.

    sigma:   current noise level (scalar tensor)
    alpha:   step size for the SURE gradient descent
    n_mc:    number of Monte Carlo samples for Jacobian trace estimate (batched)
    eps_mc:  finite-difference epsilon for Jacobian-vector product
    """
    # torch.inference_mode() cannot be overridden by enable_grad(), unlike no_grad().
    # Detect it by probing whether enable_grad actually re-enables autograd.
    try:
        with torch.enable_grad():
            _probe = torch.tensor(1.0, requires_grad=True)
            _probe2 = _probe * 2
        _autograd_available = bool(_probe2.requires_grad)
    except Exception:
        _autograd_available = False

    if not _autograd_available:
        import warnings
        warnings.warn(
            "SURE sampler: autograd is unavailable (torch.inference_mode() is active). "
            "Using analytical gradient: grad = -2 * residual with sigma-adaptive step. "
            "Pass --no-inference-mode (or omit --inference-mode) to enable exact autograd.",
            stacklevel=2,
        )

    _SAT = 4.0

    # --- main denoiser call (compiled path, no grad needed) ---
    with torch.no_grad():
        t_denoise = _sure_timer(x.device)
        x_in  = x.detach()
        x0_hat = model(x_in, sigma * s_in, **extra_args).detach()
        ms_denoise = t_denoise()

        residual   = x_in - x0_hat
        sigma_hat  = sigma.clamp(min=eps_mc)
        sigma2     = sigma_hat ** 2

        _sat_pct   = (x0_hat.abs() > _SAT).float().mean().item() * 100
        _ch_sat_x0 = [(x0_hat[0, c].abs() > _SAT).float().mean().item() * 100
                      for c in range(x0_hat.shape[1])] if x0_hat.dim() >= 2 else []
        _sure_logger.info(
            "[sure%s] x0_hat  min=%.4f  max=%.4f  mean=%.4f  std=%.4f  sat%%=%.2f  sat_per_ch=%s  nan=%s  inf=%s",
            _step_tag,
            x0_hat.min().item(), x0_hat.max().item(),
            x0_hat.mean().item(), x0_hat.std().item(), _sat_pct,
            [f"{v:.1f}%" for v in _ch_sat_x0],
            bool(torch.isnan(x0_hat).any()), bool(torch.isinf(x0_hat).any()),
        )
        _sure_logger.info(
            "[sure%s] residual  min=%.4f  max=%.4f  mean=%.4f  ||r||²=%.4f",
            _step_tag,
            residual.min().item(), residual.max().item(),
            residual.mean().item(), (residual ** 2).sum().item(),
        )

    # --- Phase 1: analytic residual gradient (-2·residual), no graph needed ---
    residual_grad = (-2.0 * residual).detach()

    # --- Phase 2: ∇_x̂₀ tr(J_D) via autograd through the eager (unwrapped) model ---
    # _mc_jac_trace_grad uses model._orig_mod to bypass torch.compile / CUDA graphs,
    # so backward() works without the "untracked pool tensors" error.
    # The grad is CPU-offloaded immediately after backward to keep GPU VRAM flat.
    jac_grad   = None
    jac_trace  = None
    ms_mc      = 0.0
    ms_grad    = 0.0

    if _autograd_available and use_jac:
        try:
            t_mc = _sure_timer(x.device)
            jac_trace, jac_grad = _mc_jac_trace_grad(
                model, x0_hat, sigma_hat, s_in, extra_args, n_mc, eps_mc)
            ms_mc = t_mc()
            # jac_grad already on CPU, jac_trace already a plain float
        except Exception as exc:
            _sure_logger.warning(
                "[sure%s] autograd jac_trace failed (%s) — falling back to residual-only gradient",
                _step_tag, exc,
            )
            jac_grad  = None
            jac_trace = None
    elif not use_jac:
        _sure_logger.info("[sure%s] jac skipped this step (interval)", _step_tag)

    _sure_logger.info(
        "[sure%s] jac_trace=%s  sigma2=%.6f  autograd=%s",
        _step_tag,
        f"{jac_trace:.4f}" if jac_trace is not None else "n/a",
        float(sigma2),
        _autograd_available and jac_grad is not None,
    )

    # --- combine: full SURE gradient or residual-only fallback ---
    with torch.no_grad():
        if jac_grad is not None:
            grad = residual_grad + 2.0 * float(sigma2) * jac_grad.to(x0_hat.device)
        else:
            grad = residual_grad

        x0_std   = x0_hat.std().item()
        grad_std = grad.std().item()
        # Std clip: scale so grad std ≤ x0_std (prevents global magnitude blowup from
        # the 1/eps amplification in jac_grad — at high σ, 2σ²·jac_grad can be ≈1e6)
        if grad_std > x0_std and grad_std > 0.0:
            grad = grad * (x0_std / grad_std)
        # Value clamp: catch heavy-tailed per-pixel outliers after std clip
        clip_val = 3.0 * x0_std
        grad = grad.clamp(-clip_val, clip_val)

        _sure_logger.info(
            "[sure%s] grad  min=%.4f  max=%.4f  mean=%.4f  std=%.4f  nan=%s  inf=%s",
            _step_tag,
            grad.min().item(), grad.max().item(),
            grad.mean().item(), grad.std().item(),
            bool(torch.isnan(grad).any()), bool(torch.isinf(grad).any()),
        )
        _sure_logger.debug(
            "[sure%s] denoise=%.1fms  mc-jac=%.1fms  backward=%.1fms",
            _step_tag, ms_denoise, ms_mc, ms_grad,
        )

        effective_alpha = alpha / (1.0 + sigma_hat.item())
        x0_corrected = x0_hat - effective_alpha * grad

        # jac contribution ratio: what fraction of the CLIPPED gradient comes from jac.
        # Computed post-clip to avoid the raw 1/eps amplification making ratio always ~1.
        # Method: compare grad (with jac) vs resid_only_clipped (without jac) in L2.
        #   ratio = ||grad - resid_only_clipped|| / (||grad|| + ε)
        # Near 0 → jac adds negligible correction; near 1 → jac dominates correction.
        if jac_grad is not None:
            _resid_only = residual_grad.clone()
            _ro_std = _resid_only.std().item()
            if _ro_std > x0_std and _ro_std > 0.0:
                _resid_only = _resid_only * (x0_std / _ro_std)
            _resid_only = _resid_only.clamp(-clip_val, clip_val)
            _jac_delta  = grad - _resid_only          # what jac actually adds post-clip
            _jac_ratio  = float(_jac_delta.norm() / (grad.norm() + 1e-8))
        else:
            _jac_ratio = None

        _ch_sat_corr = [(x0_corrected[0, c].abs() > _SAT).float().mean().item() * 100
                        for c in range(x0_corrected.shape[1])] if x0_corrected.dim() >= 2 else []
        _sure_logger.info(
            "[sure%s] effective_alpha=%.5f  jac_ratio=%.3f  x0_corrected  min=%.4f  max=%.4f  mean=%.4f  std=%.4f  sat_per_ch=%s  nan=%s  inf=%s",
            _step_tag, effective_alpha,
            _jac_ratio if _jac_ratio is not None else float('nan'),
            x0_corrected.min().item(), x0_corrected.max().item(),
            x0_corrected.mean().item(), x0_corrected.std().item(),
            [f"{v:.1f}%" for v in _ch_sat_corr],
            bool(torch.isnan(x0_corrected).any()), bool(torch.isinf(x0_corrected).any()),
        )

        # --- stats for adaptive control in the outer loop ---
        _residual_mse = float((residual ** 2).mean())
        _sigma2_val   = float(sigma2)

        _stats = {
            'residual_mse':  _residual_mse,
            'sigma2':        _sigma2_val,
            'x0_std':        x0_std,
            'grad_std':      grad_std,
            'jac_ratio':     _jac_ratio,  # None when jac was skipped or failed
        }

    return x0_corrected, _stats

@functools.lru_cache
def _pca_noise_estimate(x0_hat, patch_size=8, min_sigma=1e-3):
    """PCA noise estimation via Marchenko-Pastur bulk threshold.

    Replaces the old mean≈median heuristic with a theoretically grounded
    random-matrix estimator.  For a covariance matrix built from n_patches
    patches of dimension dim, pure-noise eigenvalues lie in the Marchenko-Pastur
    bulk  [0, σ²·(1+√γ)²]  where γ = dim/n_patches.  We iteratively solve for
    the noise variance σ² as the mean of the bulk eigenvalues.

    Two additional robustness improvements over the original:
      • Adaptive patch_size: halves until n_patches > dim (γ < 1), keeping the
        sample covariance full-rank.  This matters for small latents (SD1.5
        64×64) where the default patch_size=8 would give γ = 4.
      • All batch items: estimates are averaged across the batch rather than
        using only item 0, giving a lower-variance estimate with batch > 1.

    Shape: (B, C, H, W) → scalar float.
    """
    estimates = []
    for b in range(x0_hat.shape[0]):
        estimates.append(_pca_noise_single(x0_hat[b].detach().float(),
                                           patch_size, min_sigma))
    return float(sum(estimates) / len(estimates)) if estimates else min_sigma


def _pca_noise_single(img, patch_size=8, min_sigma=1e-3):
    """Marchenko-Pastur noise estimate for one (C, H, W) tensor."""
    C, H, W = img.shape

    # ── Adaptive patch_size: ensure n_patches > dim (γ < 1) ─────────────────
    ps = patch_size
    while ps > 2:
        ph = H // ps
        pw = W // ps
        if ph * pw > C * ps * ps:   # n_patches > dim → γ < 1, well-conditioned
            break
        ps = ps // 2

    ph = H // ps
    pw = W // ps
    if ph < 2 or pw < 2:
        return min_sigma

    n_patches = ph * pw
    dim       = C * ps * ps

    # ── Non-overlapping patches ───────────────────────────────────────────────
    cropped = img[:, :ph * ps, :pw * ps]
    unf     = cropped.unfold(1, ps, ps).unfold(2, ps, ps)
    patches = unf.permute(1, 2, 0, 3, 4).reshape(n_patches, dim)

    # ── Sample covariance eigenvalues ─────────────────────────────────────────
    mu      = patches.mean(0, keepdim=True)
    centered = patches - mu
    cov     = (centered.T @ centered) / (n_patches - 1)   # (dim, dim)
    try:
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)   # ascending
    except Exception:
        return min_sigma

    # ── Marchenko-Pastur bulk threshold ───────────────────────────────────────
    # γ = dim/n_patches < 1 by construction above.
    # Bulk upper edge: λ_+ = σ²·(1 + √γ)²
    # Iterate: σ² ← mean of eigenvalues in [0, λ_+] until convergence.
    gamma      = dim / n_patches          # < 1
    mp_edge_k  = (1.0 + gamma ** 0.5) ** 2   # λ_+ = σ² × mp_edge_k

    sigma2 = float(eigvals.mean())        # initial estimate: full mean
    for _ in range(20):
        bulk = eigvals[eigvals <= sigma2 * mp_edge_k]
        if len(bulk) == 0:
            sigma2 = float(eigvals.min())
            break
        sigma2_new = float(bulk.mean())
        if abs(sigma2_new - sigma2) / (abs(sigma2) + 1e-10) < 1e-5:
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    return max(float(sigma2 ** 0.5), min_sigma)


def _sure_effective_alpha(alpha: float, sigma_t, adam_active: bool) -> float:
    """Effective gradient-descent step size validated by Lean (SURE_verification.lean).

    Lean theorem cond4_step_closer proves descent is guaranteed for any
    effective_alpha ∈ (0, 1/2).  The formula alpha / (1 + σ_norm) satisfies
    this bound for all σ_norm ≥ 0 whenever alpha < 1/2 (standard choice).

    Scheduler auto-correction
    ─────────────────────────
    Flow-matching / VP schedulers  (σ_t ∈ [0, 1]):
        σ_norm = σ_t → effective_alpha ∈ [alpha/2, alpha]  ← unchanged.

    EDM / Karras schedulers  (σ_t up to ~80 for long schedules):
        Without correction: effective_alpha ≈ alpha/80 at early steps →
        SURE takes near-zero steps despite a valid gradient direction.
        Auto-detected when σ_t > 1; normaliser capped at 1 so that
        effective_alpha ≥ alpha/2 throughout all noise levels.

    Lean bound (effectiveAlpha_lt_half): alpha/(1 + min(σ_t, 1)) < 1/2
    holds for all σ_t ≥ 0 and alpha < 1/2, keeping cond4 always satisfied.

    Gap #2 fix: hard-clamp the result to < 0.5 with a warning.  The Lean proof
    requires η ∈ (0, 1/2) strictly; if a caller passes alpha ≥ 0.5 the formula
    can exceed this bound at low noise levels (σ_norm ≈ 0), violating Theorem A.3.

    Adam mode: step magnitude is gradient-normalised internally; sigma
    scaling would double-suppress early steps and is skipped.
    """
    _MAX_SURE_ETA = 0.499   # strict upper bound from Theorem A.3 / cond4_step_closer

    if adam_active:
        result = float(alpha)
    elif sigma_t is None or float(sigma_t) <= 0.0:
        result = float(alpha)
    else:
        sigma_t_f = float(sigma_t)
        # Scheduler auto-detection: σ_t > 1 → EDM/Karras, cap normaliser at 1
        sigma_norm = min(sigma_t_f, 1.0)
        result = float(alpha) / (1.0 + sigma_norm)

    # Gap #2: Theorem A.3 (Gaussian preservation) requires η < 1/2.
    # Clamp with a one-time warning so callers with large alpha values don't
    # silently violate the descent guarantee.
    if result >= 0.5:
        _sure_logger.warning(
            "[sure] Gap #2: effective_alpha=%.4f ≥ 0.5 (alpha=%.4f, sigma_t=%s); "
            "clamping to %.3f.  Theorem A.3 requires η < 0.5 for Gaussian "
            "preservation.  Reduce alpha below 0.5 to silence this warning.",
            result, float(alpha),
            f"{float(sigma_t):.4f}" if sigma_t is not None else "None",
            _MAX_SURE_ETA,
        )
        result = _MAX_SURE_ETA
    return result


def _sure_alpha_bo_search(
    residual,
    grad,
    sigma2,
    jac_trace,
    alpha_base,
    alpha_range=None,
    n_trials=12,
    patience=4,
    n_startup=3,
    prev_study=None,
    prev_sure=None,
    prev_prev_sure=None,
    cur_alpha=None,
):
    """Bayesian optimisation over the SURE gradient step size α.

    Objective: quadratic stop-gradient proxy (zero UNet calls per trial):
        SURE_proxy(α) = −n·σ² + ‖residual − α·grad‖² + 2σ²·jac_trace

    Under this proxy the optimum is the closed-form α* = (r·g)/‖g‖²,
    used as the analytical fallback when optuna is not installed.

    BO adds value over the closed form when the quadratic approximation is
    inaccurate (large steps, highly non-linear denoiser) and for cross-step
    landscape exploration via warm-starting from prev_study.

    When grad is collinear with residual (approx mode), the proxy degenerates
    and a SURE-monotone P-controller is used instead: alpha grows by 5% when
    SURE fell last step, shrinks by 30% when SURE rose, and holds otherwise.

    Args:
        residual:       detached tensor (x0_hat − x_hat)
        grad:           detached SURE gradient tensor
        sigma2:         float  σ̂₀²
        jac_trace:      float or None  — tr{J_D} MC estimate (0.0 if None)
        alpha_base:     float  user-set sure_alpha (centres the log-scale range)
        alpha_range:    (lo, hi) or None → auto (alpha_base×0.01, min(alpha_base×20, 0.49))
        n_trials:       BO trial budget
        patience:       early-stop after this many non-improving trials
        n_startup:      TPE random-exploration trials before fitting the model
        prev_study:     optuna Study from the previous diffusion step (warm-start)
        prev_sure:      SURE value from the previous diffusion step (P-controller)
        prev_prev_sure: SURE value from two steps ago (P-controller trend signal)
        cur_alpha:      alpha used at the previous step (P-controller state)

    Returns:
        (best_alpha: float, study_or_None)
    """
    n = residual.numel()
    jt = float(jac_trace) if jac_trace is not None else 0.0

    # Auto-range: log-scale around alpha_base; hard-cap at 0.49 (Lean bound)
    if alpha_range is None:
        lo = max(alpha_base * 0.01, 1e-5)
        hi = min(alpha_base * 20.0, 0.49)
        if lo >= hi:
            hi = min(lo * 10.0, 0.49)
    else:
        lo, hi = float(alpha_range[0]), float(alpha_range[1])

    r_flat = residual.detach().float().reshape(-1)
    g_flat = grad.detach().float().reshape(-1)

    def _proxy_sure(alpha_val):
        return float(
            -n * sigma2
            + ((r_flat - alpha_val * g_flat) ** 2).sum()
            + 2.0 * sigma2 * jt
        )

    # Analytical optimum under the quadratic proxy (closed-form line search)
    r_sq = float((r_flat * r_flat).sum()) + 1e-12
    g_sq = float((g_flat * g_flat).sum()) + 1e-12
    dot  = float((r_flat * g_flat).sum())
    alpha_analytical = float(max(lo, min(hi, dot / g_sq)))

    # Degenerate proxy guard: when grad is collinear with residual (e.g. approx
    # mode where grad = coeff·r), the proxy minimum α* = 1/coeff is
    # sigma-independent and diverges at late low-noise steps.
    # Use a SURE-monotone P-controller instead: track SURE trend across steps
    # and shrink alpha when SURE grows, grow it slightly when SURE falls.
    cos_rg = abs(dot) / (r_sq ** 0.5 * g_sq ** 0.5)
    if cos_rg > 0.99:
        _ref = cur_alpha if cur_alpha is not None else alpha_base
        if prev_sure is not None and prev_prev_sure is not None:
            if prev_sure > prev_prev_sure * 1.05:      # SURE rose >5% → pull back
                new_alpha = max(lo, _ref * 0.70)
            elif prev_sure < prev_prev_sure * 0.95:    # SURE fell >5% → push forward
                new_alpha = min(hi, _ref * 1.05)
            else:                                       # stable → hold
                new_alpha = _ref
        else:
            new_alpha = _ref  # insufficient history
        _sure_logger.info(
            "[sure_alpha_bo] grad collinear (|cos|=%.4f); P-ctrl: "
            "sure %.2f→%.2f  alpha %.5f→%.5f",
            cos_rg,
            prev_prev_sure if prev_prev_sure is not None else float('nan'),
            prev_sure      if prev_sure      is not None else float('nan'),
            _ref, new_alpha,
        )
        return new_alpha, None

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        study = _optuna.create_study(
            direction='minimize',
            sampler=_optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=n_startup,
                multivariate=False,
            ),
        )

        # Warm-start: inject previous step's observations without decay
        if prev_study is not None:
            for t in prev_study.trials:
                if t.value is not None and t.value != float('inf'):
                    try:
                        study.add_trial(
                            _optuna.trial.create_trial(
                                params=t.params,
                                distributions=t.distributions,
                                value=t.value,
                            )
                        )
                    except Exception:
                        pass  # distribution mismatch on first step — skip

        # Seed with the analytical optimum as the first trial
        study.enqueue_trial({'alpha': alpha_analytical})

        _no_improve = [0]
        _best_val   = [float('inf')]

        def _bo_callback(study, trial):
            val = study.best_value
            if val < _best_val[0]:
                _best_val[0]   = val
                _no_improve[0] = 0
            else:
                _no_improve[0] += 1
            if _no_improve[0] >= patience:
                study.stop()

        def _objective(trial):
            alpha_val = trial.suggest_float('alpha', lo, hi, log=True)
            return _proxy_sure(alpha_val)

        study.optimize(
            _objective,
            n_trials=n_trials,
            callbacks=[_bo_callback],
            show_progress_bar=False,
        )

        best_alpha = float(study.best_params['alpha'])
        _sure_logger.info(
            "[sure_alpha_bo] lo=%.5f  hi=%.5f  analytical=%.5f  bo_best=%.5f  "
            "sure_proxy=%.4f  trials=%d",
            lo, hi, alpha_analytical, best_alpha,
            study.best_value, len(study.trials),
        )
        return best_alpha, study

    except ImportError:
        _sure_logger.warning(
            "[sure_alpha_bo] optuna not installed — using analytical α*=%.5f. "
            "Install with: pip install optuna",
            alpha_analytical,
        )
        return alpha_analytical, None


def _sure_correct_x0(model, x0_hat, sigma_hat_0, s_in, extra_args,
                     alpha=0.05, n_mc=1, eps_mc=1e-3, use_jac=True,
                     sigma_t=None,
                     adam_state=None, adam_mode='none',
                     adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                     adam_wd=0.01, grad_mode='vjp',
                     approx_coeff=2.0,
                     csv_writer=None, step_idx=None,
                     alpha_bo_trials=0, alpha_bo_patience=4,
                     alpha_bo_state=None):
    """SURE gradient correction per Algorithm 1 of arXiv:2512.23232.

    grad_mode controls how ∇SURE is computed:

      'approx'  — stop-gradient approximation: grad = 2·(xnoisy − x̂)
                  No backward pass; fast but ignores J_D entirely.

      'vjp'     — exact gradient of ‖xnoisy − x̂‖² via one backward pass:
                  grad = 2·(I − J_D)^T·(xnoisy − x̂)
                  Includes the Jacobian correction from the chain rule.
                  The 2σ²·∇tr{J} term is still omitted (O(σ²) small).

      'full'    — full ∇SURE via two backward passes (through both Dθ(x) and
                  Dθ(x+εb)). Adds the MC-estimated Jacobian-trace gradient:
                  grad = 2·(I − J_D)^T·r  +  2σ²·(1/ε)·[J_D(x+εb)^T·b − J_D(x)^T·b]
                  Requires use_jac=True; falls back to 'vjp' otherwise.
                  Most expensive (three UNet passes with grad, two backward passes).

      ε       = max(|xnoisy|) / 1000   (paper §2.4 recommended choice)
      x̂      = Dθ(xnoisy, σ̂₀)
      tr{J}  ≈ b^T (Dθ(xnoisy + ε·b, σ̂₀) − x̂) · ε^{-1}   [scalar, MC averaged]
      SURE   = −n·σ̂₀² + ‖xnoisy − x̂‖² + 2·σ̂₀²·tr{J}      [logged only]
      Note: both denoiser calls use σ̂₀ (not max(ε,σ̂₀)); see SURE_sgps_full.lean §2.1.

    adam_state: dict with keys 'optimizer' (torch.optim instance) and 'param'
                (leaf tensor) — mutated in-place across steps via optimizer.step().
                Pass None to use plain gradient descent (default behaviour).
    adam_mode:  'none' = plain SGD, 'adam' = Adam, 'adamw' = AdamW.
                RAdam/RAdamW were removed: with β₂=0.999 and only ~60 diffusion
                steps, ρ_t barely clears the SGD-fallback threshold (ρ_∞=5) by
                step 6 and the variance-rectification factor rect≈0.021 at that
                point — effectively zero adaptive correction for the whole run.
    adam_beta1: first-moment decay (default 0.9).
    adam_beta2: second-moment decay (default 0.999).
    adam_eps:   denominator stabiliser (default 1e-8).
    adam_wd:    weight decay for AdamW only — decoupled from moment updates,
                shrinks x0_hat toward zero to anchor the correction near T₀.
    """
    # ε: paper says max_pixel / 1000; clamp to eps_mc for numerical safety.
    # clamp on-device before float() to avoid a CPU round-trip via Python max().
    eps = float(x0_hat.abs().max().clamp(min=eps_mc * 1000.0)) / 1000.0

    device = x0_hat.device
    sigma_denoiser = torch.tensor(sigma_hat_0, device=device, dtype=x0_hat.dtype)
    # Use sigma_denoiser for BOTH the base and perturbed calls.
    # SURE_sgps_full.lean §2.1 (Gap #1) proves that using max(ε, σ̂₀) for the
    # perturbed call while using σ̂₀ for the base call evaluates two *different*
    # denoiser functions, introducing a systematic bias of O(|σ_p − σ̂₀|/ε) in
    # the Hutchinson trace estimate (sigma_inconsistency_bias_structure).
    # Numerical stability when σ̂₀ is near zero is handled by use_jac=False below
    # (sigma2 ≤ eps_mc disables the Jacobian call entirely).
    sigma_p = sigma_denoiser
    sigma2  = float(sigma_denoiser ** 2)
    n       = x0_hat.numel()
    # Opt-2: 2σ²·tr{J} and its gradient vanish quadratically as σ̂₀→0.
    # Skip the perturbed UNet forward entirely when the contribution is negligible.
    use_jac = use_jac and (sigma2 > eps_mc)

    def _release_cache():
        # Return freed activation buffers to the OS/driver so other allocations can use them.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    # ── Pre-draw probe vectors ────────────────────────────────────────────────
    # For 'full' mode draw all n_mc probes BEFORE Pass 1 so their sum can be
    # folded into a combined scalar, replacing two backward passes with one.
    need_full_grad = (grad_mode == 'full') and use_jac
    bs = [torch.randn_like(x0_hat) for _ in range(n_mc)] if use_jac else []
    b_sum = torch.stack(bs).sum(0) if need_full_grad else None

    # ── Pass 1: x̂ = Dθ(xnoisy, σ̂₀) ─────────────────────────────────────────
    # 'approx' needs no grad graph; 'vjp'/'full' require enable_grad to override
    # any outer no_grad context (e.g. CFG wrapper).
    #
    # Flush the allocator pool before any grad-enabled forward.  The outer
    # sampling step ran a no_grad forward whose activations are freed but may
    # still sit in the MPS private pool / CUDA caching allocator.  Releasing
    # them here gives back headroom before we store activation graphs.
    if grad_mode != 'approx':
        _release_cache()

    x_in = x0_hat.detach().requires_grad_(grad_mode != 'approx')
    if grad_mode == 'approx':
        with torch.no_grad():
            x_hat = model(x_in, sigma_denoiser * s_in, **extra_args).detach()
        residual = x_in - x_hat
        grad = approx_coeff * residual  # stop-gradient approximation
    else:
        # Gradient checkpointing: recompute activations during backward instead
        # of storing them, trading one extra forward pass for ~halved peak VRAM.
        def _model_ckpt(x):
            return model(x, sigma_denoiser * s_in, **extra_args)

        with torch.enable_grad():
            x_hat    = torch.utils.checkpoint.checkpoint(
                           _model_ckpt, x_in, use_reentrant=False)
            residual = x_in - x_hat
            residual_sq = (residual ** 2).sum()

            if need_full_grad:
                # Opt-1: one backward replaces two.
                # C = ||r||² − (2σ²/(n_mc·ε))·(b_sum·x̂)
                # ∂C/∂x = 2(I−J_D)ᵀr − (2σ²/(n_mc·ε))·J_D(x)ᵀ·b_sum
                # No retain_graph needed; graph freed immediately after this pass.
                combined = residual_sq - (2.0 * sigma2 / (n_mc * eps)) * (b_sum * x_hat).sum()
                grad = torch.autograd.grad(combined, x_in)[0]
                del combined
            else:
                # 'vjp': single backward, graph freed immediately.
                grad = torch.autograd.grad(residual_sq, x_in)[0]
            residual_sq_val = float(residual_sq)  # Bug-2: save before del
            del residual_sq

        x_hat    = x_hat.detach()
        residual = residual.detach()
        _release_cache()   # free activation pool after Pass 1 backward(s)

    # ── Pass 2 (optional): tr{J} scalar + optional ∇tr{J} for 'full' mode ────
    jac_trace = 0.0 if use_jac else None
    # Bug-2: use saved scalar in vjp/full; recompute cheaply in approx (no graph).
    if grad_mode == 'approx':
        sure_val = float(-n * sigma2 + (residual ** 2).sum())
    else:
        sure_val = -n * sigma2 + residual_sq_val

    if use_jac:
        grad_jac_pert_sum = torch.zeros_like(grad) if need_full_grad else None

        for b in bs:
            x_in_pert = x_in.detach() + eps * b

            if need_full_grad:
                # Forward through Dθ(x+εb) with grad for VJP.
                # Flush pool before each grad-enabled perturbed forward.
                _release_cache()
                x_in_pert = x_in_pert.requires_grad_(True)
                def _model_pert_ckpt(x):
                    return model(x, sigma_p * s_in, **extra_args)
                with torch.enable_grad():
                    x_pert_hat   = torch.utils.checkpoint.checkpoint(
                                       _model_pert_ckpt, x_in_pert, use_reentrant=False)
                    jac_scalar_p = (b * x_pert_hat).sum() / eps
                    gj_pert      = torch.autograd.grad(jac_scalar_p, x_in_pert)[0]
                # Bug-1 fix: use (x_pert_hat − x_hat) so tr{J} matches the paper formula.
                jac_trace += float((b * (x_pert_hat - x_hat)).sum().detach()) / eps
                del x_pert_hat
                _release_cache()
                grad_jac_pert_sum = grad_jac_pert_sum + gj_pert
                del gj_pert
            else:
                with torch.no_grad():
                    x_pert_hat = model(x_in_pert, sigma_p * s_in, **extra_args).detach()
                jac_trace += float((b * (x_pert_hat - x_hat)).sum()) / eps
                del x_pert_hat

        jac_trace /= n_mc
        jac_contribution = 2.0 * sigma2 * jac_trace
        sure_val += jac_contribution

        # ── Lean-backed jac reliability gate ───────────────────────────────────
        # Lean (SURE_verification.lean, cond5_strict_descent) proves descent is
        # guaranteed with proxy gradient 2·r regardless of jac.
        #
        # Gap #6 fix: variance-based Hutchinson floor (3σ bound).
        # For b ~ N(0,I), Var(b^T J b) = 2‖J‖_F² ≤ 2n (unit operator norm).
        # With n_mc samples: Std(jac_trace estimate) ≤ sqrt(2n/n_mc).
        # Floor = 0 − 3·Std = −3·sqrt(2n/n_mc).  This is O(sqrt(n)) tighter
        # than the old −n/n_mc floor (O(n)), so noise-dominated estimates are
        # caught much earlier (e.g. n=16384, n_mc=1: −543 vs old −16384).
        _jac_floor = -3.0 * (2.0 * n / max(n_mc, 1)) ** 0.5
        if jac_trace < _jac_floor:
            _sure_logger.warning(
                "[sure_x0] Gap #6: jac_trace=%.2f < 3σ floor %.2f (n=%d, n_mc=%d); "
                "Hutchinson estimate noise-dominated (floor = −3·sqrt(2n/n_mc)). "
                "Rolling back to proxy gradient (Lean: cond5_strict_descent). "
                "Use n_mc>=8 for reliable jac estimates.",
                jac_trace, _jac_floor, n, n_mc,
            )
            sure_val -= jac_contribution   # undo the unreliable jac contribution
            jac_trace = None               # mark as unreliable for downstream logging

        if need_full_grad:
            if jac_trace is not None:
                # Opt-1: base Jac term already folded into grad via combined scalar in Pass 1.
                # ∇tr{J} ≈ (1/ε)·Σ J_D(x+εb_i)^T·b_i / n_mc  (no subtraction needed)
                grad = grad + (2.0 * sigma2 / n_mc) * grad_jac_pert_sum
            else:
                # Jac unreliable: discard jac-contaminated grad from Pass 1;
                # fall back to Lean-proven proxy gradient: 2·(x̂₀ − D_θ(x̂₀)).
                grad = (2.0 * residual).detach()
            del grad_jac_pert_sum

    # ── Gap #5: sigma self-consistency bias correction ────────────────────────
    # SURE is unbiased when σ used in the formula equals the true noise level in
    # x0_hat.  The PCA estimate σ̂₀ may differ from the residual-implied level
    # σ_res = sqrt(‖r‖²/n).  The bias (proved in SURE_sgps_full.lean §1, theorem
    # sure_sigma_bias_exact) is:
    #   SURE(σ̂₀) − SURE(σ_res) = (σ̂₀² − σ_res²)·(−n + 2·tr_J)
    # Subtracting it gives a bias-corrected value on the σ_res² basis.
    # When tr_J is unavailable (use_jac=False or jac rejected), we use 0 as a
    # conservative lower bound (true tr_J ≥ 0 for contractive denoiser).
    _sigma_sq_residual = float((residual ** 2).sum()) / max(n, 1)
    _tr_J_bc = jac_trace if jac_trace is not None else 0.0
    _sure_bias = (sigma2 - _sigma_sq_residual) * (-float(n) + 2.0 * _tr_J_bc)
    sure_val_bc = sure_val - _sure_bias
    if abs(_sure_bias) > 0.1 * (abs(sure_val) + 1.0):
        _sure_logger.debug(
            "[sure_x0] Gap #5: sigma_bias_correction=%.4f  sigma_hat_0=%.5f  "
            "sigma_residual=%.5f  sure_raw=%.4f  sure_bc=%.4f",
            _sure_bias, sigma_hat_0, _sigma_sq_residual ** 0.5,
            sure_val, sure_val_bc,
        )

    x0_std = x0_hat.std().item()
    gs     = grad.std().item()
    # Clamp before Adam: Adam's v_t accumulates grad², so a single outlier step
    # (e.g. huge SURE residual at step 0) inflates v_t and suppresses all future
    # updates for the rest of the run.  Normalising to x0_std then hard-clamping
    # to ±3σ keeps v_t in a reliable range without discarding gradient direction.
    #if gs > x0_std > 0.0:
        #grad = grad * (x0_std / gs)
    #grad = grad.clamp(-3.0 * x0_std, 3.0 * x0_std)

    # ── BO alpha search (optional) ────────────────────────────────────────────
    # Uses the quadratic stop-gradient proxy to evaluate SURE at different α
    # values without any additional UNet calls.  Warm-starts from the previous
    # diffusion step's study via alpha_bo_state['study'].
    if alpha_bo_trials > 0:
        _abs = alpha_bo_state if alpha_bo_state is not None else {}
        alpha, _new_alpha_study = _sure_alpha_bo_search(
            residual.detach(),
            grad.detach(),
            sigma2,
            jac_trace if jac_trace is not None else 0.0,
            alpha_base=alpha,
            n_trials=int(alpha_bo_trials),
            patience=int(alpha_bo_patience),
            prev_study=_abs.get('study'),
            prev_sure=_abs.get('prev_sure'),
            prev_prev_sure=_abs.get('prev_prev_sure'),
            cur_alpha=_abs.get('cur_alpha', alpha),
        )
        if alpha_bo_state is not None:
            alpha_bo_state['study']     = _new_alpha_study
            alpha_bo_state['cur_alpha'] = alpha

    # Lean (cond4_step_closer) requires effective_alpha ∈ (0, 1/2).
    # _sure_effective_alpha auto-detects EDM vs flow-matching from σ_t magnitude
    # and caps the normaliser at 1 so EDM gets effective_alpha ≥ alpha/2.
    effective_alpha = _sure_effective_alpha(
        alpha, sigma_t,
        adam_active=(adam_state is not None and adam_mode in ('adam', 'adamw')),
    )

    # --- Adam / AdamW via torch.optim ------------------------------------------
    # Each diffusion step is one optimizer iteration. torch.optim.Adam/AdamW
    # owns the moment state; we inject the SURE gradient by assigning .grad
    # directly and calling .step() — no backward pass required.
    #
    # torch.optim.AdamW uses decoupled weight decay (Loshchilov & Hutter 2019):
    #   param ← param·(1 − lr·wd) − lr·m̂/(√v̂ + ε)
    # which matches our previous manual formulation exactly.
    if adam_state is not None and adam_mode in ('adam', 'adamw'):
        if adam_state['optimizer'] is None:
            # Lazy init on first step once we know tensor shape and device
            param = x0_hat.detach().clone().requires_grad_(True)
            if adam_mode == 'adamw':
                opt = torch.optim.AdamW(
                    [param], lr=effective_alpha,
                    betas=(adam_beta1, adam_beta2),
                    eps=adam_eps, weight_decay=adam_wd
                )
            else:
                opt = torch.optim.Adam(
                    [param], lr=effective_alpha,
                    betas=(adam_beta1, adam_beta2),
                    eps=adam_eps, weight_decay=0.0, amsgrad=True
                )
            adam_state['optimizer'] = opt
            adam_state['param'] = param

        param = adam_state['param']
        opt   = adam_state['optimizer']

        # Update lr (effective_alpha can vary per step when sigma scaling is off)
        for pg in opt.param_groups:
            pg['lr'] = effective_alpha

        # Load current x0_hat into the persistent leaf tensor, inject gradient
        param.data.copy_(x0_hat.detach())
        param.grad = grad.detach()

        x_before = param.data.clone()
        opt.step()
        x0_corrected = param.data.detach().clone()

        step_delta = x_before - x0_corrected
        # step_rms = actual magnitude Adam applied (lr * adaptive_grad per pixel)
        _step_rms = float((step_delta ** 2).mean().sqrt())
        # effective_grad for adam_ratio logging: what Adam compressed the raw grad to
        effective_grad = step_delta / (effective_alpha + 1e-12)
    else:
        effective_grad = grad
        x0_corrected = (x0_hat - effective_alpha * effective_grad).detach()
        _step_rms = effective_alpha * float((effective_grad ** 2).mean().sqrt())

    _eff_grad_rms  = float((effective_grad ** 2).mean().sqrt())
    _raw_grad_rms  = float((grad ** 2).mean().sqrt())
    _residual_rms  = float((residual ** 2).mean().sqrt())
    _adam_ratio    = _eff_grad_rms / (_raw_grad_rms + 1e-8)
    _sure_logger.info(
        "[sure_x0] eps=%.5f  sigma_hat_0=%.5f  sigma_p=%.5f  lr=%.5f  step_rms=%.5f  "
        "sure=%.4f  sure_bc=%.4f  jac_trace=%s  residual_rms=%.5f  grad_rms=%.5f  "
        "eff_grad_rms=%.5f  adam_ratio=%.4f",
        eps, sigma_hat_0, float(sigma_p), effective_alpha, _step_rms,
        sure_val, sure_val_bc,
        f"{jac_trace:.4f}" if jac_trace is not None else "n/a",
        _residual_rms,
        _raw_grad_rms,
        _eff_grad_rms,
        _adam_ratio,
    )

    if csv_writer is not None and grad_mode == 'approx':
        csv_writer.writerow([
            step_idx,
            f"{sigma_hat_0:.6f}",
            f"{float(sigma_t):.6f}" if sigma_t is not None else "",
            f"{approx_coeff:.4f}",
            f"{_residual_rms:.6f}",
            f"{sure_val_bc:.6f}",
            f"{_raw_grad_rms:.6f}",
            f"{_eff_grad_rms:.6f}",
            f"{effective_alpha:.6f}",
            f"{_step_rms:.6f}",
            f"{_adam_ratio:.6f}",
        ])

    # Feed bias-corrected SURE to P-controller for more reliable alpha adaptation.
    # sure_val_bc removes the (σ̂₀²−σ_res²)·(−n+2·tr_J) bias (Gap #5) so the
    # controller tracks the true MSE trend rather than a sigma-schedule artifact.
    if alpha_bo_state is not None and alpha_bo_trials > 0:
        alpha_bo_state['prev_prev_sure'] = alpha_bo_state.get('prev_sure')
        alpha_bo_state['prev_sure']      = float(sure_val_bc)

    return x0_corrected, {'jac_ratio': None}


def _sure_correct_x0_wavelet(model, x0_hat, sigma_hat_0, s_in, extra_args,
                              alpha=0.05, n_mc=1, eps_mc=1e-3, use_jac=True,
                              sigma_t=None,
                              adam_state=None, adam_mode='bo',
                              adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                              adam_wd=0.01,
                              wavelet='db4', wavelet_level=3,
                              approx_coeff=2.0, warmup_steps=0,
                              lp_frac=1.0,
                              grad_mode='approx',
                              alpha_bo_trials=12, alpha_bo_patience=4,
                              alpha_bo_state=None):
    """SURE-Wavelet: per-subband SURE gradient correction.

    Decomposes x0_hat into orthogonal 2D wavelet subbands via ptwt.wavedec2,
    computes SURE residual gradient independently for the approximation (R) and
    each detail subband (H/V/D at each level L), applies per-subband Adam
    moment tracking, then reconstructs the corrected x0 via ptwt.waverec2.

    grad_mode controls how the per-subband gradient is computed:
      'approx'  — stop-gradient approximation: grad = approx_coeff · r_k  (fast, no backward).
      'vjp'     — one autograd backward on ‖r‖²; pixel-space gradient decomposed into
                  subbands.  Each subband k receives W_k·2(I−J_D)^T r — this mixes all
                  subbands' residuals through J_D^T.
      'vjp_sb'  — per-subband Hutchinson VJP; two no-grad forwards, no backward.
                  For each subband k: g_k = 2r_k − (2/n_mc)Σ_b(b_k^T r_k)(D_k(x+εb)−D_k(x))/ε
                  where b_k = W_k b.  Unbiased estimator of 2(I−W_k J_D W_k^T)^T r_k —
                  truly independent per-subband guidance with no cross-subband Jacobian leakage.
      'full'    — full ∇SURE: adds 2σ²·∇tr{J_D} term on top of vjp; requires
                  use_jac=True, falls back to vjp behavior otherwise.

    lp_frac : float in [0.0, 1.0] (default 1.0)
        Low-pass cutoff fraction over wavelet detail levels.
        0.0 = correct approximation subband only (structure-only).
        1.0 = correct all subbands (default behavior).
        Values between correct approximation + floor(lp_frac × wavelet_level)
        coarsest detail levels — acts as a low-pass filter over wavelet bands.
        Detail subbands beyond the cutoff are passed through unchanged.

    Structure of wavelet coefficients (output of wavedec2 at 'level' L):
      [cA,  (cH_L, cV_L, cD_L),  ...,  (cH_1, cV_1, cD_1)]
       ^^-- approximation (R)    ^^-- finest detail subbands

    adam_mode: 'none' = plain SGD with fixed alpha (sigma-scaled).
               'adam' = per-subband Adam moment tracking.
               'adamw' = per-subband AdamW (decoupled weight decay).
               'bo'   = Bayesian optimisation for alpha per step (default).
                        Requires optuna; falls back to Adam with a warning if missing.

    adam_state: dict; per-subband moments stored under 'wavelet_m'/'wavelet_v'/
                'wavelet_t' keys — built on the first call (Adam/AdamW only).

    alpha_bo_trials:  BO trial budget per step (used when adam_mode='bo').
    alpha_bo_patience: BO early-stop patience (non-improving trial count).
    alpha_bo_state:   persistent dict {'study': ...} for cross-step warm-starting.

    Falls back to pixel-space _sure_correct_x0 when ptwt is not installed.
    """
    try:
        import ptwt
        import pywt as _pywt
        _sure_logger.debug(
            "[sure_wavelet] ptwt OK  wavelet=%s  level=%d  x0_shape=%s  device=%s",
            wavelet, wavelet_level, tuple(x0_hat.shape), str(x0_hat.device),
        )
    except ImportError:
        _sure_logger.warning(
            "ptwt/pywt not installed — falling back to pixel-space SURE. "
            "Install with: pip install pytorch-wavelets pywavelets"
        )
        return _sure_correct_x0(
            model, x0_hat, sigma_hat_0, s_in, extra_args,
            alpha=alpha, n_mc=n_mc, eps_mc=eps_mc, use_jac=use_jac,
            sigma_t=sigma_t, adam_state=adam_state, adam_mode=adam_mode,
            adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
            adam_wd=adam_wd, grad_mode=grad_mode,
        )

    # ── BO mode: check optuna; warn + fall back to Adam if missing ───────────
    _effective_adam_mode = adam_mode
    if adam_mode == 'bo':
        try:
            import optuna as _optuna_check  # noqa: F401 — availability probe only
            del _optuna_check
        except ImportError:
            _sure_logger.warning(
                "[sure_wavelet] adam_mode='bo' requires optuna (pip install optuna). "
                "Falling back to Adam. "
                "Install optuna to enable per-step Bayesian alpha optimisation."
            )
            _effective_adam_mode = 'adam'
            # Ensure adam_state is initialised for the fallback
            if adam_state is None:
                adam_state = {}

    eps            = float(x0_hat.abs().max().clamp(min=eps_mc * 1000.0)) / 1000.0
    sigma_denoiser = torch.tensor(sigma_hat_0, device=x0_hat.device, dtype=x0_hat.dtype)
    # Use sigma_denoiser for both base and perturbed calls (same fix as _sure_correct_x0).
    # SURE_sgps_full.lean §2.1 proves that sigma_p ≠ sigma_denoiser introduces
    # O(|sigma_p − σ̂₀|/ε) bias in the Hutchinson trace (sigma_inconsistency_bias_structure).
    sigma_p        = sigma_denoiser
    sigma2         = float(sigma_denoiser ** 2)
    # Opt-2: 2σ²·tr{J} vanishes quadratically; skip the perturbed forward when negligible.
    use_jac = use_jac and (sigma2 > eps_mc)
    _wav           = _pywt.Wavelet(wavelet)

    _sure_logger.debug(
        "[sure_wavelet] params  sigma_hat_0=%.5f  sigma2=%.6f  eps=%.5f  "
        "sigma_p=%.5f  use_jac=%s  adam=%s  grad_mode=%s",
        sigma_hat_0, sigma2, eps, float(sigma_p), use_jac, adam_mode, grad_mode,
    )

    def _release_cache():
        if x0_hat.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif x0_hat.device.type == 'mps':
            torch.mps.empty_cache()

    # ── Pre-draw probe vectors (vjp/full/vjp_sb need them before Pass 1) ────
    need_full_grad = (grad_mode == 'full') and use_jac
    bs    = [torch.randn_like(x0_hat) for _ in range(n_mc)] if (use_jac and grad_mode in ('vjp', 'full', 'vjp_sb')) else []
    b_sum = torch.stack(bs).sum(0) if need_full_grad else None

    # ── Denoiser forward passes ──────────────────────────────────────────────
    if grad_mode == 'approx':
        with torch.no_grad():
            x_hat = model(x0_hat.detach(), sigma_denoiser * s_in, **extra_args).detach()

            # Opt-3: approx grad ignores tr{J} entirely; only compute for debug logging.
            jac_trace = None
            if use_jac and _sure_logger.isEnabledFor(_logging.DEBUG):
                b          = torch.randn_like(x0_hat)
                x_pert_hat = model(
                    (x0_hat + eps * b).detach(), sigma_p * s_in, **extra_args,
                ).detach()
                jac_trace = float((b * (x_pert_hat - x_hat)).sum()) / eps
                if n_mc > 1:
                    for _ in range(n_mc - 1):
                        b2      = torch.randn_like(x0_hat)
                        x_pert2 = model(
                            (x0_hat + eps * b2).detach(), sigma_p * s_in, **extra_args,
                        ).detach()
                        jac_trace += float((b2 * (x_pert2 - x_hat)).sum()) / eps
                    jac_trace /= n_mc

        # pixel-space residual: xnoisy − x̂
        residual    = x0_hat - x_hat
        grad_coeffs = None   # per-subband gradient computed as approx_coeff·res_c

    elif grad_mode == 'vjp_sb':
        # Per-subband Hutchinson VJP — two no-grad forwards, no autograd backward.
        #
        # For each subband k the per-subband SURE gradient is:
        #   ∇_{c_k} ‖r_k‖² = 2r_k − 2 W_k J_D W_k^T r_k
        #
        # The cross-subband term W_k J_D W_k^T r_k is estimated via a rank-1
        # Hutchinson stochastic approximation with global probe b ~ N(0,I):
        #   E_b[(b_k^T r_k) · W_k J_D b] = W_k J_D W_k^T r_k
        # where b_k = W_k b (subband k projection of the global probe).
        #
        # So:  g_k = 2r_k − (2/n_mc) Σ_b (b_k^T r_k) · (D_k(x+εb)−D_k(x))/ε
        #
        # Cost: one base forward + n_mc perturbed forwards — no backward pass.
        # The scalar jac_trace is recovered for free: Σ_k b_k^T D_diff_k = b^T d_diff.
        with torch.no_grad():
            x_hat = model(x0_hat.detach(), sigma_denoiser * s_in, **extra_args).detach()
        residual = (x0_hat - x_hat).detach()

        # Decompose residual once; reuse across probes
        _res_coeffs_sb = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode='reflect')
        _n_levels_sb   = len(_res_coeffs_sb)

        # Initialise per-subband accumulator for Jacobian correction
        _jac_corr: list | None = None
        jac_trace = 0.0 if use_jac else None

        for b in bs:
            with torch.no_grad():
                _x_pert_hat = model(
                    (x0_hat.detach() + eps * b), sigma_p * s_in, **extra_args,
                ).detach()
            _d_diff = (_x_pert_hat - x_hat) / eps  # ≈ J_D b (pixel space)

            if jac_trace is not None:
                # Global Hutchinson scalar: b^T (J_D b) = Σ_k b_k^T D_diff_k
                jac_trace += float((b * _d_diff).sum())

            # Decompose probe and Jacobian-vector product into subbands
            _b_coeffs  = ptwt.wavedec2(b.detach(), _wav, level=wavelet_level, mode='reflect')
            _dd_coeffs = ptwt.wavedec2(_d_diff,    _wav, level=wavelet_level, mode='reflect')

            if _jac_corr is None:
                _jac_corr = [torch.zeros_like(_res_coeffs_sb[0])]
                for _l in range(1, _n_levels_sb):
                    _jac_corr.append(tuple(
                        torch.zeros_like(_res_coeffs_sb[_l][_s]) for _s in range(3)
                    ))

            # Approximation subband: scale = b_0^T r_0  (scalar dot product)
            _sc0 = float((_b_coeffs[0] * _res_coeffs_sb[0]).sum())
            _jac_corr[0] = _jac_corr[0] + _sc0 * _dd_coeffs[0]

            # Detail subbands L1..L
            for _l in range(1, _n_levels_sb):
                _updated = []
                for _s in range(3):
                    _sc = float((_b_coeffs[_l][_s] * _res_coeffs_sb[_l][_s]).sum())
                    _updated.append(_jac_corr[_l][_s] + _sc * _dd_coeffs[_l][_s])
                _jac_corr[_l] = tuple(_updated)

            del _x_pert_hat, _d_diff, _b_coeffs, _dd_coeffs

        if jac_trace is not None and n_mc > 0:
            jac_trace /= n_mc
            # Gap #6 fix: variance-based floor (same as pixel-space version).
            _n_wav = x0_hat.numel()
            _jac_floor_wav = -3.0 * (2.0 * _n_wav / max(n_mc, 1)) ** 0.5
            if jac_trace < _jac_floor_wav:
                _sure_logger.warning(
                    "[sure_wavelet] Gap #6: jac_trace=%.2f < 3σ floor %.2f "
                    "(n=%d, n_mc=%d); resetting to None (noise-dominated).",
                    jac_trace, _jac_floor_wav, _n_wav, n_mc,
                )
                jac_trace = None

        # Build per-subband gradient: g_k = 2r_k − (2/n_mc)·jac_corr_k
        _inv = 2.0 / n_mc if (_jac_corr is not None and n_mc > 0) else 0.0
        _gc_list = []
        if _jac_corr is not None:
            _gc_list.append(2.0 * _res_coeffs_sb[0] - _inv * _jac_corr[0])
            for _l in range(1, _n_levels_sb):
                _gc_list.append(tuple(
                    2.0 * _res_coeffs_sb[_l][_s] - _inv * _jac_corr[_l][_s]
                    for _s in range(3)
                ))
        else:
            # use_jac=False: pure residual gradient (same as approx with coeff=2)
            _gc_list.append(2.0 * _res_coeffs_sb[0])
            for _l in range(1, _n_levels_sb):
                _gc_list.append(tuple(
                    2.0 * _res_coeffs_sb[_l][_s] for _s in range(3)
                ))
        grad_coeffs = _gc_list
        del _res_coeffs_sb, _jac_corr

    else:
        # 'vjp' / 'full' — compute exact ∇‖r‖² via one backward pass through D,
        # then decompose the pixel-space gradient into wavelet subbands so Adam
        # tracks the Jacobian-corrected direction per frequency band.
        _release_cache()
        x_in = x0_hat.detach().requires_grad_(True)

        def _model_ckpt(x):
            return model(x, sigma_denoiser * s_in, **extra_args)

        with torch.enable_grad():
            x_hat       = torch.utils.checkpoint.checkpoint(
                              _model_ckpt, x_in, use_reentrant=False)
            residual    = x_in - x_hat
            residual_sq = (residual ** 2).sum()

            if need_full_grad:
                # Opt-1: one backward replaces two (same as pixel-space _sure_correct_x0).
                # C = ||r||² − (2σ²/(n_mc·ε))·(b_sum·x̂)
                # ∂C/∂x = 2(I−J_D)ᵀr − (2σ²/(n_mc·ε))·J_D(x)ᵀ·b_sum
                combined   = residual_sq - (2.0 * sigma2 / (n_mc * eps)) * (b_sum * x_hat).sum()
                grad_pixel = torch.autograd.grad(combined, x_in)[0]
                del combined
            else:
                grad_pixel = torch.autograd.grad(residual_sq, x_in)[0]

        x_hat      = x_hat.detach()
        residual   = residual.detach()
        grad_pixel = grad_pixel.detach()
        _release_cache()

        # Jacobian trace scalar + optional ∇tr{J} for 'full' mode
        jac_trace = 0.0 if use_jac else None
        if use_jac:
            if need_full_grad:
                grad_jac_pert_sum = torch.zeros_like(grad_pixel)
                for b in bs:
                    x_in_pert = x0_hat.detach() + eps * b
                    _release_cache()
                    x_in_pert = x_in_pert.requires_grad_(True)
                    def _model_pert_ckpt(x):  # noqa: E306
                        return model(x, sigma_p * s_in, **extra_args)
                    with torch.enable_grad():
                        x_pert_hat   = torch.utils.checkpoint.checkpoint(
                                           _model_pert_ckpt, x_in_pert, use_reentrant=False)
                        jac_scalar_p = (b * x_pert_hat).sum() / eps
                        gj_pert      = torch.autograd.grad(jac_scalar_p, x_in_pert)[0]
                    # Bug-1 fix: subtract x_hat so tr{J} matches the paper formula.
                    jac_trace += float((b * (x_pert_hat - x_hat)).sum().detach()) / eps
                    del x_pert_hat
                    _release_cache()
                    grad_jac_pert_sum = grad_jac_pert_sum + gj_pert
                    del gj_pert
                jac_trace /= n_mc
                # Opt-1: base Jac term already in grad_pixel via combined scalar.
                grad_pixel = grad_pixel + (2.0 * sigma2 / n_mc) * grad_jac_pert_sum
                del grad_jac_pert_sum
            else:
                # vjp: trace via no-grad MC only (no extra backward needed)
                for b in bs:
                    with torch.no_grad():
                        x_pert_hat = model(
                            (x0_hat.detach() + eps * b), sigma_p * s_in, **extra_args,
                        ).detach()
                    jac_trace += float((b * (x_pert_hat - x_hat)).sum()) / eps
                    del x_pert_hat
                jac_trace /= n_mc

        # Clamp pixel-space gradient before wavelet decomposition — same
        # outlier-suppression as _sure_correct_x0 to keep Adam v_t stable.
        _x0_std_g = x0_hat.std().item()
        _gs       = grad_pixel.std().item()
        #if _gs > _x0_std_g > 0.0:
            #grad_pixel = grad_pixel * (_x0_std_g / _gs)
        #grad_pixel = grad_pixel.clamp(-3.0 * _x0_std_g, 3.0 * _x0_std_g)

        # Decompose pixel-space gradient into wavelet subbands — linear transform
        # commutes with the subband correction, so this is exact.
        grad_coeffs = ptwt.wavedec2(grad_pixel.detach(), _wav, level=wavelet_level, mode='reflect')
        del grad_pixel
    _sure_logger.debug(
        "[sure_wavelet] denoiser  x_hat range=[%.4f, %.4f]  "
        "residual_rms=%.5f  residual_max=%.5f  jac_trace=%s",
        float(x_hat.min()), float(x_hat.max()),
        float((residual ** 2).mean().sqrt()),
        float(residual.abs().max()),
        f"{jac_trace:.4f}" if jac_trace is not None else "n/a",
    )

    # ── Wavelet decomposition of residual and x0 ────────────────────────────
    # ptwt.wavedec2 handles [B, C, H, W] directly, orthogonal wavelet ensures
    # perfect reconstruction and energy preservation across subbands.
    residual_coeffs = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode='reflect')
    x0_coeffs       = ptwt.wavedec2(x0_hat,  _wav, level=wavelet_level, mode='reflect')

    _sure_logger.debug(
        "[sure_wavelet] wavedec2  n_bands=%d  approx_shape=%s",
        len(residual_coeffs), tuple(residual_coeffs[0].shape),
    )
    for _li in range(1, len(residual_coeffs)):
        _rH, _rV, _rD = residual_coeffs[_li]
        _sure_logger.debug(
            "[sure_wavelet]   L%d  shape=%s  res_rms H=%.5f V=%.5f D=%.5f",
            _li, tuple(_rH.shape),
            float((_rH ** 2).mean().sqrt()),
            float((_rV ** 2).mean().sqrt()),
            float((_rD ** 2).mean().sqrt()),
        )

    # ── Init per-subband Adam state on first call ────────────────────────────
    # Index 0 → approximation tensor; indices 1..L → detail tuples (H, V, D)
    # Stores per-element moments (wavelet_m / wavelet_v) only; correction
    # magnitude is fixed at base_alpha — no adaptive tier_scale (ts_val) scalar.
    # ts_val was removed because it adapted via ts_val *= exp(-alpha * ts_step)
    # where ts_step = -sure_val / (n_elem * σ²).  sure_val is always large-
    # positive at high σ, so ts_step is always negative → ts_val grew without
    # bound every step, causing pixel_delta_rms to diverge monotonically.
    n_sb = len(residual_coeffs)
    if adam_state is not None and _effective_adam_mode in ('adam', 'adamw'):
        if 'wavelet_m' not in adam_state:
            adam_state['wavelet_m'] = [None] * n_sb
            adam_state['wavelet_v'] = [None] * n_sb
            adam_state['wavelet_t'] = 0
            _sure_logger.debug("[sure_wavelet] adam state initialised  n_sb=%d", n_sb)
        adam_state['wavelet_t'] += 1
        t_adam = adam_state['wavelet_t']
        _sure_logger.debug("[sure_wavelet] adam t=%d", t_adam)
    else:
        t_adam = 1  # unused when adam is off

    x0_std = x0_hat.std().item()

    # Base alpha: sigma-scaled for plain SGD ('none') only.
    # Adam's per-element m/v normalisation already handles scale — sigma scaling
    # would double-suppress early steps and under-correct late steps.
    # BO mode: alpha is the search-space centre; BO selects the per-step value
    # without sigma pre-scaling (the proxy geometry already encodes the scale).
    if sigma_t is not None and _effective_adam_mode == 'none':
        base_alpha = alpha / (1.0 + float(sigma_t))
    else:
        base_alpha = alpha

    _sure_logger.debug(
        "[sure_wavelet] x0_std=%.5f  base_alpha=%.5f  sigma_t=%s",
        x0_std, base_alpha,
        f"{float(sigma_t):.4f}" if sigma_t is not None else "n/a",
    )

    # ── Inner helper: apply SURE gradient to each subband ────────────────────
    # Correction magnitude is fixed at base_alpha * warmup_factor.
    # Adam adapts per-element direction only via m/v moments — no outer scalar.
    # A scalar tier_scale loop was tried (ts_val) but diverged: its gradient was
    # derived from sure_val which is always positive at high σ, making the update
    # always push ts_val in one direction → unbounded exponential growth.
    # RAdam was also tried as an alternative warmup mechanism: with β₂=0.999 and
    # ~60 diffusion steps, ρ_t never grows large enough — rect factor ≈ 0.021 at
    # step 6, effectively zeroing corrections until step ~100 which never arrives.
    def _correct_coeff(x0_c, res_c, g_c, m_c, v_c):
        # g_c: pre-computed subband gradient (vjp/full); None → approx_coeff·res_c
        grad = g_c if g_c is not None else approx_coeff * res_c
        raw_grad_rms = float((grad ** 2).mean().sqrt())

        if adam_state is not None and _effective_adam_mode in ('adam', 'adamw'):
            if m_c is None:
                m_c = torch.zeros_like(grad)
                v_c = torch.zeros_like(grad)
            m_c   = m_c.mul(adam_beta1).add((1.0 - adam_beta1) * grad)
            v_c   = v_c.mul(adam_beta2).add((1.0 - adam_beta2) * grad.pow(2))
            bc1   = 1.0 - adam_beta1 ** t_adam
            bc2   = 1.0 - adam_beta2 ** t_adam
            eff_g = (m_c / bc1) / ((v_c / bc2).sqrt() + adam_eps)
        else:
            # 'none' and 'bo' — plain gradient descent
            # For 'bo' base_alpha is the BO-selected step; moments are not used
            eff_g = grad
            m_c   = None
            v_c   = None

        eff_grad_rms = float((eff_g ** 2).mean().sqrt())
        adam_ratio   = eff_grad_rms / (raw_grad_rms + 1e-8)

        # Linear warmup: ramps correction from 0 → base_alpha over warmup_steps.
        # RAdam's built-in warmup (variance rectification) was tried instead but
        # fails here — ρ_t requires ~100+ steps to rectify fully with β₂=0.999,
        # and diffusion schedules only have ~60 steps total.
        warmup_factor = min(1.0, t_adam / warmup_steps) if warmup_steps > 0 else 1.0
        step = base_alpha * warmup_factor
        if _effective_adam_mode == 'adamw' and adam_state is not None:
            corrected = (x0_c * (1.0 - step * adam_wd) - step * eff_g).detach()
        else:
            corrected = (x0_c - step * eff_g).detach()
        return corrected, m_c, v_c, raw_grad_rms, eff_grad_rms, adam_ratio

    # ── Per-subband correction ───────────────────────────────────────────────
    # lp_frac acts as a low-pass filter: only the approximation subband (idx 0)
    # and the floor(lp_frac * (n_sb-1)) coarsest detail levels are corrected.
    # Detail subbands beyond the cutoff are passed through unchanged.
    _n_detail_correct = int(lp_frac * (n_sb - 1))  # 0 = approx only, n_sb-1 = all

    # ── BO / analytical alpha search over corrected subbands ─────────────────
    # When adam_mode='bo': concatenate residuals and gradients from corrected
    # subbands into flat tensors, then find the optimal α with zero UNet calls.
    #
    #   alpha_bo_trials > 0 → Optuna TPE with warm-start from previous step.
    #   alpha_bo_trials = 0 → closed-form analytical line search: α* = (r·g)/‖g‖².
    #                         Still better than a fixed α; free to compute.
    if _effective_adam_mode == 'bo':
        _r_bo = [residual_coeffs[0].reshape(-1)]
        _g_bo = [(grad_coeffs[0] if grad_coeffs is not None
                  else approx_coeff * residual_coeffs[0]).reshape(-1)]
        for _sbi in range(1, min(_n_detail_correct + 1, n_sb)):
            for _si in range(3):
                _r_bo.append(residual_coeffs[_sbi][_si].reshape(-1))
                _g_bo.append((grad_coeffs[_sbi][_si] if grad_coeffs is not None
                               else approx_coeff * residual_coeffs[_sbi][_si]).reshape(-1))
        _r_flat = torch.cat(_r_bo).float()
        _g_flat = torch.cat(_g_bo).float()

        _bs = alpha_bo_state if alpha_bo_state is not None else {}
        if int(alpha_bo_trials) > 0:
            base_alpha, _new_study = _sure_alpha_bo_search(
                _r_flat, _g_flat, sigma2,
                jac_trace if jac_trace is not None else 0.0,
                alpha_base=alpha,
                n_trials=int(alpha_bo_trials),
                patience=int(alpha_bo_patience),
                prev_study=_bs.get('study'),
                prev_sure=_bs.get('prev_sure'),
                prev_prev_sure=_bs.get('prev_prev_sure'),
                cur_alpha=_bs.get('cur_alpha', alpha),
            )
            if alpha_bo_state is not None:
                alpha_bo_state['study'] = _new_study
        else:
            # Analytical line search — α* = (r·g) / ‖g‖²  (closed-form SURE minimum)
            # Falls back to P-controller via _sure_alpha_bo_search when grad collinear.
            _dot  = float((_r_flat * _g_flat).sum())
            _g_sq = float((_g_flat * _g_flat).sum()) + 1e-12
            _lo   = max(alpha * 0.01, 1e-5)
            _hi   = min(alpha * 20.0, 0.49)
            _r_sq_bo = float((_r_flat * _r_flat).sum()) + 1e-12
            _cos_bo  = abs(_dot) / (_r_sq_bo ** 0.5 * _g_sq ** 0.5)
            if _cos_bo > 0.99:
                # degenerate: delegate to P-controller path
                _ref = _bs.get('cur_alpha', alpha)
                _ps  = _bs.get('prev_sure')
                _pps = _bs.get('prev_prev_sure')
                if _ps is not None and _pps is not None:
                    if _ps > _pps * 1.05:
                        base_alpha = max(_lo, _ref * 0.70)
                    elif _ps < _pps * 0.95:
                        base_alpha = min(_hi, _ref * 1.05)
                    else:
                        base_alpha = _ref
                else:
                    base_alpha = _ref
                _sure_logger.info(
                    "[sure_wavelet_bo] grad collinear (|cos|=%.4f); P-ctrl: "
                    "sure %.2f→%.2f  alpha %.5f→%.5f",
                    _cos_bo,
                    _pps if _pps is not None else float('nan'),
                    _ps  if _ps  is not None else float('nan'),
                    _ref, base_alpha,
                )
            else:
                base_alpha = float(max(_lo, min(_hi, _dot / _g_sq)))

        del _r_bo, _g_bo, _r_flat, _g_flat
        if alpha_bo_state is not None:
            alpha_bo_state['cur_alpha'] = base_alpha
        _sure_logger.info(
            "[sure_wavelet_bo] base_alpha=%.5f (was %.5f)  trials=%d",
            base_alpha, alpha, int(alpha_bo_trials),
        )

    corrected_coeffs       = []
    corrected_sure_vals    = []   # SURE only for subbands that are actually corrected
    passthrough_sure_vals  = []   # SURE for pass-through subbands (diagnostic only)

    for sb_idx in range(n_sb):
        x0_sb  = x0_coeffs[sb_idx]
        res_sb = residual_coeffs[sb_idx]

        if sb_idx == 0:
            # Approximation subband — low-frequency / residual (R)
            m0 = adam_state['wavelet_m'][0] if (adam_state and 'wavelet_m' in adam_state) else None
            v0 = adam_state['wavelet_v'][0] if (adam_state and 'wavelet_v' in adam_state) else None
            sb_sure = float(-res_sb.numel() * sigma2 + (res_sb ** 2).sum())
            g0      = grad_coeffs[0] if grad_coeffs is not None else None
            c_corr, m0_new, v0_new, rg_rms, eg_rms, ar = _correct_coeff(x0_sb, res_sb, g0, m0, v0)
            if adam_state is not None and 'wavelet_m' in adam_state:
                adam_state['wavelet_m'][0] = m0_new
                adam_state['wavelet_v'][0] = v0_new
            corrected_coeffs.append(c_corr)
            corrected_sure_vals.append(sb_sure)
            _sure_logger.debug(
                "[sure_wavelet]   sb=R  shape=%s  res_rms=%.5f  sure=%.4f  "
                "grad_rms=%.5f  eff_grad_rms=%.5f  adam_ratio=%.3f  delta_rms=%.5f",
                tuple(res_sb.shape),
                float((res_sb ** 2).mean().sqrt()),
                sb_sure, rg_rms, eg_rms, ar,
                float(((c_corr - x0_sb) ** 2).mean().sqrt()),
            )
        else:
            # Detail subbands at this level — tuple(cH, cV, cD)
            if sb_idx > _n_detail_correct:
                # Beyond LP cutoff — pass through unchanged.
                # SURE is tracked separately: these fine-detail bands have large n_k
                # (the finest band has 76 800 elements at 160×160 latent) and would
                # dominate total_sure with large negative values at mid-to-late steps
                # even though they receive no gradient correction.  Including them in
                # the primary sure_val metric was causing misleading spikes (e.g. −703
                # at step 33/35) that masked the corrected-subband signal.
                corrected_coeffs.append(x0_sb)
                for sub_i in range(3):
                    passthrough_sure_vals.append(
                        float(-res_sb[sub_i].numel() * sigma2 + (res_sb[sub_i] ** 2).sum())
                    )
                continue

            prev_m = (adam_state['wavelet_m'][sb_idx]
                      if (adam_state and 'wavelet_m' in adam_state
                          and adam_state['wavelet_m'][sb_idx] is not None)
                      else (None, None, None))
            prev_v = (adam_state['wavelet_v'][sb_idx]
                      if (adam_state and 'wavelet_v' in adam_state
                          and adam_state['wavelet_v'][sb_idx] is not None)
                      else (None, None, None))

            detail_corr = []
            new_m       = []
            new_v       = []
            _sub_names  = ('H', 'V', 'D')
            for sub_i in range(3):      # H, V, D
                sb_sure = float(-res_sb[sub_i].numel() * sigma2 + (res_sb[sub_i] ** 2).sum())
                g_sb    = grad_coeffs[sb_idx][sub_i] if grad_coeffs is not None else None
                c_corr, m_new, v_new, rg_rms, eg_rms, ar = _correct_coeff(
                    x0_sb[sub_i], res_sb[sub_i], g_sb, prev_m[sub_i], prev_v[sub_i],
                )
                detail_corr.append(c_corr)
                new_m.append(m_new)
                new_v.append(v_new)
                corrected_sure_vals.append(sb_sure)
                _sure_logger.debug(
                    "[sure_wavelet]   sb=L%d_%s  shape=%s  res_rms=%.5f  sure=%.4f  "
                    "grad_rms=%.5f  eff_grad_rms=%.5f  adam_ratio=%.3f  delta_rms=%.5f",
                    sb_idx, _sub_names[sub_i],
                    tuple(res_sb[sub_i].shape),
                    float((res_sb[sub_i] ** 2).mean().sqrt()),
                    sb_sure, rg_rms, eg_rms, ar,
                    float(((c_corr - x0_sb[sub_i]) ** 2).mean().sqrt()),
                )

            if adam_state is not None and 'wavelet_m' in adam_state:
                adam_state['wavelet_m'][sb_idx] = tuple(new_m)
                adam_state['wavelet_v'][sb_idx] = tuple(new_v)
            corrected_coeffs.append(tuple(detail_corr))

    # ── Reconstruct corrected x0 ─────────────────────────────────────────────
    x0_raw = ptwt.waverec2(corrected_coeffs, _wav).detach()
    # Crop to original spatial dims — wavelet padding may add 1 pixel
    x0_corrected = x0_raw[..., :x0_hat.shape[-2], :x0_hat.shape[-1]]

    # Primary metric: SURE of corrected subbands only.  Negative values at mid-to-
    # late steps mean ‖r_k‖² < n_k·σ² — the denoiser residual is below the noise
    # floor, which is expected for a well-trained model (Lean: sure_val_negative_iff_good_denoiser).
    # Pass-through subbands are excluded: their large n_k would otherwise dominate
    # this sum with irrelevant negative values (see Supplement C in SURE_verification.lean).
    corrected_sure = sum(corrected_sure_vals)
    passthrough_sure = sum(passthrough_sure_vals)
    if jac_trace is not None:
        corrected_sure += 2.0 * sigma2 * jac_trace

    # Gap #5: bias-correct corrected_sure using the pixel-space residual.
    # residual = x0_hat − x_hat was computed above (pixel space).
    _n_wav_total = x0_hat.numel()
    _sigma_sq_res_wav = float((residual ** 2).sum()) / max(_n_wav_total, 1)
    _tr_J_bc_wav = jac_trace if jac_trace is not None else 0.0
    _sure_bias_wav = (sigma2 - _sigma_sq_res_wav) * (-float(_n_wav_total) + 2.0 * _tr_J_bc_wav)
    corrected_sure_bc = corrected_sure - _sure_bias_wav

    pixel_delta_rms = float(((x0_corrected - x0_hat) ** 2).mean().sqrt())

    _sure_logger.info(
        "[wavelet] sure=%.1f  sure_bc=%.1f  pt_sure=%.1f  pixel_δ=%.5f  base_alpha=%.5f",
        corrected_sure, corrected_sure_bc, passthrough_sure, pixel_delta_rms, base_alpha,
    )

    # Feed bias-corrected SURE into P-controller (Gap #5 fix).
    if alpha_bo_state is not None and _effective_adam_mode == 'bo':
        alpha_bo_state['prev_prev_sure'] = alpha_bo_state.get('prev_sure')
        alpha_bo_state['prev_sure']      = corrected_sure_bc

    return x0_corrected, {'sure_val': corrected_sure_bc, 'jac_ratio': None,
                          'pixel_delta_rms': pixel_delta_rms}


def sample_sure(model, x, sigmas, extra_args=None, callback=None, disable=None,
                sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                sure_jac_interval=2,
                sure_adam_mode='none', sure_adam_beta1=0.9,
                sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                sure_approx_coeff=2.0, sure_csv_path=None,
                sure_sigma_ema=0.0,
                sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """SURE Guided Posterior Sampling (SGPS) — Euler Ancestral variant.

    Implements Algorithm 1 from arXiv:2512.23232 directly.  Every step:

      1. Denoising + CFG:  x̂₀ = model(xₜ, σₜ)          [deterministic, no noise]
      2. PCA noise est.:   σ̂₀ = _pca_noise_estimate(x̂₀) [residual noise in x̂₀]
      3. SURE correction:  x̂*₀ = x̂₀ − α·∇SURE(x̂₀, σ̂₀) [gradient in x0-space]
      4. Re-add noise:     xₜ₋₁ = x̂*₀ + σₜ₋₁·ε         [Euler Ancestral]

    No preheat — SURE correction is applied at every step from T to 1.

    sure_alpha:        SURE gradient step size
    sure_n_mc:         Monte Carlo samples for Jacobian trace
    sure_eps:          finite-difference epsilon for MC Jacobian
    sure_jac_interval: compute full Jacobian every N steps; adapt from jac_ratio
    """
    import csv as _csv
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed)
    s_in = x.new_ones([x.shape[0]])

    _EMA_A = 0.35
    _dyn_jac_interval: int        = max(1, sure_jac_interval)
    _jac_ratio_ema:   float | None = None
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None
    _sigma_hat_0_ema: float | None = None   # EMA state for PCA noise estimate

    # CSV setup — only active in approx mode
    _csv_file   = None
    _csv_writer = None
    if sure_csv_path is not None and sure_grad_mode == 'approx':
        _csv_file = open(sure_csv_path, 'w', newline='')
        _csv_writer = _csv.writer(_csv_file)
        _csv_writer.writerow([
            'step', 'sigma_hat_0', 'sigma_t', 'approx_coeff',
            'residual_rms', 'sure_val', 'raw_grad_rms', 'eff_grad_rms',
            'effective_alpha', 'step_rms', 'adam_ratio',
        ])
        _sure_logger.info("SURE approx CSV: writing per-step stats to %s", sure_csv_path)

    n_steps = len(sigmas) - 1
    _sure_logger.info(
        "SURE sampler: %d steps  alpha=%.4f  n_mc=%d  jac_interval=%d  adam=%s  approx_coeff=%.4f",
        n_steps, sure_alpha, sure_n_mc, _dyn_jac_interval, sure_adam_mode, sure_approx_coeff,
    )

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_val  = sigma.item()

        # ── Step 1: Denoising + CFG (deterministic) ───────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        # ── Step 2: PCA residual noise estimate (+ ceiling + optional EMA) ──────
        _sigma_raw = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
        # Hard ceiling: residual noise in x̂₀ cannot exceed the diffusion noise
        # added at time t. Without this, the M-P estimator mistakes fine image
        # detail for noise at late steps (small sigma_t), causing the denoiser
        # to be called at a wildly wrong sigma and sure_val to blow up negative.
        _sigma_raw = min(_sigma_raw, float(sigma_val))
        if sure_sigma_ema > 0.0:
            if _sigma_hat_0_ema is None:
                _sigma_hat_0_ema = _sigma_raw
            else:
                _sigma_hat_0_ema = sure_sigma_ema * _sigma_hat_0_ema + (1.0 - sure_sigma_ema) * _sigma_raw
            # EMA memory can lag above the current sigma_t as sigma_t collapses
            # at late steps — clamp the output too, not just the raw estimate.
            sigma_hat_0 = min(_sigma_hat_0_ema, float(sigma_val))
        else:
            sigma_hat_0 = _sigma_raw

        # ── Step 3: SURE gradient correction on x0_hat at σ̂₀ ─────────────────
        _use_jac = (_dyn_jac_interval <= 1) or (i % _dyn_jac_interval == 0)
        x0_corrected, _stats = _sure_correct_x0(
            model, x0_hat, sigma_hat_0, s_in, extra_args,
            alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
            use_jac=_use_jac, sigma_t=sigma,
            adam_state=_adam_state, adam_mode=sure_adam_mode,
            adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
            adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
            approx_coeff=sure_approx_coeff,
            csv_writer=_csv_writer, step_idx=i,
            alpha_bo_trials=sure_alpha_bo_trials,
            alpha_bo_patience=sure_alpha_bo_patience,
            alpha_bo_state=_alpha_bo_state,
        )

        # Adapt jac_interval from jac_ratio EMA
        _jac_ratio_new = _stats.get('jac_ratio')
        if _jac_ratio_new is not None:
            _jac_ratio_ema = _jac_ratio_new if _jac_ratio_ema is None else \
                (1.0 - _EMA_A) * _jac_ratio_ema + _EMA_A * _jac_ratio_new
        if _jac_ratio_ema is not None and i >= 2:
            if _jac_ratio_ema < 0.05 and _dyn_jac_interval < 8:
                _dyn_jac_interval += 1
                _sure_logger.info("[adapt] jac_interval -> %d  (ratio_ema=%.3f < 0.05)",
                                  _dyn_jac_interval, _jac_ratio_ema)
            elif _jac_ratio_ema > 0.25 and _dyn_jac_interval > 1:
                _dyn_jac_interval -= 1
                _sure_logger.info("[adapt] jac_interval -> %d  (ratio_ema=%.3f > 0.25)",
                                  _dyn_jac_interval, _jac_ratio_ema)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma,
                      'sigma_hat': sigma, 'denoised': x0_corrected})

        # ── Step 4: Ancestral update from SURE-corrected x̂*₀ ────────────────
        # Mirror Euler Ancestral: split sigma_next into deterministic (sigma_down)
        # and stochastic (sigma_up) components. Pure re-noise (x0 + sigma_next·ε)
        # discards the prior state and amplifies the model's latent bias; the
        # ancestral split keeps the same trajectory dynamics as Euler a.
        if float(sigma_next) == 0:
            x = x0_corrected
        else:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=1.0)
            # Noise direction from current x using the SURE-corrected x0 estimate
            d = (x - x0_corrected) / sigma
            # Deterministic step to sigma_down, then inject sigma_up noise
            x = x + d * (sigma_down - sigma)
            x = x + sigma_up * noise_sampler(sigma, sigma_next)

    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _sure_logger.info("SURE approx CSV: closed %s", sure_csv_path)

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def sample_sure_wavelet(model, x, sigmas, extra_args=None, callback=None, disable=None,
                        sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                        sure_jac_interval=2,
                        sure_adam_mode='bo', sure_adam_beta1=0.9,
                        sure_adam_beta2=0.999, sure_adam_wd=0.01,
                        sure_wavelet='db4', sure_wavelet_level=3,
                        sure_approx_coeff=2.0, sure_csv_path=None,
                        sure_sigma_ema=0.0, sure_wavelet_warmup_steps=0,
                        sure_wavelet_lp_frac=1.0,
                        sure_grad_mode='vjp',
                        sure_alpha_bo_trials=12, sure_alpha_bo_patience=4):
    """SURE-Wavelet — Euler Ancestral variant.

    Identical control flow to sample_sure, but replaces pixel-space SURE
    correction with SURE-Wavelet:

      1. Denoising + CFG:   x̂₀ = model(xₜ, σₜ)
      2. PCA noise est.:    σ̂₀ = _pca_noise_estimate(x̂₀)
      3. Wavelet decomp.:   {cA, (cH_L,cV_L,cD_L), …} = wavedec2(x̂₀)
      4. Per-subband SURE:  correct each subband independently with its own
                            Adam moment state (freq-adaptive step size)
      5. Reconstruct:       x̂*₀ = waverec2(corrected subbands)
      6. Ancestral noise:   xₜ₋₁ = x̂*₀ + σₜ₋₁·ε

    sure_wavelet       : PyWavelets orthogonal wavelet name (default 'db4').
    sure_wavelet_level : number of DWT decomposition levels (default 3).
    sure_grad_mode     : gradient mode passed to _sure_correct_x0_wavelet.
                         'approx'  — stop-grad, no backward; grad = approx_coeff·r_k.
                         'vjp'     — one autograd backward on ‖r‖²; pixel gradient
                                     decomposed to subbands (cross-subband J_D mixing).
                         'vjp_sb'  — per-subband Hutchinson VJP; two no-grad forwards,
                                     no backward.  g_k = 2r_k − (2/n_mc)Σ(b_k^T r_k)·D_diff_k/ε.
                                     Unbiased estimator of 2(I−W_k J_D W_k^T)^T r_k —
                                     true per-subband independence. Default 'vjp_sb'.
                         'full'    — vjp + 2σ²·∇tr{J_D} trace term.
    """
    import csv as _csv
    extra_args    = {} if extra_args is None else extra_args
    seed          = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed)
    s_in          = x.new_ones([x.shape[0]])

    _EMA_A = 0.35
    _dyn_jac_interval: int         = max(1, sure_jac_interval)
    _jac_ratio_ema:   float | None = None
    _adam_state     = {} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_adam_mode == 'bo' else None
    _sigma_hat_0_ema: float | None = None

    # CSV output (aggregate per-step metrics, wavelet mode)
    _csv_fh     = None
    _csv_writer = None
    if sure_csv_path:
        _csv_fh = open(sure_csv_path, 'w', newline='')
        _csv_writer = _csv.writer(_csv_fh)
        _csv_writer.writerow([
            'step', 'sigma_hat_0', 'sigma_t', 'approx_coeff',
            'sure_val', 'pixel_delta_rms', 'effective_alpha',
        ])

    n_steps = len(sigmas) - 1
    _sure_logger.info(
        "SURE-Wavelet sampler: %d steps  alpha=%.4f  n_mc=%d  jac_interval=%d  "
        "wavelet=%s  level=%d  lp_frac=%.2f  adam=%s  grad_mode=%s  "
        "approx_coeff=%.2f  sigma_ema=%.2f  warmup=%d",
        n_steps, sure_alpha, sure_n_mc, _dyn_jac_interval,
        sure_wavelet, sure_wavelet_level, sure_wavelet_lp_frac,
        sure_adam_mode, sure_grad_mode,
        sure_approx_coeff, sure_sigma_ema, sure_wavelet_warmup_steps,
    )

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_val  = sigma.item()

        # ── Step 1: Denoising + CFG ───────────────────────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        # ── Step 2: PCA residual noise estimate + sigma_t ceiling + EMA ──────
        _sigma_raw = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
        # Clamp 1 — hard ceiling: M-P bulk eigenvalue can exceed sigma_t at late
        # steps where the denoised x̂₀ contains fine detail the estimator mistakes
        # for noise.  Without this, the SURE denoiser call uses a wrong σ̂₀ and
        # sure_val blows up negative, destabilising Adam's moment state.
        _sigma_raw = min(_sigma_raw, sigma_val)
        if sure_sigma_ema > 0.0:
            if _sigma_hat_0_ema is None:
                _sigma_hat_0_ema = _sigma_raw
            else:
                _sigma_hat_0_ema = sure_sigma_ema * _sigma_hat_0_ema + (1.0 - sure_sigma_ema) * _sigma_raw
            # Clamp 2 — EMA lag: the smoothed estimate can remain above sigma_t
            # for several steps after sigma_t collapses near the end of sampling.
            # Clamping the EMA output (not just the raw input) closes that gap.
            sigma_hat_0 = min(_sigma_hat_0_ema, sigma_val)
        else:
            sigma_hat_0 = _sigma_raw

        # ── Steps 3-5: SURE-Wavelet correction ───────────────────────────────
        _use_jac = (_dyn_jac_interval <= 1) or (i % _dyn_jac_interval == 0)
        x0_corrected, _stats = _sure_correct_x0_wavelet(
            model, x0_hat, sigma_hat_0, s_in, extra_args,
            alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
            use_jac=_use_jac, sigma_t=sigma,
            adam_state=_adam_state, adam_mode=sure_adam_mode,
            adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
            adam_wd=sure_adam_wd,
            wavelet=sure_wavelet, wavelet_level=sure_wavelet_level,
            approx_coeff=sure_approx_coeff,
            warmup_steps=sure_wavelet_warmup_steps,
            lp_frac=sure_wavelet_lp_frac,
            grad_mode=sure_grad_mode,
            alpha_bo_trials=sure_alpha_bo_trials,
            alpha_bo_patience=sure_alpha_bo_patience,
            alpha_bo_state=_alpha_bo_state,
        )

        if _csv_writer is not None:
            _csv_writer.writerow([
                i, f"{sigma_hat_0:.6f}", f"{sigma_val:.6f}",
                f"{sure_approx_coeff:.4f}",
                f"{_stats['sure_val']:.6f}",
                f"{_stats['pixel_delta_rms']:.6f}",
                f"{sure_alpha:.6f}",
            ])
            _csv_fh.flush()
        # jac_ratio adaptation is not applicable here — wavelet correction does
        # not produce a scalar jac_ratio signal; jac_interval is fixed.

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma,
                      'sigma_hat': sigma, 'denoised': x0_corrected})

        # ── Step 6: Ancestral update from SURE-Wavelet-corrected x̂*₀ ─────────
        if float(sigma_next) == 0:
            x = x0_corrected
        else:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=1.0)
            d = (x - x0_corrected) / sigma
            x = x + d * (sigma_down - sigma)
            x = x + sigma_up * noise_sampler(sigma, sigma_next)

    if _csv_fh is not None:
        _csv_fh.close()

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def _sure_score_config(snapshots, wav_name, level, lp_frac):
    """Sum SURE score for one (wavelet, level, lp_frac) config over pre-computed snapshots.

    snapshots: list of (residual_tensor, sigma2_float) pairs collected during
               calibration — one per denoising step.  All expensive model forwards
               are already done; this function only performs cheap DWT arithmetic.

    Returns a single float: sum of per-snapshot SURE values for the given config.
    """
    try:
        import ptwt
        import pywt as _pywt
    except ImportError:
        return float('inf')

    try:
        _wav = _pywt.Wavelet(wav_name)
        total = 0.0
        for residual, sigma2 in snapshots:
            res_coeffs = ptwt.wavedec2(residual, _wav, level=level, mode='reflect')
            n_sb       = len(res_coeffs)
            n_detail   = int(lp_frac * (n_sb - 1))
            total += float(-res_coeffs[0].numel() * sigma2 + (res_coeffs[0] ** 2).sum())
            for sb_idx in range(1, n_detail + 1):
                for sub_i in range(3):
                    total += float(
                        -res_coeffs[sb_idx][sub_i].numel() * sigma2
                        + (res_coeffs[sb_idx][sub_i] ** 2).sum()
                    )
        return total
    except Exception:
        return float('inf')


def sample_sure_wavelet_auto(model, x, sigmas, extra_args=None, callback=None, disable=None,
                              sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                              sure_jac_interval=2,
                              sure_adam_mode='bo', sure_adam_beta1=0.9,
                              sure_adam_beta2=0.999, sure_adam_wd=0.01,
                              sure_approx_coeff=2.0, sure_csv_path=None,
                              sure_sigma_ema=0.0, sure_wavelet_warmup_steps=0,
                              sure_grad_mode='vjp',
                              sure_wavelet_cal_steps=5,
                              sure_wavelet_cal_wavelets=('haar', 'db2', 'db4', 'db6', 'sym4'),
                              sure_wavelet_cal_levels=(1, 6),
                              sure_wavelet_bo_trials=40,
                              sure_wavelet_bo_patience=8,
                              sure_wavelet_bo_cv_warn=0.4,
                              sure_alpha_bo_trials=12, sure_alpha_bo_patience=4):
    """SURE-Wavelet Auto — Bayesian-optimised wavelet config, then SURE-Wavelet.

    Search space (mixed discrete/continuous):
      wavelet  — suggest_categorical from sure_wavelet_cal_wavelets
      level    — suggest_int over [lvl_min, lvl_max] from sure_wavelet_cal_levels
      lp_frac  — suggest_float over [0.0, 1.0]  (continuous; no fixed grid)

    Phase 1 (data collection): Run sure_wavelet_cal_steps denoising steps on an
    independent copy of x.  Each step yields one (residual, sigma2) snapshot.
    sure_wavelet_cal_steps model forwards total — no extra cost per BO trial.

    Phase 2 (BO search): Optuna TPE minimises summed SURE score across snapshots.
    Each trial is cheap (only DWT arithmetic).  Early-stop fires when the best
    value has not improved for sure_wavelet_bo_patience consecutive trials.

    Phase 3 (advisory): Compute coefficient of variation (std/|mean|) of the
    winner's per-step SURE values.  High CV → noisy objective → log a warning
    recommending more cal_steps.

    Falls back to an exhaustive grid search when optuna is not installed.

    sure_wavelet_cal_levels : (min, max) int tuple for suggest_int, or a sequence
        of ints whose min/max is used.  Default (1, 6) searches all valid levels.
    """
    extra_args    = {} if extra_args is None else extra_args
    s_in          = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x, seed=extra_args.get('seed', None))
    n_steps       = len(sigmas) - 1
    cal_steps     = min(int(sure_wavelet_cal_steps), n_steps)

    # Resolve level range
    _lvls  = list(sure_wavelet_cal_levels)
    lvl_min, lvl_max = (min(_lvls), max(_lvls)) if len(_lvls) >= 2 else (_lvls[0], _lvls[0])
    cal_wavelet_list  = list(sure_wavelet_cal_wavelets)

    _sure_logger.info(
        "SURE-Wavelet Auto: collecting %d calibration snapshots  "
        "wavelets=%s  level=[%d,%d]  bo_trials=%d  bo_patience=%d",
        cal_steps, cal_wavelet_list, lvl_min, lvl_max,
        int(sure_wavelet_bo_trials), int(sure_wavelet_bo_patience),
    )

    # ── Phase 1: collect (residual, sigma2) snapshots ───────────────────────
    snapshots = []
    x_cal     = x.clone()
    for i in range(cal_steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_val  = sigma.item()

        with torch.no_grad():
            x0_hat = model(x_cal, sigma * s_in, **extra_args).detach()

        sigma_hat_0 = min(
            _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps)),
            sigma_val,
        )
        residual = (x_cal - x0_hat).detach()
        snapshots.append((residual, sigma_hat_0 ** 2))

        if float(sigma_next) > 0:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=1.0)
            d     = (x_cal - x0_hat) / sigma
            x_cal = x_cal + d * (sigma_down - sigma)
            x_cal = x_cal + sigma_up * noise_sampler(sigma, sigma_next)

    del x_cal

    # ── Phase 2: Bayesian optimisation (or grid fallback) ────────────────────
    best_wavelet = cal_wavelet_list[0]
    best_level   = max(1, min(3, lvl_max))
    best_lp_frac = 1.0
    n_trials_run = 0

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        _no_improve = [0]
        _best_val   = [float('inf')]

        def _bo_callback(study, trial):
            val = study.best_value
            if val < _best_val[0]:
                _best_val[0]   = val
                _no_improve[0] = 0
            else:
                _no_improve[0] += 1
            if _no_improve[0] >= int(sure_wavelet_bo_patience):
                study.stop()

        def _objective(trial):
            wav   = trial.suggest_categorical('wavelet', cal_wavelet_list)
            level = trial.suggest_int('level', lvl_min, lvl_max)
            lpf   = trial.suggest_float('lp_frac', 0.0, 1.0)
            return _sure_score_config(snapshots, wav, level, lpf)

        study = _optuna.create_study(
            direction='minimize',
            sampler=_optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        )
        study.optimize(
            _objective,
            n_trials=int(sure_wavelet_bo_trials),
            callbacks=[_bo_callback],
            show_progress_bar=False,
        )

        best         = study.best_params
        best_wavelet = best['wavelet']
        best_level   = best['level']
        best_lp_frac = best['lp_frac']
        n_trials_run = len(study.trials)

        # Log top-3
        ranked = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
        )
        for rank, t in enumerate(ranked[:3], 1):
            p = t.params
            _sure_logger.info(
                "SURE-Wavelet Auto  #%d: wavelet=%s  level=%d  lp_frac=%.3f  sure=%.2f",
                rank, p['wavelet'], p['level'], p['lp_frac'], t.value,
            )

    except ImportError:
        # Fallback: exhaustive grid over wavelets × integer levels × fixed lp_frac list
        _sure_logger.warning(
            "optuna not installed — falling back to grid search. "
            "Install with: pip install optuna"
        )
        import itertools
        _lp_grid   = [0.0, 0.25, 0.5, 0.75, 1.0]
        _grid      = list(itertools.product(
            cal_wavelet_list,
            range(lvl_min, lvl_max + 1),
            _lp_grid,
        ))
        _scores    = [_sure_score_config(snapshots, w, lv, lf) for w, lv, lf in _grid]
        _best_idx  = int(min(range(len(_scores)), key=lambda j: _scores[j]))
        best_wavelet, best_level, best_lp_frac = _grid[_best_idx]
        n_trials_run = len(_grid)

        ranked_grid = sorted(range(len(_scores)), key=lambda j: _scores[j])
        for rank, idx in enumerate(ranked_grid[:3], 1):
            w, lv, lf = _grid[idx]
            _sure_logger.info(
                "SURE-Wavelet Auto (grid)  #%d: wavelet=%s  level=%d  lp_frac=%.2f  "
                "sure=%.2f", rank, w, lv, lf, _scores[idx],
            )

    _sure_logger.info(
        "SURE-Wavelet Auto: using wavelet=%s  level=%d  lp_frac=%.3f  "
        "(%d evaluations)",
        best_wavelet, best_level, best_lp_frac, n_trials_run,
    )

    # ── Phase 3: advisory — check if cal_steps was sufficient ───────────────
    if len(snapshots) > 1:
        import math as _math
        import statistics as _stat
        per_step = [
            _sure_score_config([snap], best_wavelet, best_level, best_lp_frac)
            for snap in snapshots
        ]
        finite = [v for v in per_step if _math.isfinite(v)]
        if len(finite) >= 2:
            cv = _stat.stdev(finite) / (abs(_stat.mean(finite)) + 1e-8)
            if cv > float(sure_wavelet_bo_cv_warn):
                _sure_logger.warning(
                    "SURE-Wavelet Auto: winner SURE CV=%.3f > %.2f — "
                    "objective is noisy across calibration steps; consider increasing "
                    "sure_wavelet_cal_steps (current=%d) for more reliable selection.",
                    cv, float(sure_wavelet_bo_cv_warn), cal_steps,
                )
            else:
                _sure_logger.info(
                    "SURE-Wavelet Auto: winner CV=%.3f — calibration stable.",
                    cv,
                )
        else:
            _sure_logger.warning(
                "SURE-Wavelet Auto: all per-step SURE scores are non-finite — "
                "CV advisory skipped. Verify pywt/ptwt are installed correctly.",
            )

    # ── Phase 4: main sampling from original x with winning config ───────────
    return sample_sure_wavelet(
        model, x, sigmas,
        extra_args=extra_args, callback=callback, disable=disable,
        sure_alpha=sure_alpha, sure_n_mc=sure_n_mc, sure_eps=sure_eps,
        sure_jac_interval=sure_jac_interval,
        sure_adam_mode=sure_adam_mode, sure_adam_beta1=sure_adam_beta1,
        sure_adam_beta2=sure_adam_beta2, sure_adam_wd=sure_adam_wd,
        sure_wavelet=best_wavelet, sure_wavelet_level=best_level,
        sure_wavelet_lp_frac=best_lp_frac,
        sure_approx_coeff=sure_approx_coeff, sure_csv_path=sure_csv_path,
        sure_sigma_ema=sure_sigma_ema,
        sure_wavelet_warmup_steps=sure_wavelet_warmup_steps,
        sure_grad_mode=sure_grad_mode,
        sure_alpha_bo_trials=sure_alpha_bo_trials,
        sure_alpha_bo_patience=sure_alpha_bo_patience,
    )


def sample_sure_wavelet_converge(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
    sure_jac_interval=2,
    sure_adam_mode='bo', sure_adam_beta1=0.9,
    sure_adam_beta2=0.999, sure_adam_wd=0.01,
    sure_wavelet='db4', sure_wavelet_level=3,
    sure_approx_coeff=2.0, sure_csv_path=None,
    sure_sigma_ema=0.0, sure_wavelet_warmup_steps=0,
    sure_wavelet_lp_frac=1.0,
    sure_grad_mode='vjp',
    sure_inner_steps=4,
    sure_inner_tol=1e-4,
    sure_alpha_bo_trials=12, sure_alpha_bo_patience=4,
):
    """SURE-Wavelet with convergence-driven inner loop per sigma stage.

    At each sigma σᵢ, instead of a single SURE correction, iterates up to
    sure_inner_steps times — stopping early when the correction step
    (pixel_delta_rms) falls below sure_inner_tol.  A divergence guard stops
    the inner loop if SURE value rises more than 10% vs. the previous inner step.

    Cost per sigma stage (approx mode): 1 outer + N_inner_done model calls.
    Cost per sigma stage (vjp/vjp_sb): 1 outer + N_inner_done × (1+n_mc) calls.

    Per-sigma Adam state is reset each stage so the inner steps form a clean
    gradient-descent sequence toward the SURE minimum at that noise level.

    sure_inner_steps: max SURE correction iterations per sigma stage (default 4)
    sure_inner_tol:   pixel_delta_rms early-stop threshold; 0 = always run all steps
    """
    extra_args    = {} if extra_args is None else extra_args
    seed          = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed)
    s_in          = x.new_ones([x.shape[0]])

    n_steps    = len(sigmas) - 1
    _max_inner = max(1, int(sure_inner_steps))
    _sigma_hat_0_ema: float | None = None
    _alpha_bo_state: dict | None   = {} if sure_adam_mode == 'bo' else None

    _sure_logger.info(
        "SURE-Wavelet-Converge: %d steps  alpha=%.4f  n_mc=%d  wavelet=%s  "
        "level=%d  lp_frac=%.2f  adam=%s  grad_mode=%s  "
        "inner_steps=%d  inner_tol=%.2e",
        n_steps, sure_alpha, sure_n_mc, sure_wavelet, sure_wavelet_level,
        sure_wavelet_lp_frac, sure_adam_mode, sure_grad_mode,
        _max_inner, float(sure_inner_tol),
    )

    for i in trange(n_steps, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_val  = sigma.item()

        # ── ONE outer model call: get x̂₀ ────────────────────────────────────
        with torch.no_grad():
            x0_hat = model(x, sigma * s_in, **extra_args).detach()

        # ── PCA noise estimate + ceiling + optional EMA ──────────────────────
        _sigma_raw = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
        _sigma_raw = min(_sigma_raw, sigma_val)
        if sure_sigma_ema > 0.0:
            if _sigma_hat_0_ema is None:
                _sigma_hat_0_ema = _sigma_raw
            else:
                _sigma_hat_0_ema = (sure_sigma_ema * _sigma_hat_0_ema
                                    + (1.0 - sure_sigma_ema) * _sigma_raw)
            sigma_hat_0 = min(_sigma_hat_0_ema, sigma_val)
        else:
            sigma_hat_0 = _sigma_raw

        # ── Inner SURE convergence loop ──────────────────────────────────────
        # Fresh Adam state per sigma stage so inner steps are a clean gradient-
        # descent sequence toward the SURE minimum at this noise level.
        # BO state persists across sigma stages for cross-step warm-starting.
        _inner_adam  = {} if sure_adam_mode in ('adam', 'adamw') else None
        _use_jac     = (sure_jac_interval <= 1) or (i % max(1, sure_jac_interval) == 0)
        _prev_sure   = float('inf')
        _inner_done  = 0
        _stats: dict = {'sure_val': 0.0, 'pixel_delta_rms': 0.0}

        for inner_i in range(_max_inner):
            x0_hat, _stats = _sure_correct_x0_wavelet(
                model, x0_hat, sigma_hat_0, s_in, extra_args,
                alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                use_jac=_use_jac, sigma_t=sigma,
                adam_state=_inner_adam, adam_mode=sure_adam_mode,
                adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                adam_wd=sure_adam_wd,
                wavelet=sure_wavelet, wavelet_level=sure_wavelet_level,
                approx_coeff=sure_approx_coeff,
                warmup_steps=sure_wavelet_warmup_steps,
                lp_frac=sure_wavelet_lp_frac,
                grad_mode=sure_grad_mode,
                alpha_bo_trials=sure_alpha_bo_trials,
                alpha_bo_patience=sure_alpha_bo_patience,
                alpha_bo_state=_alpha_bo_state,
            )
            _inner_done = inner_i + 1
            _sure_val   = _stats['sure_val']
            _delta_rms  = _stats['pixel_delta_rms']

            # Convergence: step size is negligible
            if sure_inner_tol > 0 and _delta_rms < float(sure_inner_tol):
                _sure_logger.debug(
                    "[converge] sigma=%.4f  inner=%d/%d  converged  "
                    "pixel_delta_rms=%.2e < tol=%.2e",
                    sigma_val, _inner_done, _max_inner, _delta_rms, float(sure_inner_tol),
                )
                break

            if _prev_sure < float('inf'):
                # Log zero-crossing: SURE went negative for the first time this stage.
                # The correction has passed through the SURE minimum — the gradient
                # now points away from the data manifold (opposite direction).
                if _prev_sure >= 0 and _sure_val < 0:
                    _sure_logger.debug(
                        "[converge] sigma=%.4f  inner=%d/%d  SURE crossed zero: %.2f → %.2f  "
                        "(overshoot candidate — watching next step)",
                        sigma_val, _inner_done, _max_inner, _prev_sure, _sure_val,
                    )

                # Positive divergence: SURE rising (correction destabilising in positive direction)
                if _prev_sure >= 0 and _sure_val > _prev_sure * 1.1:
                    _sure_logger.debug(
                        "[converge] sigma=%.4f  inner=%d/%d  +diverge-stop  "
                        "sure_val=%.2f  prev=%.2f",
                        sigma_val, _inner_done, _max_inner, _sure_val, _prev_sure,
                    )
                    break

                # Negative divergence: SURE already negative and going more extreme.
                # Both corrections apply the same gradient direction; once the SURE
                # minimum is behind us the gradient flips sign conceptually, so each
                # further step moves x̂₀ further from the data manifold.
                # prev_sure * 1.1 is more negative than prev_sure → flags a >10% drift.
                if _prev_sure < 0 and _sure_val < _prev_sure * 1.1:
                    _sure_logger.debug(
                        "[converge] sigma=%.4f  inner=%d/%d  -diverge-stop  "
                        "sure_val=%.2f  prev=%.2f  (drifting more negative)",
                        sigma_val, _inner_done, _max_inner, _sure_val, _prev_sure,
                    )
                    break

            _prev_sure = _sure_val

        _sure_logger.info(
            "[converge] sigma=%.4f  inner=%d/%d  sure_val=%.2f  pixel_delta_rms=%.2e",
            sigma_val, _inner_done, _max_inner,
            _stats['sure_val'], _stats['pixel_delta_rms'],
        )

        x0_corrected = x0_hat

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma,
                      'sigma_hat': sigma, 'denoised': x0_corrected})

        # ── Ancestral step from converged x̂*₀ ───────────────────────────────
        if float(sigma_next) == 0:
            x = x0_corrected
        else:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=1.0)
            d = (x - x0_corrected) / sigma
            x = x + d * (sigma_down - sigma)
            x = x + sigma_up * noise_sampler(sigma, sigma_next)

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x


def sample_sure_wavelet_auto_converge(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
    sure_jac_interval=2,
    sure_adam_mode='bo', sure_adam_beta1=0.9,
    sure_adam_beta2=0.999, sure_adam_wd=0.01,
    sure_approx_coeff=2.0, sure_csv_path=None,
    sure_sigma_ema=0.0, sure_wavelet_warmup_steps=0,
    sure_grad_mode='vjp',
    sure_wavelet_cal_steps=5,
    sure_wavelet_cal_wavelets=('haar', 'db2', 'db4', 'db6', 'sym4'),
    sure_wavelet_cal_levels=(1, 6),
    sure_wavelet_bo_trials=40,
    sure_wavelet_bo_patience=8,
    sure_wavelet_bo_cv_warn=0.4,
    sure_inner_steps=4,
    sure_inner_tol=1e-4,
    sure_alpha_bo_trials=12, sure_alpha_bo_patience=4,
):
    """SURE-Wavelet Auto + convergence loop.

    Phase 1-3: Same Bayesian-optimised wavelet-config search as
    sample_sure_wavelet_auto (collect snapshots → BO/grid → CV advisory).

    Phase 4: Run sample_sure_wavelet_converge with the winning config,
    so each sigma stage iterates SURE corrections until convergence rather
    than applying a single fixed correction.
    """
    extra_args    = {} if extra_args is None else extra_args
    s_in          = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x, seed=extra_args.get('seed', None))
    n_steps       = len(sigmas) - 1
    cal_steps     = min(int(sure_wavelet_cal_steps), n_steps)

    _lvls  = list(sure_wavelet_cal_levels)
    lvl_min, lvl_max = (min(_lvls), max(_lvls)) if len(_lvls) >= 2 else (_lvls[0], _lvls[0])
    cal_wavelet_list  = list(sure_wavelet_cal_wavelets)

    _sure_logger.info(
        "SURE-Wavelet Auto-Converge: collecting %d calibration snapshots  "
        "wavelets=%s  level=[%d,%d]  bo_trials=%d  bo_patience=%d  "
        "inner_steps=%d  inner_tol=%.2e",
        cal_steps, cal_wavelet_list, lvl_min, lvl_max,
        int(sure_wavelet_bo_trials), int(sure_wavelet_bo_patience),
        int(sure_inner_steps), float(sure_inner_tol),
    )

    # ── Phase 1: collect (residual, sigma2) snapshots ───────────────────────
    snapshots = []
    x_cal     = x.clone()
    for i in range(cal_steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_val  = sigma.item()

        with torch.no_grad():
            x0_hat = model(x_cal, sigma * s_in, **extra_args).detach()

        sigma_hat_0 = min(
            _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps)),
            sigma_val,
        )
        residual = (x_cal - x0_hat).detach()
        snapshots.append((residual, sigma_hat_0 ** 2))

        if float(sigma_next) > 0:
            sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=1.0)
            d     = (x_cal - x0_hat) / sigma
            x_cal = x_cal + d * (sigma_down - sigma)
            x_cal = x_cal + sigma_up * noise_sampler(sigma, sigma_next)

    del x_cal

    # ── Phase 2: Bayesian optimisation (or grid fallback) ────────────────────
    best_wavelet = cal_wavelet_list[0]
    best_level   = max(1, min(3, lvl_max))
    best_lp_frac = 1.0
    n_trials_run = 0

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        _no_improve = [0]
        _best_val   = [float('inf')]

        def _bo_callback(study, trial):
            val = study.best_value
            if val < _best_val[0]:
                _best_val[0]   = val
                _no_improve[0] = 0
            else:
                _no_improve[0] += 1
            if _no_improve[0] >= int(sure_wavelet_bo_patience):
                study.stop()

        def _objective(trial):
            wav   = trial.suggest_categorical('wavelet', cal_wavelet_list)
            level = trial.suggest_int('level', lvl_min, lvl_max)
            lpf   = trial.suggest_float('lp_frac', 0.0, 1.0)
            return _sure_score_config(snapshots, wav, level, lpf)

        study = _optuna.create_study(
            direction='minimize',
            sampler=_optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        )
        study.optimize(
            _objective,
            n_trials=int(sure_wavelet_bo_trials),
            callbacks=[_bo_callback],
            show_progress_bar=False,
        )

        best         = study.best_params
        best_wavelet = best['wavelet']
        best_level   = best['level']
        best_lp_frac = best['lp_frac']
        n_trials_run = len(study.trials)

        ranked = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
        )
        for rank, t in enumerate(ranked[:3], 1):
            p = t.params
            _sure_logger.info(
                "SURE-Wavelet Auto-Converge  #%d: wavelet=%s  level=%d  lp_frac=%.3f  sure=%.2f",
                rank, p['wavelet'], p['level'], p['lp_frac'], t.value,
            )

    except ImportError:
        _sure_logger.warning(
            "optuna not installed — falling back to grid search. "
            "Install with: pip install optuna"
        )
        import itertools
        _lp_grid   = [0.0, 0.25, 0.5, 0.75, 1.0]
        _grid      = list(itertools.product(
            cal_wavelet_list,
            range(lvl_min, lvl_max + 1),
            _lp_grid,
        ))
        _scores    = [_sure_score_config(snapshots, w, lv, lf) for w, lv, lf in _grid]
        _best_idx  = int(min(range(len(_scores)), key=lambda j: _scores[j]))
        best_wavelet, best_level, best_lp_frac = _grid[_best_idx]
        n_trials_run = len(_grid)

        ranked_grid = sorted(range(len(_scores)), key=lambda j: _scores[j])
        for rank, idx in enumerate(ranked_grid[:3], 1):
            w, lv, lf = _grid[idx]
            _sure_logger.info(
                "SURE-Wavelet Auto-Converge (grid)  #%d: wavelet=%s  level=%d  lp_frac=%.2f  "
                "sure=%.2f", rank, w, lv, lf, _scores[idx],
            )

    _sure_logger.info(
        "SURE-Wavelet Auto-Converge: using wavelet=%s  level=%d  lp_frac=%.3f  "
        "(%d evaluations)",
        best_wavelet, best_level, best_lp_frac, n_trials_run,
    )

    # ── Phase 3: advisory — check if cal_steps was sufficient ───────────────
    if len(snapshots) > 1:
        import math as _math
        import statistics as _stat
        per_step = [
            _sure_score_config([snap], best_wavelet, best_level, best_lp_frac)
            for snap in snapshots
        ]
        finite = [v for v in per_step if _math.isfinite(v)]
        if len(finite) >= 2:
            cv = _stat.stdev(finite) / (abs(_stat.mean(finite)) + 1e-8)
            if cv > float(sure_wavelet_bo_cv_warn):
                _sure_logger.warning(
                    "SURE-Wavelet Auto-Converge: winner SURE CV=%.3f > %.2f — "
                    "objective is noisy; consider increasing sure_wavelet_cal_steps (current=%d).",
                    cv, float(sure_wavelet_bo_cv_warn), cal_steps,
                )
            else:
                _sure_logger.info(
                    "SURE-Wavelet Auto-Converge: winner CV=%.3f — calibration stable.", cv,
                )

    # ── Phase 4: main sampling with winning config + convergence loop ────────
    return sample_sure_wavelet_converge(
        model, x, sigmas,
        extra_args=extra_args, callback=callback, disable=disable,
        sure_alpha=sure_alpha, sure_n_mc=sure_n_mc, sure_eps=sure_eps,
        sure_jac_interval=sure_jac_interval,
        sure_adam_mode=sure_adam_mode, sure_adam_beta1=sure_adam_beta1,
        sure_adam_beta2=sure_adam_beta2, sure_adam_wd=sure_adam_wd,
        sure_wavelet=best_wavelet, sure_wavelet_level=best_level,
        sure_wavelet_lp_frac=best_lp_frac,
        sure_approx_coeff=sure_approx_coeff, sure_csv_path=sure_csv_path,
        sure_sigma_ema=sure_sigma_ema,
        sure_wavelet_warmup_steps=sure_wavelet_warmup_steps,
        sure_grad_mode=sure_grad_mode,
        sure_inner_steps=sure_inner_steps,
        sure_inner_tol=sure_inner_tol,
        sure_alpha_bo_trials=sure_alpha_bo_trials,
        sure_alpha_bo_patience=sure_alpha_bo_patience,
    )


def sample_sure_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None,
                          rtol=0.05, atol=0.0078, h_init=0.05,
                          pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81,
                          sure_alpha=0.05, sure_n_mc=1, sure_eps=1e-3,
                          sure_preheat_frac=0.3, sure_jac_interval=2,
                          sure_adam_mode='none', sure_adam_beta1=0.9,
                          sure_adam_beta2=0.999, sure_adam_wd=0.01, sure_grad_mode='vjp',
                          sure_alpha_bo_trials=0, sure_alpha_bo_patience=4):
    """SURE sampler with adaptive step size (PID controller on DPM-Solver-2 error).

    Combines:
    - DPM-Solver-2 local truncation error for step size control (via PID)
    - SURE trajectory correction applied after the preheat phase

    The sigma schedule is computed automatically — only sigma_min and sigma_max
    are required (no steps parameter).

    Step size control:
      rtol, atol:      error tolerances for PID controller
      h_init:          initial step size in log-sigma space
      pcoeff/icoeff/dcoeff: PID gains (default: I-only)
      accept_safety:   step accepted if PID factor >= this

    SURE correction:
      sure_alpha:        gradient step size
      sure_n_mc:         Monte Carlo samples for Jacobian trace
      sure_eps:          finite-difference epsilon
      sure_preheat_frac: fraction of t-range to run without correction (default 0.3)
      sure_jac_interval: run full Jacobian every N correction steps (default 2)
    """
    if sigma_max <= 0:
        raise ValueError('sigma_max must be > 0')
    # sigma_min=0 is valid for zero-SNR models (e.g. Illustrious).
    # Log-space integration requires sigma > 0, so clamp to a small floor.
    _sigma_min_log = max(float(sigma_min), 1e-6)
    if sigma_min <= 0:
        _sure_logger.warning(
            "SURE-Adaptive: sigma_min=0 (zero-SNR model) — clamping "
            "log-space endpoint to 1e-6; final denoised output is unaffected"
        )

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Work in log-sigma space: t = -log(sigma), sigma = exp(-t)
    # t increases as sigma decreases (t_start < t_end)
    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    t_start = t_fn(torch.tensor(sigma_max,      dtype=x.dtype, device=x.device))
    t_end   = t_fn(torch.tensor(_sigma_min_log, dtype=x.dtype, device=x.device))
    t_preheat = t_start + sure_preheat_frac * (t_end - t_start)

    pid    = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, order=2, accept_safety=accept_safety)
    atol_t = torch.tensor(atol, dtype=x.dtype, device=x.device)
    rtol_t = torch.tensor(rtol, dtype=x.dtype, device=x.device)

    info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}
    x_prev = x.clone()
    s = t_start.clone()

    _sure_logger.info(
        "SURE-Adaptive: sigma [%.4f → %.4f]  preheat_frac=%.2f  alpha=%.4f  jac_interval=%d",
        sigma_max, sigma_min, sure_preheat_frac, sure_alpha, sure_jac_interval,
    )

    # jac_interval tracking (fixed for adaptive sampler — not adaptive, to keep control predictable)
    _corr_count = 0
    _adam_state     = {'optimizer': None, 'param': None} if sure_adam_mode in ('adam', 'adamw') else None
    _alpha_bo_state = {} if sure_alpha_bo_trials > 0 else None

    with tqdm(disable=disable) as pbar:
        while s < t_end - 1e-5:
            t_next = torch.minimum(t_end, s + pid.h)
            sigma_s    = sigma_fn(s)
            sigma_next = sigma_fn(t_next)

            in_preheat = float(s) < float(t_preheat)
            _tag = f" [adap] sigma={float(sigma_s):.4f}"

            # --- Primary x̂₀ at current state (with SURE after preheat) ---
            if in_preheat:
                with torch.no_grad():
                    x0_s = model(x, sigma_s * s_in, **extra_args).detach()
            else:
                with torch.no_grad():
                    x0_hat = model(x, sigma_s * s_in, **extra_args).detach()
                sigma_hat_0 = _pca_noise_estimate(x0_hat, min_sigma=float(sure_eps))
                _use_jac = (sure_jac_interval <= 1) or (_corr_count % sure_jac_interval == 0)
                x0_s, _ = _sure_correct_x0(
                    model, x0_hat, sigma_hat_0, s_in, extra_args,
                    alpha=sure_alpha, n_mc=sure_n_mc, eps_mc=sure_eps,
                    use_jac=_use_jac, sigma_t=sigma_s,
                    adam_state=_adam_state, adam_mode=sure_adam_mode,
                    adam_beta1=sure_adam_beta1, adam_beta2=sure_adam_beta2,
                    adam_wd=sure_adam_wd, grad_mode=sure_grad_mode,
                    alpha_bo_trials=sure_alpha_bo_trials,
                    alpha_bo_patience=sure_alpha_bo_patience,
                    alpha_bo_state=_alpha_bo_state,
                )
                _corr_count += 1

            # --- DPM-Solver-1 step (low order) ---
            # eps_s = (x - x̂₀_s) / sigma_s
            # x_low = x - sigma_next * expm1(h) * eps_s   where h = t_next - s
            h = t_next - s
            eps_s = (x - x0_s) / sigma_s
            x_low = x - sigma_next * h.expm1() * eps_s

            # --- DPM-Solver-2 midpoint step (high order) ---
            # midpoint t: s1 = s + h/2, sigma_mid = sigma_fn(s1)
            r = 0.5
            s1 = s + r * h
            sigma_mid = sigma_fn(s1)
            u1 = x - sigma_mid * (r * h).expm1() * eps_s
            with torch.no_grad():
                x0_mid = model(u1, sigma_mid * s_in, **extra_args).detach()
            eps_mid = (u1 - x0_mid) / sigma_mid
            # 2nd-order correction term
            x_high = x - sigma_next * h.expm1() * eps_s \
                       - sigma_next / (2 * r) * h.expm1() * (eps_mid - eps_s)

            # --- PID error estimate ---
            delta = torch.maximum(atol_t, rtol_t * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5

            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high
                s = t_next
                info['n_accept'] += 1
                pbar.update()
                if callback is not None:
                    callback({'x': x, 'i': info['steps'], 'sigma': sigma_s, 'sigma_hat': sigma_s,
                              'denoised': x0_s, 'error': error, 'h': pid.h, **info})
            else:
                info['n_reject'] += 1

            info['nfe']   += 2   # x0_s + x0_mid (SURE counts more internally but we log 2)
            info['steps'] += 1

            _sure_logger.info(
                "[adap] step=%d  sigma=%.4f→%.4f  error=%.4f  h=%.4f  accept=%s  preheat=%s",
                info['steps'], float(sigma_s), float(sigma_next),
                float(error), float(pid.h), accept, in_preheat,
            )

            if info['steps'] > 10000:
                _sure_logger.warning("SURE-Adaptive: step limit reached, stopping early")
                break

    _sure_logger.info(
        "SURE-Adaptive done: %d accepted / %d rejected  nfe=%d",
        info['n_accept'], info['n_reject'], info['nfe'],
    )

    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    return x

