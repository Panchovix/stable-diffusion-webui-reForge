import dataclasses
import torch
import numpy as np
from scipy import stats
import math

from modules import shared

if shared.opts.sd_sampling == "A1111":
    import k_diffusion
    from k_diffusion.sampling import append_zero
    from k_diffusion import sampling
elif shared.opts.sd_sampling == "ldm patched (Comfy)":
    import ldm_patched.k_diffusion
    from ldm_patched.k_diffusion.sampling import append_zero
    from ldm_patched.k_diffusion import sampling

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


if shared.opts.sd_sampling == "A1111":
    k_diffusion.sampling.to_d = to_d
elif shared.opts.sd_sampling == "ldm patched (Comfy)":
    ldm_patched.k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any

    default_rho: float = -1
    need_inner_model: bool = False
    aliases: list = None


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [
        inner_model.t_to_sigma(ts)
        for ts in torch.linspace(start, end, n + 1)[:-1]
    ]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    rho = shared.opts.karras_rho
    return ldm_patched.k_diffusion.sampling.get_sigmas_karras(n, sigma_min, sigma_max, rho, device)

def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    shrink_factor = shared.opts.exponential_shrink_factor
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    sigmas = sigmas * torch.exp(shrink_factor * torch.linspace(0, 1, n, device=device))
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, device='cpu'):
    rho = shared.opts.polyexponential_rho
    return ldm_patched.k_diffusion.sampling.get_sigmas_polyexponential(n, sigma_min, sigma_max, rho, device)


def get_sigmas_sinusoidal_sf(n, sigma_min, sigma_max, device='cpu'):
    sf = shared.opts.sinusoidal_sf_factor
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return sigmas

def get_sigmas_invcosinusoidal_sf(n, sigma_min, sigma_max, device='cpu'):
    sf = shared.opts.invcosinusoidal_sf_factor
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return sigmas

def get_sigmas_react_cosinusoidal_dynsf(n, sigma_min, sigma_max, device='cpu'):
    sf = shared.opts.react_cosinusoidal_dynsf_factor
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(n*x/n))
    sigmas = sigmas * sigma_max
    return sigmas


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    if shared.sd_model.is_sdxl:
        sigmas = sigmas = [sigma_max, sigma_max/2.314, sigma_max/3.875, sigma_max/6.701, sigma_max/10.89, sigma_max/16.954, sigma_max/26.333, sigma_max/38.46, sigma_max/62.457, sigma_max/129.336, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [sigma_max, sigma_max/2.257, sigma_max/3.785, sigma_max/5.418, sigma_max/7.749, sigma_max/10.469, sigma_max/15.176, sigma_max/22.415, sigma_max/36.629, sigma_max/96.151, 0.029]


    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def get_sigmas_ays_custom(n, sigma_min, sigma_max, device='cpu'):
    try:
        sigmas_str = shared.opts.ays_custom_sigmas
        sigmas_values = sigmas_str.strip('[]').split(',')
        sigmas = np.array([float(x.strip()) for x in sigmas_values])
        
        if n != len(sigmas):
            sigmas = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sigmas)), sigmas)
        sigmas = np.append(sigmas, [0.0])
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        print(f"Error parsing custom sigmas: {e}")
        print("Falling back to default AYS sigmas")
        return get_align_your_steps_sigmas(n, sigma_min, sigma_max, device)

def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas

def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    """
    Beta scheduler, based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    
    total_timesteps = (len(inner_model.sigmas) - 1)
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = unet.model.model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)

def get_align_your_steps_sigmas_GITS(n, sigma_min, sigma_max, device):
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    if shared.sd_model.is_sdxl:
        sigmas = [sigma_max, sigma_max/3.087, sigma_max/5.693, sigma_max/9.558, sigma_max/14.807, sigma_max/22.415, sigma_max/34.964, sigma_max/54.533, sigma_max/81.648, sigma_max/115.078, 0.029]

    else:
        sigmas = [sigma_max, sigma_max/3.165, sigma_max/5.829, sigma_max/11.824, sigma_max/20.819, sigma_max/36.355, sigma_max/60.895, sigma_max/93.685, sigma_max/140.528, sigma_max/155.478, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def ays_11_sigmas(n, sigma_min, sigma_max, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    if shared.sd_model.is_sdxl:
        sigmas = [sigma_max, sigma_max/2.314, sigma_max/3.875, sigma_max/6.701, sigma_max/10.89, sigma_max/16.954, sigma_max/26.333, sigma_max/38.46, sigma_max/62.457, sigma_max/129.336, 0.029]
    else:
        sigmas = [sigma_max, sigma_max/2.257, sigma_max/3.785, sigma_max/5.418, sigma_max/7.749, sigma_max/10.469, sigma_max/15.176, sigma_max/22.415, sigma_max/36.629, sigma_max/96.151, 0.029]


    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def ays_32_sigmas(n, sigma_min, sigma_max, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])
        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys
    if shared.sd_model.is_sdxl:
        sigmas = [sigma_max, sigma_max/1.310860875657935, sigma_max/1.718356235075352, sigma_max/2.252525958180810, sigma_max/2.688026675053433, sigma_max/3.174423075322040, sigma_max/3.748832539417044, sigma_max/4.463856789920335, sigma_max/5.326233593328242, sigma_max/6.355213820679800, sigma_max/7.477672611007930, sigma_max/8.745803592589411, sigma_max/10.228995682978878, sigma_max/11.864653584709637, sigma_max/13.685783347784952, sigma_max/15.786441921021279, sigma_max/18.202564111697559, sigma_max/20.980440157432400, sigma_max/24.182245076323649, sigma_max/27.652401723193991, sigma_max/31.246429590323925, sigma_max/35.307579021272943, sigma_max/40.308138967569972, sigma_max/47.132212095147923, sigma_max/55.111585405517003, sigma_max/65.460441760115945, sigma_max/82.786347724072168, sigma_max/104.698036963744033, sigma_max/138.041693219503482, sigma_max/264.794761864988552, sigma_max/507.935470821253285, 0.015000000000000000]
    else:
        sigmas = [sigma_max, sigma_max/1.300323183382763, sigma_max/1.690840379611262, sigma_max/2.198638945761486, sigma_max/2.622696705671493, sigma_max/3.098705619671305, sigma_max/3.661108232617473, sigma_max/4.152506637972936, sigma_max/4.662023756728857, sigma_max/5.234059175875519, sigma_max/5.874818853387466, sigma_max/6.593316416277412, sigma_max/7.399687115002039, sigma_max/8.213824943635682, sigma_max/9.050917900247738, sigma_max/9.973321246245751, sigma_max/11.115344803852001, sigma_max/12.529738625194212, sigma_max/14.124109921351757, sigma_max/15.959814856974724, sigma_max/18.099481611774999, sigma_max/20.526004748634670, sigma_max/23.506648288108032, sigma_max/27.541589307433523, sigma_max/32.269132736422456, sigma_max/38.982216080970984, sigma_max/53.219344283057142, sigma_max/72.656173487928834, sigma_max/103.609326413189740, sigma_max/218.693105563304210, sigma_max/461.605857767280530, 0.015000000000000000]
           
    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)
    return torch.FloatTensor(sigmas).to(device)

def cosine_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sf = shared.opts.cosine_sf_factor
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C * sf
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler(n, sigma_min, sigma_max, device='cpu'):
    decay = shared.opts.cosexpblend_exp_decay
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = decay ** (1/(n-1))
        E = sigma_max
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

def phi_scheduler(n, sigma_min, sigma_max, device='cpu'):
    power = shared.opts.phi_power
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(n-1))**(phi**power))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_laplace(n, sigma_min, sigma_max, device='cpu'):
    mu = shared.opts.laplace_mu
    beta = shared.opts.laplace_beta
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_karras_dynamic(n, sigma_min, sigma_max, device='cpu'):
    rho = shared.opts.karras_dynamic_rho
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/n)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('karras', 'Karras', sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sinusoidal_sf', 'Sinusoidal SF', get_sigmas_sinusoidal_sf),
    Scheduler('invcosinusoidal_sf', 'Invcosinusoidal SF', get_sigmas_invcosinusoidal_sf),
    Scheduler('react_cosinusoidal_dynsf', 'React Cosinusoidal DynSF', get_sigmas_react_cosinusoidal_dynsf),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
    Scheduler('align_your_steps_custom', 'Align Your Steps Custom', get_sigmas_ays_custom),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
    Scheduler('turbo', 'Turbo', turbo_scheduler, need_inner_model=True),
    Scheduler('cosine', 'Cosine', cosine_scheduler),
    Scheduler('cosine-exponential blend', 'Cosine-exponential Blend', cosexpblend_scheduler),
    Scheduler('phi', 'Phi', phi_scheduler),
    Scheduler('laplace', 'Laplace', get_sigmas_laplace),
    Scheduler('karras dynamic', 'Karras Dynamic', get_sigmas_karras_dynamic),
    Scheduler('align_your_steps_GITS', 'Align Your Steps GITS', get_align_your_steps_sigmas_GITS),
    Scheduler('align_your_steps_11', 'Align Your Steps 11', ays_11_sigmas),
    Scheduler('align_your_steps_32', 'Align Your Steps 32', ays_32_sigmas),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
