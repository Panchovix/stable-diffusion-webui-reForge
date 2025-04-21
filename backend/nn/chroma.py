# implementation of Chroma for Forge, inspired by https://github.com/lodestone-rock/ComfyUI_FluxMod

from dataclasses import dataclass

import math
import torch

from torch import nn
from einops import rearrange, repeat
from backend.attention import attention_function
from backend.utils import fp16_fix, tensor2parameter


def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)
    x = attention_function(q, k, v, q.shape[1], skip_reshape=True)
    return x


def rope(pos, dim, theta):
    if pos.device.type == "mps" or pos.device.type == "xpu":
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    else:
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)

    # out = torch.einsum("...n,d->...nd", pos, omega)
    out = pos.unsqueeze(-1) * omega.unsqueeze(0)

    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    del cos_out, sin_out

    # out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    b, n, d, _ = out.shape
    out = out.view(b, n, d, 2, 2)

    return out.float()


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    del xq_, xk_
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2

    # TODO: Once A trainer for flux get popular, make timestep_embedding consistent to that trainer

    # Do not block CUDA steam, but having about 1e-4 differences with Flux official codes:
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    # Block CUDA steam, but consistent with official codes:
    # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    del freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    del args
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        del ids, n_axes
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


if hasattr(torch, 'rms_norm'):
    functional_rms_norm = torch.rms_norm
else:
    def functional_rms_norm(x, normalized_shape, weight, eps):
        if x.dtype in [torch.bfloat16, torch.float32]:
            n = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps) * weight
        else:
            n = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps).to(x.dtype) * weight
        return x * n


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = None  # to trigger module_profile
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6
        self.normalized_shape = [dim]

    def forward(self, x):
        if self.scale.dtype != x.dtype:
            self.scale = tensor2parameter(self.scale.to(dtype=x.dtype))
        return functional_rms_norm(x, self.normalized_shape, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        del v
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(k), k.to(q)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pe):
        qkv = self.qkv(x)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)

        x = attention(q, k, v, pe=pe)
        del q, k, v

        x = self.proj(x)
        return x

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm( hidden_dim) for x in range( n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x

@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor

class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, mod, pe):
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = mod
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = fp16_fix(txt)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, mod, pe):
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del x_mod

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        del q, k, v, pe
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        del attn, mlp

        x = x + mod.gate * output
        x = fp16_fix(x)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, mod):
        shift, scale = mod
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class IntegratedChromaTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int, vec_in_dim: int, context_in_dim: int, hidden_size: int, mlp_ratio: float, num_heads: int, depth: int, depth_single_blocks: int, axes_dim: list[int], theta: int, qkv_bias: bool, guidance_embed):
        super().__init__()
        
        guidance_out_dim = 3072
        guidance_hidden_dim = 5120
        guidance_n_layers = 5

        self.in_channels = in_channels * 4
        self.out_channels = self.in_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.distilled_guidance_layer = Approximator(64, guidance_out_dim, guidance_hidden_dim, guidance_n_layers)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        
    @staticmethod
    def distribute_modulations(tensor, single_block_count: int = 38, double_blocks_count: int = 19):
        """
        Distributes slices of the tensor into the block_dict as ModulationOut objects.

        Args:
            tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
        """
        batch_size, vectors, dim = tensor.shape
        block_dict = {}
        for i in range(single_block_count):
            key = f"single_blocks.{i}.modulation.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.img_mod.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.txt_mod.lin"
            block_dict[key] = None
        block_dict["final_layer.adaLN_modulation.1"] = None
        idx = 0  # Index to keep track of the vector slices
        for key in block_dict.keys():
            if "single_blocks" in key:
                # Single block: 1 ModulationOut
                block_dict[key] = ModulationOut(
                    shift=tensor[:, idx:idx+1, :],
                    scale=tensor[:, idx+1:idx+2, :],
                    gate=tensor[:, idx+2:idx+3, :]
                )
                idx += 3  # Advance by 3 vectors
            elif "img_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block
            elif "txt_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block
            elif "final_layer" in key:
                # Final layer: 1 ModulationOut
                block_dict[key] = [
                    tensor[:, idx:idx+1, :],
                    tensor[:, idx+1:idx+2, :],
                ]
                idx += 2  # Advance by 2 vectors
        return block_dict
        
    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps, guidance=None):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        device = img.device
        dtype = img.dtype
        nb_double_block = len(self.double_blocks)
        nb_single_block = len(self.single_blocks)
        
        mod_index_length = nb_double_block*12 + nb_single_block*3 + 2
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(device=device, dtype=dtype)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(device=device, dtype=dtype)
        modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(device=device, dtype=dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
        mod_vectors = self.distilled_guidance_layer(input_vec)
        mod_vectors_dict = self.distribute_modulations(mod_vectors, nb_single_block, nb_double_block)
        
        txt = self.txt_in(txt)
        del guidance
        ids = torch.cat((txt_ids, img_ids), dim=1)
        del txt_ids, img_ids
        pe = self.pe_embedder(ids)
        del ids
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]
            img, txt = block(img=img, txt=txt, mod=double_mod, pe=pe)
        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            img = block(img, mod=single_mod, pe=pe)
        del pe
        img = img[:, txt.shape[1]:, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, final_mod)
        return img

    def forward(self, x, timestep, context, guidance=None, **kwargs):
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        patch_size = 2
        pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
        pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        del x, pad_h, pad_w
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
        del input_device, input_dtype
        out = self.inner_forward(img, img_ids, context, txt_ids, timestep, guidance)
        del img, img_ids, txt_ids, timestep, context
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        del h_len, w_len, bs
        return out
