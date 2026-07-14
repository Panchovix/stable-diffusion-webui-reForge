"""
test/test_jax_pipeline_convert.py

Unit tests for jax_pipeline/convert.py.

Requires torch (a hard project dependency) and jax (optional — see
requirements_jax.txt); both are skipped via pytest.importorskip so this
file collects cleanly in environments without jax installed.

Stubs diffusers/accelerate/modules_forge internals that diff_pipeline.pipeline
touches at import time (same list as test_diff_pipeline_pipeline.py), but
deliberately does NOT stub ldm_patched.modules.utils — convert.py's whole
point is to reuse the *real* unet_to_diffusers() key mapping, so the test
exercises it for real against a small synthetic SDXL-like config.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _make_stub(*parts):
    root_name = parts[0]
    root = sys.modules.setdefault(root_name, MagicMock())
    current = root
    for part in parts[1:]:
        child = getattr(current, part, None)
        if child is None or not isinstance(child, MagicMock):
            child = MagicMock()
            setattr(current, part, child)
        full_name = ".".join(parts[: parts.index(part) + 1])
        sys.modules.setdefault(full_name, child)
        current = child
    return root


# Stub everything diff_pipeline.pipeline touches at import time EXCEPT
# ldm_patched.modules.utils — we want the real unet_to_diffusers().
_make_stub("modules", "shared_cmd_options")
_make_stub("modules_forge", "unet_patcher")
_make_stub("modules_forge", "stream")
_make_stub("ldm_patched", "modules", "model_management")
_make_stub("ldm_patched", "modules", "args_parser")
_make_stub("ldm_patched", "modules", "weight_adapter", "lora")
_make_stub("diffusers", "models", "unets", "unet_2d_condition")
_make_stub("diffusers", "models", "attention_processor")
_make_stub("accelerate")

_stream_stub = sys.modules["modules_forge.stream"]
_stream_stub.using_stream = False
_stream_stub.current_stream = None

from jax_pipeline import convert  # noqa: E402


# A tiny SDXL-shaped config (2 down blocks instead of 3, depth-1 res blocks)
# so unet_to_diffusers() produces a small-but-real key map fast.
_TINY_UNET_CFG = {
    "num_res_blocks": [1, 1],
    "channel_mult": [1, 2],
    "transformer_depth": [0, 2],
    # popped once per (num_res_blocks[x] + 1) per up block => 2+2 = 4 entries needed
    "transformer_depth_output": [0, 2, 0, 2],
    "transformer_depth_middle": 2,
}


class TestIsConvWeight:
    def test_conv_in_is_conv(self):
        assert convert._is_conv_weight("conv_in.weight", 4)

    def test_downsampler_is_conv(self):
        assert convert._is_conv_weight("down_blocks.0.downsamplers.0.conv.weight", 4)

    def test_upsampler_is_conv(self):
        assert convert._is_conv_weight("up_blocks.0.upsamplers.0.conv.weight", 4)

    def test_resnet_conv_is_conv(self):
        assert convert._is_conv_weight("down_blocks.0.resnets.0.conv1.weight", 4)

    def test_norm_weight_is_not_conv(self):
        assert not convert._is_conv_weight("down_blocks.0.resnets.0.norm1.weight", 1)

    def test_linear_weight_is_not_conv(self):
        assert not convert._is_conv_weight("down_blocks.0.resnets.0.time_emb_proj.weight", 2)

    def test_bias_is_not_conv(self):
        # 1-D bias tensors never match regardless of key content
        assert not convert._is_conv_weight("conv_in.bias", 1)


class TestLdmSdToHf:
    def test_resolves_known_keys(self):
        from diff_pipeline.pipeline import _SDXL_LDM_UNET_CONFIG
        from ldm_patched.modules.utils import unet_to_diffusers

        merged = dict(_SDXL_LDM_UNET_CONFIG)
        merged.update(_TINY_UNET_CFG)
        key_map = unet_to_diffusers(merged)

        # conv_in.weight is a fixed 1:1 identity mapping (UNET_MAP_BASIC)
        assert key_map["conv_in.weight"] == "input_blocks.0.0.weight"

        ldm_sd = {ldm_key: torch.zeros(1) for ldm_key in key_map.values()}
        hf_sd = convert.ldm_sd_to_hf(ldm_sd, _TINY_UNET_CFG)

        assert "conv_in.weight" in hf_sd
        assert torch.equal(hf_sd["conv_in.weight"], ldm_sd["input_blocks.0.0.weight"])

    def test_missing_ldm_keys_are_dropped_not_fatal(self):
        # Empty ldm_state_dict -> every HF key is "missing"; should not raise.
        hf_sd = convert.ldm_sd_to_hf({}, _TINY_UNET_CFG)
        assert hf_sd == {}


class TestHfSdToJax:
    def test_conv_weight_transposed_oihw_to_hwio(self):
        w = torch.randn(8, 4, 3, 3)  # O=8, I=4, kH=3, kW=3
        hf_sd = {"conv_in.weight": w}
        params = convert.hf_sd_to_jax(hf_sd, dtype=jnp.float32)

        out = params["conv_in.weight"]
        assert out.shape == (3, 3, 4, 8)  # kH, kW, I, O
        # spot-check a value survives the transpose at its new coordinates
        assert float(out[1, 2, 3, 5]) == pytest.approx(float(w[5, 3, 1, 2]), abs=1e-5)

    def test_linear_weight_not_transposed(self):
        w = torch.randn(1280, 320)
        hf_sd = {"down_blocks.0.resnets.0.time_emb_proj.weight": w}
        params = convert.hf_sd_to_jax(hf_sd, dtype=jnp.float32)

        out = params["down_blocks.0.resnets.0.time_emb_proj.weight"]
        assert out.shape == (1280, 320)
        assert float(out[10, 20]) == pytest.approx(float(w[10, 20]), abs=1e-5)

    def test_default_dtype_is_bfloat16(self):
        hf_sd = {"conv_norm_out.weight": torch.randn(320)}
        params = convert.hf_sd_to_jax(hf_sd)
        assert params["conv_norm_out.weight"].dtype == jnp.bfloat16

    def test_bad_tensor_is_skipped_not_fatal(self):
        # A value that isn't a torch.Tensor should be skipped, not raise.
        hf_sd = {"conv_in.weight": torch.randn(8, 4, 3, 3), "broken.weight": object()}
        params = convert.hf_sd_to_jax(hf_sd, dtype=jnp.float32)
        assert "conv_in.weight" in params
        assert "broken.weight" not in params
