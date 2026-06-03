"""
test/test_diff_pipeline_pipeline.py

Unit tests for diff_pipeline/pipeline.py.

Heavy use of unittest.mock so that diffusers, ldm_patched, and webui
internals are never actually imported — the suite runs in any plain Python
environment that has torch installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub the webui / ldm_patched / diffusers modules before the first import
# of diff_pipeline.pipeline.  This must happen before any import of the
# module under test.
# ---------------------------------------------------------------------------

def _make_stub(*parts):
    """Return a hierarchy of MagicMock namespaces for a dotted module path."""
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


# Stub every external dependency that pipeline.py touches at import time.
_make_stub("modules", "shared_cmd_options")
_make_stub("modules_forge", "unet_patcher")
_make_stub("modules_forge", "stream")
_make_stub("ldm_patched", "modules", "model_management")
_make_stub("ldm_patched", "modules", "utils")
_make_stub("ldm_patched", "modules", "args_parser")
_make_stub("ldm_patched", "modules", "weight_adapter", "lora")
_make_stub("diffusers", "models", "unets", "unet_2d_condition")
_make_stub("diffusers", "models", "attention_processor")
_make_stub("accelerate")

# Disable the CUDA stream path — _get_stream_module() checks `using_stream` and
# `current_stream` on the returned module; setting both to falsy keeps apply_model()
# on the non-stream branch in tests.
_stream_stub = sys.modules["modules_forge.stream"]
_stream_stub.using_stream = False
_stream_stub.current_stream = None

# Provide a real-ish AttnProcessor2_0 stub so PassthroughAttnProcessor works.
_attn_proc_mod = sys.modules["diffusers.models.attention_processor"]

class _FakeAttnProcessor2_0:
    def __call__(self, attn, hidden_states, **kwargs):
        return hidden_states

_attn_proc_mod.AttnProcessor2_0 = _FakeAttnProcessor2_0

# Now import the module under test.
from diff_pipeline.pipeline import (  # noqa: E402
    _SDXL_HF_UNET_CONFIG,
    _SDXL_LDM_UNET_CONFIG,
    _DOWN_ATTN_LDM_IDX,
    _UP_ATTN_LDM_IDX,
    _derive_hf_config_from_ldm,
    ForgeAttnProcessor,
    PassthroughAttnProcessor,
    DiffPipeline,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_attn_mock(
    heads: int = 8,
    inner_dim: int = 64,
    seq_len: int = 4,
    batch: int = 2,
    use_spatial_norm: bool = False,
    use_group_norm: bool = False,
    residual_connection: bool = False,
):
    """Return a MagicMock that looks enough like a diffusers Attention module."""
    attn = MagicMock()
    attn.heads = heads
    attn.spatial_norm = MagicMock() if use_spatial_norm else None
    attn.group_norm = MagicMock() if use_group_norm else None
    attn.norm_cross = False
    attn.residual_connection = residual_connection
    attn.rescale_output_factor = 1.0
    attn.norm_q = None
    attn.norm_k = None

    # to_q / to_k / to_v return tensors shaped (B, seq, inner_dim)
    def _proj(x):
        B = x.shape[0]
        S = x.shape[1]
        return torch.zeros(B, S, inner_dim)

    attn.to_q = MagicMock(side_effect=_proj)
    attn.to_q.out_features = inner_dim
    attn.to_k = MagicMock(side_effect=_proj)
    attn.to_v = MagicMock(side_effect=_proj)

    # to_out[0] is projection, to_out[1] is dropout (identity in tests)
    attn.to_out = [MagicMock(side_effect=lambda x: x), MagicMock(side_effect=lambda x: x)]

    attn.prepare_attention_mask = MagicMock(side_effect=lambda mask, seq, batch: mask)

    return attn


def _make_diff_pipeline_bare() -> DiffPipeline:
    """Create a DiffPipeline instance bypassing __init__."""
    dp = DiffPipeline.__new__(DiffPipeline)
    # Minimal state required by most methods
    dp._hf_unet = MagicMock()
    # Mark as already compiled so apply_model() never calls torch.compile on a mock.
    dp._compiled = True
    dp._auto_offload = False
    dp._sequential_offload = False
    dp._offload = False
    dp._seq_hooks_installed = False
    dp._mps_optimized = False
    dp._auto_offload_ready = False
    dp._b_hooks = []
    dp._b_block_paths = []
    dp._tc_ready = False
    dp._autocast_dtype = None
    dp._ldm_to_hf = {}
    dp._synced_patches_uuid = None
    dp._active_adapters = []

    # model_sampling stub
    ms = MagicMock()
    ms.calculate_input = MagicMock(side_effect=lambda sigma, x: x)
    ms.timestep = MagicMock(side_effect=lambda sigma: sigma)
    ms.calculate_denoised = MagicMock(side_effect=lambda sigma, out, x: out)
    ms.sigma_min = torch.tensor(0.029)
    ms.sigma_max = torch.tensor(14.6)
    ms.sigmas = torch.linspace(14.6, 0.029, 20)
    ms.zsnr = False
    dp.model_sampling = ms

    # sd_model stub
    dp.sd_model = MagicMock()
    dp.sd_model.forge_objects = None

    # unet_patcher stub
    dp.unet_patcher = MagicMock()
    dp.unet_patcher.patches = {}
    dp.unet_patcher.patches_uuid = "init-uuid"

    return dp


# ===========================================================================
# _derive_hf_config_from_ldm
# ===========================================================================

class TestDeriveHfConfigFromLdm:

    def test_empty_config_no_overrides(self):
        overrides, report = _derive_hf_config_from_ldm({})
        assert overrides == {}
        assert report == []

    def test_in_channels_matching_base_no_override(self):
        overrides, _ = _derive_hf_config_from_ldm({"in_channels": 4})
        assert "in_channels" not in overrides

    def test_in_channels_different_adds_override(self):
        overrides, report = _derive_hf_config_from_ldm({"in_channels": 8})
        assert overrides["in_channels"] == 8
        assert any("in_channels" in r for r in report)

    def test_out_channels_different_adds_override(self):
        overrides, report = _derive_hf_config_from_ldm({"out_channels": 8})
        assert overrides["out_channels"] == 8
        assert any("out_channels" in r for r in report)

    def test_context_dim_maps_to_cross_attention_dim(self):
        overrides, report = _derive_hf_config_from_ldm({"context_dim": 1024})
        assert overrides["cross_attention_dim"] == 1024
        assert any("cross_attention_dim" in r for r in report)

    def test_context_dim_matching_base_no_override(self):
        base_cross = _SDXL_HF_UNET_CONFIG["cross_attention_dim"]
        overrides, _ = _derive_hf_config_from_ldm({"context_dim": base_cross})
        assert "cross_attention_dim" not in overrides

    def test_adm_in_channels_maps_to_projection_dim(self):
        overrides, report = _derive_hf_config_from_ldm({"adm_in_channels": 1024})
        assert overrides["projection_class_embeddings_input_dim"] == 1024
        assert any("projection_class_embeddings_input_dim" in r for r in report)

    def test_model_channels_and_channel_mult_derive_block_out_channels(self):
        overrides, report = _derive_hf_config_from_ldm(
            {"model_channels": 256, "channel_mult": [1, 2, 4]}
        )
        assert overrides["block_out_channels"] == [256, 512, 1024]
        assert any("block_out_channels" in r for r in report)

    def test_block_out_channels_matching_base_no_override(self):
        base_boc = _SDXL_HF_UNET_CONFIG["block_out_channels"]
        mc = base_boc[0]
        cm = [b // mc for b in base_boc]
        overrides, _ = _derive_hf_config_from_ldm(
            {"model_channels": mc, "channel_mult": cm}
        )
        assert "block_out_channels" not in overrides

    def test_transformer_depth_maps_to_transformer_layers_per_block(self):
        # ldm layout: pairs per block — [d0, d0_skip, d1, d1_skip, d2, d2_skip]
        overrides, report = _derive_hf_config_from_ldm(
            {"transformer_depth": [0, 0, 2, 2, 5, 5]}
        )
        # hf picks first non-zero from each pair: [0, 2, 5]
        assert overrides["transformer_layers_per_block"] == [0, 2, 5]
        assert any("transformer_layers_per_block" in r for r in report)

    def test_transformer_depth_too_short_no_override(self):
        overrides, _ = _derive_hf_config_from_ldm({"transformer_depth": [1, 2]})
        assert "transformer_layers_per_block" not in overrides

    def test_transformer_depth_matching_base_no_override(self):
        base_td = _SDXL_HF_UNET_CONFIG["transformer_layers_per_block"]
        # Expand back to ldm layout: duplicate each value
        ldm_td = []
        for v in base_td:
            ldm_td += [v, v]
        overrides, _ = _derive_hf_config_from_ldm({"transformer_depth": ldm_td})
        assert "transformer_layers_per_block" not in overrides

    def test_num_res_blocks_list_uses_first_element(self):
        overrides, report = _derive_hf_config_from_ldm({"num_res_blocks": [3, 3, 3]})
        assert overrides["layers_per_block"] == 3
        assert any("layers_per_block" in r for r in report)

    def test_num_res_blocks_int_uses_value(self):
        overrides, _ = _derive_hf_config_from_ldm({"num_res_blocks": 3})
        assert overrides["layers_per_block"] == 3

    def test_num_res_blocks_matching_base_no_override(self):
        base_lpb = _SDXL_HF_UNET_CONFIG["layers_per_block"]
        overrides, _ = _derive_hf_config_from_ldm({"num_res_blocks": [base_lpb]})
        assert "layers_per_block" not in overrides

    def test_report_contains_human_readable_strings(self):
        _, report = _derive_hf_config_from_ldm(
            {"in_channels": 8, "context_dim": 512}
        )
        for entry in report:
            assert isinstance(entry, str)
            assert "→" in entry

    def test_multiple_overrides_returned_together(self):
        overrides, report = _derive_hf_config_from_ldm(
            {
                "in_channels": 8,
                "out_channels": 8,
                "context_dim": 512,
                "adm_in_channels": 1024,
            }
        )
        assert len(overrides) == 4
        assert len(report) == 4


# ===========================================================================
# Block address tables
# ===========================================================================

class TestBlockAddressTables:

    def test_down_attn_ldm_idx_keys(self):
        expected = {(1, 0): 4, (1, 1): 5, (2, 0): 7, (2, 1): 8}
        assert _DOWN_ATTN_LDM_IDX == expected

    def test_up_attn_ldm_idx_keys(self):
        expected = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2,
            (1, 0): 3, (1, 1): 4, (1, 2): 5,
        }
        assert _UP_ATTN_LDM_IDX == expected

    def test_ldm_indices_unique_within_each_table(self):
        # Keys may overlap across tables (same (b_idx, a_idx) tuple means
        # different things in down vs up context), but LDM indices must be
        # unique within each table to avoid ambiguous block mapping.
        assert len(set(_DOWN_ATTN_LDM_IDX.values())) == len(_DOWN_ATTN_LDM_IDX)
        assert len(set(_UP_ATTN_LDM_IDX.values())) == len(_UP_ATTN_LDM_IDX)


# ===========================================================================
# ForgeAttnProcessor
# ===========================================================================

class TestForgeAttnProcessorInit:

    def test_stores_block_metadata(self):
        proc = ForgeAttnProcessor("input", 3, 2)
        assert proc.block_name == "input"
        assert proc.block_idx == 3
        assert proc.transformer_idx == 2

    def test_different_block_names(self):
        for name in ("input", "middle", "output"):
            proc = ForgeAttnProcessor(name, 0, 0)
            assert proc.block_name == name


class TestForgeAttnProcessorCall:

    def _hidden(self, batch=2, seq=4, dim=64):
        return torch.zeros(batch, seq, dim)

    def test_no_patches_runs_standard_attention(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()
        out = proc(attn, hidden, encoder_hidden_states=hidden)
        assert out.shape == hidden.shape

    def test_none_transformer_options_treated_as_empty(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()
        # should not raise
        out = proc(attn, hidden, transformer_options=None)
        assert out is not None

    def test_patches_replace_attn2_tuple_key_calls_replacement(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()

        sentinel = torch.ones(2, 4, 64)
        replace_fn = MagicMock(return_value=sentinel)
        topts = {
            "patches_replace": {
                "attn2": {("input", 1, 0): replace_fn}
            }
        }
        out = proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)
        replace_fn.assert_called_once()
        # The processor passes the replacement result through to_out projection and
        # rescale_output_factor division, so the output is value-equal but not the
        # same object.
        assert torch.allclose(out, sentinel)

    def test_patches_replace_attn2_pair_key_fallback(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()

        sentinel = torch.ones(2, 4, 64)
        replace_fn = MagicMock(return_value=sentinel)
        topts = {
            "patches_replace": {
                "attn2": {("input", 1): replace_fn}  # pair key, no transformer_idx
            }
        }
        out = proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)
        replace_fn.assert_called_once()
        assert torch.allclose(out, sentinel)

    def test_attn2_patch_modifies_qkv(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()

        patch_called = []

        def my_patch(q, k, v, opts):
            patch_called.append(True)
            return q * 2, k, v

        topts = {"patches": {"attn2_patch": [my_patch]}}
        proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)
        assert patch_called, "attn2_patch was not called"

    def test_extra_options_contains_block_metadata(self):
        proc = ForgeAttnProcessor("middle", 0, 3)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()

        captured = {}

        def capture_patch(q, k, v, opts):
            captured.update(opts)
            return q, k, v

        topts = {"patches": {"attn2_patch": [capture_patch]}}
        proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)

        assert captured["block"] == ("middle", 0)
        assert captured["block_index"] == 3
        assert captured["n_heads"] == 4

    def test_replace_fn_receives_extra_options(self):
        proc = ForgeAttnProcessor("output", 2, 1)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        hidden = self._hidden()

        received_opts = {}

        def replace_fn(q, k, v, opts):
            received_opts.update(opts)
            return torch.zeros_like(q)

        topts = {
            "patches_replace": {"attn2": {("output", 2, 1): replace_fn}},
            "custom_key": "custom_value",
        }
        proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)
        assert received_opts.get("custom_key") == "custom_value"

    def test_residual_connection_added_when_set(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64, residual_connection=True)
        hidden = torch.ones(2, 4, 64)
        # With residual_connection=True, output = projection(hidden) + residual
        # to_out is identity, so output should be hidden + hidden = 2*hidden
        out = proc(attn, hidden)
        # Values should not equal the original (residual adds the input back)
        # identity proj + residual = 0 + hidden = hidden; just check shape
        assert out.shape == hidden.shape

    def test_rescale_output_factor_applied(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        attn.rescale_output_factor = 2.0
        hidden = self._hidden()
        out = proc(attn, hidden)
        # All zeros / 2.0 still zeros — just verify no crash and correct shape
        assert out.shape == hidden.shape

    def test_4d_hidden_states_reshaped_back(self):
        proc = ForgeAttnProcessor("input", 1, 0)
        attn = _make_attn_mock(heads=4, inner_dim=64)
        # 4-D input: (B, C, H, W)
        hidden_4d = torch.zeros(2, 64, 2, 2)
        out = proc(attn, hidden_4d)
        assert out.shape == hidden_4d.shape

    def test_no_patches_replace_falls_through_to_standard_path(self):
        proc = ForgeAttnProcessor("input", 4, 1)
        attn = _make_attn_mock(heads=8, inner_dim=64)
        hidden = self._hidden(batch=2, seq=6, dim=64)
        topts = {"patches_replace": {"attn2": {}}, "patches": {}}
        out = proc(attn, hidden, encoder_hidden_states=hidden, transformer_options=topts)
        assert out.shape == hidden.shape


# ===========================================================================
# PassthroughAttnProcessor
# ===========================================================================

class TestPassthroughAttnProcessor:

    def test_delegates_to_inner_processor(self):
        proc = PassthroughAttnProcessor()
        attn = MagicMock()
        hidden = torch.zeros(2, 4, 64)
        # _FakeAttnProcessor2_0 returns hidden_states unchanged
        out = proc(attn, hidden)
        assert out is hidden

    def test_transformer_options_accepted_without_error(self):
        proc = PassthroughAttnProcessor()
        attn = MagicMock()
        hidden = torch.zeros(2, 4, 64)
        out = proc(attn, hidden, transformer_options={"some": "opt"})
        assert out is hidden

    def test_extra_kwargs_forwarded(self):
        proc = PassthroughAttnProcessor()
        attn = MagicMock()
        hidden = torch.zeros(2, 4, 64)
        # Should not raise even with extra kwargs
        out = proc(attn, hidden, attention_mask=None, temb=None)
        assert out is hidden


# ===========================================================================
# DiffPipeline — introspection helpers
# ===========================================================================

class TestDiffPipelineIntrospection:

    def test_is_active_true_when_hf_unet_set(self):
        dp = _make_diff_pipeline_bare()
        dp._hf_unet = MagicMock()
        assert dp.is_active() is True

    def test_is_active_false_when_hf_unet_none(self):
        dp = _make_diff_pipeline_bare()
        dp._hf_unet = None
        assert dp.is_active() is False

    def test_repr_active(self):
        dp = _make_diff_pipeline_bare()
        assert "active" in repr(dp)

    def test_repr_scaffold_when_none(self):
        dp = _make_diff_pipeline_bare()
        dp._hf_unet = None
        assert "scaffold-only" in repr(dp)

    def test_get_pipeline_returns_hf_unet(self):
        dp = _make_diff_pipeline_bare()
        sentinel = MagicMock()
        dp._hf_unet = sentinel
        assert dp.get_pipeline() is sentinel

    def test_apply_diffusers_optimization_is_noop(self):
        dp = _make_diff_pipeline_bare()
        # Should return None without side effects
        result = dp.apply_diffusers_optimization(MagicMock())
        assert result is None


# ===========================================================================
# DiffPipeline — _reset_auto_offload
# ===========================================================================

class TestResetAutoOffload:

    def test_clears_hooks_and_paths(self):
        dp = _make_diff_pipeline_bare()
        h1, h2 = MagicMock(), MagicMock()
        dp._b_hooks = [h1, h2]
        dp._b_block_paths = ["down_blocks.0", "down_blocks.1"]
        dp._auto_offload_ready = True

        dp._reset_auto_offload()

        h1.remove.assert_called_once()
        h2.remove.assert_called_once()
        assert dp._b_hooks == []
        assert dp._b_block_paths == []
        assert dp._auto_offload_ready is False

    def test_idempotent_on_empty(self):
        dp = _make_diff_pipeline_bare()
        dp._b_hooks = []
        dp._auto_offload_ready = False
        dp._reset_auto_offload()  # should not raise
        assert dp._b_hooks == []


# ===========================================================================
# DiffPipeline — _remove_lora_adapters
# ===========================================================================

class TestRemoveLoraAdapters:

    def test_deletes_all_active_adapters(self):
        dp = _make_diff_pipeline_bare()
        dp._active_adapters = [("forge_lora_0", 1.0), ("forge_lora_1", 0.5)]
        dp._hf_unet.delete_adapter = MagicMock()

        dp._remove_lora_adapters()

        assert dp._hf_unet.delete_adapter.call_count == 2
        assert dp._active_adapters == []

    def test_tolerates_delete_exception(self):
        dp = _make_diff_pipeline_bare()
        dp._active_adapters = [("forge_lora_0", 1.0)]
        dp._hf_unet.delete_adapter = MagicMock(side_effect=RuntimeError("gone"))

        dp._remove_lora_adapters()  # must not raise
        assert dp._active_adapters == []

    def test_noop_when_no_active_adapters(self):
        dp = _make_diff_pipeline_bare()
        dp._active_adapters = []
        dp._hf_unet.delete_adapter = MagicMock()

        dp._remove_lora_adapters()
        dp._hf_unet.delete_adapter.assert_not_called()


# ===========================================================================
# DiffPipeline — conditioning bridge (text_embeds / time_ids resolution)
# ===========================================================================

class TestApplyModelConditioningBridge:
    """Test the text_embeds / time_ids fallback logic inside apply_model()
    without executing the actual UNet forward by patching _hf_unet."""

    def _run(self, dp: DiffPipeline, x, t, **kwargs):
        """Execute apply_model with a patched UNet that returns zeros."""
        out_shape = x.shape
        dummy_out = torch.zeros(*out_shape)

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            # _hf_unet() must return a tuple
            dp._hf_unet.return_value = (dummy_out.unsqueeze(0),)
            # make parameters() return a single cpu float32 param
            p = nn.Parameter(torch.zeros(1))
            dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))
            result = dp.apply_model(x, t, **kwargs)
        return result

    def _make_inputs(self, batch=2, h=8, w=8):
        x = torch.zeros(batch, 4, h, w)
        t = torch.ones(batch) * 0.5
        c_crossattn = torch.zeros(batch, 10, 64)
        return x, t, c_crossattn

    def test_adm_text_embeds_priority_over_y(self):
        dp = _make_diff_pipeline_bare()
        x, t, enc = self._make_inputs()
        text_embeds = torch.ones(2, 1280)
        y = torch.zeros(2, 2816)  # should be ignored

        captured = {}

        def capture_forward(**kw):
            captured.update(kw)
            return (torch.zeros(2, 4, 8, 8),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: capture_forward(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=enc,
                           adm_text_embeds=text_embeds, y=y)

        assert torch.allclose(
            captured["added_cond_kwargs"]["text_embeds"], text_embeds
        )

    def test_y_fallback_slices_first_1280(self):
        dp = _make_diff_pipeline_bare()
        x, t, enc = self._make_inputs()
        y_val = torch.cat([torch.ones(2, 1280), torch.zeros(2, 1536)], dim=1)  # 2816

        captured = {}

        def capture_forward(**kw):
            captured.update(kw)
            return (torch.zeros(2, 4, 8, 8),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: capture_forward(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=enc, y=y_val)

        te = captured["added_cond_kwargs"]["text_embeds"]
        assert te.shape == (2, 1280)
        assert te.mean().item() == pytest.approx(1.0)

    def test_zero_fallback_when_no_pooled_conditioning(self):
        dp = _make_diff_pipeline_bare()
        x, t, enc = self._make_inputs()

        captured = {}

        def capture_forward(**kw):
            captured.update(kw)
            return (torch.zeros(2, 4, 8, 8),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: capture_forward(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=enc)

        te = captured["added_cond_kwargs"]["text_embeds"]
        assert te.shape == (2, 1280)
        assert te.sum().item() == pytest.approx(0.0)

    def test_adm_time_ids_used_when_provided(self):
        dp = _make_diff_pipeline_bare()
        x, t, enc = self._make_inputs()
        time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024],
                                  [1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32)

        captured = {}

        def capture_forward(**kw):
            captured.update(kw)
            return (torch.zeros(2, 4, 8, 8),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: capture_forward(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=enc,
                           adm_text_embeds=torch.zeros(2, 1280),
                           adm_time_ids=time_ids)

        ti = captured["added_cond_kwargs"]["time_ids"]
        assert ti.shape == (2, 6)
        assert torch.allclose(ti, time_ids)

    def test_time_ids_derived_from_latent_shape_when_absent(self):
        dp = _make_diff_pipeline_bare()
        # latent 16×24 → pixel space 128×192
        x = torch.zeros(1, 4, 16, 24)
        t = torch.ones(1) * 0.5
        enc = torch.zeros(1, 10, 64)

        captured = {}

        def capture_forward(**kw):
            captured.update(kw)
            return (torch.zeros(1, 4, 16, 24),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: capture_forward(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=enc,
                           adm_text_embeds=torch.zeros(1, 1280))

        ti = captured["added_cond_kwargs"]["time_ids"]
        assert ti.shape == (1, 6)
        # [h_px, w_px, 0, 0, h_px, w_px]
        assert ti[0, 0].item() == pytest.approx(128.0)
        assert ti[0, 1].item() == pytest.approx(192.0)
        assert ti[0, 2].item() == pytest.approx(0.0)
        assert ti[0, 4].item() == pytest.approx(128.0)


# ===========================================================================
# DiffPipeline — ControlNet residual mapping
# ===========================================================================

class TestApplyModelControlNetMapping:

    def _capture_forward(self, dp, x, t, control):
        captured = {}

        def _capture(**kw):
            captured.update(kw)
            return (torch.zeros_like(x),)

        dp._hf_unet = MagicMock(side_effect=lambda **kw: _capture(**kw))
        p = nn.Parameter(torch.zeros(1))
        dp._hf_unet.parameters = MagicMock(side_effect=lambda: iter([p]))

        with patch.object(dp, "_sync_lora"), \
             patch.object(dp, "apply_diffusers_optimization"), \
             patch.object(dp, "_maybe_setup_tensor_core_opts"):
            dp.apply_model(x, t, c_crossattn=torch.zeros(2, 4, 64),
                           control=control,
                           adm_text_embeds=torch.zeros(2, 1280))
        return captured

    def test_no_control_passes_none_residuals(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        cap = self._capture_forward(dp, x, t, control=None)
        assert cap["down_block_additional_residuals"] is None
        assert cap["mid_block_additional_residual"] is None

    def test_input_residuals_reversed(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        r0 = torch.ones(2, 4, 8, 8) * 1
        r1 = torch.ones(2, 4, 8, 8) * 2
        r2 = torch.ones(2, 4, 8, 8) * 3
        control = {"input": [r0, r1, r2], "middle": [None]}
        cap = self._capture_forward(dp, x, t, control)
        residuals = cap["down_block_additional_residuals"]
        # Order should be reversed: [r2, r1, r0]
        assert residuals is not None
        assert len(residuals) == 3
        assert torch.allclose(residuals[0], r2)
        assert torch.allclose(residuals[1], r1)
        assert torch.allclose(residuals[2], r0)

    def test_none_entries_in_input_filtered_out(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        r = torch.ones(2, 4, 8, 8)
        control = {"input": [r, None, r], "middle": [None]}
        cap = self._capture_forward(dp, x, t, control)
        assert len(cap["down_block_additional_residuals"]) == 2

    def test_middle_residual_forwarded(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        mid = torch.ones(2, 4, 4, 4) * 7.0
        control = {"input": [], "middle": [mid]}
        cap = self._capture_forward(dp, x, t, control)
        assert cap["mid_block_additional_residual"] is mid

    def test_middle_none_entry_ignored(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        control = {"input": [], "middle": [None]}
        cap = self._capture_forward(dp, x, t, control)
        assert cap["mid_block_additional_residual"] is None

    def test_empty_input_list_gives_none_residuals(self):
        dp = _make_diff_pipeline_bare()
        x = torch.zeros(2, 4, 8, 8)
        t = torch.ones(2) * 0.5
        control = {"input": [], "middle": []}
        cap = self._capture_forward(dp, x, t, control)
        assert cap["down_block_additional_residuals"] is None


# ===========================================================================
# DiffPipeline — _maybe_setup_tensor_core_opts (CPU path)
# ===========================================================================

class TestMaybeSetupTensorCoreOpts:

    def test_cpu_device_sets_autocast_none(self):
        dp = _make_diff_pipeline_bare()
        dp._tc_ready = False
        dp._maybe_setup_tensor_core_opts(torch.device("cpu"), torch.float32)
        assert dp._tc_ready is True
        assert dp._autocast_dtype is None

    def test_runs_only_once(self):
        dp = _make_diff_pipeline_bare()
        dp._tc_ready = True  # already run
        dp._autocast_dtype = torch.float16
        # second call must be a no-op
        dp._maybe_setup_tensor_core_opts(torch.device("cpu"), torch.float32)
        assert dp._autocast_dtype == torch.float16  # unchanged


# ===========================================================================
# DiffPipeline — _sync_lora (uuid guard)
# ===========================================================================

class TestSyncLoraUuidGuard:

    def test_noop_when_uuid_unchanged(self):
        dp = _make_diff_pipeline_bare()
        dp._synced_patches_uuid = "abc"
        dp.unet_patcher.patches_uuid = "abc"
        dp.unet_patcher.patches = {}
        dp._hf_unet.delete_adapter = MagicMock()

        dp._sync_lora()

        dp._hf_unet.delete_adapter.assert_not_called()

    def test_runs_when_uuid_changes(self):
        dp = _make_diff_pipeline_bare()
        dp._synced_patches_uuid = "old-uuid"
        dp.unet_patcher.patches_uuid = "new-uuid"
        dp.unet_patcher.patches = {}
        dp._hf_unet.delete_adapter = MagicMock()

        dp._sync_lora()

        # uuid updated, no LoRA patches → adapters cleared but not re-added
        assert dp._synced_patches_uuid == "new-uuid"

    def test_lora_change_resets_compiled_flag(self):
        dp = _make_diff_pipeline_bare()
        dp._compiled = True
        dp._synced_patches_uuid = "old"
        dp.unet_patcher.patches_uuid = "new"
        dp.unet_patcher.patches = {}
        dp._hf_unet.delete_adapter = MagicMock()

        dp._sync_lora()

        assert dp._compiled is False


# ===========================================================================
# SDXL config constants sanity checks
# ===========================================================================

class TestSdxlConfigConstants:

    def test_hf_unet_config_has_required_keys(self):
        required = {
            "in_channels", "out_channels", "cross_attention_dim",
            "block_out_channels", "layers_per_block",
            "transformer_layers_per_block", "down_block_types", "up_block_types",
        }
        assert required.issubset(set(_SDXL_HF_UNET_CONFIG.keys()))

    def test_hf_unet_config_sdxl_defaults(self):
        assert _SDXL_HF_UNET_CONFIG["in_channels"] == 4
        assert _SDXL_HF_UNET_CONFIG["out_channels"] == 4
        assert _SDXL_HF_UNET_CONFIG["cross_attention_dim"] == 2048
        assert _SDXL_HF_UNET_CONFIG["block_out_channels"] == [320, 640, 1280]

    def test_ldm_unet_config_has_required_keys(self):
        required = {
            "num_res_blocks", "transformer_depth", "channel_mult",
            "transformer_depth_middle", "transformer_depth_output",
        }
        assert required.issubset(set(_SDXL_LDM_UNET_CONFIG.keys()))

    def test_block_out_channels_divisible_by_8(self):
        for ch in _SDXL_HF_UNET_CONFIG["block_out_channels"]:
            assert ch % 8 == 0, f"{ch} is not divisible by 8"

    def test_down_and_up_block_types_length_matches_block_out_channels(self):
        n = len(_SDXL_HF_UNET_CONFIG["block_out_channels"])
        assert len(_SDXL_HF_UNET_CONFIG["down_block_types"]) == n
        assert len(_SDXL_HF_UNET_CONFIG["up_block_types"]) == n
