import os

import gradio as gr

from modules import scripts
from modules.prompt_parser import get_token_subspaces
from ldm_patched.k_diffusion.sure_token_guidance import (
    TokenGuidanceConfig,
    patch_model_with_token_guidance,
)

# scripts.basedir() returns the EXTENSION's own root directory (e.g.
# "extensions-builtin/sd_forge_sure_token_ag"), not the "scripts/" subfolder
# this file lives in — only two levels up reaches the repo root.
_RULE_DIR = os.path.join(scripts.basedir(), "..", "..", "configs", "prompt_rules")
_RULE_DIR = os.path.normpath(_RULE_DIR)
_PRESET_CHOICES = ["danbooru", "pony", "illustrious"]


class SureTokenSubspaceGuidanceForForge(scripts.Script):
    sorting_priority = 12.7

    def title(self):
        return "SURE Token Subspace Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown(
                "Token-level conditional-space guidance: fixes vanishing tags, "
                "attribute leakage between entities (e.g. '1girl, brown hair, 1boy, "
                "blue hair'), and bias toward common tags. Zero extra NFE — corrects "
                "the attn2 cross-attention matrix already computed each step. "
                "Debug prints go to the console; look for `[TokenSubspaceGuidance]`, "
                "`[RuleEngine]`, `[SemanticAnalyzer]`, and `[RefineEngine]`."
            )
            enabled = gr.Checkbox(label="Enabled", value=False)
            with gr.Row():
                use_intention_tree = gr.Checkbox(
                    label="Use intention tree (Plan 04)", value=True,
                    info="Rule engine + Matrix-Tree semantic analysis + refine engine, "
                         "instead of the flat 8-keyword heuristic. Falls back "
                         "automatically if the pipeline errors for this generation.",
                )
                tag_presets = gr.CheckboxGroup(
                    label="Tag rule presets", choices=_PRESET_CHOICES, value=["danbooru"],
                    info="danbooru is the base ruleset; pony/illustrious are overlays — "
                         "load alongside danbooru, not instead of it.",
                )
            with gr.Row():
                tau_vanish = gr.Slider(
                    label="Vanish Threshold", minimum=0.0, maximum=2.0, step=0.05, value=0.4,
                    info="Multiplier of a tag's 'fair share' of attention (width/context_len). "
                         "A tag is 'vanishing' if its peak attention anywhere is below this "
                         "multiple of its fair share.",
                )
                beta_vanish = gr.Slider(
                    label="Vanish Boost", minimum=0.0, maximum=0.5, step=0.01, value=0.15,
                )
            with gr.Row():
                leak_strength = gr.Slider(
                    label="Leak Attenuation", minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    info="Fraction of a rival tag's attention removed outside its own region.",
                )
                bias_strength = gr.Slider(
                    label="Bias Boost", minimum=0.0, maximum=0.3, step=0.01, value=0.05,
                    info="Boost for rare tags relative to the example frequency table.",
                )
            with gr.Row():
                leak_min_confidence = gr.Slider(
                    label="Leak Confidence Floor", minimum=0.0, maximum=2.0, step=0.05, value=0.3,
                    info="How strongly an entity tag must dominate a region before leak "
                         "correction applies there. Too low = corrects background/ambiguous "
                         "pixels too (dilutes the effect); too high = only corrects the most "
                         "obvious regions.",
                )
            with gr.Row():
                attn_blocks = gr.Radio(
                    label="Attention Blocks",
                    choices=["all", "middle", "mid+out", "input", "output"],
                    value="middle",
                    info="Which UNet blocks to hook. 'middle' is cheapest and least console spam.",
                )
                debug = gr.Checkbox(label="Debug prints", value=True)

        return (enabled, use_intention_tree, tag_presets, tau_vanish, beta_vanish,
                leak_strength, bias_strength, leak_min_confidence, attn_blocks, debug)

    def _run_intention_tree_pipeline(self, p, tag_presets):
        """Returns (final_tokens, groups_info) or (None, groups_info) on any
        failure — callers must fall back to the flat heuristic rather than
        crash the whole generation. This pipeline (Plan 04) is newer and has
        only been verified against synthetic data this session, not a live
        model — see plans/04-matrix-tree-prompt-structure.md."""
        from modules.rule_engine.schema import load_rule_yaml
        from modules.rule_engine.pipeline import run_pipeline

        preset_paths = [os.path.join(_RULE_DIR, f"{name}.yaml") for name in tag_presets]
        preset_paths = [pth for pth in preset_paths if os.path.isfile(pth)]
        if not preset_paths:
            print(f"[TokenSubspaceGuidance] no valid tag-rule preset files found in "
                  f"{_RULE_DIR!r} for {tag_presets!r}; falling back to the keyword heuristic.")
            return None, None

        print(f"[TokenSubspaceGuidance] running Plan 04 intention-tree pipeline with presets "
              f"{[os.path.basename(pth) for pth in preset_paths]}...")
        try:
            ruleset = load_rule_yaml(*preset_paths)
            final_tokens, groups_info = run_pipeline(p.sd_model, p, ruleset)
            print(f"[TokenSubspaceGuidance] intention-tree pipeline SUCCEEDED: "
                  f"{len(final_tokens) if final_tokens else 0} segment(s) reconciled.")
            return final_tokens, groups_info
        except Exception as e:  # noqa: BLE001 — this pipeline is new/unverified; never let it crash generation
            import traceback
            print(f"[TokenSubspaceGuidance] intention-tree pipeline FAILED ({type(e).__name__}: {e}); "
                  f"falling back to the Plan 03 keyword heuristic for this generation. "
                  f"Full traceback (report this if it keeps happening):")
            traceback.print_exc()
            return None, None

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (enabled, use_intention_tree, tag_presets, tau_vanish, beta_vanish,
         leak_strength, bias_strength, leak_min_confidence, attn_blocks, debug) = script_args

        if not enabled:
            return

        final_tokens = None
        groups_info = None
        if use_intention_tree:
            final_tokens, groups_info = self._run_intention_tree_pipeline(p, tag_presets)

        if groups_info is None:
            # Fallback path: still use the FINAL (style-expanded) prompt text,
            # never the raw p.prompt, even when the intention-tree pipeline is
            # disabled or failed.
            from modules.rule_engine.embedding_extraction import get_final_prompt_text
            final_prompt = get_final_prompt_text(p)
            groups_info = get_token_subspaces(p.sd_model, final_prompt)

        if not groups_info["groups"]:
            print("[TokenSubspaceGuidance] no groups found for prompt; guidance will no-op this generation.")

        cfg = TokenGuidanceConfig(
            tau_vanish=float(tau_vanish),
            beta_vanish=float(beta_vanish),
            leak_strength=float(leak_strength),
            bias_strength=float(bias_strength),
            leak_min_confidence=float(leak_min_confidence),
            debug=bool(debug),
        )

        unet = p.sd_model.forge_objects.unet
        unet, _diag_store = patch_model_with_token_guidance(
            unet, groups_info, cfg, attn_blocks, final_tokens=final_tokens,
        )
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            sure_token_ag_enabled=True,
            sure_token_ag_use_intention_tree=bool(final_tokens is not None),
            sure_token_ag_tag_presets=",".join(tag_presets),
            sure_token_ag_tau_vanish=tau_vanish,
            sure_token_ag_beta_vanish=beta_vanish,
            sure_token_ag_leak_strength=leak_strength,
            sure_token_ag_bias_strength=bias_strength,
            sure_token_ag_leak_min_confidence=leak_min_confidence,
            sure_token_ag_attn_blocks=attn_blocks,
        ))
