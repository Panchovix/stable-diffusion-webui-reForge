"""Extract LoRA trigger/activation words from `<lora:name:weight>` syntax in
a prompt, using THIS REPO's own on-disk metadata convention (confirmed this
session by reading the actual code — see
`lean_proofs_rfv/THEOREM_MATRIX_TREE_BUFFER.md` §0): each LoRA's sidecar
`<basename>.json` file stores trigger words as `user_metadata["activation
text"]`, a comma-separated STRING (NOT Civitai's `trainedWords` array) — see
`extensions-builtin/Lora/ui_edit_user_metadata.py` and
`modules/ui_extra_networks_user_metadata.py`.

This reads the sidecar file directly via `networks.available_networks[name]
.filename`, bypassing the gradio UI metadata-editor class (`UserMetadataEditor`)
entirely — that class is built around a live extra-networks page/gradio
state we don't have at prompt-classification time, and its own
`get_user_metadata` reads from an in-memory `page.items` cache rather than
disk directly.

CAVEAT (untested against a real LoRA in this session — no live WebUI/torch
available in this sandbox): the `import networks` line below assumes
`extensions-builtin/Lora`'s own `networks` module is importable at prompt-
classification time, which is only true once Forge's extension loader has
put that directory on `sys.path` (it does this at WebUI startup for all
extensions-builtin, so this should hold at generation time, but has not
been exercised against a real LoRA file this session — verify on first real
use, per this project's established "flag what's unverified" convention).
"""

from __future__ import annotations

import json
import os
import re

_RE_LORA_TAG = re.compile(r"<lora:([^:>]+)(?::[^>]*)?>")
_RE_COMMA = re.compile(r" *, *")


def extract_lora_names(prompt: str) -> list:
    """Every `<lora:name:weight>` reference in the prompt, in order, name only."""
    return _RE_LORA_TAG.findall(prompt)


def _read_activation_text(lora_name: str) -> str:
    try:
        import networks  # extensions-builtin/Lora's own module; see CAVEAT above
    except ImportError:
        print(f"[RuleEngine] LoRA <{lora_name}>: could not import the Lora extension's "
              f"`networks` module (not on sys.path yet?) — skipping trigger-word lookup.")
        return ""

    entry = networks.available_networks.get(lora_name)
    if entry is None or not getattr(entry, "filename", None):
        print(f"[RuleEngine] LoRA <{lora_name}>: not found in networks.available_networks "
              f"— skipping trigger-word lookup.")
        return ""

    metadata_path = os.path.splitext(entry.filename)[0] + ".json"
    if not os.path.isfile(metadata_path):
        return ""  # no saved metadata sidecar — not an error, just nothing to read

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[RuleEngine] LoRA <{lora_name}>: failed to read {metadata_path!r}: {e}")
        return ""

    return data.get("activation text", "") or ""


def extract_lora_trigger_words(prompt: str) -> dict:
    """Returns {lora_name: [trigger_word, ...]} for every `<lora:...>` tag in
    the prompt that has a saved "activation text" sidecar. LoRAs with no
    saved trigger words (or missing/unreadable metadata files) map to an
    empty list — not an error, just nothing to classify from that LoRA."""
    result = {}
    for name in extract_lora_names(prompt):
        activation_text = _read_activation_text(name)
        words = [w for w in _RE_COMMA.split(activation_text.strip()) if w] if activation_text else []
        result[name] = words
        print(f"[RuleEngine] LoRA <{name}>: activation text -> "
              f"{words if words else '(none saved)'}")
    return result
