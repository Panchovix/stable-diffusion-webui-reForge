#!/usr/bin/env python3
"""
tools/debug_hijack.py

Debug client for the diffusers path hijack.

What this does
--------------
1. Registers the SDXL path hijack (so it is active when the server loads a model).
2. Calls the webui API to set sd_model_checkpoint → triggers load_model() →
   triggers maybe_apply_path_hijack() → pdb breakpoint fires in the server terminal.
3. Prints the server's response so you can see whether the hijack message appeared.

Run the server first in a separate terminal with:
    ./venv/bin/python launch.py --api --forge-diffusers-pipeline --nowebui

Then in another terminal:
    ./venv/bin/python tools/debug_hijack.py

pdb will pause the SERVER process — interact with it in the server terminal.

Useful pdb commands once you hit the breakpoint in maybe_apply_path_hijack():
    (Pdb) p cmd_opts.forge_diffusers_pipeline   # should be True
    (Pdb) p _PATH_HIJACK_REGISTRY               # should be non-empty
    (Pdb) p checkpoint_info.name                # model filename
    (Pdb) p checkpoint_info.metadata            # safetensors metadata dict
    (Pdb) n                                     # step to next line
    (Pdb) c                                     # continue execution
"""

import sys
import time
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:7860"
CHECKPOINT = "NoobAI-XL-v1.1.safetensors"   # model file present in models/Stable-diffusion/


def api(method: str, path: str, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        raise


def wait_for_server(retries=30, delay=2):
    print("Waiting for server …", end="", flush=True)
    for _ in range(retries):
        try:
            api("GET", "/sdapi/v1/cmd-flags")
            print(" ready.")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(delay)
    print()
    sys.exit("Server did not start in time.")


def main():
    wait_for_server()

    # Confirm --forge-diffusers-pipeline is set
    flags = api("GET", "/sdapi/v1/cmd-flags")
    fp = flags.get("forge_diffusers_pipeline", False)
    print(f"forge_diffusers_pipeline flag on server: {fp}")
    if not fp:
        print(
            "\n[!] --forge-diffusers-pipeline is NOT set on the server.\n"
            "    Restart with: ./venv/bin/python launch.py --api --forge-diffusers-pipeline --nowebui\n"
        )
        sys.exit(1)

    # List available models
    models = api("GET", "/sdapi/v1/sd-models")
    names = [m["model_name"] for m in models]
    print(f"Available models: {names}")

    # Find our target
    target = next((m["title"] for m in models if CHECKPOINT in m.get("filename", "")), None)
    if target is None:
        # Try by model_name
        target = next((m["title"] for m in models if CHECKPOINT.split(".")[0] in m["model_name"]), None)
    if target is None:
        print(f"[!] Could not find '{CHECKPOINT}' in model list. Available: {names}")
        sys.exit(1)

    print(f"\nTriggering model load: {target!r}")
    print(">>> pdb breakpoint will fire in the SERVER terminal — switch to it now.")
    print("    Useful commands:")
    print("      (Pdb) p cmd_opts.forge_diffusers_pipeline")
    print("      (Pdb) p _PATH_HIJACK_REGISTRY")
    print("      (Pdb) p checkpoint_info.name")
    print("      (Pdb) p checkpoint_info.metadata")
    print("      (Pdb) n   # step")
    print("      (Pdb) c   # continue\n")

    # POST options → triggers reload_model_weights → load_model → hijack
    api("POST", "/sdapi/v1/options", {"sd_model_checkpoint": target})

    print("Model load complete (or hijack intercepted).")


if __name__ == "__main__":
    # Register the SDXL hijack on the CLIENT side — this is just documentation.
    # The actual registration must happen inside the SERVER process.
    # See the note below on how to do that via a startup script or extension.
    print(__doc__)
    print("=" * 60)
    print(
        "NOTE: register_path_hijack() must be called INSIDE the server process.\n"
        "      The easiest way is to add this to a file the server imports at startup,\n"
        "      e.g. create  extensions/diffusers_hijack_ext/scripts/activate.py :\n\n"
        "        from diff_pipeline.load_model import (\n"
        "            register_path_hijack, is_sdxl_checkpoint, dummy_sdxl_hijack\n"
        "        )\n"
        "        register_path_hijack(is_sdxl_checkpoint, dummy_sdxl_hijack)\n\n"
        "      Then restart the server before running this script.\n"
    )
    print("=" * 60)
    main()
