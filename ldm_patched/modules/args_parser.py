# Original code from Comfy, https://github.com/comfyanonymous/ComfyUI
import argparse
import enum
import ldm_patched.modules.options

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

#parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0")
#parser.add_argument("--port", type=int, default=8188)
parser.add_argument("--disable-header-check", type=str, default=None, metavar="ORIGIN", nargs="?", const="*")
parser.add_argument("--web-upload-size", type=float, default=100)

parser.add_argument("--external-working-path", type=str, default=None, metavar="PATH", nargs='+', action='append')
parser.add_argument("--output-path", type=str, default=None)
parser.add_argument("--temp-path", type=str, default=None)
parser.add_argument("--cache-path", type=str, default=None)
parser.add_argument("--in-browser", action="store_true")
parser.add_argument("--disable-in-browser", action="store_true")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

parser.add_argument("--disable-attention-upcast", action="store_true")

cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--all-in-fp32", action="store_true")
fp_group.add_argument("--all-in-fp16", action="store_true")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--unet-in-bf16", action="store_true")
fpunet_group.add_argument("--unet-in-fp16", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--vae-in-fp16", action="store_true")
fpvae_group.add_argument("--vae-in-fp32", action="store_true")
fpvae_group.add_argument("--vae-in-bf16", action="store_true")

parser.add_argument("--vae-in-cpu", action="store_true")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
fpte_group.add_argument("--clip-in-fp16", action="store_true")
fpte_group.add_argument("--clip-in-fp32", action="store_true")


parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)

parser.add_argument("--disable-ipex-hijack", action="store_true")

parser.add_argument("--torch-compile", action='store_true', help="Enable torch.compile for potential speedups")
parser.add_argument("--torch-compile-backend", type=str, default="inductor", choices=["inductor", "cudagraphs"], help="Backend for torch.compile")
parser.add_argument("--torch-compile-mode", type=str, default="default", 
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], 
                    help="Compilation mode for torch.compile. Only applies to inductor backend")
parser.add_argument("--torch-compile-epilogue-fusion", action='store_true', help="Enable epilogue fusion (requires max-autotune)")
parser.add_argument("--torch-compile-max-autotune", action='store_true', help="Enable max autotune")
parser.add_argument("--torch-compile-fallback-random", action='store_true', help="Enable fallback random")
parser.add_argument("--torch-compile-shape-padding", action='store_true', help="Enable shape padding")
parser.add_argument("--torch-compile-cudagraphs", action='store_true', help="Enable CUDA graphs")
parser.add_argument("--torch-compile-trace", action='store_true', help="Enable tracing")
parser.add_argument("--torch-compile-graph-diagram", action='store_true', help="Enable graph diagram")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "fast"
    TAESD = "taesd"

parser.add_argument("--preview-option", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, action=EnumAction)

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true")
attn_group.add_argument("--attention-quad", action="store_true")
attn_group.add_argument("--attention-pytorch", action="store_true")
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")

parser.add_argument("--disable-xformers", action="store_true")

parser.add_argument("--allow-fp16-accumulation", action="store_true", help="Enable FP16 accumulation in cuBLAS operations")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true")
vram_group.add_argument("--always-high-vram", action="store_true")
vram_group.add_argument("--always-normal-vram", action="store_true")
vram_group.add_argument("--always-low-vram", action="store_true")
vram_group.add_argument("--always-no-vram", action="store_true")
vram_group.add_argument("--always-cpu", action="store_true")


parser.add_argument("--always-offload-from-vram", action="store_true")
parser.add_argument("--pytorch-deterministic", action="store_true")

parser.add_argument("--disable-server-log", action="store_true")
parser.add_argument("--debug-mode", action="store_true")
parser.add_argument("--is-windows-embedded-python", action="store_true")

parser.add_argument("--nightly-builds", action="store_true", help="Use nightly PyTorch builds for compatible GPUs")

parser.add_argument("--disable-server-info", action="store_true")

parser.add_argument("--multi-user", action="store_true")

parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")

if ldm_patched.modules.options.args_parsing:
    args = parser.parse_args([])
else:
    args = parser.parse_args([])

if args.is_windows_embedded_python:
    args.in_browser = True

if args.disable_in_browser:
    args.in_browser = False
