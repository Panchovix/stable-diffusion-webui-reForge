# Original code from Comfy, https://github.com/comfyanonymous/ComfyUI
import argparse
import enum
import os
from typing import Optional
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

#parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::", help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
#parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
#parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
#parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")

parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")

parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the output directory.")
parser.add_argument("--temp-directory", type=str, default=None)
parser.add_argument("--input-directory", type=str, default=None)
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID", help="Set the id of the cuda device this instance will use.")

parser.add_argument("--disable-attention-upcast", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")

parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")

parser.add_argument("--oneapi-device-selector", type=str, default=None, metavar="SELECTOR_STRING", help="Sets the oneAPI device(s) this instance will use.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--all-in-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--all-in-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--unet-in-bf16", action="store_true", help="Run the UNET in bf16. This should only be used for testing stuff.")
fpunet_group.add_argument("--unet-in-fp16", action="store_true", help="Store unet weights in fp16.")
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--vae-in-fp16", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--vae-in-fp32", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--vae-in-bf16", action="store_true", help="Run the VAE in bf16.")
parser.add_argument("--vae-in-cpu", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--clip-in-fp16", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--clip-in-fp32", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

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


parser.add_argument("--supports-fp8-compute", action="store_true", help="ComfyUI will act like if the device supports fp8 compute.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, help="Default preview method for sampler nodes.", action=EnumAction)

parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-classic", action="store_true", help="Use the old style (aggressive) caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--attention-quad", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--attention-pytorch", action="store_true", help="Use the new pytorch 2.0 cross attention function.")
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-sage-attention3", action="store_true", help="Use sage attention 3. Supported only on blackwell GPUs.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")
parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")

parser.add_argument("--allow-fp16-accumulation", action="store_true", help="Enable FP16 accumulation in cuBLAS operations")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--always-high-vram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--always-normal-vram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--always-low-vram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--always-no-vram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--always-cpu", action="store_true", help="To use the CPU for everything (slow).")

parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reverved depending on your OS.")


parser.add_argument("--async-offload", action="store_true", help="Use async weight offloading.")

parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

parser.add_argument("--always-offload-from-vram", action="store_true", help="Force reForge to agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--pytorch-deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"

parser.add_argument("--fast", nargs="*", type=PerformanceFeature, help="Enable some untested and potentially quality deteriorating optimizations. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: fp16_accumulation fp8_matrix_mult cublas_ops")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors.")

cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--disable-server-log", action="store_true", help="Don't print server output.")
parser.add_argument("--debug-mode", action="store_true", help="Enables more debug prints.")
parser.add_argument("--is-windows-embedded-python", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")
parser.add_argument("--disable-server-info", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")
parser.add_argument("--nightly-builds", action="store_true", help="Use nightly PyTorch builds for compatible GPUs")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")

parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")
parser.add_argument("--verbose", default='INFO', const='DEBUG', nargs="?", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")

def is_valid_directory(path: Optional[str]) -> Optional[str]:
    """Validate if the given path is a directory."""
    if path is None:
        return None

    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")
    return path

if ldm_patched.modules.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.is_windows_embedded_python:
    args.in_browser = True

import logging
logging_level = logging.INFO
if args.debug_mode:
    logging_level = logging.DEBUG
    logging.basicConfig(format="%(message)s", level=logging_level)
    