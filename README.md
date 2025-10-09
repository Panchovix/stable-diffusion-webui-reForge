# Suggested repos instead of reForge

* Forge Classic: https://github.com/Haoming02/sd-webui-forge-classic, from @Haoming02 with a lot of optimizations and features, from reforge, forge, etc based on old backend of forge.
* Forge Neo: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo, from @Haoming02.
It is a continuation of Forge2 (so Flux, fp8, gguf, etc) but with more features (wan 2.2, Qwen Image, Nunchaku, etc), aimed on optimizations and new features.
* ersatzForge: https://github.com/DenOfEquity/ersatzForge, from DenOfEquity, based on Forge2, but as he says, with (experimental, opinionated) changes to Forge2 webUI.

# Stable Diffusion WebUI Forge/reForge

Stable Diffusion WebUI Forge/reForge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (based on [Gradio](https://www.gradio.app/)) to make development easier, optimize resource management, speed up inference, and study experimental features.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI's Forge.

# Forge2/reForge2

* newmain_newforge: Based on latest forge2 (gradio4, flux, etc) with some small changes that I plan to add very slowly. For now it has python 3.12 support, sage/flash attention support, all the samplers and schedulers from reForge (1), and recently, support for CFG++ samplers.

# Suggestion: For stability based on old forge, use forge classic

reForge(1) is not really stable for all tasks sadly.

So if you want to keep using old forge backend as it is, for sd1.x,2.x and SDXL, I suggest to use forge classic by @Haoming02 instead https://github.com/Haoming02/sd-webui-forge-classic, as at the moment that is the real succesor to old forge.

# Other branches:
* main: Main branch with multiple changes and updates. But not stable as main-old branch.
* dev: Similar to main but with more unstable changes. I.e. using comfy/ldm_patched backend for sd1.x and sdxl instead of A1111.
* dev2: More unstable than dev, for now same as dev.
* experimental: same as dev2 but with gradio 4.
* main-old: Branch with old forge backend. Possibly the most stable and older one (2025-03)

# Installing Forge/reForge

### (Suggested) Clean install.

For this, you will need Python (Python 3.7 up to 3.12 works fine, 3.13 still has some issues)
If you know what you are doing, you can install Forge/reForge using same method as SD-WebUI. (Install Git, Python, Git Clone the reForge repo `https://github.com/Panchovix/stable-diffusion-webui-reForge.git` and then run webui-user.bat):

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
```
Then run webui-user.bat (Windows) or webui-user.sh (Linux, for this one make sure to uncomment the lines according of your folder, paths and setting you need).

When you want to update:
```bash
cd stable-diffusion-webui-reForge
git pull
```

### If using Windows 7 and/or CUDA 11.x

For this, way to install is a bit different, since it uses another req file. We will rename the original req file to a backup, and then copy the legacy one renmaed as the original, to keep updates working.
For Windows CMD, it would be:

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
ren requirements_versions.txt requirements_versions_backup.txt
copy requirements_versions_legacy.txt requirements_versions.txt
```

Windows PS1

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
Rename-Item requirements_versions.txt requirements_versions_backup.txt
Copy-Item requirements_versions_legacy.txt requirements_versions.txt
```

Then run webui-user.bat (Windows).

### You have A1111 and you know Git
Tutorial from: https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git
If you have already had OG A1111 and you are familiar with git, An option is go to `/path/to/stable-diffusion-webui` and
```bash
git remote add reForge https://github.com/Panchovix/stable-diffusion-webui-reForge
git branch Panchovix/main
git checkout Panchovix/main
git fetch reForge
git branch -u reForge/main
git stash
git pull
```
To go back to OG A1111, just do `git checkout master` or `git checkout main`.

If you got stuck in a merge to resolve conflicts, you can go back with `git merge --abort`

-------

Pre-done package is planned, but I'm not sure how to do it. Any PR or help with this is appreciated.

# Forge/reForge Backend

Forge/reForge backend removes all WebUI's codes related to resource management and reworked everything. All previous CMD flags like `medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet`, ... are all **REMOVED**. Adding these flags will not cause error but they will not do anything now.

Without any cmd flag, Forge/reForge can run SDXL with 4GB vram and SD1.5 with 2GB vram.

**Some flags that you may still pay attention to:** 

1. `--always-offload-from-vram` (This flag will make things **slower** but less risky). This option will let Forge/reForge always unload models from VRAM. This can be useful if you use multiple software together and want Forge/reForge to use less VRAM and give some VRAM to other software, or when you are using some old extensions that will compete vram with Forge/reForge, or (very rarely) when you get OOM.

2. `--cuda-malloc` (This flag will make things **faster** but more risky). This will ask pytorch to use *cudaMallocAsync* for tensor malloc. On some profilers I can observe performance gain at millisecond level, but the real speed up on most my devices are often unnoticed (about or less than 0.1 second per image). This cannot be set as default because many users reported issues that the async malloc will crash the program. Users need to enable this cmd flag at their own risk.

3. `--cuda-stream` (This flag will make things **faster** but more risky). This will use pytorch CUDA streams (a special type of thread on GPU) to move models and compute tensors simultaneously. This can almost eliminate all model moving time, and speed up SDXL on 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc) by about 15\% to 25\%. However, this unfortunately cannot be set as default because I observe higher possibility of pure black images (Nan outputs) on 2060, and higher chance of OOM on 1080 and 2060. When the resolution is large, there is a chance that the computation time of one single attention layer is longer than the time for moving entire model to GPU. When that happens, the next attention layer will OOM since the GPU is filled with the entire model, and no remaining space is available for computing another attention layer. Most overhead detecting methods are not robust enough to be reliable on old devices (in my tests). Users need to enable this cmd flag at their own risk.

4. `--pin-shared-memory` (This flag will make things **faster** but more risky). Effective only when used together with `--cuda-stream`. This will offload modules to Shared GPU Memory instead of system RAM when offloading models. On some 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc), I can observe significant (at least 20\%) speed-up for SDXL. However, this unfortunately cannot be set as default because the OOM of Shared GPU Memory is a much more severe problem than common GPU memory OOM. Pytorch does not provide any robust method to unload or detect Shared GPU Memory. Once the Shared GPU Memory OOM, the entire program will crash (observed with SDXL on GTX 1060/1050/1066), and there is no dynamic method to prevent or recover from the crash. Users need to enable this cmd flag at their own risk.

Some extra flags that can help with performance or save VRAM, or more, depending of your needs. Most of them are found on ldm_patched/modules/args_parser.py and on the normal A1111 path (modules/cmd_args.py):

    --disable-xformers
        Disables xformers, to use other attentions like SDP.
    --use-sage-attention
        Uses SAGE attention implementation, from https://github.com/thu-ml/SageAttention. You need to install the library separately, as it needs triton.
    --attention-split
        Use the split cross attention optimization. Ignored when xformers is used.
    --attention-quad
        Use the sub-quadratic cross attention optimization . Ignored when xformers is used.
    --attention-pytorch
        Use the new pytorch 2.0 cross attention function.
    --disable-attention-upcast
        Disable all upcasting of attention. Should be unnecessary except for debugging.
    --force-channels-last
        Force channels last format when inferencing the models.
    --disable-cuda-malloc
        Disable cudaMallocAsync.
    --gpu-device-id
        Set the id of the cuda device this instance will use.
    --force-upcast-attention
        Force enable attention upcasting.

(VRAM related)

    --always-gpu
        Store and run everything (text encoders/CLIP models, etc... on the GPU).
    --always-high-vram
        By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.
    --always-normal-vram
        Used to force normal vram use if lowvram gets automatically enabled.
    --always-low-vram
        Split the unet in parts to use less vram.
    --always-no-vram
        When lowvram isn't enough.
    --always-cpu
        To use the CPU for everything (slow).

(float point type)

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

(rare platforms)

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

# Lora ctl (Control)

I've added this repo adapted for reforge.

This wouldn't be possible to do without the original ones!

Huge credits to cheald for Lora ctl (Control). Link for the reforge extension is: https://github.com/Panchovix/sd_webui_loractl_reforge_y.git

Many thanks to @1rre for his work for preliminary working version for lora control!

You can see how to use them on their respective repos

https://github.com/cheald/sd-webui-loractl

## Moved built-it extensions to separate repos

Since the UI got really cluttered with built it extensions, I have removed some of them and made them separate repos. You can install them by the extension installer on the UI or doing `git clone repo.git` replacing `repo.git` with the following links, in the extensions folder.

* RAUNet-MSW-MSA (HiDiffusion): https://github.com/Panchovix/reforge_jankhidiffusion.git
* Skimmed CFG: https://github.com/Panchovix/reForge-SkimmedCFG.git
* Forge Style Align: https://github.com/Panchovix/sd_forge_stylealign.git
* reForge Sigmas Merge: https://github.com/Panchovix/reForge-Sigmas_merge.git
* Differential Diffusion: https://github.com/Panchovix/reForge-DifferentialDiffusion.git
* Auomatic CFG: https://github.com/Panchovix/reForge-AutomaticCFG.git
* reForge_Advanced_CLIP_Text_Encode (not working yet): https://github.com/Panchovix/reForge_Advanced_CLIP_Text_Encode.git
* Hunyuan-DiT-for-webUI-main: https://github.com/Panchovix/Hunyuan-DiT-for-webUI-main.git
* PixArt-Sigma-for-webUI-main: https://github.com/Panchovix/PixArt-Sigma-for-webUI-main.git
* StableCascade-for-webUI-main: https://github.com/Panchovix/StableCascade-for-webUI-main.git
* StableDiffusion3-for-webUI-main: https://github.com/Panchovix/StableDiffusion3-for-webUI-main.git

# Last "Old" Forge commit (https://github.com/lllyasviel/stable-diffusion-webui-forge/commit/bfee03d8d9415a925616f40ede030fe7a51cbcfd) before forge2.

# Support

Some people have been asking how to donate or support the project, and I'm really grateful for that! I did this buymeacoffe link from some suggestions!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Panchovix)