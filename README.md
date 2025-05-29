a backup of my local (experimental, opinionated) changes to Forge2 webUI

* auto selection of VAE and text encoders per model / UI setting
* Chroma (based on https://github.com/croquelois/forgeChroma)
* extended Checkpoint Merger, including UI for nArn0's embedding convertor (based on https://github.com/nArn0/sdxl-embedding-converter)
* Hypernetworks
* various other minor tweaks: UI, embedding filtering, code consolidation, dead code removal, performance improvements for me
* tiling (sd1, 2, xl) (based on https://github.com/spinagon/ComfyUI-seamless-tiling)
* all embeddings everywhere all at once: SD1.5 embeddings (CLIP-L only) can be used with SDXL, SD3, maybe Flux (haven't tested); SDXL embeddings can be used with SD1 (applies CLIP-L only, CLIP-G ignored), SD3
* new preprocessors for IPAdapter, including tiling, noising (for uncond) and sharpening of inputs. And multi-input.
* Latent NeuralNet upscaler by city96 (based on https://github.com/city96/SD-Latent-Upscaler)
* ResAdapter support: download models to `models/other_modules`, load via 'Additional modules' selector (as VAE, text encoder), LoRA as usual (https://huggingface.co/jiaxiangc/res-adapter)
* long CLIP
* distilled T5 models for Flux by LifuWang (see https://huggingface.co/LifuWang/DistillT5)
* lama and MAT inpainting models usable in img2img, both as processing options and as infill options
