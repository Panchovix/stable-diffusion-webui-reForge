# HuggingFace Guess

A simple tool to guess an HuggingFace repo URL from a state dict.

This repo does almost the same thing as `from diffusers.loaders.single_file_utils import fetch_diffusers_config` but a bit stronger and more robust.

The main model detection logics are extracted from Diffusers and stolen from ComfyUI.

```python
import safetensors.torch as sf
import huggingface_guess


state_dict = sf.load_file('./realisticVisionV51_v51VAE.safetensors')
repo_name = huggingface_guess.guess_repo_name(state_dict)
print(repo_name)
```

The above codes will print `runwayml/stable-diffusion-v1-5`. 

Then you can download (or prefetch configs) from HuggingFace to instantiate models and load weights.
