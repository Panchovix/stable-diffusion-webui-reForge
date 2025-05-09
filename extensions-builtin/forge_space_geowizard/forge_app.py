import spaces

import gradio as gr
import numpy as np
import torch
from PIL import Image

from diffusers import DDIMScheduler, AutoencoderKL
from geo_models.geowizard_pipeline import DepthNormalEstimationPipeline
from geo_models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


with spaces.capture_gpu_object() as gpu_object:
    vae = AutoencoderKL.from_pretrained(spaces.convert_root_path(), subfolder='vae')
    scheduler = DDIMScheduler.from_pretrained(spaces.convert_root_path(), subfolder='scheduler')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(spaces.convert_root_path(), subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(spaces.convert_root_path(), subfolder="feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(spaces.convert_root_path(), subfolder="unet")

pipe = DepthNormalEstimationPipeline(vae=vae,
                                     image_encoder=image_encoder,
                                     feature_extractor=feature_extractor,
                                     unet=unet,
                                     scheduler=scheduler)

spaces.automatically_move_pipeline_components(pipe)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.encoder, target_model=pipe.vae)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.decoder, target_model=pipe.vae)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.post_quant_conv, target_model=pipe.vae)
# spaces.change_attention_from_diffusers_to_forge(vae)
# spaces.change_attention_from_diffusers_to_forge(unet)


@spaces.GPU(gpu_objects=gpu_object, manual_load=True)
def depth_normal(img,
                 denoising_steps,
                 ensemble_size,
                 processing_res,
                 seed,
                 domain):
    seed = int(seed)
    if seed >= 0:
        torch.manual_seed(seed)

    pipe_out = pipe(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=0,
        domain=domain,
        show_progress_bar=True,
    )

    depth_colored = Image.fromarray(((1. - pipe_out.depth_np) * 255.0).clip(0, 255).astype(np.uint8))
    normal_colored = pipe_out.normal_colored

    return depth_colored, normal_colored


TITLE = "# GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image"
DESCRIPTION = "Generate consistent depth and normal from single image. High quality and rich details. https://github.com/fuxiao0719/GeoWizard/"
GPU_ID = 0

with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            input_image = gr.Image(type='pil', image_mode='RGBA', height=320, label='Input image')

            with gr.Accordion('Advanced options', open=True):
                with gr.Column():
                    domain = gr.Radio(
                        [
                            ("Outdoor", "outdoor"),
                            ("Indoor", "indoor"),
                            ("Object", "object"),
                        ],
                        label="Data type (select the type that matches your image)",
                        value="indoor",
                    )
                    denoising_steps = gr.Slider(
                        label="Number of denoising steps (more steps, better quality)",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=10,
                    )
                    ensemble_size = gr.Slider(
                        label="Ensemble size (more steps, higher accuracy)",
                        minimum=1,
                        maximum=15,
                        step=1,
                        value=3,
                    )
                    seed = gr.Number(0, label='Seed')

                    processing_res = gr.Radio(
                        [
                            ("Native", 0),
                            ("Recommended", 768),
                        ],
                        label="Processing resolution",
                        value=768,
                    )

        with gr.Column(scale=1):
            run_btn = gr.Button('Generate', variant='primary', interactive=True)

            depth = gr.Image(interactive=False, label="Depth", height=360)
            normal = gr.Image(interactive=False, label="Normal", height=360)

            run_btn.click(fn=depth_normal, inputs=[input_image, denoising_steps, ensemble_size, processing_res, seed, domain], outputs=[depth, normal])

if __name__ == '__main__':
    demo.queue().launch(share=True, max_threads=80)
