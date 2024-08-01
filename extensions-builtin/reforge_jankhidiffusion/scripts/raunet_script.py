import logging

import gradio as gr
from modules import scripts

# Now import from your package
from HiDiffusion.raunet import ApplyRAUNet, ApplyRAUNetSimple, UPSCALE_METHODS
from HiDiffusion.msw_msa_attention import ApplyMSWMSAAttention, ApplyMSWMSAAttentionSimple

logging.debug("Imports successful in RAUNet script")
opApplyRAUNet = ApplyRAUNet()
opApplyRAUNetSimple = ApplyRAUNetSimple()
opApplyMSWMSA = ApplyMSWMSAAttention()
opApplyMSWMSASimple = ApplyMSWMSAAttentionSimple()
class RAUNetScript(scripts.Script):
    sorting_priority = 15  # Adjust this as needed

    def title(self):
        return "RAUNet and MSW-MSA for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            gr.HTML("<p><i>Make sure to use only either the simple or the advanced version.</i></p>")
            with gr.Tab("RAUNet Simple"):
                gr.Markdown("Simplified RAUNet for easier setup. Helps avoid artifacts at high resolutions.")
                raunet_simple_enabled = gr.Checkbox(label="RAUNet Simple Enabled", value=False)
                raunet_simple_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                res_mode = gr.Radio(choices=["high (1536-2048)", "low (1024 or lower)", "ultra (over 2048)"], value="high (1536-2048)", label="Resolution Mode")
                gr.Markdown("Note: Resolution mode is a preset, exact match to your resolution is not necessary.")
                simple_upscale_mode = gr.Dropdown(choices=["default"] + list(UPSCALE_METHODS), value="default", label="Upscale Mode")
                simple_ca_upscale_mode = gr.Dropdown(choices=["default"] + list(UPSCALE_METHODS), value="default", label="CA Upscale Mode")

            with gr.Tab("RAUNet Advanced"):
                gr.Markdown("Advanced RAUNet settings. For fine-tuning artifact reduction at high resolutions.")
                raunet_enabled = gr.Checkbox(label="RAUNet Enabled", value=False)
                raunet_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                input_blocks = gr.Text(label="Input Blocks", value="3")
                output_blocks = gr.Text(label="Output Blocks", value="8")
                gr.Markdown("For SD1.5: Input 3 corresponds to Output 8, Input 6 to Output 5, Input 9 to Output 2")
                gr.Markdown("For SDXL: Input 3 corresponds to Output 5, Input 6 to Output 2")
                time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                gr.Markdown("Time mode: Controls format of start/end times. Use percent if unsure.")
                start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.45)
                two_stage_upscale = gr.Checkbox(label="Two Stage Upscale", value=False)
                upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="Upscale Mode")
                gr.Markdown("Recommended upscale mode: bicubic or bislerp")
                
                with gr.Accordion(open=False, label="Cross-Attention Settings"):
                    ca_start_time = gr.Slider(label="CA Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                    ca_end_time = gr.Slider(label="CA End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                    ca_input_blocks = gr.Text(label="CA Input Blocks", value="4")
                    ca_output_blocks = gr.Text(label="CA Output Blocks", value="8")
                    ca_upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="CA Upscale Mode")

            with gr.Tab("MSW-MSA Simple"):
                gr.Markdown("Simplified MSW-MSA for easier setup. Can improve performance and quality at high resolutions.")
                mswmsa_simple_enabled = gr.Checkbox(label="MSW-MSA Simple Enabled", value=False)
                mswmsa_simple_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")

            with gr.Tab("MSW-MSA Advanced"):
                gr.Markdown("Advanced MSW-MSA settings. For fine-tuning performance and quality improvements.")
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Enabled", value=False)
                mswmsa_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                mswmsa_input_blocks = gr.Text(label="Input Blocks", value="1,2")
                mswmsa_middle_blocks = gr.Text(label="Middle Blocks", value="")
                mswmsa_output_blocks = gr.Text(label="Output Blocks", value="9,10,11")
                gr.Markdown("Recommended SD15: input 1,2, output 9,10,11")
                gr.Markdown("Recommended SDXL: input 4,5, output 4,5")
                mswmsa_time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                mswmsa_start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                mswmsa_end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                gr.Markdown("Note: For very high resolutions (>2048), try starting at 0.2 or after other scaling effects end.")

            gr.HTML("<p><i>Note: Make sure you use the options corresponding to your model type (SD1.5 or SDXL). Otherwise, it may have no effect or fail.</i></p>")
            gr.Markdown("Compatibility: These methods may not work with other attention modifications or scaling effects targeting the same blocks.")

        # Add JavaScript to handle visibility and model-specific settings
        def update_raunet_settings(model_type):
            if model_type == "SD15":
                return "3", "8", "4", "8", 0.0, 0.45, 0.0, 0.3
            else:  # SDXL
                return "3", "5", "2", "7", 1.0, 1.0, 1.0, 1.0  # Disabling both patches by default for SDXL

        raunet_model_type.change(
            fn=update_raunet_settings,
            inputs=[raunet_model_type],
            outputs=[input_blocks, output_blocks, ca_input_blocks, ca_output_blocks, start_time, end_time, ca_start_time, ca_end_time]
        )

        def update_mswmsa_settings(model_type):
            if model_type == "SD15":
                return "1,2", "", "9,10,11"
            else:  # SDXL
                return "4,5", "", "4,5"

        mswmsa_model_type.change(
            fn=update_mswmsa_settings,
            inputs=[mswmsa_model_type],
            outputs=[mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks]
        )

        return (enabled,raunet_simple_enabled, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode,
                raunet_enabled, raunet_model_type, input_blocks, output_blocks, time_mode, start_time, end_time, 
                two_stage_upscale, upscale_mode, ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode,
                mswmsa_simple_enabled, mswmsa_simple_model_type,
                mswmsa_enabled, mswmsa_model_type, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, 
                mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (enabled,raunet_simple_enabled, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode,
        raunet_enabled, raunet_model_type, input_blocks, output_blocks, time_mode, start_time, end_time, 
        two_stage_upscale, upscale_mode, ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode,
        mswmsa_simple_enabled, mswmsa_simple_model_type,
        mswmsa_enabled, mswmsa_model_type, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, 
        mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time) = script_args

        if not enabled:
            # Apply RAUNet patch with enabled=False to reset any modifications
            unet = opApplyRAUNet.patch(False, unet, "", "", "", 0, 0, False, "", 0, 0, "", "", "")[0]
            p.extra_generation_params.update(dict(raunet_enabled=False, raunet_simple_enabled=False))

            # Apply MSW-MSA patch with empty block settings to reset any modifications
            unet = opApplyMSWMSA.patch(unet, "", "", "", mswmsa_time_mode, 0, 0)[0]
            p.extra_generation_params.update(dict(mswmsa_enabled=False, mswmsa_simple_enabled=False))
            return
        
        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        # Handle RAUNet
        if raunet_simple_enabled or raunet_enabled:
            if raunet_simple_enabled:
                unet = opApplyRAUNetSimple.go(
                    True, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode, unet
                )[0]
                p.extra_generation_params.update(
                    dict(
                        raunet_simple_enabled=True,
                        raunet_model_type=raunet_simple_model_type,
                        raunet_res_mode=res_mode,
                        raunet_simple_upscale_mode=simple_upscale_mode,
                        raunet_simple_ca_upscale_mode=simple_ca_upscale_mode,
                    )
                )
            else:
                unet = opApplyRAUNet.patch(
                    True, unet, input_blocks, output_blocks, time_mode, start_time, end_time, two_stage_upscale, upscale_mode,
                    ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode
                )[0]
                p.extra_generation_params.update(
                    dict(
                        raunet_enabled=True,
                        raunet_model_type=raunet_model_type,
                        raunet_input_blocks=input_blocks,
                        raunet_output_blocks=output_blocks,
                        raunet_time_mode=time_mode,
                        raunet_start_time=start_time,
                        raunet_end_time=end_time,
                        raunet_two_stage_upscale=two_stage_upscale,
                        raunet_upscale_mode=upscale_mode,
                        raunet_ca_start_time=ca_start_time,
                        raunet_ca_end_time=ca_end_time,
                        raunet_ca_input_blocks=ca_input_blocks,
                        raunet_ca_output_blocks=ca_output_blocks,
                        raunet_ca_upscale_mode=ca_upscale_mode,
                    )
                )
        else:
            # Apply RAUNet patch with enabled=False to reset any modifications
            unet = opApplyRAUNet.patch(False, unet, "", "", "", 0, 0, False, "", 0, 0, "", "", "")[0]
            p.extra_generation_params.update(dict(raunet_enabled=False, raunet_simple_enabled=False))

        # Handle MSW-MSA
        if mswmsa_simple_enabled or mswmsa_enabled:
            if mswmsa_simple_enabled:
                unet = opApplyMSWMSASimple.go(mswmsa_simple_model_type, unet)[0]
                p.extra_generation_params.update(
                    dict(
                        mswmsa_simple_enabled=True,
                        mswmsa_model_type=mswmsa_simple_model_type,
                    )
                )
            else:
                unet = opApplyMSWMSA.patch(
                    unet, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time
                )[0]
                p.extra_generation_params.update(
                    dict(
                        mswmsa_enabled=True,
                        mswmsa_model_type=mswmsa_model_type,
                        mswmsa_input_blocks=mswmsa_input_blocks,
                        mswmsa_middle_blocks=mswmsa_middle_blocks,
                        mswmsa_output_blocks=mswmsa_output_blocks,
                        mswmsa_time_mode=mswmsa_time_mode,
                        mswmsa_start_time=mswmsa_start_time,
                        mswmsa_end_time=mswmsa_end_time,
                    )
                )
        else:
            # Apply MSW-MSA patch with empty block settings to reset any modifications
            unet = opApplyMSWMSA.patch(unet, "", "", "", mswmsa_time_mode, 0, 0)[0]
            p.extra_generation_params.update(dict(mswmsa_enabled=False, mswmsa_simple_enabled=False))

        # Always update the unet
        p.sd_model.forge_objects.unet = unet

        # Add debug logging
        logging.debug(f"RAUNet Simple enabled: {raunet_simple_enabled}, Model Type: {raunet_simple_model_type}")
        logging.debug(f"RAUNet enabled: {raunet_enabled}, Model Type: {raunet_model_type}")
        logging.debug(f"MSW-MSA Simple enabled: {mswmsa_simple_enabled}, Model Type: {mswmsa_simple_model_type}")
        logging.debug(f"MSW-MSA enabled: {mswmsa_enabled}, Model Type: {mswmsa_model_type}")
        logging.debug(f"MSW-MSA settings: Input Blocks: {mswmsa_input_blocks}, Output Blocks: {mswmsa_output_blocks}")

        return
