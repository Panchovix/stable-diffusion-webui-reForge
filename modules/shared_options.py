import os
import gradio as gr

from modules import localization, ui_components, shared_items, shared, interrogate, shared_gradio_themes, util, sd_emphasis
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir, default_output_dir  # noqa: F401
from modules.shared_cmd_options import cmd_opts
from modules.options import options_section, OptionInfo, OptionHTML, categories
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling

options_templates = {}
hide_dirs = shared.hide_dirs

restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images",
    "temp_dir",
    "clean_temp_dir_at_start",
}

categories.register_category("saving", "Saving images")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")
categories.register_category("training", "Training")

options_templates.update(options_section(('saving-images', "Saving images/grids", "saving"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images', ui_components.DropdownEditable, {"choices": ("png", "jpg", "jpeg", "webp", "avif")}).info("manual input of <a href='https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html' target='_blank'>other formats</a> is possible, but compatibility is not guaranteed"),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "save_images_replace_action": OptionInfo("Replace", "Saving the image to an existing file", gr.Radio, {"choices": ["Replace", "Add number suffix"], **hide_dirs}),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids', ui_components.DropdownEditable, {"choices": ("png", "jpg", "jpeg", "webp", "avif")}).info("manual input of <a href='https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html' target='_blank'>other formats</a> is possible, but compatibility is not guaranteed"),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "grid_zip_filename_pattern": OptionInfo("", "Archive filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "font": OptionInfo("", "Font for image grids that have text"),
    "grid_text_active_color": OptionInfo("#000000", "Text color for image grids", ui_components.FormColorPicker, {}),
    "grid_text_inactive_color": OptionInfo("#999999", "Inactive text color for image grids", ui_components.FormColorPicker, {}),
    "grid_background_color": OptionInfo("#ffffff", "Background color for image grids", ui_components.FormColorPicker, {}),

    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg and avif images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    "img_max_size_mp": OptionInfo(200, "Maximum image size", gr.Number).info("in megapixels"),

    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "save_write_log_csv": OptionInfo(True, "Write log.csv when saving images using 'Save' button"),
    "save_init_img": OptionInfo(False, "Save init images when using img2img"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

    "save_incomplete_images": OptionInfo(False, "Save incomplete images").info("save images that has been interrupted in mid-generation; even if not saved, they will still show up in webui output."),

    "notification_audio": OptionInfo(True, "Play notification sound after image generation").info("notification.mp3 should be present in the root directory").needs_reload_ui(),
    "notification_volume": OptionInfo(100, "Notification sound volume", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("in %"),
}))

options_templates.update(options_section(('saving-paths', "Paths for saving", "saving"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'txt2img-images')), 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'img2img-images')), 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'extras-images')), 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'txt2img-grids')), 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'img2img-grids')), 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo(util.truncate_path(os.path.join(data_path, 'log', 'images')), "Directory for saving images using the Save button", component_args=hide_dirs),
    "outdir_init_images": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'init-images')), "Directory for saving init images when using img2img", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory", "saving"), {
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling", "postprocessing"), {
    "unload_sd_during_upscale": OptionInfo(False, "Unload SD Model from VRAM to RAM during upscale"),
    "ESRGAN_tile": OptionInfo(256, "Tile size for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 16}).info("0 = no tiling"),
    "ESRGAN_tile_overlap": OptionInfo(32, "Tile overlap for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 8}).info("Low values = visible seam"),
    "RCAN_tile": OptionInfo(512, "Tile size for RCAN upscaler. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 16}),
    "RCAN_tile_overlap": OptionInfo(32, "Tile overlap for RCAN upscaler. Higher values = fewer artifacts but slower processing.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 8}),
    "PLKSR_tile": OptionInfo(512, "Tile size for PLKSR upscaler. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
    "PLKSR_tile_overlap": OptionInfo(32, "Tile overlap for PLKSR upscaler. Higher values = fewer artifacts but slower processing.", gr.Slider, {"minimum": 0, "maximum": 256, "step": 16}),
    "DAT_tile": OptionInfo(256, "Tile size for DAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 32}).info("0 = no tiling"),
    "DAT_tile_overlap": OptionInfo(32, "Tile overlap for DAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 32}).info("Low values = visible seam"),
    "HAT_tile": OptionInfo(256, "Tile size for HAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 16}).info("0 = no tiling"),
    "HAT_tile_overlap": OptionInfo(32, "Tile overlap for HAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 16}).info("Low values = visible seam"),
    "SRFormer_tile": OptionInfo(176, "Tile size for SRFormer upscalers. Recommended: Multiple of 22 for SRFormer, Multiple of 16 for SRFormerLight", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 2}).info("0 = no tiling"),
    "SRFormer_tile_overlap": OptionInfo(32, "Tile overlap for SRFormer upscalers. Recommended: Multiple of 22 for SRFormer, Multiple of 16 for SRFormerLight", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 2}).info("Low values = visible seam"),
    "GRL_tile": OptionInfo(256, "Tile size for GRL upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 8}).info("0 = no tiling"),
    "GRL_tile_overlap": OptionInfo(32, "Tile overlap for GRL upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 8}).info("Low values = visible seam"),
    "OmniSR_tile": OptionInfo(256, "Tile size for OmniSR upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 8}).info("0 = no tiling"),
    "OmniSR_tile_overlap": OptionInfo(32, "Tile overlap for OmniSR upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 8}).info("Low values = visible seam"),
    "SPAN_tile": OptionInfo(256, "Tile size for SPAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 32}).info("0 = no tiling"),
    "SPAN_tile_overlap": OptionInfo(32, "Tile overlap for SPAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 32}).info("Low values = visible seam"),
    "COMPACT_tile": OptionInfo(0, "Tile size for COMPACT upscalers.", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 16}).info("0 = no tiling"),
    "COMPACT_tile_overlap": OptionInfo(32, "Tile overlap for COMPACT upscalers.", gr.Slider, {"minimum": 0, "maximum": 2048, "step": 16}).info("Low values = visible seam"),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "dat_enabled_models": OptionInfo(["DAT x2", "DAT x3", "DAT x4"], "Select which DAT models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.dat_models_names()}),
    "DAT_tile": OptionInfo(192, "Tile size for DAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}),
    "set_scale_by_when_changing_upscaler": OptionInfo(False, "Automatically set the Scale by factor based on the name of the selected Upscaler."),
}))

options_templates.update(options_section(('face-restoration', "Face restoration", "postprocessing"), {
    "face_restoration": OptionInfo(False, "Restore faces", infotext='Face restoration').info("will use a third-party model on generation result to reconstruct faces"),
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in shared.face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('system', "System", "system"), {
    "auto_launch_browser": OptionInfo("Local", "Automatically open webui in browser on startup", gr.Radio, lambda: {"choices": ["Disable", "Local", "Remote"]}),
    "enable_console_prompts": OptionInfo(shared.cmd_opts.enable_console_prompts, "Print prompts to console when generating with txt2img and img2img."),
    "show_warnings": OptionInfo(False, "Show warnings in console.").needs_reload_ui(),
    "show_gradio_deprecation_warnings": OptionInfo(True, "Show gradio deprecation warnings in console.").needs_reload_ui(),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    "enable_upscale_progressbar": OptionInfo(True, "Show a progress bar in the console for tiled upscaling."),
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
    "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info("directory is hidden if its name starts with \".\""),
    "disable_mmap_load_safetensors": OptionInfo(False, "Disable memmapping for loading .safetensors files.").info("fixes very slow loading speed in some cases"),
    "hide_ldm_prints": OptionInfo(True, "Prevent Stability-AI's ldm/sgm modules from printing noise to console."),
    "dump_stacks_on_signal": OptionInfo(False, "Print stack traces before exiting the program with ctrl+c."),
    "concurrent_git_fetch_limit": OptionInfo(16, "Number of simultaneous extension update checks ", gr.Slider, {"step": 1, "minimum": 1, "maximum": 100}).info("reduce extension update check time"),
}))

options_templates.update(options_section(('profiler', "Profiler", "system"), {
    "profiling_explanation": OptionHTML("""
Those settings allow you to enable torch profiler when generating pictures.
Profiling allows you to see which code uses how much of computer's resources during generation.
Each generation writes its own profile to one file, overwriting previous.
The file can be viewed in <a href="chrome:tracing">Chrome</a>, or on a <a href="https://ui.perfetto.dev/">Perfetto</a> web site.
Warning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB in size.
"""),
    "profiling_enable": OptionInfo(False, "Enable profiling"),
    "profiling_activities": OptionInfo(["CPU"], "Activities", gr.CheckboxGroup, {"choices": ["CPU", "CUDA"]}),
    "profiling_record_shapes": OptionInfo(True, "Record shapes"),
    "profiling_profile_memory": OptionInfo(True, "Profile memory"),
    "profiling_with_stack": OptionInfo(True, "Include python stack"),
    "profiling_filename": OptionInfo("trace.json", "Profile filename"),
}))

options_templates.update(options_section(('API', "API", "system"), {
    "api_enable_requests": OptionInfo(True, "Allow http:// and https:// URLs for input images in API", restrict_api=True),
    "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
    "api_useragent": OptionInfo("", "User agent for requests", restrict_api=True),
}))

options_templates.update(options_section(('training', "Training", "training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    "pin_memory": OptionInfo(False, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file."),
    "save_training_settings_to_txt": OptionInfo(True, "Save textual inversion and hypernet settings to a text file whenever training starts."),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    "training_xattention_optimizations": OptionInfo(False, "Use cross attention optimizations while training"),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging."),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard."),
    "training_tensorboard_flush_every": OptionInfo(120, "How often, in seconds, to flush the pending tensorboard events and summaries to disk."),
}))

options_templates.update(options_section(('sd', "Stable Diffusion", "sd"), {
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}, refresh=shared_items.refresh_checkpoints, infotext='Model hash'),
    "sd_checkpoints_limit": OptionInfo(1, "Maximum number of checkpoints loaded at the same time", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "sd_checkpoints_keep_in_cpu": OptionInfo(True, "Only keep one model on device").info("will keep models other than the currently used one in RAM rather than VRAM"),
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}).info("obsolete; set to 0 and use the two settings above instead"),
    "sd_unet": OptionInfo("Automatic", "SD Unet", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list).info("choose Unet model: Automatic = use one with same filename as checkpoint; None = use Unet from checkpoint"),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds").needs_reload_ui(),
    "emphasis": OptionInfo("Original", "Emphasis mode", gr.Radio, lambda: {"choices": [x.name for x in sd_emphasis.options]}, infotext="Emphasis").info("makes it possible to make model to pay (more:1.1) or (less:0.9) attention to text when you use the syntax in prompt; " + sd_emphasis.get_options_descriptions()),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
    "sdxl_clip_l_skip": OptionInfo(False, "Clip skip SDXL", gr.Checkbox).info("Enable Clip skip for the secondary clip model in sdxl. Has no effect on SD 1.5 or SD 2.0/2.1."),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}, infotext="Clip skip").link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer"),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU", "NV"]}, infotext="RNG").info("changes seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"),
    "tiling": OptionInfo(False, "Tiling", infotext='Tiling').info("produce a tileable picture"),
    "hires_fix_refiner_pass": OptionInfo("second pass", "Hires fix: which pass to enable refiner for", gr.Radio, {"choices": ["first pass", "second pass", "both passes"]}, infotext="Hires refiner"),
    "cond_stage_model_device_compatibility_check": OptionInfo(False, "Perform device compatibility check for conditional stage model. Enables broader hardware compatibility by falling back to CPU if GPU doesn't support required data types. May improve stability on some systems, but can significantly slow down model loading and potentially impact generation speed.", gr.Checkbox, {"interactive": True}),
}))

options_templates.update(options_section(('sdxl', "Stable Diffusion XL", "sd"), {
    "sdxl_crop_top": OptionInfo(0, "crop top coordinate"),
    "sdxl_crop_left": OptionInfo(0, "crop left coordinate"),
    "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "SDXL low aesthetic score", gr.Number).info("used for refiner model negative prompt"),
    "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "SDXL high aesthetic score", gr.Number).info("used for refiner model prompt"),
}))

options_templates.update(options_section(('sd3', "Stable Diffusion 3", "sd"), {
    "sd3_enable_t5": OptionInfo(False, "Enable T5").info("load T5 text encoder; increases VRAM use by a lot, potentially improving quality of generation; requires model reload to apply"),
}))

options_templates.update(options_section(('vae', "VAE", "sd"), {
    "sd_vae_explanation": OptionHTML("""
<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>
image into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling
(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.
For img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.
"""),
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list, infotext='VAE').info("choose VAE model: Automatic = use one with same filename as checkpoint; None = use VAE from checkpoint"),
    "sd_vae_overrides_per_model_preferences": OptionInfo(True, "Selected VAE overrides per-model preferences").info("you can set per-model VAE either by editing user metadata for checkpoints, or by making the VAE have same name as checkpoint"),
    "auto_vae_precision_bfloat16": OptionInfo(False, "Automatically convert VAE to bfloat16").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image; if enabled, overrides the option below"),
    "auto_vae_precision": OptionInfo(True, "Automatically revert VAE to 32-bit floats").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image"),
    "sd_vae_encode_method": OptionInfo("Full", "VAE type for encode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Encoder').info("method to encode image to latent (use in img2img, hires-fix or inpaint mask)"),
    "sd_vae_decode_method": OptionInfo("Full", "VAE type for decode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Decoder').info("method to decode latent to image"),
}))

options_templates.update(options_section(('sd_sampling', "SD Sampling backend for A1111 samplers", "sd"), {
    "sd_sampling": OptionInfo(
        "A1111",
        "SD Sampling Backend for A1111 samplers",
        gr.Radio,
        lambda: {"choices": ["A1111", "ldm patched (Comfy)"]}
    ).info(
        """<p>Choose the SD sampling backend for A1111:</p>
        <p><strong>A restart of the UI is required for changes to apply effect.</strong></p>
        <p><strong>A1111:</strong> Uses the implementation found on repositories/k_diffusion to use on A1111 samplers. It can help to reproduce old seeds and get some desired outputs, but code hasn't been updated in a while.<br>
        <p><strong>ldm patched (Comfy):</strong> Uses the implementation of ldm_patched.k_diffusion to use on A1111 samplers. It does add more determinism and optimizations to samplers. Uses latest code but can change the results not as expected<br>
        """
    ),
}))

options_templates.update(options_section(('img2img', "img2img", "sd"), {
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Conditional mask weight'),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.001}, infotext='Noise multiplier'),
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img and hires fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Extra noise').info("0 = disabled (default); should be lower than denoising strength"),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies.").info("normally you'd do less with less denoising"),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill transparent parts of the input image with this color.", ui_components.FormColorPicker, {}),
    "img2img_editor_height": OptionInfo(720, "Height of the image editor", gr.Slider, {"minimum": 80, "maximum": 1600, "step": 1}).info("in pixels").needs_reload_ui(),
    "img2img_sketch_default_brush_color": OptionInfo("#ffffff", "Sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img sketch").needs_reload_ui(),
    "img2img_inpaint_mask_brush_color": OptionInfo("#ffffff", "Inpaint mask brush color", ui_components.FormColorPicker,  {}).info("brush color of inpaint mask").needs_reload_ui(),
    "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ffffff", "Inpaint sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img inpaint sketch").needs_reload_ui(),
    "img2img_inpaint_mask_high_contrast": OptionInfo(True, "For inpainting, use a high-contrast brush pattern").info("use a checkerboard brush pattern instead of color brush").needs_reload_ui(),
    "img2img_inpaint_mask_scribble_alpha": OptionInfo(75, "Inpaint mask alpha (transparency)", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("only affects non-high-contrast brush").needs_reload_ui(),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 1000, "step": 1}).info('0: disable, -1: show all images. Too many images can cause lag'),
    "overlay_inpaint": OptionInfo(True, "Overlay original for inpaint").info("when inpainting, overlay the original image over the areas that weren't inpainted."),
    "img2img_autosize": OptionInfo(False, "After loading into Img2img, automatically update Width and Height"),
    "img2img_batch_use_original_name": OptionInfo(False, "Save using original filename in img2img batch. Applies to 'Upload' and 'From directory' tabs.").info("Warning: overwriting is possible, based on Settings > Saving images/grids > Saving the image to an existing file.")
}))

options_templates.update(options_section(('optimizations', "Optimizations", "sd"), {
    "cross_attention_optimization": OptionInfo("Automatic", "Cross attention optimization", gr.Dropdown, lambda: {"choices": shared_items.cross_attention_optimizations()}),
    "s_min_uncond": OptionInfo(0.0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 15.0, "step": 0.01}, infotext='NGMS').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster"),
    "s_min_uncond_all": OptionInfo(False, "Negative Guidance minimum sigma all steps", infotext='NGMS all steps').info("By default, NGMS above skips every other step; this makes it skip all steps"),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256").info("0=disable, higher=faster"),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for high-res pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio hr').info("only applies if non-zero and overrides above"),
    "pad_cond_uncond": OptionInfo(False, "Pad prompt/negative prompt", infotext='Pad conds').info("improves performance when prompt and negative prompt have different lengths; changes seeds"),
    "pad_cond_uncond_v0": OptionInfo(False, "Pad prompt/negative prompt (v0)", infotext='Pad conds v0').info("alternative implementation for the above; used prior to 1.6.0 for DDIM sampler; overrides the above if set; WARNING: truncates negative prompt if it's too long; changes seeds"),
    "persistent_cond_cache": OptionInfo(True, "Persistent cond cache").info("do not recalculate conds from prompts if prompts have not changed since previous calculation"),
    "batch_cond_uncond": OptionInfo(True, "Batch cond/uncond").info("do both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed; previously this was controlled by --always-batch-cond-uncond commandline argument"),
    "fp8_storage": OptionInfo("Disable", "FP8 weight", gr.Radio, {"choices": ["Disable", "Enable for SDXL", "Enable"]}).info("Use FP8 to store Linear/Conv layers' weight. Require pytorch>=2.1.0."),
    "cache_fp16_weight": OptionInfo(False, "Cache FP16 weight for LoRA").info("Cache fp16 weight when enabling FP8, will increase the quality of LoRA. Use more system ram."),
}))

options_templates.update(options_section(('compatibility', "Compatibility", "sd"), {
    "auto_backcompat": OptionInfo(True, "Automatic backward compatibility").info("automatically enable options for backwards compatibility when importing generation parameters from infotext that has program version."),
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
    "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
    "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
    "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
    "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; old: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"),
    "use_downcasted_alpha_bar": OptionInfo(False, "Downcast model alphas_cumprod to fp16 before sampling. For reproducing old seeds.", infotext="Downcast alphas_cumprod"),
    "refiner_switch_by_sample_steps": OptionInfo(False, "Switch to refiner by sampling steps instead of model timesteps. Old behavior for refiner.", infotext="Refiner switch by sampling steps"),
    "use_old_clip_g_load_and_ztsnr_application": OptionInfo(False, "Use old (incorrect) CLIP G load and ztSNR application. For reproducing old seeds.", infotext="Old CLIP G load and ztSNR application"),
}))

options_templates.update(options_section(('interrogate', "Interrogate"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(False, "Include ranks of model tags matches in results.").info("booru only"),
    "interrogate_clip_num_beams": OptionInfo(1, "BLIP: num_beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "BLIP: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "BLIP: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file").info("0 = No limit"),
    "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": interrogate.category_types()}, refresh=interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "deepbooru: score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "deepbooru: sort tags alphabetically").info("if not: sort by score"),
    "deepbooru_use_spaces": OptionInfo(True, "deepbooru: use spaces in tags").info("if not: use underscores"),
    "deepbooru_escape": OptionInfo(True, "deepbooru: escape (\\) brackets").info("so they are used as literal brackets and not for emphasis"),
    "deepbooru_filter_tags": OptionInfo("", "deepbooru: filter out those tags").info("separate by comma"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks", "sd"), {
    "extra_networks_show_hidden_directories": OptionInfo(True, "Show hidden directories").info("directory is hidden if its name starts with \".\"."),
    "extra_networks_dir_button_function": OptionInfo(False, "Add a '/' to the beginning of directory buttons").info("Buttons will display the contents of the selected directory without acting as a search filter."),
    "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string has 4 characters or more'),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
    "extra_networks_card_text_scale": OptionInfo(1.0, "Card text scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}).info("1 = original size"),
    "extra_networks_card_show_desc": OptionInfo(True, "Show description on card"),
    "extra_networks_card_description_is_html": OptionInfo(False, "Treat card description as HTML"),
    "extra_networks_card_order_field": OptionInfo("Path", "Default order field for Extra Networks cards", gr.Dropdown, {"choices": ['Path', 'Name', 'Date Created', 'Date Modified']}).needs_reload_ui(),
    "extra_networks_card_order": OptionInfo("Ascending", "Default order for Extra Networks cards", gr.Dropdown, {"choices": ['Ascending', 'Descending']}).needs_reload_ui(),
    "extra_networks_tree_view_style": OptionInfo("Tree", "Extra Networks directory view style", gr.Radio, {"choices": ["Tree", "Dirs"]}).needs_reload_ui(),
    "extra_networks_tree_view_default_enabled": OptionInfo(True, "Show the Extra Networks directory view by default").needs_reload_ui(),
    "extra_networks_tree_view_default_width": OptionInfo(180, "Default width for the Extra Networks directory tree view", gr.Number).needs_reload_ui(),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_reload_ui(),
    "textual_inversion_print_at_load": OptionInfo(False, "Print a list of Textual Inversion embeddings when loading model"),
    "textual_inversion_add_hashes_to_infotext": OptionInfo(True, "Add Textual Inversion hashes to infotext"),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None", *shared.hypernetworks]}, refresh=shared_items.reload_hypernetworks),
    "textual_inversion_image_embedding_data_cache": OptionInfo(False, 'Cache the data of image embeddings').info('potentially increase TI load time at the cost some disk space'),
}))

options_templates.update(options_section(('ui_prompt_editing', "Prompt editing", "ui"), {
    "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "Word delimiters when editing the prompt with Ctrl+up/down"),
    "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Ctrl+up/down whitespace delimiters", gr.CheckboxGroup, lambda: {"choices": ["Tab", "Carriage Return", "Line Feed"]}),
    "keyedit_move": OptionInfo(True, "Alt+left/right moves prompt elements"),
    "disable_token_counters": OptionInfo(False, "Disable prompt token counters"),
    "include_styles_into_token_counters": OptionInfo(True, "Count tokens of enabled styles").info("When calculating how many tokens the prompt has, also consider tokens added by enabled styles."),
}))

options_templates.update(options_section(('ui_gallery', "Gallery", "ui"), {
    "return_grid": OptionInfo(True, "Show grid in gallery"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in gallery"),
    "js_modal_lightbox": OptionInfo(True, "Full page image viewer: enable"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Full page image viewer: show images zoomed in by default"),
    "js_modal_lightbox_gamepad": OptionInfo(False, "Full page image viewer: navigate with gamepad"),
    "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "Full page image viewer: gamepad repeat period").info("in milliseconds"),
    "sd_webui_modal_lightbox_icon_opacity": OptionInfo(1, "Full page image viewer: control icon unfocused opacity", gr.Slider, {"minimum": 0.0, "maximum": 1, "step": 0.01}, onchange=shared.reload_gradio_theme).info('for mouse only').needs_reload_ui(),
    "sd_webui_modal_lightbox_toolbar_opacity": OptionInfo(0.9, "Full page image viewer: tool bar opacity", gr.Slider, {"minimum": 0.0, "maximum": 1, "step": 0.01}, onchange=shared.reload_gradio_theme).info('for mouse only').needs_reload_ui(),
    "gallery_height": OptionInfo("", "Gallery height", gr.Textbox).info("can be any valid CSS value, for example 768px or 20em").needs_reload_ui(),
    "open_dir_button_choice": OptionInfo("Subdirectory", "What directory the [📂] button opens", gr.Radio, {"choices": ["Output Root", "Subdirectory", "Subdirectory (even temp dir)"]}),
    "hires_button_gallery_inset": OptionInfo(False, "Insert [✨] hires button results to gallery").info("when False the original first pass image is replaced by the results"),
}))

options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
    "compact_prompt_box": OptionInfo(False, "Compact prompt layout").info("puts prompt and negative prompt inside the Generate tab, leaving more vertical space for the image on the right").needs_reload_ui(),
    "samplers_in_dropdown": OptionInfo(True, "Use dropdown for sampler selection instead of radio group").needs_reload_ui(),
    "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_reload_ui(),
    "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Checkpoint dropdown: use filenames without paths").info("models in subdirectories like photo/sd15.ckpt will be listed as just sd15.ckpt"),
    "hires_fix_show_sampler": OptionInfo(False, "Hires fix: show hires checkpoint and sampler selection").needs_reload_ui(),
    "hires_fix_show_prompts": OptionInfo(False, "Hires fix: show hires prompt and negative prompt").needs_reload_ui(),
    "txt2img_settings_accordion": OptionInfo(False, "Settings in txt2img hidden under Accordion").needs_reload_ui(),
    "img2img_settings_accordion": OptionInfo(False, "Settings in img2img hidden under Accordion").needs_reload_ui(),
    "interrupt_after_current": OptionInfo(True, "Don't Interrupt in the middle").info("when using Interrupt button, if generating more than one image, stop after the generation of an image has finished, instead of immediately"),
}))

options_templates.update(options_section(('ui', "User interface", "ui"), {
    "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
    "quicksettings_list": OptionInfo(["sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_reload_ui(),
    "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "ui_reorder_list": OptionInfo([], "UI item order for txt2img/img2img tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
    "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + shared_gradio_themes.gradio_hf_hub_themes}).info("you can also manually enter any of themes from the <a href='https://huggingface.co/spaces/gradio/theme-gallery'>gallery</a>.").needs_reload_ui(),
    "gradio_themes_cache": OptionInfo(True, "Cache gradio themes locally").info("disable to update the selected Gradio theme"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "enable_reloading_ui_scripts": OptionInfo(False, "Reload UI scripts when using Reload UI option").info("useful for developing: if you make changes to UI scripts code, it is applied when the UI is reloded."),
}))


options_templates.update(options_section(('infotext', "Infotext", "ui"), {
    "infotext_explanation": OptionHTML("""
Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.
It is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.
"""),
    "enable_pnginfo": OptionInfo(True, "Write infotext to metadata of the generated image"),
    "stealth_pnginfo_option": OptionInfo("Alpha", "Stealth infotext mode", gr.Radio, {"choices": ["Alpha", "RGB", "None"]}).info("Ignored if infotext is disabled"),
    "save_txt": OptionInfo(False, "Create a text file with infotext next to every generated image"),

    "add_model_name_to_info": OptionInfo(True, "Add model name to infotext"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to infotext"),
    "add_vae_name_to_info": OptionInfo(True, "Add VAE name to infotext"),
    "add_vae_hash_to_info": OptionInfo(True, "Add VAE hash to infotext"),
    "add_user_name_to_info": OptionInfo(False, "Add user name to infotext when authenticated"),
    "add_version_to_infotext": OptionInfo(True, "Add program version to infotext"),
    "disable_weights_auto_swap": OptionInfo(True, "Disregard checkpoint information from pasted infotext").info("when reading generation parameters from text into UI"),
    "infotext_skip_pasting": OptionInfo([], "Disregard fields from pasted infotext", ui_components.DropdownMulti, lambda: {"choices": shared_items.get_infotext_names()}),
    "infotext_styles": OptionInfo("Apply if any", "Infer styles from prompts of pasted infotext", gr.Radio, {"choices": ["Ignore", "Apply", "Discard", "Apply if any"]}).info("when reading generation parameters from text into UI)").html("""<ul style='margin-left: 1.5em'>
<li>Ignore: keep prompt and styles dropdown as it is.</li>
<li>Apply: remove style text from prompt, always replace styles dropdown value with found styles (even if none are found).</li>
<li>Discard: remove style text from prompt, keep styles dropdown as it is.</li>
<li>Apply if any: remove style text from prompt; if any styles are found in prompt, put them into styles dropdown, otherwise keep it as it is.</li>
</ul>"""),

}))

options_templates.update(options_section(('ui', "Live previews", "ui"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "live_previews_image_format": OptionInfo("png", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
    "show_progress_type": OptionInfo("Approx NN", "Live preview method", gr.Radio, {"choices": ["Approx NN", "Approx cheap", "TAESD"]}).info("Approx NN: fast preview; TAESD = high-quality preview; Approx cheap = fastest but low-quality preview"),
    "live_preview_allow_lowvram_full": OptionInfo(False, "Allow Full live preview method with lowvram/medvram").info("If not, Approx NN will be used instead; Full live preview method is very detrimental to speed if lowvram/medvram optimizations are enabled"),
    "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period").info("in milliseconds"),
    "live_preview_fast_interrupt": OptionInfo(False, "Return image with chosen live preview method on interrupt").info("makes interrupts faster"),
    "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show Live preview in full page image viewer"),
    "prevent_screen_sleep_during_generation": OptionInfo(True, "Prevent screen sleep during generation"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters", "sd"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
    "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta DDIM').info("noise multiplier; higher = more unpredictable results"),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta').info("noise multiplier; applies to ancestral samplers (Euler a, DPM++ 2S a) and SDE samplers"),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext='Sigma churn').info('amount of stochasticity; only applies to Euler, Heun, and DPM2'),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext='Sigma tmin').info('enable stochasticity; start value of the sigma range; only applies to Euler, Heun, and DPM2'),
    's_tmax':  OptionInfo(0.0, "sigma tmax",  gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext='Sigma tmax').info("0 = inf; end value of the sigma range; only applies to Euler, Heun, and DPM2"),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext='Sigma noise').info('amount of additional noise to counteract loss of detail during sampling'),
    'dpmpp_sde_r': OptionInfo(0.5, "DPM++ SDE r value", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='DPM++ SDE r').info("midpoint value for the SDE solver; only applies to DPM++ SDE sampler"),
    'dpmpp_2m_sde_solver': OptionInfo('midpoint', "DPM++ 2M SDE solver type", gr.Radio, {"choices": ['midpoint', 'heun']}, infotext='DPM++ 2M solver').info("solver algorithm type; only applies to DPM++ 2M SDE sampler"),
    'sigma_min': OptionInfo(0.0, "sigma min", gr.Number, infotext='Schedule min sigma').info("0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"),
    'sigma_max': OptionInfo(0.0, "sigma max", gr.Number, infotext='Schedule max sigma').info("0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"),
    'rho':  OptionInfo(0.0, "rho", gr.Number, infotext='Schedule rho').info("0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext='ENSD').info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma", infotext='Discard penultimate sigma').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
    'sgm_noise_multiplier': OptionInfo(False, "SGM noise multiplier", infotext='SGM noise multiplier').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818").info("Match initial noise to official SDXL implementation - only useful for reproducing images"),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}, infotext='UniPC variant'),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}, infotext='UniPC skip type'),
    'uni_pc_order': OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, infotext='UniPC order').info("must be < sampling steps"),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final", infotext='UniPC lower order final'),
    'sd_noise_schedule': OptionInfo("Default", "Noise schedule for sampling", gr.Radio, {"choices": ["Default", "Zero Terminal SNR"]}, infotext="Noise Schedule").info("for use with zero terminal SNR trained models"),
    'skip_early_cond': OptionInfo(0.0, "Ignore negative prompt during early sampling", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Skip Early CFG").info("disables CFG on a proportion of steps at the beginning of generation; 0=skip none; 1=skip all; can both improve sample diversity/quality and speed up sampling; XYZ plot: Skip Early CFG"),
}))

options_templates.update(options_section(('sampler-params', "Scheduler parameters", "sd"), {
    "A1111_schedulers_group": OptionHTML("""<br><h2 style='text-align: center'>Scheduler configuration for A1111 samplers</h2>
        Configuration options for schedulers for A1111 samplers (DPM++ SDE, Euler a, Euler, DPM++ 2M, Euler SMEA/DY, Kohaku_LoNyu_Yog, up to normal UniPC below DDIM)"""),
    
    "karras_rho": OptionInfo(7.0, "Karras scheduler - rho", gr.Slider, {"minimum": 1.0, "maximum": 20.0, "step": 0.1}, infotext='Karras scheduler rho').info('Default = 7.0; controls the shape of the noise schedule'),
    
    "exponential_shrink_factor": OptionInfo(0.0, "Exponential scheduler - shrink factor", gr.Slider, {"minimum": -1.0, "maximum": 1.0, "step": 0.01}, infotext='Exponential scheduler shrink factor').info('Default = 0.0; controls the rate of decay in the noise schedule'),
    
    "polyexponential_rho": OptionInfo(1.0, "Polyexponential scheduler - rho", gr.Slider, {"minimum": 0.1, "maximum": 5.0, "step": 0.1}, infotext='Polyexponential scheduler rho').info('Default = 1.0; controls the curvature of the noise schedule'),
    
    "sinusoidal_sf_factor": OptionInfo(3.5, "Sinusoidal SF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.1}, infotext='Sinusoidal SF scheduler factor').info('Default = 3.5; controls the shape of the sinusoidal curve'),
    
    "invcosinusoidal_sf_factor": OptionInfo(3.5, "Invcosinusoidal SF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.1}, infotext='Invcosinusoidal SF scheduler factor').info('Default = 3.5; controls the shape of the inverse cosinusoidal curve'),
    
    "react_cosinusoidal_dynsf_factor": OptionInfo(2.15, "React Cosinusoidal DynSF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.05}, infotext='React Cosinusoidal DynSF scheduler factor').info('Default = 2.15; controls the dynamic scaling factor'),
    
    "beta_dist_alpha": OptionInfo(0.6, "Beta scheduler - alpha", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext='Beta scheduler alpha').info('Default = 0.6; the alpha parameter of the beta distribution used in Beta sampling'),
    "beta_dist_beta": OptionInfo(0.6, "Beta scheduler - beta", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext='Beta scheduler beta').info('Default = 0.6; the beta parameter of the beta distribution used in Beta sampling'),
    
    "cosine_sf_factor": OptionInfo(1.0, "Cosine scheduler - scale factor", gr.Slider, {"minimum": 0.1, "maximum": 5.0, "step": 0.1}, infotext='Cosine scheduler scale factor').info('Default = 1.0; controls the scaling of the cosine curve'),
    
    "cosexpblend_exp_decay": OptionInfo(0.9, "Cosine-exponential Blend scheduler - exponential decay", gr.Slider, {"minimum": 0.1, "maximum": 0.99, "step": 0.01}, infotext='Cosine-exponential Blend scheduler exponential decay').info('Default = 0.9; controls the rate of exponential decay'),
    
    "phi_power": OptionInfo(2.0, "Phi scheduler - power", gr.Slider, {"minimum": 1.0, "maximum": 5.0, "step": 0.1}, infotext='Phi scheduler power').info('Default = 2.0; controls the power of the phi-based curve'),
    
    "laplace_mu": OptionInfo(0.0, "Laplace scheduler - mu", gr.Slider, {"minimum": -1.0, "maximum": 1.0, "step": 0.1}, infotext='Laplace scheduler mu').info('Default = 0.0; controls the location parameter of the Laplace distribution'),
    "laplace_beta": OptionInfo(0.5, "Laplace scheduler - beta", gr.Slider, {"minimum": 0.1, "maximum": 2.0, "step": 0.1}, infotext='Laplace scheduler beta').info('Default = 0.5; controls the scale parameter of the Laplace distribution'),
    
    "karras_dynamic_rho": OptionInfo(7.0, "Karras Dynamic scheduler - base rho", gr.Slider, {"minimum": 1.0, "maximum": 20.0, "step": 0.1}, infotext='Karras Dynamic scheduler base rho').info('Default = 7.0; controls the base shape of the dynamic noise schedule'),
    
    "ays_custom_sigmas": OptionInfo("[14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]", "Align Your Steps Custom - sigma values", gr.Textbox, {}, infotext='AYS Custom sigmas').info('Custom sigma values for the A1111 AYS custom scheduler. Modify to create your own schedule.'),

    "reforge_schedulers_group": OptionHTML("""<br><h2 style='text-align: center'>Scheduler configuration for reForge samplers</h2>
        Configuration options for schedulers for reforge samplers (All the rest, CFG++ Samplers, DPM++ SDE Comfy, Euler Ancestral Comfy, ODE, DPM++ 2M DY, DDPM, etc)"""),
    
    "reforge_karras_rho": OptionInfo(7.0, "Reforge Karras scheduler - rho", gr.Slider, {"minimum": 1.0, "maximum": 20.0, "step": 0.1}, infotext='Reforge Karras scheduler rho').info('Default = 7.0; controls the shape of the noise schedule for reforge Karras scheduler'),
    
    "reforge_exponential_shrink_factor": OptionInfo(0.0, "Reforge Exponential scheduler - shrink factor", gr.Slider, {"minimum": -1.0, "maximum": 1.0, "step": 0.01}, infotext='Reforge Exponential scheduler shrink factor').info('Default = 0.0; controls the rate of decay in the noise schedule for reforge Exponential scheduler'),
    
    "reforge_polyexponential_rho": OptionInfo(1.0, "Reforge Polyexponential scheduler - rho", gr.Slider, {"minimum": 0.1, "maximum": 5.0, "step": 0.1}, infotext='Reforge Polyexponential scheduler rho').info('Default = 1.0; controls the curvature of the noise schedule for reforge Polyexponential scheduler'),
    
    "reforge_ays_custom_sigmas": OptionInfo("[14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]", "Reforge Align Your Steps Custom - sigma values", gr.Textbox, {}, infotext='Reforge AYS Custom sigmas').info('Custom sigma values for the reforge AYS custom scheduler. Modify to create your own schedule.'),
    
    "reforge_normal_sgm": OptionInfo(False, "Reforge Normal scheduler - use SGM", gr.Checkbox, {}, infotext='Reforge Normal scheduler SGM').info('If checked, uses SGM uniform sampling for the reforge Normal scheduler'),
    
    "reforge_beta_dist_alpha": OptionInfo(0.6, "Reforge Beta scheduler - alpha", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext='Reforge Beta scheduler alpha').info('Default = 0.6; the alpha parameter of the beta distribution used in reforge Beta sampling'),
    "reforge_beta_dist_beta": OptionInfo(0.6, "Reforge Beta scheduler - beta", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext='Reforge Beta scheduler beta').info('Default = 0.6; the beta parameter of the beta distribution used in reforge Beta sampling'),
    
    "reforge_cosine_sf_factor": OptionInfo(1.0, "Reforge Cosine scheduler - scale factor", gr.Slider, {"minimum": 0.1, "maximum": 5.0, "step": 0.1}, infotext='Reforge Cosine scheduler scale factor').info('Default = 1.0; controls the scaling of the cosine curve for reforge Cosine scheduler'),
    
    "reforge_cosexpblend_exp_decay": OptionInfo(0.9, "Reforge Cosine-exponential Blend scheduler - exponential decay", gr.Slider, {"minimum": 0.1, "maximum": 0.99, "step": 0.01}, infotext='Reforge Cosine-exponential Blend scheduler exponential decay').info('Default = 0.9; controls the rate of exponential decay for reforge Cosine-exponential Blend scheduler'),
    
    "reforge_phi_power": OptionInfo(2.0, "Reforge Phi scheduler - power", gr.Slider, {"minimum": 1.0, "maximum": 5.0, "step": 0.1}, infotext='Reforge Phi scheduler power').info('Default = 2.0; controls the power of the phi-based curve for reforge Phi scheduler'),
    
    "reforge_laplace_mu": OptionInfo(0.0, "Reforge Laplace scheduler - mu", gr.Slider, {"minimum": -1.0, "maximum": 1.0, "step": 0.1}, infotext='Reforge Laplace scheduler mu').info('Default = 0.0; controls the location parameter of the Laplace distribution for reforge Laplace scheduler'),
    "reforge_laplace_beta": OptionInfo(0.5, "Reforge Laplace scheduler - beta", gr.Slider, {"minimum": 0.1, "maximum": 2.0, "step": 0.1}, infotext='Reforge Laplace scheduler beta').info('Default = 0.5; controls the scale parameter of the Laplace distribution for reforge Laplace scheduler'),
    
    "reforge_karras_dynamic_rho": OptionInfo(7.0, "Reforge Karras Dynamic scheduler - base rho", gr.Slider, {"minimum": 1.0, "maximum": 20.0, "step": 0.1}, infotext='Reforge Karras Dynamic scheduler base rho').info('Default = 7.0; controls the base shape of the dynamic noise schedule for reforge Karras Dynamic scheduler'),
    
    "reforge_sinusoidal_sf_factor": OptionInfo(3.5, "Reforge Sinusoidal SF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.1}, infotext='Reforge Sinusoidal SF scheduler factor').info('Default = 3.5; controls the shape of the sinusoidal curve for reforge Sinusoidal SF scheduler'),
    
    "reforge_invcosinusoidal_sf_factor": OptionInfo(3.5, "Reforge Invcosinusoidal SF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.1}, infotext='Reforge Invcosinusoidal SF scheduler factor').info('Default = 3.5; controls the shape of the inverse cosinusoidal curve for reforge Invcosinusoidal SF scheduler'),
    
    "reforge_react_cosinusoidal_dynsf_factor": OptionInfo(2.15, "Reforge React Cosinusoidal DynSF scheduler - factor", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.05}, infotext='Reforge React Cosinusoidal DynSF scheduler factor').info('Default = 2.15; controls the dynamic scaling factor for reforge React Cosinusoidal DynSF scheduler'),
}))

options_templates.update(options_section(('sampler-params', "reForge Sampler Parameters", "sd"), {
    # Basic Samplers Section
    "basic_samplers_group": OptionHTML("""<br><h2 style='text-align: center'>Basic Samplers</h2>
        Configuration options for fundamental sampling methods."""),

    "ancestral_group": OptionHTML("<br><h3>Ancestral Eta Setting</h3>"),
    "ancestral_eta": OptionInfo(1.0, "Ancestral sampling eta", gr.Slider, {"minimum": -1.0, "maximum": 3.0, "step": 0.01}, infotext='Ancestral eta').info("Controls noise levels in ancestral sampling. 0 = no noise, 1 = default, higher values = more noise. Applies only to ancestral samplers"),
    
    # Euler Parameters
    "euler_group": OptionHTML("<br><h3>Euler Settings</h3>"),
    "euler_og_s_churn": OptionInfo(0.0, "Euler - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler s_churn'),
    "euler_og_s_tmin": OptionInfo(0.0, "Euler - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler s_tmin'),
    "euler_og_s_noise": OptionInfo(1.0, "Euler - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler s_noise'),

    # Euler Ancestral Parameters
    "euler_ancestral_group": OptionHTML("<br><h3>Euler Ancestral Settings</h3>"),
    "euler_ancestral_og_eta": OptionInfo(1.0, "Euler Ancestral - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler Ancestral eta'),
    "euler_ancestral_og_s_noise": OptionInfo(1.0, "Euler Ancestral - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler Ancestral s_noise'),

    # Heun Parameters
    "heun_group": OptionHTML("<br><h3>Heun Settings</h3>"),
    "heun_og_s_churn": OptionInfo(0.0, "Heun - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Heun s_churn'),
    "heun_og_s_tmin": OptionInfo(0.0, "Heun - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Heun s_tmin'),
    "heun_og_s_noise": OptionInfo(1.0, "Heun - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Heun s_noise'),

    # Advanced DPM-Solver Section
    "dpm_solver_group": OptionHTML("""<br><h2 style='text-align: center'>DPM-Solver Family</h2>
        Advanced configurations for DPM-based samplers."""),

    # DPM++ 2S Parameters
    "dpm_2s_ancestral_group": OptionHTML("<br><h3>DPM++ 2S Ancestral Settings</h3>"),
    "dpm_2s_ancestral_og_eta": OptionInfo(1.0, "DPM++ 2S Ancestral - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ 2S Ancestral eta'),
    "dpm_2s_ancestral_og_s_noise": OptionInfo(1.0, "DPM++ 2S Ancestral - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2S Ancestral s_noise'),

    # DPM++ SDE Parameters
    "dpm_sde_group": OptionHTML("<br><h3>DPM++ SDE Settings</h3>"),
    "dpmpp_sde_og_eta": OptionInfo(1.0, "DPM++ SDE - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ SDE eta'),
    "dpmpp_sde_og_s_noise": OptionInfo(1.0, "DPM++ SDE - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ SDE s_noise'),
    "dpmpp_sde_og_r": OptionInfo(0.5, "DPM++ SDE - r", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ SDE r'),

    # DPM++ 2M Parameters
    "dpm_2m_group": OptionHTML("<br><h3>DPM++ 2M Settings</h3>"),
    "dpmpp_2m_sde_og_eta": OptionInfo(1.0, "DPM++ 2M SDE - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ 2M SDE eta'),
    "dpmpp_2m_sde_og_s_noise": OptionInfo(1.0, "DPM++ 2M SDE - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2M SDE s_noise'),
    "dpmpp_2m_sde_og_solver_type": OptionInfo("midpoint", "DPM++ 2M SDE - solver_type", gr.Dropdown, {"choices": ["heun", "midpoint"]}, infotext='DPM++ 2M SDE solver_type'),

    # DPM++ 3M Parameters
    "dpm_3m_group": OptionHTML("<br><h3>DPM++ 3M SDE Settings</h3>"),
    "dpmpp_3m_sde_og_eta": OptionInfo(1.0, "DPM++ 3M SDE - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ 3M SDE eta'),
    "dpmpp_3m_sde_og_s_noise": OptionInfo(1.0, "DPM++ 3M SDE - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 3M SDE s_noise'),

    # Other Advanced Samplers Section
    "dpm_solver_group": OptionHTML("""<br><h2 style='text-align: center'>Other Advanced Samplers Family</h2>
        Advanced configurations for some samplers."""),

    "heunpp2_group": OptionHTML("<br><h3>HeunPP2 Settings</h3>"),
    "heunpp2_s_churn": OptionInfo(0.0, "HeunPP2 - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='HeunPP2 s_churn'),
    "heunpp2_s_tmin": OptionInfo(0.0, "HeunPP2 - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='HeunPP2 s_tmin'),
    "heunpp2_s_noise": OptionInfo(1.0, "HeunPP2 - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='HeunPP2 s_noise'),

    # IPNDM Parameters
    "ipndm_group": OptionHTML("<br><h3>IPNDM Settings</h3>"),
    "ipndm_max_order": OptionInfo(4, "IPNDM - max_order", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}, infotext='IPNDM max_order'),

    # IPNDM_V Parameters
    "ipndm_v_group": OptionHTML("<br><h3>IPNDM-V Settings</h3>"),
    "ipndm_v_max_order": OptionInfo(4, "IPNDM-V - max_order", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}, infotext='IPNDM-V max_order'),

    # DEIS Parameters
    "deis_group": OptionHTML("<br><h3>DEIS Settings</h3>"),
    "deis_max_order": OptionInfo(3, "DEIS - max_order", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}, infotext='DEIS max_order'),
    "deis_mode": OptionInfo("tab", "DEIS - mode", gr.Dropdown, {"choices": ["tab", "newton"]}, infotext='DEIS mode'),

    # Kohaku LoNyu Parameters
    "kohaku_group": OptionHTML("<br><h3>Kohaku LoNyu Settings</h3>"),
    "kohaku_lonyu_yog_s_churn": OptionInfo(0.0, "Kohaku LoNyu Yog - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog s_churn').info('Default = 0.0; amount of noise to add during sampling'),
    "kohaku_lonyu_yog_s_tmin": OptionInfo(0.0, "Kohaku LoNyu Yog - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog s_tmin').info('Default = 0.0; minimum sigma threshold for noise'),
    "kohaku_lonyu_yog_s_noise": OptionInfo(1.0, "Kohaku LoNyu Yog - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Kohaku LoNyu Yog s_noise').info('Default = 1.0; noise scaling factor'),
    "kohaku_lonyu_yog_eta": OptionInfo(1.0, "Kohaku LoNyu Yog - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog eta').info('Default = 1.0; eta parameter for controlling the stochastic sampling process'),

    # CFG++ Enhanced Section
    "cfg_enhanced_group": OptionHTML("""<br><h2 style='text-align: center'>CFG++ Enhanced Samplers</h2>
        Advanced configurations for samplers with CFG++ enhancement."""),

    # Euler CFG++ Parameters
    "euler_cfg_group": OptionHTML("<br><h3>Euler Ancestral CFG++ Settings</h3>"),
    "euler_ancestral_cfg_pp_eta": OptionInfo(1.0, "Euler Ancestral CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='Euler Ancestral CFG++ eta'),
    "euler_ancestral_cfg_pp_s_noise": OptionInfo(1.0, "Euler Ancestral CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler Ancestral CFG++ s_noise'),

    # DPM++ CFG++ Parameters
    "dpmpp_2s_ancestral_cfg_pp_group": OptionHTML("<br><h3>DPM++ 2S Ancestral CFG++ Settings</h3>"),
    "dpmpp_2s_ancestral_cfg_pp_eta": OptionInfo(1.0, "DPM++ 2S Ancestral CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='DPM++ 2S Ancestral CFG++ eta'),
    "dpmpp_2s_ancestral_cfg_pp_s_noise": OptionInfo(1.0, "DPM++ 2S Ancestral CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2S Ancestral CFG++ s_noise'),

    # DPM++ SDE CFG++ Parameters
    "dpm_sde_cfg_group": OptionHTML("<br><h3>DPM++ SDE CFG++ Settings</h3>"),
    "dpmpp_sde_cfg_pp_eta": OptionInfo(1.0, "DPM++ SDE CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='DPM++ SDE CFG++ eta'),
    "dpmpp_sde_cfg_pp_s_noise": OptionInfo(1.0, "DPM++ SDE CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ SDE CFG++ s_noise'),
    "dpmpp_sde_cfg_pp_r": OptionInfo(0.5, "DPM++ SDE CFG++ - r", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ SDE CFG++ r'),

    "dpmpp_2s_ancestral_dyn_group": OptionHTML("<br><h3>DPM++ 2S Ancestral Dyn CFG++ Settings</h3>"),
    "dpmpp_2s_ancestral_dyn_eta": OptionInfo(1.0, "DPM++ 2S Ancestral Dynamic CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='DPM++ 2S Ancestral Dynamic CFG++ eta').info('Default = 1.0; eta for DPM++ 2S Ancestral Dynamic sampler'),
    "dpmpp_2s_ancestral_dyn_s_noise": OptionInfo(1.0, "DPM++ 2S Ancestral Dynamic CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2S Ancestral Dynamic CFG++ s_noise').info('Default = 1.0; s_noise for DPM++ 2S Ancestral Dynamic sampler'),

    "dpmpp_2s_ancestral_intern_group": OptionHTML("<br><h3>DPM++ 2S Ancestral Intern CFG++ Settings</h3>"),
    "dpmpp_2s_ancestral_intern_eta": OptionInfo(1.0, "DPM++ 2S Ancestral Internal CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='DPM++ 2S Ancestral Internal CFG++ eta').info('Default = 1.0; eta for DPM++ 2S Ancestral Internal sampler'),
    "dpmpp_2s_ancestral_intern_s_noise": OptionInfo(1.0, "DPM++ 2S Ancestral Internal CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2S Ancestral Internal CFG++ s_noise').info('Default = 1.0; s_noise for DPM++ 2S Ancestral Internal sampler'),

    "dpmpp_3m_sde_cfg_pp_group": OptionHTML("<br><h3>DPM++ 3M SDE CFG++ Settings</h3>"),
    "dpmpp_3m_sde_cfg_pp_eta": OptionInfo(1.0, "DPM++ 3M SDE CFG++ - eta", gr.Slider, {"minimum": -1.0001, "maximum": 2.0, "step": 0.0001}, infotext='DPM++ 3M SDE CFG++ eta').info('Default = 1.0; eta for DPM++ 3M SDE sampler with CFG++'),
    "dpmpp_3m_sde_cfg_pp_s_noise": OptionInfo(1.0, "DPM++ 3M SDE CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 3M SDE CFG++ s_noise').info('Default = 1.0; s_noise for DPM++ 3M SDE sampler with CFG++'),

    # # Kohaku LoNyu CFG++ Parameters
    # "kohaku_lonyugroup": OptionHTML("<br><h3>Kohaku LoNyu CFG++ Settings</h3>"),
    # "kohaku_lonyu_yog_s_cfgpp_churn": OptionInfo(0.0, "Kohaku LoNyu Yog - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog CFG++ s_churn').info('Default = 0.0; amount of noise to add during sampling'),
    # "kohaku_lonyu_yog_s_cfgpp_tmin": OptionInfo(0.0, "Kohaku LoNyu Yog - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog CFG++ s_tmin').info('Default = 0.0; minimum sigma threshold for noise'), 
    # "kohaku_lonyu_yog_s_cfgpp_noise": OptionInfo(1.0, "Kohaku LoNyu Yog - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Kohaku LoNyu Yog CFG++ s_noise').info('Default = 1.0; noise scaling factor'),
    # "kohaku_lonyu_yog_cfgpp_eta": OptionInfo(1.0, "Kohaku LoNyu Yog - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Kohaku LoNyu Yog CFG++ eta').info('Default = 1.0; eta parameter'),

    # Dynamic Samplers Section
    "dynamic_samplers_group": OptionHTML("""<br><h2 style='text-align: center'>Dynamic Samplers</h2>
        Advanced configurations for samplers with dynamic thresholding."""),

    # Euler Dynamic Parameters
    "euler_dynamic_group": OptionHTML("<br><h3>Euler Dynamic Settings</h3>"),
    "euler_dy_s_churn": OptionInfo(0.0, "Euler DY - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY s_churn'),
    "euler_dy_s_tmin": OptionInfo(0.0, "Euler DY - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY s_tmin'),
    "euler_dy_s_noise": OptionInfo(1.0, "Euler DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler DY s_noise'),

    # Euler SMEA Dynamic Parameters
    "euler_smea_dynamic_group": OptionHTML("<br><h3>Euler SMEA Dynamic Settings</h3>"),
    "euler_smea_dy_s_churn": OptionInfo(0.0, "Euler SMEA DY - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler SMEA DY s_churn'),
    "euler_smea_dy_s_tmin": OptionInfo(0.0, "Euler SMEA DY - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler SMEA DY s_tmin'),
    "euler_smea_dy_s_noise": OptionInfo(1.0, "Euler SMEA DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler SMEA DY s_noise'),

    # Euler Negative Parameters
    "euler_negative_group": OptionHTML("<br><h3>Euler Negative Settings</h3>"),
    "euler_negative_s_churn": OptionInfo(0.0, "Euler Negative - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler Negative s_churn').info('Default = 0.0; s_churn for Euler Negative sampler'),
    "euler_negative_s_tmin": OptionInfo(0.0, "Euler Negative - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler Negative s_tmin').info('Default = 0.0; s_tmin for Euler Negative sampler'),
    "euler_negative_s_noise": OptionInfo(1.0, "Euler Negative - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler Negative s_noise').info('Default = 1.0; s_noise for Euler Negative sampler'),

    # Euler Dynamic Negative Parameters
    "euler_dy_negative_group": OptionHTML("<br><h3>Euler Dynamic Negative Settings</h3>"),
    "euler_dy_negative_s_churn": OptionInfo(0.0, "Euler DY Negative - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY Negative s_churn').info('Default = 0.0; s_churn for Euler DY Negative sampler'),
    "euler_dy_negative_s_tmin": OptionInfo(0.0, "Euler DY Negative - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY Negative s_tmin').info('Default = 0.0; s_tmin for Euler DY Negative sampler'),
    "euler_dy_negative_s_noise": OptionInfo(1.0, "Euler DY Negative - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler DY Negative s_noise').info('Default = 1.0; s_noise for Euler DY Negative sampler'),

    # DPM++ Dynamic Parameters
    "dpmpp_2m_dy_group": OptionHTML("<br><h3>DPM++ 2m Dynamic Settings</h3>"),
    "dpmpp_2m_dy_s_noise": OptionInfo(1.0, "DPM++ 2M DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2M DY s_noise'),
    "dpmpp_2m_dy_s_dy_pow": OptionInfo(-1.0, "DPM++ 2M DY - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='DPM++ 2M DY dynamic power'),
    "dpmpp_2m_dy_s_extra_steps": OptionInfo(True, "DPM++ 2M DY - extra steps", gr.Checkbox, {}, infotext='DPM++ 2M DY extra steps'),

    # DPM++ 3M Dynamic Parameters
    "dpm_3m_dynamic_group": OptionHTML("<br><h3>DPM++ 3M Dynamic Settings</h3>"),
    "dpmpp_3m_dy_s_noise": OptionInfo(1.0, "DPM++ 3M DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 3M DY s_noise'),
    "dpmpp_3m_dy_s_dy_pow": OptionInfo(-1.0, "DPM++ 3M DY - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='DPM++ 3M DY dynamic power'),
    "dpmpp_3m_dy_s_extra_steps": OptionInfo(True, "DPM++ 3M DY - extra steps", gr.Checkbox, {}, infotext='DPM++ 3M DY extra steps'),

    #DPM++ 2M SDE Dynamic Parameters
    "dpm_2M_SDE_DY_group": OptionHTML("<br><h3>DPM++ 2M SDE Dynamic Settings</h3>"),
    "dpmpp_2m_sde_dy_eta": OptionInfo(1.0, "DPM++ 2M SDE DY - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ 2M SDE DY eta').info('Default = 1.0; eta for DPM++ 2M SDE dynamic sampler'),
    "dpmpp_2m_sde_dy_s_noise": OptionInfo(1.0, "DPM++ 2M SDE DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2M SDE DY s_noise').info('Default = 1.0; noise scaling for DPM++ 2M SDE dynamic sampler'),
    "dpmpp_2m_sde_dy_solver_type": OptionInfo("midpoint", "DPM++ 2M SDE DY - solver type", gr.Dropdown, {"choices": ["heun", "midpoint"]}, infotext='DPM++ 2M SDE DY solver type').info('Default = midpoint; solver type for DPM++ 2M SDE dynamic sampler'),
    "dpmpp_2m_sde_dy_s_dy_pow": OptionInfo(-1.0, "DPM++ 2M SDE DY - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='DPM++ 2M SDE DY dynamic power').info('Default = -1.0; power for dynamic thresholding'),
    "dpmpp_2m_sde_dy_s_extra_steps": OptionInfo(True, "DPM++ 2M SDE DY - extra steps", gr.Checkbox, {}, infotext='DPM++ 2M SDE DY extra steps').info('Whether to use extra steps in DPM++ 2M SDE dynamic sampler'),

    #DPM++ 3M SDE Dynamic Parameters
    "dpm_3M_SDE_DY_group": OptionHTML("<br><h3>DPM++ 3M SDE Dynamic Settings</h3>"),
    "dpmpp_3m_sde_dy_eta": OptionInfo(1.0, "DPM++ 3M SDE DY - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='DPM++ 3M SDE DY eta').info('Default = 1.0; eta for DPM++ 3M SDE dynamic sampler'),
    "dpmpp_3m_sde_dy_s_noise": OptionInfo(1.0, "DPM++ 3M SDE DY - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 3M SDE DY s_noise').info('Default = 1.0; noise scaling for DPM++ 3M SDE dynamic sampler'),
    "dpmpp_3m_sde_dy_s_dy_pow": OptionInfo(-1.0, "DPM++ 3M SDE DY - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='DPM++ 3M SDE DY dynamic power').info('Default = -1.0; power for dynamic thresholding'),
    "dpmpp_3m_sde_dy_s_extra_steps": OptionInfo(True, "DPM++ 3M SDE DY - extra steps", gr.Checkbox, {}, infotext='DPM++ 3M SDE DY extra steps').info('Whether to use extra steps in DPM++ 3M SDE dynamic sampler'),

    # Dynamic CFG++ Section
    "dynamic_cfg_group": OptionHTML("""<br><h2 style='text-align: center'>Dynamic CFG++ Samplers</h2>
        Advanced configurations for dynamic samplers with CFG++ enhancement."""),

    # Euler Dynamic CFG++ Parameters
    "euler_dynamic_cfg_group": OptionHTML("<br><h3>Euler Dynamic CFG++ Settings</h3>"),
    "euler_dy_cfg_pp_s_churn": OptionInfo(0.0, "Euler DY CFG++ - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY CFG++ s_churn'),
    "euler_dy_cfg_pp_s_tmin": OptionInfo(0.0, "Euler DY CFG++ - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler DY CFG++ s_tmin'),
    "euler_dy_cfg_pp_s_noise": OptionInfo(1.0, "Euler DY CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler DY CFG++ s_noise'),
    "euler_dy_cfg_pp_s_dy_pow": OptionInfo(-1.0, "Euler DY CFG++ - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='Euler DY CFG++ dynamic power'),
    "euler_dy_cfg_pp_s_extra_steps": OptionInfo(True, "Euler DY CFG++ - extra steps", gr.Checkbox, {}, infotext='Euler DY CFG++ extra steps'),

    # Euler SMEA Dynamic CFG++ Parameters
    "euler_smea_dy_cfg_pp_group": OptionHTML("<br><h3>Euler SMEA DY CFG++ Settings</h3>"),
    "euler_smea_dy_cfg_pp_s_churn": OptionInfo(0.0, "Euler SMEA DY CFG++ - s_churn", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler SMEA DY CFG++ s_churn').info('Default = 0.0; amount of noise to add during sampling'),
    "euler_smea_dy_cfg_pp_s_tmin": OptionInfo(0.0, "Euler SMEA DY CFG++ - s_tmin", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler SMEA DY CFG++ s_tmin').info('Default = 0.0; minimum sigma threshold for noise'),
    "euler_smea_dy_cfg_pp_s_noise": OptionInfo(1.0, "Euler SMEA DY CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler SMEA DY CFG++ s_noise').info('Default = 1.0; noise scaling factor'),
    "euler_smea_dy_cfg_pp_s_dy_pow": OptionInfo(-1.0, "Euler SMEA DY CFG++ - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='Euler SMEA DY CFG++ dynamic power').info('Default = -1.0; power for dynamic thresholding'),
    "euler_smea_dy_cfg_pp_s_extra_steps": OptionInfo(True, "Euler SMEA DY CFG++ - extra steps", gr.Checkbox, {}, infotext='Euler SMEA DY CFG++ extra steps').info('Whether to use extra steps in sampling'),

    # Euler a Dynamic CFG++ Parameters
    "euler_ancestral_dy_cfg_pp_group": OptionHTML("<br><h3>Euler Ancestral DY CFG++ Settings</h3>"),
    "euler_ancestral_dy_cfg_pp_eta": OptionInfo(1.0, "Euler Ancestral DY CFG++ - eta", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.01}, infotext='Euler Ancestral DY CFG++ eta').info('Default = 1.0; eta parameter for noise schedule'),
    "euler_ancestral_dy_cfg_pp_s_noise": OptionInfo(1.0, "Euler Ancestral DY CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='Euler Ancestral DY CFG++ s_noise').info('Default = 1.0; noise scaling factor'),
    "euler_ancestral_dy_cfg_pp_s_dy_pow": OptionInfo(-1.0, "Euler Ancestral DY CFG++ - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='Euler Ancestral DY CFG++ dynamic power').info('Default = -1.0; power for dynamic thresholding'),
    "euler_ancestral_dy_cfg_pp_s_extra_steps": OptionInfo(True, "Euler Ancestral DY CFG++ - extra steps", gr.Checkbox, {}, infotext='Euler Ancestral DY CFG++ extra steps').info('Whether to use extra steps in sampling'),

    # DPM++ 2M DY CFG++ Parameters
    "dpmpp_2m_dy_cfg_pp_group": OptionHTML("<br><h3>Euler Ancestral DY CFG++ Settings</h3>"),
    "dpmpp_2m_dy_cfg_pp_s_noise": OptionInfo(1.0, "DPM++ 2M DY CFG++ - s_noise", gr.Slider, {"minimum": -1.0, "maximum": 2.0, "step": 0.1}, infotext='DPM++ 2M DY CFG++ s_noise').info('Default = 1.0; noise scaling factor'),
    "dpmpp_2m_dy_cfg_pp_s_dy_pow": OptionInfo(-1.0, "DPM++ 2M DY CFG++ - dynamic power", gr.Slider, {"minimum": -2.0, "maximum": 5.0, "step": 0.1}, infotext='DPM++ 2M DY CFG++ dynamic power').info('Default = -1.0; power for dynamic thresholding'),
    "dpmpp_2m_dy_cfg_pp_s_extra_steps": OptionInfo(True, "DPM++ 2M DY CFG++ - extra steps", gr.Checkbox, {}, infotext='DPM++ 2M DY CFG++ extra steps').info('Whether to use extra steps in sampling'),

    # ODE Solvers Section
    "ode_solvers_group": OptionHTML("""<br><h2 style='text-align: center'>ODE Solvers</h2>
        Configuration options for Ordinary Differential Equation based samplers."""),

    # ODE Bosh3 Parameters
    "ode_bosh3_group": OptionHTML("<br><h3>ODE Bosh3 Settings</h3>"),
    "ode_bosh3_rtol": OptionInfo(-2.5, "ODE Bosh3 - log relative tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Bosh3 rtol').info('Default = -2.5; log10 of relative tolerance for Bosh3 ODE solver'),
    "ode_bosh3_atol": OptionInfo(-3.5, "ODE Bosh3 - log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Bosh3 atol').info('Default = -3.5; log10 of absolute tolerance for Bosh3 ODE solver'),
    "ode_bosh3_max_steps": OptionInfo(250, "ODE Bosh3 - max steps", gr.Slider, {"minimum": 1, "maximum": 500, "step": 1}, infotext='ODE Bosh3 max steps').info('Default = 250; maximum number of steps for Bosh3 ODE solver'),

    # ODE Fehlberg2 Parameters
    "ode_fehlberg2_group": OptionHTML("<br><h3>ODE Fehlberg2 Settings</h3>"),
    "ode_fehlberg2_rtol": OptionInfo(-4.0, "ODE Fehlberg2 - log relative tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Fehlberg2 rtol').info('Default = -4.0; log10 of relative tolerance for Fehlberg2 ODE solver'),
    "ode_fehlberg2_atol": OptionInfo(-6.0, "ODE Fehlberg2 - log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Fehlberg2 atol').info('Default = -6.0; log10 of absolute tolerance for Fehlberg2 ODE solver'),
    "ode_fehlberg2_max_steps": OptionInfo(250, "ODE Fehlberg2 - max steps", gr.Slider, {"minimum": 1, "maximum": 500, "step": 1}, infotext='ODE Fehlberg2 max steps').info('Default = 250; maximum number of steps for Fehlberg2 ODE solver'),

    # ODE Adaptive Heun Parameters
    "ode_adapt_heun_group": OptionHTML("<br><h3>ODE Adaptive Heun Settings</h3>"),
    "ode_adaptive_heun_rtol": OptionInfo(-2.5, "ODE Adaptive Heun - log relative tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Adaptive Heun rtol').info('Default = -2.5; log10 of relative tolerance for Adaptive Heun ODE solver'),
    "ode_adaptive_heun_atol": OptionInfo(-3.5, "ODE Adaptive Heun - log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Adaptive Heun atol').info('Default = -3.5; log10 of absolute tolerance for Adaptive Heun ODE solver'),
    "ode_adaptive_heun_max_steps": OptionInfo(250, "ODE Adaptive Heun - max steps", gr.Slider, {"minimum": 1, "maximum": 500, "step": 1}, infotext='ODE Adaptive Heun max steps').info('Default = 250; maximum number of steps for Adaptive Heun ODE solver'),

    # ODE Dopri5 Parameters
    "ode_dopri5_group": OptionHTML("<br><h3>ODE Dopri5 Settings</h3>"),
    "ode_dopri5_rtol": OptionInfo(-2.0, "ODE Dopri5 - log relative tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Dopri5 rtol').info('Default = -2.0; log10 of relative tolerance for Dopri5 ODE solver'),
    "ode_dopri5_atol": OptionInfo(-3.0, "ODE Dopri5 - log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Dopri5 atol').info('Default = -3.0; log10 of absolute tolerance for Dopri5 ODE solver'),
    "ode_dopri5_max_steps": OptionInfo(250, "ODE Dopri5 - max steps", gr.Slider, {"minimum": 1, "maximum": 500, "step": 1}, infotext='ODE Dopri5 max steps').info('Default = 250; maximum number of steps for Dopri5 ODE solver'),

    # Custom ODE Parameters
    "ode_custom_group": OptionHTML("<br><h3>Custom ODE Settings</h3>"),
    "ode_custom_solver": OptionInfo("dopri5", "ODE Custom - Solver", gr.Dropdown, {"choices": k_diffusion_sampling.ALL_SOLVERS}, infotext='ODE Custom solver').info('Choose the ODE solver method'),
    "ode_custom_rtol": OptionInfo(-3.0, "ODE Custom - log relative tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Custom rtol').info('Default = -3.0; log10 of relative tolerance for adaptive ODE solvers'),
    "ode_custom_atol": OptionInfo(-4.0, "ODE Custom - log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": 0, "step": 0.1}, infotext='ODE Custom atol').info('Default = -4.0; log10 of absolute tolerance for adaptive ODE solvers'),
    "ode_custom_max_steps": OptionInfo(250, "ODE Custom - max steps", gr.Slider, {"minimum": 1, "maximum": 500, "step": 1}, infotext='ODE Custom max steps').info('Default = 250; maximum number of steps for ODE solver'),

    "dpm_fast_options": OptionHTML("<br><h3>DPM/DPM2 options</h3>"),
    "dpm_fast_s_noise": OptionInfo(1.0, "DPM Fast s_noise", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "dpm_adaptive_s_noise": OptionInfo(1.0, "DPM Adaptive s_noise", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "dpm2_ancestral_s_noise": OptionInfo(1.0, "DPM2 Ancestral s_noise", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),

    }))

options_templates.update(options_section(('sampler-params', "Custom Sampler Parameters", "sd"), {
    # Common sampler parameters
    "custom_sampler_name": OptionInfo("euler_comfy", "Custom Sampler - Type", gr.Dropdown, {
        "choices": [
            "euler_comfy", "euler_ancestral_comfy", "heun_comfy", 
            "dpmpp_2s_ancestral_comfy", "dpmpp_sde_comfy", "dpmpp_2m_comfy",
            "dpmpp_2m_sde_comfy", "dpmpp_3m_sde_comfy", "euler_ancestral_turbo",
            "dpmpp_2m_turbo", "dpmpp_2m_sde_turbo", "ddpm", "heunpp2",
            "ipndm", "ipndm_v", "deis", "euler_cfg_pp", "euler_ancestral_cfg_pp",
            "sample_euler_ancestral_RF", "dpmpp_2s_ancestral_cfg_pp",
            "sample_dpmpp_2s_ancestral_RF", "dpmpp_2s_ancestral_cfg_pp_dyn",
            "dpmpp_2s_ancestral_cfg_pp_intern", "dpmpp_sde_cfg_pp",
            "dpmpp_2m_cfg_pp", "dpmpp_3m_sde_cfg_pp", "dpmpp_2m_dy",
            "dpmpp_3m_dy", "dpmpp_3m_sde_dy", "euler_dy_cfg_pp",
            "euler_smea_dy_cfg_pp", "euler_ancestral_dy_cfg_pp",
            "dpmpp_2m_dy_cfg_pp", "clyb_4m_sde_momentumized",
            "res_solver", "kohaku_lonyu_yog_cfg_pp"
        ]
    }, infotext='Custom sampler type').info('The sampling algorithm to use'),

    # Sampler specific parameters
    "custom_sampler_eta": OptionInfo(1.0, "Custom Sampler - eta", gr.Slider, {"minimum": -2.0, "maximum": 2.0, "step": 0.01}, infotext='Custom sampler eta').info('Default = 1.0; Controls the scheduler randomness/noise level'),
    
    "custom_sampler_s_noise": OptionInfo(1.0, "Custom Sampler - s_noise", gr.Slider, {"minimum": -2.0, "maximum": 3.0, "step": 0.1}, infotext='Custom sampler s_noise').info('Default = 1.0; Controls the noise level during sampling'),
    
    "custom_sampler_solver_type": OptionInfo("midpoint", "Custom Sampler - solver type", gr.Dropdown, {"choices": ["midpoint", "heun"]}, infotext='Custom sampler solver type').info('Default = midpoint; The type of solver to use'),
    
    "custom_sampler_r": OptionInfo(0.5, "Custom Sampler - r value", gr.Slider, {"minimum": -2.0, "maximum": 2.0, "step": 0.1}, infotext='Custom sampler r').info('Default = 0.5; Controls the step size ratio'),

    # CFG parameters  
    "custom_cfg_conds": OptionInfo(8.0, "Custom Sampler - CFG scale", gr.Slider, {"minimum": -2.0, "maximum": 100.0, "step": 0.1}, infotext='Custom CFG scale').info('Default = 8.0; Controls the strength of the guidance'),
    
    "custom_cfg_cond2_negative": OptionInfo(8.0, "Custom Sampler - Secondary CFG scale", gr.Slider, {"minimum": -2.0, "maximum": 100.0, "step": 0.1}, infotext='Custom secondary CFG scale').info('Default = 8.0; Controls the strength of the secondary guidance'),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing", "postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts(filter_out_extra_only=True)]}),
    'postprocessing_disable_in_extras': OptionInfo([], "Disable postprocessing operations in extras tab", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts(filter_out_main_ui_only=True)]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts(filter_out_main_ui_only=True)]}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    'postprocessing_existing_caption_action': OptionInfo("Ignore", "Action for existing captions", gr.Radio, {"choices": ["Ignore", "Keep", "Prepend", "Append"]}).info("when generating captions using postprocessing; Ignore = use generated; Keep = use original; Prepend/Append = combine both"),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
    "restore_config_state_file": OptionInfo("", "Config state file to restore from, under 'config-states/' folder"),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))

options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
    "forge_canvas_plain": OptionInfo(False, "ForgeCanvas: use plain background").needs_reload_ui(),
    "forge_canvas_toolbar_always": OptionInfo(False, "ForgeCanvas: toolbar always visible").needs_reload_ui(),
}))