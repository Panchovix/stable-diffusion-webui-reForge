import gradio as gr
from gradio import processing_utils
import warnings
import PIL.ImageOps

from modules import scripts, ui_tempdir, patches


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]

    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop('tooltip', None)

    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_IOComponent_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Block_get_config(self):
    config = original_Block_get_config(self)

    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    config.pop('example_inputs', None)

    return config


def BlockContext_init(self, *args, **kwargs):
    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Blocks_get_config_file(self, *args, **kwargs):
    config = original_Blocks_get_config_file(self, *args, **kwargs)

    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}

    return config


def Image_upload_handler(self, x):
    """Handles conversion of uploaded images to RGB"""
    if isinstance(x, dict) and 'image' in x:
        output_image = x['image'].convert('RGB')
        return output_image
    return x

def Image_custom_preprocess(self, x):
    """Custom preprocessing for images with masks"""
    if x is None:
        return x
        
    mask = ""
    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        if isinstance(x, dict):
            x, mask = x["image"], x["mask"]
            
    if not isinstance(x, str):
        return x
        
    im = processing_utils.decode_base64_to_image(x)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = im.convert(self.image_mode)
        
    if self.shape is not None:
        im = processing_utils.resize_and_crop(im, self.shape)
        
    if self.invert_colors:
        im = PIL.ImageOps.invert(im)
        
    if (self.source == "webcam" 
        and self.mirror_webcam is True 
        and self.tool != "color-sketch"):
        im = PIL.ImageOps.mirror(im)
        
    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        mask_im = None
        if mask is not None:
            mask_im = processing_utils.decode_base64_to_image(mask)
        return {
            "image": self._format_image(im),
            "mask": self._format_image(mask_im)
        }
        
    return self._format_image(im)

def Image_init_extension(self, *args, **kwargs):
    """Extended initialization for Image components"""
    res = original_Image_init(self, *args, **kwargs)
    
    # Only apply to inpaint with mask component for now
    if getattr(self, 'elem_id', None) == 'img2maskimg':
        self.upload(
            fn=Image_upload_handler.__get__(self, gr.Image),
            inputs=self,
            outputs=self
        )
        self.preprocess = Image_custom_preprocess.__get__(self, gr.Image)
    
    return res


original_IOComponent_init = patches.patch(__name__, obj=gr.components.IOComponent, field="__init__", replacement=IOComponent_init)
original_Block_get_config = patches.patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
original_BlockContext_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)
original_Image_init = patches.patch(__name__, obj=gr.components.Image, field="__init__", replacement=Image_init_extension)


ui_tempdir.install_ui_tempdir_override()
