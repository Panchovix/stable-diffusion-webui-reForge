import ldm_patched.utils.path_utils
import ldm_patched.modules.sd
import ldm_patched.modules.model_management


class QuadrupleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (ldm_patched.utils.path_utils.get_filename_list("text_encoders"), ),
                              "clip_name2": (ldm_patched.utils.path_utils.get_filename_list("text_encoders"), ),
                              "clip_name3": (ldm_patched.utils.path_utils.get_filename_list("text_encoders"), ),
                              "clip_name4": (ldm_patched.utils.path_utils.get_filename_list("text_encoders"), )
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nhidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4):
        clip_path1 = ldm_patched.utils.path_utils.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = ldm_patched.utils.path_utils.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = ldm_patched.utils.path_utils.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = ldm_patched.utils.path_utils.get_full_path_or_raise("text_encoders", clip_name4)
        clip = ldm_patched.modules.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4], embedding_directory=ldm_patched.utils.path_utils.get_folder_paths("embeddings"))
        return (clip,)

class CLIPTextEncodeHiDream:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "llama": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, clip_l, clip_g, t5xxl, llama):

        tokens = clip.tokenize(clip_g)
        tokens["l"] = clip.tokenize(clip_l)["l"]
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        tokens["llama"] = clip.tokenize(llama)["llama"]
        return (clip.encode_from_tokens_scheduled(tokens), )

# Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
NODE_CLASS_MAPPINGS = {
    "QuadrupleCLIPLoader": QuadrupleCLIPLoader,
    "CLIPTextEncodeHiDream": CLIPTextEncodeHiDream,
}
