from ldm_patched.helpers.torch_compile import set_torch_compile_wrapper


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "backend": (["inductor", "cudagraphs"],),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend):
        m = model.clone()
        set_torch_compile_wrapper(model=m, backend=backend)
        return (m, )

# Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
