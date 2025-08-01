# import torch
# import ldm_patched.modules.model_management

# from kornia.morphology import dilation, erosion, opening, closing, gradient, top_hat, bottom_hat
# import kornia.color


# class Morphology:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"image": ("IMAGE",),
#                                 "operation": (["erode",  "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],),
#                                 "kernel_size": ("INT", {"default": 3, "min": 3, "max": 999, "step": 1}),
#                                 }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process"

#     CATEGORY = "image/postprocessing"

#     def process(self, image, operation, kernel_size):
#         device = ldm_patched.modules.model_management.get_torch_device()
#         kernel = torch.ones(kernel_size, kernel_size, device=device)
#         image_k = image.to(device).movedim(-1, 1)
#         if operation == "erode":
#             output = erosion(image_k, kernel)
#         elif operation == "dilate":
#             output = dilation(image_k, kernel)
#         elif operation == "open":
#             output = opening(image_k, kernel)
#         elif operation == "close":
#             output = closing(image_k, kernel)
#         elif operation == "gradient":
#             output = gradient(image_k, kernel)
#         elif operation == "top_hat":
#             output = top_hat(image_k, kernel)
#         elif operation == "bottom_hat":
#             output = bottom_hat(image_k, kernel)
#         else:
#             raise ValueError(f"Invalid operation {operation} for morphology. Must be one of 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'")
#         img_out = output.to(ldm_patched.modules.model_management.intermediate_device()).movedim(1, -1)
#         return (img_out,)


# class ImageRGBToYUV:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "image": ("IMAGE",),
#                               }}

#     RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
#     RETURN_NAMES = ("Y", "U", "V")
#     FUNCTION = "execute"

#     CATEGORY = "image/batch"

#     def execute(self, image):
#         out = kornia.color.rgb_to_ycbcr(image.movedim(-1, 1)).movedim(1, -1)
#         return (out[..., 0:1].expand_as(image), out[..., 1:2].expand_as(image), out[..., 2:3].expand_as(image))

# class ImageYUVToRGB:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"Y": ("IMAGE",),
#                              "U": ("IMAGE",),
#                              "V": ("IMAGE",),
#                               }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "execute"

#     CATEGORY = "image/batch"

#     def execute(self, Y, U, V):
#         image = torch.cat([torch.mean(Y, dim=-1, keepdim=True), torch.mean(U, dim=-1, keepdim=True), torch.mean(V, dim=-1, keepdim=True)], dim=-1)
#         out = kornia.color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1)
#         return (out,)

# # Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
#NODE_CLASS_MAPPINGS = {
#     "Morphology": Morphology,
#     "ImageRGBToYUV": ImageRGBToYUV,
#     "ImageYUVToRGB": ImageYUVToRGB,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Morphology": "ImageMorphology",
# }
