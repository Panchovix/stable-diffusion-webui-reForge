from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
import torch

from modules import devices, errors, face_restoration
from modules_forge.utils import prepare_free_memory

if TYPE_CHECKING:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

logger = logging.getLogger(__name__)


def bgr_image_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR NumPy image in [0..1] range to a PyTorch RGB float32 tensor."""
    assert img.shape[2] == 3, "image must be RGB"
    if img.dtype == "float64":
        img = img.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def rgb_tensor_to_bgr_image(tensor: torch.Tensor, *, min_max=(0.0, 1.0)) -> np.ndarray:
    """
    Convert a PyTorch RGB tensor in range `min_max` to a BGR NumPy image in [0..1] range.
    """
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    assert tensor.dim() == 3, "tensor must be RGB"
    img_np = tensor.numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image, no RGB/BGR required
        return np.squeeze(img_np, axis=2)
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


def create_face_helper(device) -> FaceRestoreHelper:
    from facexlib.detection import retinaface
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    if hasattr(retinaface, 'device'):
        retinaface.device = device
    return FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        # pad_blur=True,
        use_parse=True,
        device=device,
    )

from facexlib.utils.misc import img2tensor
# from torchvision.transforms.functional import normalize

# patched to blend back better, by github.com/sixcu (pr 32 facexlib)
@torch.no_grad()
def patched_paste_faces_to_input_image(self, upsample_img=None):
    h, w, _ = self.input_img.shape
    h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

    if upsample_img is None:
        # simply resize the background
        upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
    else:
        upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

    assert len(self.restored_faces) == len(
        self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
    for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
        # Add an offset to inverse affine matrix, for more precise back alignment
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

        mask = np.ones(self.face_size, dtype=np.float32)
        if self.use_parse:
            # inference
            face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
            # normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_input = torch.unsqueeze(face_input, 0).to(self.device)
            with torch.no_grad():
                out = self.face_parse(face_input)[0]
            out = out.argmax(dim=1).squeeze().cpu().numpy()

            mask = np.zeros(out.shape)
            MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
            for idx, color in enumerate(MASK_COLORMAP):
                mask[out == idx] = color
            mask = mask / 255.
            mask = cv2.resize(mask, restored_face.shape[:2], interpolation=cv2.INTER_NEAREST)

        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
        # remove the black borders
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)  # // 3
        # compute the fusion edge based on the area of face
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2

        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        if len(upsample_img.shape) == 2:  # upsample_img is gray image
            upsample_img = upsample_img[:, :, None]
        inv_soft_mask = inv_soft_mask[:, :, None]

        if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
            alpha = upsample_img[:, :, 3:]
            upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
            upsample_img = np.concatenate((upsample_img, alpha), axis=2)
        else:
            upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

    if np.max(upsample_img) > 256:  # 16-bit image
        upsample_img = upsample_img.astype(np.uint16)
    else:
        upsample_img = upsample_img.astype(np.uint8)

    return upsample_img


def restore_with_face_helper(
    np_image: np.ndarray,
    face_helper: FaceRestoreHelper,
    restore_face: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """
    Find faces in the image using face_helper, restore them using restore_face, and paste them back into the image.

    `restore_face` should take a cropped face image and return a restored face image.
    """
    from torchvision.transforms.functional import normalize
    np_image = np_image[:, :, ::-1]
    original_resolution = np_image.shape[0:2]

    try:
        logger.debug("Detecting faces...")
        face_helper.clean_all()
        face_helper.read_image(np_image)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        logger.debug("Found %d faces, restoring", len(face_helper.cropped_faces))
        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = bgr_image_to_rgb_tensor(cropped_face / 255.0)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(devices.device_face_restore)

            try:
                with torch.no_grad():
                    cropped_face_t = restore_face(cropped_face_t)
                devices.torch_gc()
            except Exception:
                errors.report('Failed face-restoration inference', exc_info=True)

            restored_face = rgb_tensor_to_bgr_image(cropped_face_t, min_max=(-1, 1))
            restored_face = (restored_face * 255.0).astype('uint8')
            face_helper.add_restored_face(restored_face)
            
        logger.debug("Merging restored faces into image")
        face_helper.get_inverse_affine(None)
        # img = face_helper.paste_faces_to_input_image()
        img = patched_paste_faces_to_input_image(face_helper)
        img = img[:, :, ::-1]
        if original_resolution != img.shape[0:2]:
            img = cv2.resize(
                img,
                (0, 0),
                fx=original_resolution[1] / img.shape[1],
                fy=original_resolution[0] / img.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )
        logger.debug("Face restoration complete")
    finally:
        face_helper.clean_all()
    return img


class CommonFaceRestoration(face_restoration.FaceRestoration):
    net: torch.Module | None
    model_url: str
    model_download_name: str

    def __init__(self, model_path: str):
        super().__init__()
        self.net = None
        self.model_path = model_path
        if model_path:
            os.makedirs(model_path, exist_ok=True)

    def get_device(self):
        return devices.device_face_restore

    @cached_property
    def face_helper(self) -> FaceRestoreHelper:
        return create_face_helper(self.get_device())

    def send_model_to(self, device):
        if self.net:
            logger.debug("Sending %s to %s", self.net, device)
            self.net.to(device)
        if self.face_helper:
            logger.debug("Sending face helper to %s", device)
            self.face_helper.face_det.to(device)
            self.face_helper.face_parse.to(device)

    def load_net(self) -> torch.Module:
        raise NotImplementedError("load_net must be implemented by subclasses")

    def restore_with_helper(
        self,
        np_image: np.ndarray,
        restore_face: Callable[[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        try:
            if self.net is None:
                self.net = self.load_net()
        except Exception:
            logger.warning("Unable to load face-restoration model", exc_info=True)
            return np_image

        try:
            prepare_free_memory()
            self.send_model_to(self.get_device())
            return restore_with_face_helper(np_image, self.face_helper, restore_face)
        finally:
            self.send_model_to(devices.cpu)


def patch_facexlib(dirname: str) -> None:
    import facexlib.detection
    import facexlib.parsing

    det_facex_load_file_from_url = facexlib.detection.load_file_from_url
    par_facex_load_file_from_url = facexlib.parsing.load_file_from_url

    def update_kwargs(kwargs):
        return dict(kwargs, save_dir=dirname, model_dir=None)

    def facex_load_file_from_url(**kwargs):
        return det_facex_load_file_from_url(**update_kwargs(kwargs))

    def facex_load_file_from_url2(**kwargs):
        return par_facex_load_file_from_url(**update_kwargs(kwargs))

    facexlib.detection.load_file_from_url = facex_load_file_from_url
    facexlib.parsing.load_file_from_url = facex_load_file_from_url2
