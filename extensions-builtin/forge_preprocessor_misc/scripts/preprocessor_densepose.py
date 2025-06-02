from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices

import os
import torch
import numpy
import cv2
from einops import rearrange

from typing import Tuple


class MatrixVisualizer:
    """
    Base visualizer for matrix data
    """

    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        if self.inplace:
            image_target_bgr = image_bgr
        else:
            image_target_bgr = image_bgr * 0
        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = numpy.tile((mask == 0)[:, :, numpy.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(numpy.float32) * self.val_scale
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(numpy.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(numpy.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == numpy.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == numpy.uint8

class DensePoseMaskedColormapResultsVisualizer:
    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )

    def visualize(
        self,
        image_bgr: numpy.ndarray,
        results,
    ) -> numpy.ndarray:
        context = image_bgr
        for i, result in enumerate(results):
            boxes_xywh, labels, uv = result
            iuv_array = torch.cat(
                (labels[None].type(torch.float32), uv * 255.0)
            ).type(torch.uint8)

            matrix = iuv_array.cpu().numpy()[0, :, :]
            mask = numpy.zeros(matrix.shape, dtype=numpy.uint8)
            mask[matrix > 0] = 1
            image_bgr = self.mask_visualizer.visualize(context, mask, matrix, boxes_xywh)

        image_bgr = context
        return image_bgr


def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs: Tuple[int, int, int, int]
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = torch.nn.functional.interpolate(
        coarse_segm,
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        torch.nn.functional.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        * (coarse_segm_bbox > 0).long()
    )
    return labels

def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs: Tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = torch.nn.functional.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = torch.nn.functional.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv

def densepose_chart_predictor_output_to_result_with_confidences(
    boxes: torch.Tensor,
    coarse_segm,
    fine_segm,
    u, v

):
    boxes_xywh_abs = boxes.clone()  # currently xyxy
    boxes_xywh_abs[:, 2] -= boxes_xywh_abs[:, 0]
    boxes_xywh_abs[:, 3] -= boxes_xywh_abs[:, 1]

    box_xywh = tuple(boxes_xywh_abs[0].long().tolist())

    labels = resample_fine_and_coarse_segm_tensors_to_bbox(fine_segm, coarse_segm, box_xywh).squeeze(0)
    uv = resample_uv_tensors_to_bbox(u, v, labels, box_xywh)

    return box_xywh, labels, uv


class PreprocessorDensepose(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['OpenPose']
        self.model_filename_filters = ['pose']
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100
        self.model = None
        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "densepose")

        remote_model_path = 'https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript'
        model_path = os.path.join(model_dir, 'densepose_r50_fpn_dl.torchscript')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        self.model = torch.jit.load(model_path, map_location="cpu").eval()
        

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        H, W  = image.shape[:2]
        hint_image_canvas = numpy.zeros([H, W, 3], dtype=numpy.uint8)
        # hint_image_canvas = numpy.tile(hint_image_canvas[:, :, numpy.newaxis], [1, 1, 3])
        image = rearrange(torch.from_numpy(image).to(devices.get_device_for("controlnet")), 'h w c -> c h w')
        pred_boxes, coarse_segm, fine_segm, u, v = self.model(image)

        extractor = densepose_chart_predictor_output_to_result_with_confidences
        densepose_results = [extractor(pred_boxes[i:i+1], coarse_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

        N_PART_LABELS = 24
        result_visualizer = DensePoseMaskedColormapResultsVisualizer(
            alpha=1,
            val_scale = 255.0 / N_PART_LABELS
        )

        if self.name == 'densepose_viridis':
            result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
            hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
            hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
            hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
            hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
            result = hint_image
        else:
            result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
            hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
            result = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

        self.model.to('cpu')
        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorDensepose('densepose_viridis'))
add_supported_preprocessor(PreprocessorDensepose('densepose_parula'))
