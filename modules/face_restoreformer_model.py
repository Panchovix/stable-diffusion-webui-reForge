from __future__ import annotations

import logging

import torch

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
)

logger = logging.getLogger(__name__)

model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
model_download_name = 'RestoreFormer.pth'

# used by e.g. postprocessing_codeformer.py
restoreformer: face_restoration.FaceRestoration | None = None


class FaceRestorerFormer(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "RestoreFormer"

    def load_net(self) -> torch.Module:
        for model_path in modelloader.load_models(
            model_path=self.model_path,
            model_url=model_url,
            command_path=self.model_path,
            download_name=model_download_name,
            ext_filter=['.pth'],
        ):
            return modelloader.load_spandrel_model(
                model_path,
                device=devices.device_face_restore,
            ).model
        raise ValueError("No restoreformer model found")

    def restore(self, np_image):
        def restore_face(cropped_face_t):
            assert self.net is not None
            with torch.no_grad():
                return self.net(cropped_face_t)[0]

        return self.restore_with_helper(np_image, restore_face)


def setup_model(dirname: str) -> None:
    global restoreformer
    try:
        restoreformer = FaceRestorerFormer(dirname)
        shared.face_restorers.append(restoreformer)
    except Exception:
        errors.report("Error setting up RestoreFormer", exc_info=True)
