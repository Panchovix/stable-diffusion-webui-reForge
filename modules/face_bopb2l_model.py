from __future__ import annotations

import logging
import os

import torch
import numpy
from PIL import Image
import importlib  

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
    codeformer_model,
)
from modules.paths_internal import extensions_dir

logger = logging.getLogger(__name__)
bopb2l_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerBOPB2L(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "MS-BOPB2L"

    def face_helper(self) -> FaceRestoreHelper:
        pass

    def send_model_to(self, device):
        pass

    def load_net(self) -> torch.Module:
        pass


    @torch.no_grad()
    def restore(self, np_image):
        image = Image.fromarray(np_image)

        detect = importlib.import_module("extensions.sd-webui-old-photo-restoration.lib_bopb2l.Face_Detection.detect_all_dlib_HR")
        warp = importlib.import_module("extensions.sd-webui-old-photo-restoration.lib_bopb2l.Face_Detection.align_warp_back_multiple_dlib_HR")
        face = importlib.import_module("extensions.sd-webui-old-photo-restoration.lib_bopb2l.Face_Enhancement.test_face")

        faces = detect.detect_hr(image)
        if len(faces) == 0:
            return np_image

        FACE_CHECKPOINTS_FOLDER = os.path.join(
            extensions_dir, "sd-webui-old-photo-restoration", "lib_bopb2l", "Face_Enhancement", "checkpoints"
        )

        FACE_ENHANCEMENT_CHECKPOINTS = ("Setting_9_epoch_100", "FaceSR_512")

        gpu_id = 0 if 'cuda' in str(self.get_device()) else -1

        args = {
            "checkpoints_dir": FACE_CHECKPOINTS_FOLDER,
            "name": FACE_ENHANCEMENT_CHECKPOINTS[1],
            "gpu_ids": str(gpu_id),
            "load_size": 512,
            "label_nc": 18,
            "no_instance": True,
            "preprocess_mode": "resize",
            "batchSize": 1,
            "no_parsing_map": True,
        }
        restored_faces = face.test_face(faces, args)

        result = warp.align_warp_hr(image, restored_faces)

        return codeformer_model.codeformer.restore( # 2nd light pass with codeformer improves
            numpy.array(result), w=1.0
        )
        # return numpy.array(result)


def bopb2l_fix_faces(np_image): # needed if add new accordion to Extras (or just leave as part of Old Photo Restoration)
    global bopb2l_face_restorer
    if bopb2l_face_restorer:
        return bopb2l_face_restorer.restore(np_image)   # best followed by minimum strength codeformer, at least for ReActor
    logger.warning("MS-BOPB2L face restorer not set up")
    return np_image


def setup_model() -> None:
    global bopb2l_face_restorer

    # relies on Haoming's Old Photo Restoration extension being installed, to avoid duplicating models
    if os.path.isdir(os.path.join(extensions_dir, "sd-webui-old-photo-restoration")):
        try:
            bopb2l_face_restorer = FaceRestorerBOPB2L(model_path=None)
            shared.face_restorers.append(bopb2l_face_restorer)
        except Exception:
            errors.report("Error setting up MS-BOPB2L", exc_info=True)
