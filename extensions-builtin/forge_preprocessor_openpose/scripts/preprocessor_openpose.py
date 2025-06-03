from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices
from modules.modelloader import load_file_from_url
from modules.paths_internal import models_path

import os
import torch
import numpy


# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


from openpose import util
from openpose.body import Body, BodyResult, Keypoint
from openpose.hand import Hand
from openpose.face import Face
from openpose.types import HandResult, FaceResult, HumanPoseResult, AnimalPoseResult
from openpose.animalpose import draw_animalposes
from openpose.util import draw_poses, decode_json_as_poses, encode_poses_as_json

from typing import Tuple, List, Callable, Union, Optional


class PreprocessorOpenPose(Preprocessor):
    def __init__(self, name, body, hand, face):
        super().__init__()
        self.name = name
        self.tags = ['OpenPose']
        self.model_filename_filters = ['pose']
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None
        
        self.include_body = body
        self.include_hand = hand
        self.include_face = face

        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "openpose")

        remote_body_model = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
        remote_hand_model = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
        remote_face_model = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

        body_model_path = os.path.join(model_dir, "body_pose_model.pth")
        hand_model_path = os.path.join(model_dir, "hand_pose_model.pth")
        face_model_path = os.path.join(model_dir, "facenet.pth")

        if not os.path.exists(body_model_path):
            load_file_from_url(remote_body_model, model_dir=model_dir)

        if not os.path.exists(hand_model_path):
            load_file_from_url(remote_hand_model, model_dir=model_dir)

        if not os.path.exists(face_model_path):
            load_file_from_url(remote_face_model, model_dir=model_dir)

        self.body_estimation = Body(body_model_path)
        self.hand_estimation = Hand(hand_model_path)
        self.face_estimation = Face(face_model_path)

    def unload_model(self):
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(
        self, body: BodyResult, oriImg
    ) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y : y + w, x : x + w, :]).astype(
                numpy.float32
            )
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = numpy.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(
                    W
                )
                peaks[:, 1] = numpy.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(
                    H
                )

                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None

        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y : y + w, x : x + w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(
            numpy.float32
        )
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = numpy.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = numpy.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

        return None

    def detect_poses(
        self, oriImg, include_hand=False, include_face=False
    ) -> List[HumanPoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
        """

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)

                results.append(
                    HumanPoseResult(
                        BodyResult(
                            keypoints=[
                                Keypoint(
                                    x=keypoint.x / float(W), y=keypoint.y / float(H)
                                )
                                if keypoint is not None
                                else None
                                for keypoint in body.keypoints
                            ],
                            total_score=body.total_score,
                            total_parts=body.total_parts,
                        ),
                        left_hand,
                        right_hand,
                        face,
                    )
                )

            return results

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, json_pose_callback: Callable[[str], None] = None,
 **kwargs):
        if self.body_estimation is None:
            self.load_model()
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        H, W, _ = image.shape
        poses = []

        poses = self.detect_poses(image, self.include_hand, self.include_face)

        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, [], H, W))

        result = draw_poses(
            poses,
            H,
            W,
            draw_body=self.include_body,
            draw_hand=self.include_hand,
            draw_face=self.include_face,
        )

        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))

                                                                     #body, hand, face
add_supported_preprocessor(PreprocessorOpenPose('openpose',           True,  False, False))
add_supported_preprocessor(PreprocessorOpenPose('openpose_face',      True,  False, True))
add_supported_preprocessor(PreprocessorOpenPose('openpose_face_only', False, False, True))
add_supported_preprocessor(PreprocessorOpenPose('openpose_full',      True,  True,  True))
add_supported_preprocessor(PreprocessorOpenPose('openpose_hand',      True,  True,  False))


class PreprocessorDWPose(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['OpenPose']
        self.model_filename_filters = ['pose']
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.dw_pose_estimation = None
        
        # self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_dw_model(self):
        from openpose.wholebody import Wholebody
        model_dir = os.path.join(preprocessor_dir, "openpose")

        remote_onnx_det = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
        remote_onnx_pose = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"

        onnx_det = os.path.join(model_dir, "yolox_l.onnx")
        onnx_pose = os.path.join(model_dir, "dw-ll_ucoco_384.onnx")

        if not os.path.exists(onnx_det):
            load_file_from_url(remote_onnx_det, model_dir=model_dir)

        if not os.path.exists(onnx_pose):
            load_file_from_url(remote_onnx_pose, model_dir=model_dir)

        self.dw_pose_estimation = Wholebody(onnx_det, onnx_pose)

    def detect_poses_dw(self, oriImg) -> List[HumanPoseResult]:
        from openpose.wholebody import Wholebody

        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)


    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, json_pose_callback: Callable[[str], None] = None,
 **kwargs):
        if self.dw_pose_estimation is None:
            self.load_dw_model()

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        H, W, _ = image.shape
        poses = []

        poses = self.detect_poses_dw(image)

        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, [], H, W))

        result = draw_poses(
            poses,
            H,
            W,
            draw_body=True,
            draw_hand=True,
            draw_face=True,
        )

        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))

add_supported_preprocessor(PreprocessorDWPose('dw_openpose_full'))


class PreprocessorAnimalPose(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['OpenPose']
        self.model_filename_filters = ['pose']
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.animal_pose_estimation = None
        
        # self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_animalpose_model(self):
        from openpose.animalpose import AnimalPose
        model_dir = os.path.join(preprocessor_dir, "openpose")

        remote_onnx_det = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
        remote_animal_pose ="https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx"


        onnx_det = os.path.join(model_dir, "yolox_l.onnx")
        onnx_animal = os.path.join(model_dir, "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx")

        if not os.path.exists(onnx_det):
            load_file_from_url(remote_onnx_det, model_dir=model_dir)

        if not os.path.exists(onnx_animal):
            load_file_from_url(remote_animal_pose, model_dir=model_dir)

        self.animal_pose_estimation = AnimalPose(onnx_det, onnx_animal)

    def detect_poses_animal(self, oriImg) -> List[AnimalPoseResult]:
        with torch.no_grad():
            return self.animal_pose_estimation(oriImg.copy())

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, json_pose_callback: Callable[[str], None] = None,
 **kwargs):
        if self.animal_pose_estimation is None:
            self.load_animalpose_model()

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        H, W, _ = image.shape
        animals = []

        animals = self.detect_poses_animal(image)

        if json_pose_callback:
            json_pose_callback(encode_poses_as_json([], animals, H, W))

        result = draw_animalposes(animals, H, W)

        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))

add_supported_preprocessor(PreprocessorAnimalPose('animal_openpose'))
