from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import resize_image_with_pad, HWC3

import torch
import mediapipe
import numpy
import os


# from typing import Mapping


right_iris_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(10, 200, 250), thickness=2, circle_radius=1)
right_eye_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(10, 200, 180), thickness=2, circle_radius=1)
right_eyebrow_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(10, 220, 180), thickness=2, circle_radius=1)
left_iris_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(250, 200, 10), thickness=2, circle_radius=1)
left_eye_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(180, 200, 10), thickness=2, circle_radius=1)
left_eyebrow_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(180, 220, 10), thickness=2, circle_radius=1)
mouth_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(10, 180, 10), thickness=2, circle_radius=1)
head_draw = mediapipe.solutions.drawing_styles.DrawingSpec(color=(10, 200, 10), thickness=2, circle_radius=1)

# mediapipe.solutions.face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mediapipe.solutions.face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_LEFT_IRIS:
   face_connection_spec[edge] = left_iris_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_RIGHT_IRIS:
   face_connection_spec[edge] = right_iris_draw
for edge in mediapipe.solutions.face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mediapipe.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None

        if drawing_spec.get(idx) is None:
            continue
        else:
            draw_color = drawing_spec[idx].color

        # if isinstance(drawing_spec, Mapping):
            # if drawing_spec.get(idx) is None:
                # continue
            # else:
                # draw_color = drawing_spec[idx].color
        # elif isinstance(drawing_spec, mediapipe.solutions.drawing_styles.DrawingSpec):
            # draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(
        img_rgb,
        max_faces: int,
        min_confidence: float,
        min_face_size_pixels = 64
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    with mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
    ) as facemesh:
        img_height, img_width, img_channels = img_rgb.shape
        assert(img_channels == 3)

        results = facemesh.process(img_rgb).multi_face_landmarks

        if results is None:
            print("No faces detected in controlnet image for Mediapipe face annotator.")
            return numpy.zeros_like(img_rgb)

        # Filter faces that are too small
        filtered_landmarks = []
        for lm in results:
            landmarks = lm.landmark
            face_rect = [
                landmarks[0].x,
                landmarks[0].y,
                landmarks[0].x,
                landmarks[0].y,
            ]  # Left, up, right, down.
            for i in range(len(landmarks)):
                face_rect[0] = min(face_rect[0], landmarks[i].x)
                face_rect[1] = min(face_rect[1], landmarks[i].y)
                face_rect[2] = max(face_rect[2], landmarks[i].x)
                face_rect[3] = max(face_rect[3], landmarks[i].y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * img_width
                face_height_pixels = face_height * img_height
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(lm)
            else:
                filtered_landmarks.append(lm)

        # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
        draw_face = numpy.zeros_like(img_rgb)

        # Draw detected faces:
        for face_landmarks in filtered_landmarks:
            mediapipe.solutions.drawing_utils.draw_landmarks(
                draw_face,
                face_landmarks,
                connections=face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec
            )
            draw_pupils(draw_face, face_landmarks, iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        draw_face = reverse_channels(draw_face).copy()

        return draw_face


class PreprocessorMediaPipeFace(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'mediapipe_face'
        self.tags = ['Misc']
        self.model_filename_filters = ['mediapipe']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(minimum=1, maximum=10, step=1, value=1, label='Maximum faces', visible=True)
        self.slider_2 = PreprocessorParameter(minimum=0.01, maximum=1.0, step=0.11, value=0.5, label='Minimum confidence', visible=True)
        self.sorting_priority = 100

        self.cache = None
        self.cacheHash = None

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)

        result = generate_annotation(image, int(slider_1), slider_2)

        return HWC3(remove_pad(result))

add_supported_preprocessor(PreprocessorMediaPipeFace())
