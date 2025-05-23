from modules import shared


class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def restore_faces(np_image):
    if shared.opts.face_restoration_model != "None":
        for fr in shared.face_restorers:
            if fr.name() == shared.opts.face_restoration_model:
                return fr.restore(np_image)

    return np_image
