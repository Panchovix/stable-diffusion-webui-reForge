import tempfile
from collections import namedtuple
from pathlib import Path
from os import walk, environ

import gradio.components

from PIL import PngImagePlugin
from PIL import Image
import gradio.image_utils

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def register_tmp_file(gradio, filename: Path):
    if hasattr(gradio, "temp_file_sets"):  # gradio 3.15
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {
            str(Path(filename).resolve())
        }

    if hasattr(gradio, "temp_dirs"):  # gradio 3.9
        gradio.temp_dirs = gradio.temp_dirs | {str(Path(filename).resolve().parent)}


def check_tmp_file(gradio, filename):
    if hasattr(gradio, "temp_file_sets"):
        return any(filename in fileset for fileset in gradio.temp_file_sets)

    if hasattr(gradio, "temp_dirs"):
        return any(
            Path(temp_dir).resolve() in Path(filename).resolve().parents
            for temp_dir in gradio.temp_dirs
        )

    return False


def save_pil_to_file(image: Image.Image | str | Path, cache_dir: str, format="webp"):
    if isinstance(image, (str, Path)):
        return image
    already_saved_as: str | None = getattr(image, "already_saved_as", None)
    if already_saved_as and Path(already_saved_as).is_file():
        already_saved_as_path = Path(already_saved_as)
        register_tmp_file(shared.demo, already_saved_as_path)
        filename_with_mtime = (
            f"{already_saved_as_path}?{already_saved_as_path.stat().st_mtime}"
        )
        register_tmp_file(shared.demo, Path(filename_with_mtime))
        return filename_with_mtime
    if shared.opts.temp_dir:
        cache_dir = str(shared.opts.temp_dir)
    else:
        Path(cache_dir).mkdir(exist_ok=True)

    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=cache_dir)
    image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj.name


def install_ui_tempdir_override():
    """override save to file function so that it also writes PNG info"""
    gradio.image_utils.save_image = save_pil_to_file


def on_tmpdir_changed():
    if shared.opts.temp_dir == "" or shared.demo is None:
        return
    tmpdir = Path(shared.opts.temp_dir)
    tmpdir.mkdir(exist_ok=True)
    # os.makedirs(, exist_ok=True)

    register_tmp_file(shared.demo, tmpdir / "x")


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not Path(temp_dir).is_dir():
        return
    temp_dir = Path(temp_dir)

    for root, _, files in walk(temp_dir, topdown=False):
        for name in files:
            file = Path(name)
            if not file.suffix.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            file_solved = Path(root) / file
            if file_solved.is_file():
                file_solved.unlink()
            else:
                raise Exception(
                    f"{file_solved} Attempted to cleanup tmpdr but file didn't exist when checked."
                )


def is_gradio_temp_path(path):
    """
    Check if the path is a temp dir used by gradio
    """
    path = Path(path)
    if shared.opts.temp_dir and path.is_relative_to(shared.opts.temp_dir):
        return True
    if gradio_temp_dir := environ.get("GRADIO_TEMP_DIR"):
        if path.is_relative_to(gradio_temp_dir):
            return True
    if path.is_relative_to(Path(tempfile.gettempdir()) / "gradio"):
        return True
    return False
