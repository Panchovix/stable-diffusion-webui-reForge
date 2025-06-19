import os
import subprocess
import datetime

from PIL import Image
import cv2

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, infotext_utils
from modules.shared import opts


# video split / combine from sd-webui-video-extras-tab extension by light-and-ray
try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
except Exception as e:
    FFMPEG = 'ffmpeg'


def separate_video_into_frames(video_path, frames_path):
    ffmpeg_cmd = [
        FFMPEG,
        '-loglevel', 'quiet',
        '-i', video_path,
        '-y',
        os.path.join(frames_path, '%05d.png'),
    ]

    rc = subprocess.run(ffmpeg_cmd).returncode
    if rc != 0:
        raise Exception(f'ffmpeg exited with code {rc}. See console for details')

    return


def getVideoFrames(video_path, frames_path):
    separate_video_into_frames(video_path, frames_path)
    return


def getVideoFPS(video_path):
    name = os.path.splitext(video_path)
    ext = name[1].lower()

    if ext == '.webp' or ext == '.apng' or ext == '.png':
        #from gif2gif extension
        with Image.open(video_path) as im:
            try:
                fps = round (1000 / im.info["duration"], 3)
            except:
                fps = 'unknown'
    else:
        video = cv2.VideoCapture(video_path)
        fps = round(video.get(cv2.CAP_PROP_FPS), 3)
        video.release()

    return fps
        
def save_video(frames_dir, fps, original, output_path, interpolate):
    if original != '':  # order of parameters is fussy
        ffmpeg_cmd = [
            FFMPEG,
            '-loglevel', 'quiet',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%5d.png'),
            '-r', str(fps*interpolate),
            '-i', original,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-c:a', 'copy',
            '-vf', f'fps={fps * interpolate}',
            '-profile:v', 'high444',
            '-pix_fmt', 'yuv444p',
            '-shortest',
            '-y',
            output_path
        ]
        ffmpeg_cmd[19:19] = ['-filter:v', 'mblend'] if interpolate > 1 else []
    else:
        ffmpeg_cmd = [
            FFMPEG,
            '-loglevel', 'quiet',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%5d.png'),
            '-r', str(fps * interpolate),
            '-map', '0:v:0',
            '-c:v', 'libx264',
            '-vf', f'fps={fps * interpolate}',
            '-profile:v', 'high444',
            '-pix_fmt', 'yuv444p',
            '-shortest',
            '-y',
            output_path
        ]
        ffmpeg_cmd[13:13] = ['-filter:v', 'mblend'] if interpolate > 1 else []

    # print(' '.join(f"'{str(v)}'" if ' ' in str(v) else str(v) for v in ffmpeg_cmd))
    rc = subprocess.run(ffmpeg_cmd).returncode
    if rc != 0:
        raise Exception(f'ffmpeg exited with code {rc}. See console for details')


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, \
                        input_video, output_frames, input_frames, output_fps, output_video, interpolate, \
                        *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = images.fix_image(img)
                    fn = ''
                else:
                    image = images.read(os.path.abspath(img.name))
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            assert input_dir, 'input directory not selected'

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, 'image not selected'
            yield image, None

    if extras_mode == 3:
        output_text = ''
        input_fps = 'unknown'

        if input_video != '':
            if input_video[0] == '\"':
                input_video = input_video[1:-1]

            input_fps = getVideoFPS(input_video)

        if input_video != '' and input_frames == '':
            name = os.path.splitext(input_video)
            ext = name[1].lower()

            if output_frames == '':
                output_frames = name[0] + '_frames'

            # Create the frames folder if it doesn't exist
            os.makedirs(output_frames, exist_ok=True)
                
            if ext == '.webp' or ext == '.apng' or ext == '.png':
                #from gif2gif extension
                with Image.open(input_video) as im:
                    frame_count = 0
                    try:
                        while True:
                            frame = im.copy()
                            frame.save(os.path.join(output_frames, f'{frame_count:05}.png'))
                            frame_count += 1
                            im.seek(frame_count)
                    except EOFError:
                        pass
                output_text += f'{frame_count} '
            else:
                getVideoFrames(input_video, output_frames)

            output_text += f'frames (fps: {input_fps}) saved to {output_frames}\n'

        if input_frames != '':
            if output_video == '':
                timestamp = int(datetime.datetime.now().timestamp())
                output_video = os.path.join(input_frames, f'output_{timestamp}.mp4')
            if output_fps == 0:
                if input_fps != 'unknown':
                    output_fps = input_fps
                else:
                    output_fps = 25

            save_video(input_frames, output_fps, input_video, output_video, interpolate)
            output_text += f'video saved to {output_video}'
            
        return '', ui_common.plaintext_to_html(output_text), ''

    elif extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ''

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
            except Exception:
                continue
        else:
            image_data = image_placeholder

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ''
                forced_filename = None

            infotext = ", ".join([k if k == v else f'{k}: {infotext_utils.quote(v)}' for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

                if pp.caption:
                    caption_filename = os.path.splitext(fullfn)[0] + ".txt"
                    try:
                        with open(caption_filename, encoding="utf8") as file:
                            caption = file.read().strip() + " "
                    except FileNotFoundError:
                        caption = ""

                    caption += pp.caption.strip()

                    if caption:
                        with open(caption_filename, "w", encoding="utf8") as file:
                            file.write(caption)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

    devices.torch_gc()
    shared.state.end()
    return outputs, ui_common.plaintext_to_html(infotext), ''


def run_postprocessing_webui(id_task, *args, **kwargs):
    return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    """old handler for API"""

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_enabled": True,
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "enable": True,
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "enable": True,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
