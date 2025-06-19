import gradio as gr
from modules import scripts, shared, ui_common, postprocessing, call_queue, ui_toprow
import modules.infotext_utils as parameters_copypaste
from modules.ui_components import ResizeHandleRow, ToolButton


def create_ui():
    dummy_component = gr.Textbox(visible=False)
    tab_index = gr.State(value=0)

    with ResizeHandleRow(equal_height=False, variant='compact'):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", interactive=True, type="pil", elem_id="extras_image", image_mode="RGBA", height="60vh")

                with gr.TabItem('Batch Process', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

                with gr.TabItem('Video', id='video', elem_id='extras_video') as tab_video:
                    gr.Markdown("## combine frames to video")
                    with gr.Row():
                        input_frames  = gr.Textbox(label="Input frames", placeholder="A directory of images on the same machine where the server is running.", max_lines=1)
                        clear_combine = ToolButton('\U0001f5d1\ufe0f')
                    with gr.Row():
                        output_video  = gr.Textbox(label="Output video filename", placeholder="Blank: default name", max_lines=1)
                        output_fps    = gr.Number(label="Output fps", value=0, minimum=0, scale=0)
                        interpolate   = gr.Number(label="Interpolation", value=1, minimum=1, step=1, scale=0)
                    gr.Markdown("### if *Input video file* is provided, it will be used for **audio** and, if *Output fps* is zero, for **fps**.")
                    gr.Markdown("## split video to frames")
                    with gr.Row():
                        input_video   = gr.Textbox(label="Input video file", placeholder="A video on the same machine where the server is running.", max_lines=1)
                    with gr.Row():
                        output_frames = gr.Textbox(label="Output directory", placeholder="Blank: directory of input.", max_lines=1)
                        clear_split   = ToolButton('\U0001f5d1\ufe0f')
                    gr.Markdown("---")
                    gr.Markdown("### further post-processing is ignored - switch to 'Batch from Directory'")

                    def clearI():
                        return "", ""
                    def clearO():
                        return "", 0, "", 1

                    clear_split.click(fn=clearI, inputs=None, outputs=[input_video, output_frames], show_progress=False)
                    clear_combine.click(fn=clearO, inputs=None, outputs=[input_frames, output_fps, output_video, interpolate], show_progress=False)


            script_inputs = scripts.scripts_postproc.setup_ui()

        with gr.Column():
            toprow = ui_toprow.Toprow(id_part="extras")
            toprow.create_inline_toprow_image()
            submit = toprow.submit

            output_panel = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)

    tab_single.select(fn=lambda: 0, inputs=None, outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=None, outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=None, outputs=[tab_index])
    tab_video.select(fn=lambda: 3, inputs=None, outputs=[tab_index])

    submit_click_inputs = [
        dummy_component,
        tab_index,
        extras_image,
        image_batch,
        extras_batch_input_dir,
        extras_batch_output_dir,
        show_extras_results,
        input_video,
        output_frames,
        input_frames,
        output_fps,
        output_video,
        interpolate,
        *script_inputs
    ]

    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing_webui, extra_outputs=[None, '']),
        _js="submit_extras",
        inputs=submit_click_inputs,
        outputs=[
            output_panel.gallery,
            output_panel.generation_info,
            output_panel.html_log,
        ],
        show_progress=False,
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=None, outputs=None
    )
