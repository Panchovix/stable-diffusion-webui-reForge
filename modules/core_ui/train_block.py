import gradio as gr

import modules.hypernetworks.ui as hypernetworks_ui
import modules.shared as shared
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.textual_inversion.ui as textual_inversion_ui
from modules import script_callbacks, sd_hijack
from modules.call_queue import wrap_gradio_gpu_call
from modules.core_ui.common_elements import create_refresh_button
from modules.core_ui.components import (
    FormRow,
    ResizeHandleRow,
)


def create_interface(txt2img_preview_params: list[gr.components.Component]):
    with gr.Blocks(analytics_enabled=False) as train_interface:
        dummy_component = gr.Textbox(visible=False)
        with gr.Row(equal_height=False):
            gr.HTML(
                value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>"
            )

        with ResizeHandleRow(variant="compact", equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):
                with gr.Tab(label="Create embedding", id="create_embedding"):
                    new_embedding_name = gr.Textbox(
                        label="Name", elem_id="train_new_embedding_name"
                    )
                    initialization_text = gr.Textbox(
                        label="Initialization text",
                        value="*",
                        elem_id="train_initialization_text",
                    )
                    nvpt = gr.Slider(
                        label="Number of vectors per token",
                        minimum=1,
                        maximum=75,
                        step=1,
                        value=1,
                        elem_id="train_nvpt",
                    )
                    overwrite_old_embedding = gr.Checkbox(
                        value=False,
                        label="Overwrite Old Embedding",
                        elem_id="train_overwrite_old_embedding",
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(
                                value="Create embedding",
                                variant="primary",
                                elem_id="train_create_embedding",
                            )

                with gr.Tab(label="Create hypernetwork", id="create_hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(
                        label="Name", elem_id="train_new_hypernetwork_name"
                    )
                    new_hypernetwork_sizes = gr.CheckboxGroup(
                        label="Modules",
                        value=["768", "320", "640", "1280"],
                        choices=["768", "1024", "320", "640", "1280"],
                        elem_id="train_new_hypernetwork_sizes",
                    )
                    new_hypernetwork_layer_structure = gr.Textbox(
                        "1, 2, 1",
                        label="Enter hypernetwork layer structure",
                        placeholder="1st and last digit must be 1. ex:'1, 2, 1'",
                        elem_id="train_new_hypernetwork_layer_structure",
                    )
                    new_hypernetwork_activation_func = gr.Dropdown(
                        value="linear",
                        label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)",
                        choices=hypernetworks_ui.keys,
                        elem_id="train_new_hypernetwork_activation_func",
                    )
                    new_hypernetwork_initialization_option = gr.Dropdown(
                        value="Normal",
                        label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise",
                        choices=[
                            "Normal",
                            "KaimingUniform",
                            "KaimingNormal",
                            "XavierUniform",
                            "XavierNormal",
                        ],
                        elem_id="train_new_hypernetwork_initialization_option",
                    )
                    new_hypernetwork_add_layer_norm = gr.Checkbox(
                        label="Add layer normalization",
                        elem_id="train_new_hypernetwork_add_layer_norm",
                    )
                    new_hypernetwork_use_dropout = gr.Checkbox(
                        label="Use dropout",
                        elem_id="train_new_hypernetwork_use_dropout",
                    )
                    new_hypernetwork_dropout_structure = gr.Textbox(
                        "0, 0, 0",
                        label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15",
                        placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'",
                    )
                    overwrite_old_hypernetwork = gr.Checkbox(
                        value=False,
                        label="Overwrite Old Hypernetwork",
                        elem_id="train_overwrite_old_hypernetwork",
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(
                                value="Create hypernetwork",
                                variant="primary",
                                elem_id="train_create_hypernetwork",
                            )

                def get_textual_inversion_template_names():
                    return sorted(textual_inversion.textual_inversion_templates)

                with gr.Tab(label="Train", id="train"):
                    gr.HTML(
                        value='<p style=\'margin-bottom: 0.7em\'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion" style="font-weight:bold;">[wiki]</a></p>'
                    )
                    with FormRow():
                        train_embedding_name = gr.Dropdown(
                            label="Embedding",
                            elem_id="train_embedding",
                            choices=sorted(
                                sd_hijack.model_hijack.embedding_db.word_embeddings.keys()
                            ),
                        )
                        create_refresh_button(
                            train_embedding_name,
                            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings,
                            lambda: {
                                "choices": sorted(
                                    sd_hijack.model_hijack.embedding_db.word_embeddings.keys()
                                )
                            },
                            "refresh_train_embedding_name",
                        )

                        train_hypernetwork_name = gr.Dropdown(
                            label="Hypernetwork",
                            elem_id="train_hypernetwork",
                            choices=sorted(shared.hypernetworks),
                        )
                        create_refresh_button(
                            train_hypernetwork_name,
                            shared.reload_hypernetworks,
                            lambda: {"choices": sorted(shared.hypernetworks)},
                            "refresh_train_hypernetwork_name",
                        )

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(
                            label="Embedding Learning rate",
                            placeholder="Embedding Learning rate",
                            value="0.005",
                            elem_id="train_embedding_learn_rate",
                        )
                        hypernetwork_learn_rate = gr.Textbox(
                            label="Hypernetwork Learning rate",
                            placeholder="Hypernetwork Learning rate",
                            value="0.00001",
                            elem_id="train_hypernetwork_learn_rate",
                        )

                    with FormRow():
                        clip_grad_mode = gr.Dropdown(
                            value="disabled",
                            label="Gradient Clipping",
                            choices=["disabled", "value", "norm"],
                        )
                        clip_grad_value = gr.Textbox(
                            placeholder="Gradient clip value",
                            value="0.1",
                            show_label=False,
                        )

                    with FormRow():
                        batch_size = gr.Number(
                            label="Batch size",
                            value=1,
                            precision=0,
                            elem_id="train_batch_size",
                        )
                        gradient_step = gr.Number(
                            label="Gradient accumulation steps",
                            value=1,
                            precision=0,
                            elem_id="train_gradient_step",
                        )

                    dataset_directory = gr.Textbox(
                        label="Dataset directory",
                        placeholder="Path to directory with input images",
                        elem_id="train_dataset_directory",
                    )
                    log_directory = gr.Textbox(
                        label="Log directory",
                        placeholder="Path to directory where to write outputs",
                        value="textual_inversion",
                        elem_id="train_log_directory",
                    )

                    with FormRow():
                        template_file = gr.Dropdown(
                            label="Prompt template",
                            value="style_filewords.txt",
                            elem_id="train_template_file",
                            choices=get_textual_inversion_template_names(),
                        )
                        create_refresh_button(
                            template_file,
                            textual_inversion.list_textual_inversion_templates,
                            lambda: {"choices": get_textual_inversion_template_names()},
                            "refrsh_train_template_file",
                        )

                    training_width = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        step=8,
                        label="Width",
                        value=512,
                        elem_id="train_training_width",
                    )
                    training_height = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        step=8,
                        label="Height",
                        value=512,
                        elem_id="train_training_height",
                    )
                    varsize = gr.Checkbox(
                        label="Do not resize images",
                        value=False,
                        elem_id="train_varsize",
                    )
                    steps = gr.Number(
                        label="Max steps",
                        value=100000,
                        precision=0,
                        elem_id="train_steps",
                    )

                    with FormRow():
                        create_image_every = gr.Number(
                            label="Save an image to log directory every N steps, 0 to disable",
                            value=500,
                            precision=0,
                            elem_id="train_create_image_every",
                        )
                        save_embedding_every = gr.Number(
                            label="Save a copy of embedding to log directory every N steps, 0 to disable",
                            value=500,
                            precision=0,
                            elem_id="train_save_embedding_every",
                        )

                    use_weight = gr.Checkbox(
                        label="Use PNG alpha channel as loss weight",
                        value=False,
                        elem_id="use_weight",
                    )

                    save_image_with_stored_embedding = gr.Checkbox(
                        label="Save images with embedding in PNG chunks",
                        value=True,
                        elem_id="train_save_image_with_stored_embedding",
                    )
                    preview_from_txt2img = gr.Checkbox(
                        label="Read parameters (prompt, etc...) from txt2img tab when making previews",
                        value=False,
                        elem_id="train_preview_from_txt2img",
                    )

                    shuffle_tags = gr.Checkbox(
                        label="Shuffle tags by ',' when creating prompts.",
                        value=False,
                        elem_id="train_shuffle_tags",
                    )
                    tag_drop_out = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        label="Drop out tags when creating prompts.",
                        value=0,
                        elem_id="train_tag_drop_out",
                    )

                    latent_sampling_method = gr.Radio(
                        label="Choose latent sampling method",
                        value="once",
                        choices=["once", "deterministic", "random"],
                        elem_id="train_latent_sampling_method",
                    )

                    with gr.Row():
                        train_embedding = gr.Button(
                            value="Train Embedding",
                            variant="primary",
                            elem_id="train_train_embedding",
                        )
                        interrupt_training = gr.Button(
                            value="Interrupt", elem_id="train_interrupt_training"
                        )
                        train_hypernetwork = gr.Button(
                            value="Train Hypernetwork",
                            variant="primary",
                            elem_id="train_train_hypernetwork",
                        )

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id="ti_gallery_container"):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                gr.Gallery(
                    label="Output", show_label=False, elem_id="ti_gallery", columns=4
                )
                gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=textual_inversion_ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ],
        )

        create_hypernetwork.click(
            fn=hypernetworks_ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure,
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ],
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(
                textual_inversion_ui.train_embedding, extra_outputs=[gr.update()]
            ),
            js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ],
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(
                hypernetworks_ui.train_hypernetwork, extra_outputs=[gr.update()]
            ),
            js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ],
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )
    return train_interface
