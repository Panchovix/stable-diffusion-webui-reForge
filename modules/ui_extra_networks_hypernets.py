import os
import gradio as gr

from modules import shared, ui_extra_networks, ui_extra_networks_user_metadata
from modules.ui_extra_networks import quote_js
from modules.hashes import sha256_from_cache


class HypernetUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        self.edit_activation_text = None
        self.slider_preferred_weight = None

    def save_hypernet_user_metadata(self, name, desc, activation_text, preferred_weight, notes):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["activation text"] = activation_text
        user_metadata["preferred weight"] = preferred_weight
        user_metadata["notes"] = notes

        self.write_user_metadata(name, user_metadata)


    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)

        return [
            *values[0:5],
            user_metadata.get('activation text', ''),
            float(user_metadata.get('preferred weight', 0.0)),
        ]


    def create_editor(self):
        self.create_default_editor_elems()

        self.edit_activation_text = gr.Text(label='Activation text', info="Will be added to prompt along with Hypernetwork")
        self.slider_preferred_weight = gr.Slider(label='Preferred weight', info="Set to 0 to disable", minimum=0.0, maximum=2.0, value=1.0, step=0.01)

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.edit_activation_text,
            self.slider_preferred_weight,
        ]

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=None, outputs=[self.box])

        edited_components = [
            self.edit_description,
            self.edit_activation_text,
            self.slider_preferred_weight,
            self.edit_notes,
        ]

        self.setup_save_handler(self.button_save, self.save_hypernet_user_metadata, edited_components)

class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetworks')

    def refresh(self):
        shared.reload_hypernetworks()

    def create_item(self, name, index=None, enable_filter=True):
        full_path = shared.hypernetworks.get(name)
        if full_path is None:
            return

        path, ext = os.path.splitext(full_path)
        sha256 = sha256_from_cache(full_path, f'hypernet/{name}')
        shorthash = sha256[0:10] if sha256 else None
        search_terms = [self.search_terms_from_path(path)]
        if sha256:
            search_terms.append(sha256)
        item =  {
            "name": name,
            "filename": full_path,
            "shorthash": shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(path + ext)},
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", shared.opts.extra_networks_default_multiplier)

        item["prompt"] = quote_js(f"<hypernet:{name}:{preferred_weight}>" + ((" " + activation_text) if activation_text else ""))

        return item


    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(shared.hypernetworks)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.hypernetwork_dir]

    def create_user_metadata_editor(self, ui, tabname):
        return HypernetUserMetadataEditor(ui, tabname, self)
