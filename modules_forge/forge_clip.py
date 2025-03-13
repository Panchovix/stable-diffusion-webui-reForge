from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from ldm_patched.modules import model_management
from modules import sd_models
from modules.shared import opts


def move_clip_to_gpu():
    if sd_models.model_data.sd_model is None:
        print('Error: CLIP called before SD is loaded!')
        return

    model_management.load_model_gpu(sd_models.model_data.sd_model.forge_objects.clip.patcher)
    return


class CLIP_SD_15_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.minimal_clip_skip = 1

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        return z


class CLIP_SD_21_H(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)

        return z


class CLIP_SD_XL_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]

        return z


class CLIP_SD_XL_G(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]

        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[2]
        text_projection = self.wrapped.text_projection
        pooled_output = pooled_output.float().to(text_projection.device) @ text_projection.float()
        z.pooled = pooled_output
        return z
    