from ldm_patched.modules import sd1_clip
import torch
import os

class SDXLClipG(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None, dtype=None, model_options={}):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2

        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config_bigg.json")
        model_options = {**model_options, "model_name": "clip_g"}
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype,
                         special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False, 
                         return_projected_pooled=True, model_options=model_options)

    def load_sd(self, sd):
        return super().load_sd(sd)

class SDXLClipGTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None, tokenizer_data={}):
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g', tokenizer_data=tokenizer_data)


class SDXLTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.clip_g = SDXLClipGTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

class SDXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, model_options=model_options)
        self.clip_g = SDXLClipG(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        
        g_out = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out = self.clip_l.encode_token_weights(token_weight_pairs_l)
        
        # Handle the case where encode_token_weights returns more than 2 values
        if isinstance(g_out, tuple) and len(g_out) > 2:
            g_out_tensor, g_pooled = g_out[:2]
        else:
            g_out_tensor, g_pooled = g_out
            
        if isinstance(l_out, tuple) and len(l_out) > 2:
            l_out_tensor, l_pooled = l_out[:2]
        else:
            l_out_tensor, l_pooled = l_out
        
        # Ensure the tensors have compatible dimensions before concatenating
        cut_to = min(l_out_tensor.shape[1], g_out_tensor.shape[1])
        
        # Return the concatenated output and the pooled output from g
        return torch.cat([l_out_tensor[:,:cut_to], g_out_tensor[:,:cut_to]], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)

class SDXLRefinerClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, clip_name="g", clip_model=SDXLClipG, model_options=model_options)


class StableCascadeClipGTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None, tokenizer_data={}):
        super().__init__(tokenizer_path, pad_with_end=True, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g', tokenizer_data=tokenizer_data)

class StableCascadeTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="g", tokenizer=StableCascadeClipGTokenizer)

class StableCascadeClipG(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=77, freeze=True, layer="hidden", layer_idx=-1, dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config_bigg.json")
        model_options = {**model_options, "model_name": "clip_g"}
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype,
                         special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=False, enable_attention_masks=True, return_projected_pooled=True, model_options=model_options)

    def load_sd(self, sd):
        return super().load_sd(sd)

class StableCascadeClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, clip_name="g", clip_model=StableCascadeClipG, model_options=model_options)
        