from ldm_patched.modules import sd1_clip
import ldm_patched.modules.model_management
import ldm_patched.modules.text_encoders.llama
from transformers import LlamaTokenizerFast
import torch
import os
import numbers


def llama_detect(state_dict, prefix=""):
    out = {}
    t5_key = "{}model.norm.weight".format(prefix)
    if t5_key in state_dict:
        out["dtype_llama"] = state_dict[t5_key].dtype

    scaled_fp8_key = "{}scaled_fp8".format(prefix)
    if scaled_fp8_key in state_dict:
        out["llama_scaled_fp8"] = state_dict[scaled_fp8_key].dtype

    return out


class LLAMA3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, min_length=256, pad_token=128258):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "llama_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='llama', tokenizer_class=LlamaTokenizerFast, has_start_token=True, has_end_token=False, pad_to_max_length=False, max_length=99999999, pad_token=pad_token, min_length=min_length, tokenizer_data=tokenizer_data)

class LLAMAModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-3, dtype=None, attention_mask=True, model_options={}, special_tokens={"start": 128000, "pad": 128258}):
        llama_scaled_fp8 = model_options.get("llama_scaled_fp8", None)
        if llama_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = llama_scaled_fp8

        textmodel_json_config = {}
        vocab_size = model_options.get("vocab_size", None)
        if vocab_size is not None:
            textmodel_json_config["vocab_size"] = vocab_size

        model_options = {**model_options, "model_name": "llama"}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens=special_tokens, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Llama2, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class HunyuanVideoTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = """<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"""  # 95 tokens
        self.llama = LLAMA3Tokenizer(embedding_directory=embedding_directory, min_length=1, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, image_embeds=None, image_interleave=1, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)

        if llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)
        llama_text_tokens = self.llama.tokenize_with_weights(llama_text, return_word_ids, **kwargs)
        embed_count = 0
        for r in llama_text_tokens:
            for i in range(len(r)):
                if r[i][0] == 128257:
                    if image_embeds is not None and embed_count < image_embeds.shape[0]:
                        r[i] = ({"type": "embedding", "data": image_embeds[embed_count], "original_type": "image", "image_interleave": image_interleave},) + r[i][1:]
                        embed_count += 1
        out["llama"] = llama_text_tokens
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class HunyuanVideoClipModel(torch.nn.Module):
    def __init__(self, dtype_llama=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_llama = ldm_patched.modules.model_management.pick_weight_dtype(dtype_llama, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.llama = LLAMAModel(device=device, dtype=dtype_llama, model_options=model_options)
        self.dtypes = set([dtype, dtype_llama])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.llama.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.llama.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_llama = token_weight_pairs["llama"]

        llama_out, llama_pooled, llama_extra_out = self.llama.encode_token_weights(token_weight_pairs_llama)

        template_end = 0
        extra_template_end = 0
        extra_sizes = 0
        user_end = 9999999999999
        images = []

        tok_pairs = token_weight_pairs_llama[0]
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if not torch.is_tensor(elem):
                if isinstance(elem, numbers.Integral):
                    if elem == 128006:
                        if tok_pairs[i + 1][0] == 882:
                            if tok_pairs[i + 2][0] == 128007:
                                template_end = i + 2
                                user_end = -1
                    if elem == 128009 and user_end == -1:
                        user_end = i + 1
                else:
                    if elem.get("original_type") == "image":
                        elem_size = elem.get("data").shape[0]
                        if template_end > 0:
                            if user_end == -1:
                                extra_template_end += elem_size - 1
                        else:
                            image_start = i + extra_sizes
                            image_end = i + elem_size + extra_sizes
                            images.append((image_start, image_end, elem.get("image_interleave", 1)))
                            extra_sizes += elem_size - 1

        if llama_out.shape[1] > (template_end + 2):
            if tok_pairs[template_end + 1][0] == 271:
                template_end += 2
        llama_output = llama_out[:, template_end + extra_sizes:user_end + extra_sizes + extra_template_end]
        llama_extra_out["attention_mask"] = llama_extra_out["attention_mask"][:, template_end + extra_sizes:user_end + extra_sizes + extra_template_end]
        if llama_extra_out["attention_mask"].sum() == torch.numel(llama_extra_out["attention_mask"]):
            llama_extra_out.pop("attention_mask")  # attention mask is useless if no masked elements

        if len(images) > 0:
            out = []
            for i in images:
                out.append(llama_out[:, i[0]: i[1]: i[2]])
            llama_output = torch.cat(out + [llama_output], dim=1)

        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return llama_output, l_pooled, llama_extra_out

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.llama.load_sd(sd)


def hunyuan_video_clip(dtype_llama=None, llama_scaled_fp8=None):
    class HunyuanVideoClipModel_(HunyuanVideoClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "llama_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["llama_scaled_fp8"] = llama_scaled_fp8
            super().__init__(dtype_llama=dtype_llama, device=device, dtype=dtype, model_options=model_options)
    return HunyuanVideoClipModel_
