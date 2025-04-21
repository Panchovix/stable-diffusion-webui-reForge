import torch
import math
import struct


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            params += w.nelement()
    return params


def weight_dtype(sd, prefix=""):
    dtypes = {}
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            dtypes[w.dtype] = dtypes.get(w.dtype, 0) + 1

    return max(dtypes, key=dtypes.get)


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x:shape_from * (x + 1)]

    return sd


def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp).transpose(0, 1).contiguous()
    return sd


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}


def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


MMDIT_MAP_BASIC = {
    ("context_embedder.bias", "context_embedder.bias"),
    ("context_embedder.weight", "context_embedder.weight"),
    ("t_embedder.mlp.0.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
    ("t_embedder.mlp.0.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
    ("t_embedder.mlp.2.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
    ("t_embedder.mlp.2.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
    ("x_embedder.proj.bias", "pos_embed.proj.bias"),
    ("x_embedder.proj.weight", "pos_embed.proj.weight"),
    ("y_embedder.mlp.0.bias", "time_text_embed.text_embedder.linear_1.bias"),
    ("y_embedder.mlp.0.weight", "time_text_embed.text_embedder.linear_1.weight"),
    ("y_embedder.mlp.2.bias", "time_text_embed.text_embedder.linear_2.bias"),
    ("y_embedder.mlp.2.weight", "time_text_embed.text_embedder.linear_2.weight"),
    ("pos_embed", "pos_embed.pos_embed"),
    ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
    ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
    ("final_layer.linear.bias", "proj_out.bias"),
    ("final_layer.linear.weight", "proj_out.weight"),
}

MMDIT_MAP_BLOCK = {
    ("context_block.adaLN_modulation.1.bias", "norm1_context.linear.bias"),
    ("context_block.adaLN_modulation.1.weight", "norm1_context.linear.weight"),
    ("context_block.attn.proj.bias", "attn.to_add_out.bias"),
    ("context_block.attn.proj.weight", "attn.to_add_out.weight"),
    ("context_block.mlp.fc1.bias", "ff_context.net.0.proj.bias"),
    ("context_block.mlp.fc1.weight", "ff_context.net.0.proj.weight"),
    ("context_block.mlp.fc2.bias", "ff_context.net.2.bias"),
    ("context_block.mlp.fc2.weight", "ff_context.net.2.weight"),
    ("x_block.adaLN_modulation.1.bias", "norm1.linear.bias"),
    ("x_block.adaLN_modulation.1.weight", "norm1.linear.weight"),
    ("x_block.attn.proj.bias", "attn.to_out.0.bias"),
    ("x_block.attn.proj.weight", "attn.to_out.0.weight"),
    ("x_block.mlp.fc1.bias", "ff.net.0.proj.bias"),
    ("x_block.mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("x_block.mlp.fc2.bias", "ff.net.2.bias"),
    ("x_block.mlp.fc2.weight", "ff.net.2.weight"),
}


def mmdit_to_diffusers(mmdit_config, output_prefix=""):
    key_map = {}

    depth = mmdit_config.get("depth", 0)
    num_blocks = mmdit_config.get("num_blocks", depth)
    for i in range(num_blocks):
        block_from = "transformer_blocks.{}".format(i)
        block_to = "{}joint_blocks.{}".format(output_prefix, i)

        offset = depth * 64

        for end in ("weight", "bias"):
            k = "{}.attn.".format(block_from)
            qkv = "{}.x_block.attn.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

            qkv = "{}.context_block.attn.qkv.{}".format(block_to, end)
            key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

        for k in MMDIT_MAP_BLOCK:
            key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

    map_basic = MMDIT_MAP_BASIC.copy()
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.bias".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.bias".format(depth - 1), swap_scale_shift))
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.weight".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.weight".format(depth - 1), swap_scale_shift))

    for k in map_basic:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


def auraflow_to_diffusers(mmdit_config, output_prefix=""):
    n_double_layers = mmdit_config.get("n_double_layers", 0)
    n_layers = mmdit_config.get("n_layers", 0)

    key_map = {}
    for i in range(n_layers):
        if i < n_double_layers:
            index = i
            prefix_from = "joint_transformer_blocks"
            prefix_to = "{}double_layers".format(output_prefix)
            block_map = {
                "attn.to_q.weight": "attn.w2q.weight",
                "attn.to_k.weight": "attn.w2k.weight",
                "attn.to_v.weight": "attn.w2v.weight",
                "attn.to_out.0.weight": "attn.w2o.weight",
                "attn.add_q_proj.weight": "attn.w1q.weight",
                "attn.add_k_proj.weight": "attn.w1k.weight",
                "attn.add_v_proj.weight": "attn.w1v.weight",
                "attn.to_add_out.weight": "attn.w1o.weight",
                "ff.linear_1.weight": "mlpX.c_fc1.weight",
                "ff.linear_2.weight": "mlpX.c_fc2.weight",
                "ff.out_projection.weight": "mlpX.c_proj.weight",
                "ff_context.linear_1.weight": "mlpC.c_fc1.weight",
                "ff_context.linear_2.weight": "mlpC.c_fc2.weight",
                "ff_context.out_projection.weight": "mlpC.c_proj.weight",
                "norm1.linear.weight": "modX.1.weight",
                "norm1_context.linear.weight": "modC.1.weight",
            }
        else:
            index = i - n_double_layers
            prefix_from = "single_transformer_blocks"
            prefix_to = "{}single_layers".format(output_prefix)

            block_map = {
                "attn.to_q.weight": "attn.w1q.weight",
                "attn.to_k.weight": "attn.w1k.weight",
                "attn.to_v.weight": "attn.w1v.weight",
                "attn.to_out.0.weight": "attn.w1o.weight",
                "norm1.linear.weight": "modCX.1.weight",
                "ff.linear_1.weight": "mlp.c_fc1.weight",
                "ff.linear_2.weight": "mlp.c_fc2.weight",
                "ff.out_projection.weight": "mlp.c_proj.weight"
            }

        for k in block_map:
            key_map["{}.{}.{}".format(prefix_from, index, k)] = "{}.{}.{}".format(prefix_to, index, block_map[k])

    MAP_BASIC = {
        ("positional_encoding", "pos_embed.pos_embed"),
        ("register_tokens", "register_tokens"),
        ("t_embedder.mlp.0.weight", "time_step_proj.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "time_step_proj.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "time_step_proj.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "time_step_proj.linear_2.bias"),
        ("cond_seq_linear.weight", "context_embedder.weight"),
        ("init_x_linear.weight", "pos_embed.proj.weight"),
        ("init_x_linear.bias", "pos_embed.proj.bias"),
        ("final_linear.weight", "proj_out.weight"),
        ("modF.1.weight", "norm_out.linear.weight", swap_scale_shift),
    }

    for k in MAP_BASIC:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output


def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict


def safetensors_header(safetensors_path, max_size=100 * 1024 * 1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack('<Q', header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj, attr, value):
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj
