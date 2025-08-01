import ldm_patched.contrib.nodes_model_merging

class ModelMergeSD1(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(12):
            arg_dict["input_blocks.{}.".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}.".format(i)] = argument

        for i in range(12):
            arg_dict["output_blocks.{}.".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}


class ModelMergeSDXL(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(9):
            arg_dict["input_blocks.{}".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}".format(i)] = argument

        for i in range(9):
            arg_dict["output_blocks.{}".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}

class ModelMergeSD3_2B(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embed."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["context_embedder."] = argument
        arg_dict["y_embedder."] = argument
        arg_dict["t_embedder."] = argument

        for i in range(24):
            arg_dict["joint_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}


class ModelMergeAuraflow(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["init_x_linear."] = argument
        arg_dict["positional_encoding"] = argument
        arg_dict["cond_seq_linear."] = argument
        arg_dict["register_tokens"] = argument
        arg_dict["t_embedder."] = argument

        for i in range(4):
            arg_dict["double_layers.{}.".format(i)] = argument

        for i in range(32):
            arg_dict["single_layers.{}.".format(i)] = argument

        arg_dict["modF."] = argument
        arg_dict["final_linear."] = argument

        return {"required": arg_dict}

class ModelMergeFlux1(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["img_in."] = argument
        arg_dict["time_in."] = argument
        arg_dict["guidance_in"] = argument
        arg_dict["vector_in."] = argument
        arg_dict["txt_in."] = argument

        for i in range(19):
            arg_dict["double_blocks.{}.".format(i)] = argument

        for i in range(38):
            arg_dict["single_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeSD35_Large(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embed."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["context_embedder."] = argument
        arg_dict["y_embedder."] = argument
        arg_dict["t_embedder."] = argument

        for i in range(38):
            arg_dict["joint_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeMochiPreview(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_frequencies."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["t5_y_embedder."] = argument
        arg_dict["t5_yproj."] = argument

        for i in range(48):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeLTXV(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["patchify_proj."] = argument
        arg_dict["adaln_single."] = argument
        arg_dict["caption_projection."] = argument

        for i in range(28):
            arg_dict["transformer_blocks.{}.".format(i)] = argument

        arg_dict["scale_shift_table"] = argument
        arg_dict["proj_out."] = argument

        return {"required": arg_dict}

class ModelMergeCosmos7B(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embedder."] = argument
        arg_dict["extra_pos_embedder."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["affline_norm."] = argument


        for i in range(28):
            arg_dict["blocks.block{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeCosmos14B(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embedder."] = argument
        arg_dict["extra_pos_embedder."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["affline_norm."] = argument


        for i in range(36):
            arg_dict["blocks.block{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeWAN2_1(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "1.3B model has 30 blocks, 14B model has 40 blocks. Image to video model has the extra img_emb."

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["patch_embedding."] = argument
        arg_dict["time_embedding."] = argument
        arg_dict["time_projection."] = argument
        arg_dict["text_embedding."] = argument
        arg_dict["img_emb."] = argument

        for i in range(40):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["head."] = argument

        return {"required": arg_dict}

class ModelMergeCosmosPredict2_2B(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embedder."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["t_embedding_norm."] = argument


        for i in range(28):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeCosmosPredict2_14B(ldm_patched.contrib.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embedder."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["t_embedding_norm."] = argument


        for i in range(36):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

# Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
NODE_CLASS_MAPPINGS = {
    "ModelMergeSD1": ModelMergeSD1,
    "ModelMergeSD2": ModelMergeSD1, #SD1 and SD2 have the same blocks
    "ModelMergeSDXL": ModelMergeSDXL,
    "ModelMergeSD3_2B": ModelMergeSD3_2B,
    "ModelMergeAuraflow": ModelMergeAuraflow,
    "ModelMergeFlux1": ModelMergeFlux1,
    "ModelMergeSD35_Large": ModelMergeSD35_Large,
    "ModelMergeMochiPreview": ModelMergeMochiPreview,
    "ModelMergeLTXV": ModelMergeLTXV,
    "ModelMergeCosmos7B": ModelMergeCosmos7B,
    "ModelMergeCosmos14B": ModelMergeCosmos14B,
    "ModelMergeWAN2_1": ModelMergeWAN2_1,
    "ModelMergeCosmosPredict2_2B": ModelMergeCosmosPredict2_2B,
    "ModelMergeCosmosPredict2_14B": ModelMergeCosmosPredict2_14B,
}
