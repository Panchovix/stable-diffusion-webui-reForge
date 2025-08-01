# from __future__ import annotations

# import av
# import torchaudio
# import torch
# import ldm_patched.modules.model_management
# import folder_paths
# import os
# import io
# import json
# import random
# import hashlib
# import ldm_patched.contrib.node_helpers
# # from ldm_patched.modules.args_parser import args
# from ldm_patched.modules.ldmpatched_types.node_typing import FileLocator

# class EmptyLatentAudio:
#     def __init__(self):
#         self.device = ldm_patched.modules.model_management.intermediate_device()

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1}),
#                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
#                              }}
#     RETURN_TYPES = ("LATENT",)
#     FUNCTION = "generate"

#     CATEGORY = "latent/audio"

#     def generate(self, seconds, batch_size):
#         length = round((seconds * 44100 / 2048) / 2) * 2
#         latent = torch.zeros([batch_size, 64, length], device=self.device)
#         return ({"samples":latent, "type": "audio"}, )

# class ConditioningStableAudio:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"positive": ("CONDITIONING", ),
#                              "negative": ("CONDITIONING", ),
#                              "seconds_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
#                              "seconds_total": ("FLOAT", {"default": 47.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
#                              }}

#     RETURN_TYPES = ("CONDITIONING","CONDITIONING")
#     RETURN_NAMES = ("positive", "negative")

#     FUNCTION = "append"

#     CATEGORY = "conditioning"

#     def append(self, positive, negative, seconds_start, seconds_total):
#         positive = ldm_patched.contrib.node_helpers.conditioning_set_values(positive, {"seconds_start": seconds_start, "seconds_total": seconds_total})
#         negative = ldm_patched.contrib.node_helpers.conditioning_set_values(negative, {"seconds_start": seconds_start, "seconds_total": seconds_total})
#         return (positive, negative)

# class VAEEncodeAudio:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "audio": ("AUDIO", ), "vae": ("VAE", )}}
#     RETURN_TYPES = ("LATENT",)
#     FUNCTION = "encode"

#     CATEGORY = "latent/audio"

#     def encode(self, vae, audio):
#         sample_rate = audio["sample_rate"]
#         if 44100 != sample_rate:
#             waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
#         else:
#             waveform = audio["waveform"]

#         t = vae.encode(waveform.movedim(1, -1))
#         return ({"samples":t}, )

# class VAEDecodeAudio:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
#     RETURN_TYPES = ("AUDIO",)
#     FUNCTION = "decode"

#     CATEGORY = "latent/audio"

#     def decode(self, vae, samples):
#         audio = vae.decode(samples["samples"]).movedim(-1, 1)
#         std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
#         std[std < 1.0] = 1.0
#         audio /= std
#         return ({"waveform": audio, "sample_rate": 44100}, )


# def save_audio(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None, quality="128k"):

#     filename_prefix += self.prefix_append
#     full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
#     results: list[FileLocator] = []

#     # # Prepare metadata dictionary
#     # metadata = {}
#     # if not args.disable_metadata:
#     #     if prompt is not None:
#     #         metadata["prompt"] = json.dumps(prompt)
#     #     if extra_pnginfo is not None:
#     #         for x in extra_pnginfo:
#     #             metadata[x] = json.dumps(extra_pnginfo[x])

#     # Opus supported sample rates
#     OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

#     for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
#         filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
#         file = f"{filename_with_batch_num}_{counter:05}_.{format}"
#         output_path = os.path.join(full_output_folder, file)

#         # Use original sample rate initially
#         sample_rate = audio["sample_rate"]

#         # Handle Opus sample rate requirements
#         if format == "opus":
#             if sample_rate > 48000:
#                 sample_rate = 48000
#             elif sample_rate not in OPUS_RATES:
#                 # Find the next highest supported rate
#                 for rate in sorted(OPUS_RATES):
#                     if rate > sample_rate:
#                         sample_rate = rate
#                         break
#                 if sample_rate not in OPUS_RATES:  # Fallback if still not supported
#                     sample_rate = 48000

#             # Resample if necessary
#             if sample_rate != audio["sample_rate"]:
#                 waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

#         # Create output with specified format
#         output_buffer = io.BytesIO()
#         output_container = av.open(output_buffer, mode='w', format=format)

#         # # Set metadata on the container
#         # for key, value in metadata.items():
#         #     output_container.metadata[key] = value

#         # Set up the output stream with appropriate properties
#         if format == "opus":
#             out_stream = output_container.add_stream("libopus", rate=sample_rate)
#             if quality == "64k":
#                 out_stream.bit_rate = 64000
#             elif quality == "96k":
#                 out_stream.bit_rate = 96000
#             elif quality == "128k":
#                 out_stream.bit_rate = 128000
#             elif quality == "192k":
#                 out_stream.bit_rate = 192000
#             elif quality == "320k":
#                 out_stream.bit_rate = 320000
#         elif format == "mp3":
#             out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
#             if quality == "V0":
#                 #TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
#                 out_stream.codec_context.qscale = 1
#             elif quality == "128k":
#                 out_stream.bit_rate = 128000
#             elif quality == "320k":
#                 out_stream.bit_rate = 320000
#         else: #format == "flac":
#             out_stream = output_container.add_stream("flac", rate=sample_rate)

#         frame = av.AudioFrame.from_ndarray(waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format='flt', layout='mono' if waveform.shape[0] == 1 else 'stereo')
#         frame.sample_rate = sample_rate
#         frame.pts = 0
#         output_container.mux(out_stream.encode(frame))

#         # Flush encoder
#         output_container.mux(out_stream.encode(None))

#         # Close containers
#         output_container.close()

#         # Write the output to file
#         output_buffer.seek(0)
#         with open(output_path, 'wb') as f:
#             f.write(output_buffer.getbuffer())

#         results.append({
#             "filename": file,
#             "subfolder": subfolder,
#             "type": self.type
#         })
#         counter += 1

#     return { "ui": { "audio": results } }

# class SaveAudio:
#     def __init__(self):
#         self.output_dir = folder_paths.get_output_directory()
#         self.type = "output"
#         self.prefix_append = ""

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "audio": ("AUDIO", ),
#                             "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
#                             },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

#     RETURN_TYPES = ()
#     FUNCTION = "save_flac"

#     OUTPUT_NODE = True

#     CATEGORY = "audio"

#     def save_flac(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None):
#         return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo)

# class SaveAudioMP3:
#     def __init__(self):
#         self.output_dir = folder_paths.get_output_directory()
#         self.type = "output"
#         self.prefix_append = ""

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "audio": ("AUDIO", ),
#                             "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
#                             "quality": (["V0", "128k", "320k"], {"default": "V0"}),
#                             },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

#     RETURN_TYPES = ()
#     FUNCTION = "save_mp3"

#     OUTPUT_NODE = True

#     CATEGORY = "audio"

#     def save_mp3(self, audio, filename_prefix="ComfyUI", format="mp3", prompt=None, extra_pnginfo=None, quality="128k"):
#         return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo, quality)

# class SaveAudioOpus:
#     def __init__(self):
#         self.output_dir = folder_paths.get_output_directory()
#         self.type = "output"
#         self.prefix_append = ""

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "audio": ("AUDIO", ),
#                             "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
#                             "quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
#                             },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

#     RETURN_TYPES = ()
#     FUNCTION = "save_opus"

#     OUTPUT_NODE = True

#     CATEGORY = "audio"

#     def save_opus(self, audio, filename_prefix="ComfyUI", format="opus", prompt=None, extra_pnginfo=None, quality="V3"):
#         return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo, quality)

# class PreviewAudio(SaveAudio):
#     def __init__(self):
#         self.output_dir = folder_paths.get_temp_directory()
#         self.type = "temp"
#         self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"audio": ("AUDIO", ), },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

# def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
#     """Convert audio to float 32 bits PCM format."""
#     if wav.dtype.is_floating_point:
#         return wav
#     elif wav.dtype == torch.int16:
#         return wav.float() / (2 ** 15)
#     elif wav.dtype == torch.int32:
#         return wav.float() / (2 ** 31)
#     raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

# def load(filepath: str) -> tuple[torch.Tensor, int]:
#     with av.open(filepath) as af:
#         if not af.streams.audio:
#             raise ValueError("No audio stream found in the file.")

#         stream = af.streams.audio[0]
#         sr = stream.codec_context.sample_rate
#         n_channels = stream.channels

#         frames = []
#         length = 0
#         for frame in af.decode(streams=stream.index):
#             buf = torch.from_numpy(frame.to_ndarray())
#             if buf.shape[0] != n_channels:
#                 buf = buf.view(-1, n_channels).t()

#             frames.append(buf)
#             length += buf.shape[1]

#         if not frames:
#             raise ValueError("No audio frames decoded.")

#         wav = torch.cat(frames, dim=1)
#         wav = f32_pcm(wav)
#         return wav, sr

# class LoadAudio:
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
#         return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

#     CATEGORY = "audio"

#     RETURN_TYPES = ("AUDIO", )
#     FUNCTION = "load"

#     def load(self, audio):
#         audio_path = folder_paths.get_annotated_filepath(audio)
#         waveform, sample_rate = load(audio_path)
#         audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
#         return (audio, )

#     @classmethod
#     def IS_CHANGED(s, audio):
#         image_path = folder_paths.get_annotated_filepath(audio)
#         m = hashlib.sha256()
#         with open(image_path, 'rb') as f:
#             m.update(f.read())
#         return m.digest().hex()

#     @classmethod
#     def VALIDATE_INPUTS(s, audio):
#         if not folder_paths.exists_annotated_filepath(audio):
#             return "Invalid audio file: {}".format(audio)
#         return True

# # Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "EmptyLatentAudio": EmptyLatentAudio,
#     "VAEEncodeAudio": VAEEncodeAudio,
#     "VAEDecodeAudio": VAEDecodeAudio,
#     "SaveAudio": SaveAudio,
#     "SaveAudioMP3": SaveAudioMP3,
#     "SaveAudioOpus": SaveAudioOpus,
#     "LoadAudio": LoadAudio,
#     "PreviewAudio": PreviewAudio,
#     "ConditioningStableAudio": ConditioningStableAudio,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "EmptyLatentAudio": "Empty Latent Audio",
#     "VAEEncodeAudio": "VAE Encode Audio",
#     "VAEDecodeAudio": "VAE Decode Audio",
#     "PreviewAudio": "Preview Audio",
#     "LoadAudio": "Load Audio",
#     "SaveAudio": "Save Audio (FLAC)",
#     "SaveAudioMP3": "Save Audio (MP3)",
#     "SaveAudioOpus": "Save Audio (Opus)",
# }
