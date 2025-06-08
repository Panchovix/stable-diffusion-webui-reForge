from typing import Callable

import torch
import tqdm
from PIL import Image

from modules import devices, images, shared, torch_utils
from modules_forge.utils import pil_image_to_torch_bgr, torch_bgr_to_pil_image


# Forge-Classic PR #83 by LoganBooker - no measureable improvement for me
# def try_apply_spandrel_patches():
    # try:
        # from spandrel.architectures.__arch_helpers.block import ResidualDenseBlock_5C, RRDB

        # _orig_init = ResidualDenseBlock_5C.__init__

        # def ResidualDenseBlock_5C_init(self, *args, **kwargs):
            # _orig_init(self, *args, **kwargs)
            # self.nf, self.gc = kwargs.get('nf', 64), kwargs.get('gc', 32)

        # def ResidualDenseBlock_5C_forward(self, x):
            # B, _, H, W = x.shape
            # nf, gc = self.nf, self.gc

            # buf = torch.empty((B, nf + 4 * gc, H, W), dtype=x.dtype, device=x.device)
            # buf[:, :nf].copy_(x)

            # x1 = self.conv1(x)
            # buf[:, nf:nf+gc].copy_(x1)

            # x2 = self.conv2(buf[:, :nf+gc])
            # if self.conv1x1: x2.add_(self.conv1x1(x))
            # buf[:, nf+gc:nf+2*gc].copy_(x2)

            # x3 = self.conv3(buf[:, :nf+2*gc])
            # buf[:, nf+2*gc:nf+3*gc].copy_(x3)

            # x4 = self.conv4(buf[:, :nf+3*gc])
            # if self.conv1x1: x4.add_(x2)
            # buf[:, nf+3*gc:nf+4*gc].copy_(x4)

            # x5 = self.conv5(buf)
            # return x5.mul_(0.2).add_(x)

        # def RRDB_forward(self, x):
            # return self.RDB3(self.RDB2(self.RDB1(x))).mul_(0.2).add_(x)

        # ResidualDenseBlock_5C.__init__ = ResidualDenseBlock_5C_init
        # ResidualDenseBlock_5C.forward = ResidualDenseBlock_5C_forward
        # RRDB.forward = RRDB_forward

        # print("[Upscalers] Patched Spandrel blocks with optimized forward passes")
    # except Exception as e:
        # print(f"[Upscalers] Failed to patch Spandrel blocks: {type(e).__name__}: {e}")

# try_apply_spandrel_patches()

def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    if tile_size <= 0:
        output = upscale_pil_patch(model, img)
        return output

    grid = images.split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []

    with tqdm.tqdm(total=grid.tile_count, desc=desc, disable=not shared.opts.enable_upscale_progressbar) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                if shared.state.interrupted:
                    return img
                output = upscale_pil_patch(model, tile)
                scale_factor = output.width // tile.width
                newrow.append([x * scale_factor, w * scale_factor, output])
                p.update(1)
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(
        newtiles,
        tile_w=int(grid.tile_w * scale_factor),
        tile_h=int(grid.tile_h * scale_factor),
        image_w=int(grid.image_w * scale_factor),
        image_h=int(grid.image_h * scale_factor),
        overlap=int(grid.overlap * scale_factor),
    )
    return images.combine_grid(newgrid)


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
):
    # Alternative implementation of `upscale_with_model` originally used by
    # SwinIR and ScuNET.  It differs from `upscale_with_model` in that tiling and
    # weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    # Pillow space without weighting.

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )

    weights = torch.zeros_like(result)

    with tqdm.tqdm(total=len(h_idx_list) * len(w_idx_list), desc=desc, disable=not shared.opts.enable_upscale_progressbar) as pbar:
        for h_idx in h_idx_list:
            if shared.state.interrupted or shared.state.skipped:
                break

            for w_idx in w_idx_list:
                if shared.state.interrupted or shared.state.skipped:
                    break

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx : h_idx + tile_size,
                    w_idx : w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                result[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch)

                out_patch_mask = torch.ones_like(out_patch)

                weights[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.
    """
    param = torch_utils.get_param(model)
    tensor = pil_image_to_torch_bgr(img).to(dtype=param.dtype).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = tiled_upscale_2(
            tensor,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            desc=desc,
            device=param.device,
        )
    return torch_bgr_to_pil_image(output)
