"""
Temporary debug probe for image display diagnosis.
Exposes /debug/image-probe — open in browser after a generation attempt.
Remove once the issue is identified.
"""
import os
import traceback
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def setup_debug_probe(app: FastAPI, demo) -> None:
    """Patch file-serving and add diagnostic endpoint. Call from app_started_callback."""
    _patch_file_route(app, demo)
    _add_probe_endpoint(app, demo)


def _patch_file_route(app: FastAPI, demo) -> None:
    """Wrap the /file= route to print every request + access-check result."""
    import gradio.routes as gr_routes
    import gradio.utils as gr_utils

    original_routes = [r for r in app.routes if getattr(r, "path", None) == "/file={path_or_url:path}"]
    if not original_routes:
        print("[DEBUG PROBE] WARNING: could not find /file= route to patch")
        return

    @app.get("/file={path_or_url:path}", include_in_schema=False)
    async def file_probe(path_or_url: str):
        from fastapi import HTTPException
        import gradio.routes

        blocks = app.get_blocks()
        abs_path = gr_utils.abspath(Path(path_or_url))

        in_blocklist = any(gr_utils.is_in_or_equal(abs_path, bp) for bp in blocks.blocked_paths)
        is_dotfile = any(part.startswith(".") for part in abs_path.parts)
        is_dir = abs_path.is_dir()
        exists = abs_path.exists()
        in_app_dir = gr_utils.is_in_or_equal(abs_path, app.cwd)
        created_by_app = str(abs_path) in set().union(*blocks.temp_file_sets) if blocks.temp_file_sets else False
        in_allowlist = any(gr_utils.is_in_or_equal(abs_path, ap) for ap in blocks.allowed_paths)
        was_uploaded = gr_utils.is_in_or_equal(abs_path, app.uploaded_file_dir)
        allowed = (in_app_dir or created_by_app or in_allowlist or was_uploaded) and not in_blocklist and not is_dotfile and not is_dir

        print(
            f"[DEBUG /file=] {path_or_url}\n"
            f"  abs: {abs_path}\n"
            f"  exists={exists} in_app_dir={in_app_dir} created_by_app={created_by_app} "
            f"in_allowlist={in_allowlist} was_uploaded={was_uploaded}\n"
            f"  blocked={in_blocklist} dotfile={is_dotfile} dir={is_dir} → ALLOWED={allowed}\n"
            f"  temp_file_sets count: {len(blocks.temp_file_sets)}, "
            f"  total tracked files: {sum(len(s) for s in blocks.temp_file_sets)}"
        )

        if not allowed:
            raise HTTPException(403, f"[PROBE] File not allowed: {path_or_url}")
        if not exists:
            raise HTTPException(404, f"[PROBE] File not found: {path_or_url}")

        from starlette.responses import FileResponse
        return FileResponse(abs_path, headers={"Accept-Ranges": "bytes"})

    # Also patch Gallery.postprocess to log every real call
    import gradio.components.gallery as _gallery_mod
    _orig_postprocess = _gallery_mod.Gallery.postprocess

    def _patched_postprocess(self, y):
        result = _orig_postprocess(self, y)
        n_in = len(y) if y else 0
        print(f"[DEBUG Gallery.postprocess] input={n_in} items, output={len(result)} items")
        if y:
            for i, img in enumerate(y[:3]):
                print(f"  in[{i}] type={type(img).__name__} size={getattr(img, 'size', '?')} mode={getattr(img, 'mode', '?')}")
        if result:
            for i, item in enumerate(result[:3]):
                print(f"  out[{i}] = {item}")
        return result

    _gallery_mod.Gallery.postprocess = _patched_postprocess
    print("[DEBUG PROBE] Gallery.postprocess patched with logging")


def _add_probe_endpoint(app: FastAPI, demo) -> None:

    @app.get("/debug/last-output", response_class=HTMLResponse, include_in_schema=False)
    async def last_output():
        """Inspect the last saved output image directly."""
        import glob, io, base64
        from PIL import Image

        lines = ["<h2>Last Output Image Probe</h2><pre>"]
        pattern = "/Users/nickzeng/Documents/stable-diffusion-webui-reForge/outputs/txt2img-images/**/*.png"
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            return HTMLResponse("<pre>No output files found</pre>")

        last = files[-1]
        lines.append(f"File: {last}")
        try:
            img = Image.open(last)
            lines.append(f"Mode: {img.mode}  Size: {img.size}")
            import numpy as np
            arr = np.array(img)
            lines.append(f"dtype={arr.dtype}  shape={arr.shape}")
            lines.append(f"min={arr.min()}  max={arr.max()}")
            if img.mode == "RGBA":
                alpha = arr[:, :, 3]
                lines.append(f"Alpha channel: min={alpha.min()}  max={alpha.max()}  mean={alpha.mean():.1f}")
                lines.append(f"  (0=transparent, 255=opaque — if max==0, image is invisible!)")
            # Show inline
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            lines.append(f'</pre><p>Inline preview (data URI):</p><img src="data:image/png;base64,{b64}" style="max-width:400px;border:2px solid blue"/>')
            lines.append(f'<p>Via /file= route: <a href="/file={last}" target="_blank">/file={last}</a></p>')
            # Also convert to RGB and show
            rgb = img.convert("RGB")
            buf2 = io.BytesIO()
            rgb.save(buf2, format="PNG")
            b64_rgb = base64.b64encode(buf2.getvalue()).decode()
            lines.append(f'<p>Converted to RGB (flattens alpha):</p><img src="data:image/png;base64,{b64_rgb}" style="max-width:400px;border:2px solid red"/><pre>')
        except Exception as e:
            import traceback
            lines.append(f"Error: {traceback.format_exc()}")
        lines.append("</pre>")
        return HTMLResponse("".join(lines))

    @app.get("/debug/image-probe", response_class=HTMLResponse, include_in_schema=False)
    async def image_probe():
        from PIL import Image
        import io, base64

        lines = ["<h2>Image Display Probe</h2><pre>"]

        # 1. Create a test PIL image
        try:
            img = Image.new("RGB", (200, 200), color=(180, 60, 60))
            lines.append("✓ PIL image created (200x200 red square)")
        except Exception as e:
            lines.append(f"✗ PIL image creation failed: {e}")
            return HTMLResponse("".join(lines) + "</pre>")

        # 2. Run through Gallery.postprocess
        try:
            blocks = app.get_blocks()
            # Find the first Gallery component
            gallery_comp = None
            for comp in blocks.blocks.values():
                if type(comp).__name__ == "Gallery":
                    gallery_comp = comp
                    break

            if gallery_comp is None:
                lines.append("✗ No Gallery component found in blocks")
            else:
                result = gallery_comp.postprocess([img])
                lines.append(f"✓ Gallery.postprocess returned {len(result)} item(s)")
                if result:
                    item = result[0]
                    file_path = item["name"] if isinstance(item, dict) else item
                    lines.append(f"  file_path: {file_path}")
                    lines.append(f"  file exists on disk: {os.path.exists(file_path)}")

                    # 3. Check security gates
                    from pathlib import Path
                    import gradio.utils as gr_utils
                    abs_path = gr_utils.abspath(Path(file_path))
                    in_app_dir = gr_utils.is_in_or_equal(abs_path, app.cwd)
                    created_by_app = str(abs_path) in set().union(*blocks.temp_file_sets) if blocks.temp_file_sets else False
                    in_allowlist = any(gr_utils.is_in_or_equal(abs_path, ap) for ap in blocks.allowed_paths)
                    was_uploaded = gr_utils.is_in_or_equal(abs_path, app.uploaded_file_dir)
                    is_dotfile = any(part.startswith(".") for part in abs_path.parts)

                    lines.append(f"\n  Security gate check:")
                    lines.append(f"    app.cwd = {app.cwd}")
                    lines.append(f"    in_app_dir    = {in_app_dir}")
                    lines.append(f"    created_by_app= {created_by_app}  (temp_file_sets count: {len(blocks.temp_file_sets)})")
                    lines.append(f"    in_allowlist  = {in_allowlist}")
                    lines.append(f"    was_uploaded  = {was_uploaded}")
                    lines.append(f"    is_dotfile    = {is_dotfile}")
                    allowed = (in_app_dir or created_by_app or in_allowlist or was_uploaded) and not is_dotfile
                    lines.append(f"    → WOULD BE ALLOWED: {allowed}")

                    # 4. Inline base64 preview (bypasses file serving entirely)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    lines.append(f"\n  Inline base64 preview (no file serving):")
                    lines.append(f'</pre><img src="data:image/png;base64,{b64}" style="border:2px solid green"/><pre>')

                    # 5. Link via /file= route
                    lines.append(f'\n  Via /file= route: <a href="/file={file_path}" target="_blank">/file={file_path}</a>')
                    lines.append(f"  (open link above — if it 403s or 404s, the file serving is the bug)")

        except Exception as e:
            lines.append(f"✗ Gallery test failed:\n{traceback.format_exc()}")

        lines.append("</pre>")
        return HTMLResponse("".join(lines))

    print("[DEBUG PROBE] /debug/image-probe endpoint registered")
