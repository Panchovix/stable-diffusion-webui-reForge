"""
Lazy backend accessors for the k-diffusion sampling library.

Both backends (k_diff and ldm_patched) are forks of the same k-diffusion
library and expose identical APIs. The active backend is chosen at runtime
via opts.sd_sampling, which is not available at module-import time.

Use these helpers instead of conditional imports at module level.
"""

from modules import shared


def get_sampling():
    """Return the k-diffusion sampling module for the active backend."""
    if shared.opts.sd_sampling == "A1111":
        from k_diff.k_diffusion import sampling
    else:
        from ldm_patched.k_diffusion import sampling
    return sampling


def get_k_diffusion():
    """Return the top-level k_diffusion package for the active backend."""
    if shared.opts.sd_sampling == "A1111":
        import k_diff.k_diffusion as k_diffusion
    else:
        import ldm_patched.k_diffusion as k_diffusion
    return k_diffusion


def get_external():
    """Return the k-diffusion external module for the active backend."""
    if shared.opts.sd_sampling == "A1111":
        from k_diff.k_diffusion import external
    else:
        from ldm_patched.k_diffusion import external
    return external
