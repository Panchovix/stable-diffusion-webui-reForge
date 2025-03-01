from __future__ import annotations
from modules.shared import opts

if opts.sd_processing == "reForge OG":
    from processing_processors.forge import *

elif opts.sd_processing == "reForge A1111":
    from processing_processors.automatic import *
else:
    raise ValueError(f"Unknown SD Processing option: {opts.sd_processing}")
