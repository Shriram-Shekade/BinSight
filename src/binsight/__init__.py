"""BinSight: zero-shot smart bin demo."""

import os

# Must be set before torch is imported: unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
