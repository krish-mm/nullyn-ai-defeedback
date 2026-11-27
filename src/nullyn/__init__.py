"""Nullyn AI: acoustic feedback suppression for live sound.

The audio path is a bank of per-sample IIR notch filters, optionally preceded
by a closed-loop canceller (see ``nullyn.closedloop``). A side-chain detector
decides which spectral peaks are howl. The detector never delays the audio.

The closed-loop canceller lives in the separate ``nullyn.closedloop`` package
so it can be omitted from a public release if needed.
"""

from .constants import DEFAULT_BLOCK_MS, RESEARCH_FS
from .engine import NullynEngine
from .models import load_classic_config, load_detector_params
from .notch import ClassicNotch, NullynV0, NotchBank

__version__ = "0.3.1"
__all__ = [
    "NullynEngine",
    "NullynV0",
    "ClassicNotch",
    "NotchBank",
    "load_detector_params",
    "load_classic_config",
    "__version__",
    "RESEARCH_FS",
    "DEFAULT_BLOCK_MS",
]
