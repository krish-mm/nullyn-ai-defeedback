"""Closed-loop (reference-input) feedback cancellation for Nullyn AI.

This package is intentionally separate from the notch layer. A public release
can omit ``nullyn/closedloop/`` and the rest of the package still provides
the Nullyn AI v0 notch suppressor and the classic detector.

The canceller is a PEM-prewhitened partitioned-block frequency-domain NLMS
adaptive filter that uses a speaker-feed reference, with the notch layer as
fallback. See ``ClosedLoopAFC``.
"""

from .afc import ClosedLoopAFC, afc_block_size

__all__ = ["ClosedLoopAFC", "afc_block_size"]
