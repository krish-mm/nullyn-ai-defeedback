"""Load packaged Nullyn AI detector weights and classic-notch thresholds."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np

_DATA = files("nullyn") / "data"

MODEL_FILES = {
    "v0": "nullyn_v0.npz",
    "nullyn_v0": "nullyn_v0.npz",
    "v0.1": "nullyn_v0_1.npz",
    "nullyn_v0.1": "nullyn_v0_1.npz",
    "nullyn_v0_1": "nullyn_v0_1.npz",
}


def load_detector_params(name="v0", path=None):
    """Return a TinyMLP parameter dict (mu, sd, W, b, thr, nparams).

    ``name`` is ``v0`` (default; the kept model) or ``v0.1`` (the retrained
    variant that was not adopted). Pass ``path`` to load an ``.npz`` file
    from disk instead.
    """
    if path is not None:
        src = Path(path)
    else:
        key = str(name).lower()
        if key not in MODEL_FILES:
            raise KeyError(f"unknown model {name!r}; expected one of {sorted(MODEL_FILES)}")
        src = _DATA / MODEL_FILES[key]
    with src.open("rb") as fh:
        z = np.load(fh)
        params = dict(
            mu=np.asarray(z["mu"]),
            sd=np.asarray(z["sd"]),
            W=[np.asarray(z["W0"]), np.asarray(z["W1"]), np.asarray(z["W2"])],
            b=[np.asarray(z["b0"]), np.asarray(z["b1"]), np.asarray(z["b2"])],
            thr=float(z["thr"]),
            nparams=int(z["nparams"]),
        )
    return params


def load_classic_config(path=None):
    """Return the tuned classic-notch thresholds (pnpr, phpr, papr, need)."""
    src = Path(path) if path is not None else Path(str(_DATA / "classic_tuned.json"))
    if path is None:
        text = (_DATA / "classic_tuned.json").read_text()
        return json.loads(text)
    return json.loads(src.read_text())
