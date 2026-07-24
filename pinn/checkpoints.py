"""Saving and reloading trained fields, so figures never require retraining.

Every figure in this repo that shows a *field* -- the heat and Burgers error
heatmaps, the Burgers slices, the spectral-bias profile at k=32, the adaptive
-collocation error panels, the hard-BC comparison -- is a picture of a trained
network evaluated on a grid. A CSV of the loss history cannot regenerate it;
only the weights can. Training those models takes about an hour of CPU, which
is exactly the kind of cost that makes a reader take a repo's figures on faith.

So the weights ship with the repo, and this module is the contract for them.

A checkpoint is a dict holding the state dict *plus the constructor arguments
that rebuild the module*, because a bare ``state_dict`` is not enough to
reconstruct anything: it records tensor shapes but not that
``FourierMLP(sigma=(5.0, 1.0))`` drew a specific frozen ``B`` from a specific
seed, and rebuilding with the wrong ``sigma`` would silently give a different
function that still loads cleanly. Storing the config makes the rebuild exact
and turns a mismatch into a loud error.

Design notes
------------
- ``weights_only=True`` on load. Checkpoints hold tensors and plain
  dicts/lists/numbers, never pickled objects, so nothing here needs the
  unrestricted unpickler (which executes arbitrary code on an untrusted file).
- Buffers ride along in the state dict, which is what makes ``FourierMLP``
  reloadable: its random projection ``B`` is a registered buffer, so the saved
  matrix is restored rather than redrawn.
- The loaded module is returned in ``eval()`` mode but with gradients enabled:
  the derivative helpers differentiate *through* the network with respect to
  its inputs, so a checkpoint loaded under ``torch.no_grad()`` would be useless
  for the residual figures.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch
from torch import nn

from pinn.features import FourierMLP
from pinn.model import MLP

FORMAT = 1

# The module types a checkpoint may rebuild, keyed by class name. Anything else
# raises on load rather than being reconstructed by guesswork.
BUILDERS: dict[str, type[nn.Module]] = {
    "MLP": MLP,
    "FourierMLP": FourierMLP,
}


def save_model(
    model: nn.Module,
    path: str,
    config: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Write ``model`` to ``path`` together with the kwargs that rebuild it.

    Parameters
    ----------
    model : nn.Module
        A trained :class:`~pinn.model.MLP` or
        :class:`~pinn.features.FourierMLP`.
    config : mapping
        Exactly the keyword arguments its constructor was called with. Passed
        back verbatim by :func:`load_model`, so a wrong entry here becomes a
        shape mismatch at load time instead of a wrong figure.
    meta : mapping, optional
        Free-form provenance for the reader: the experiment that produced it,
        the step count, the error it reached. Never used to rebuild anything.
    """
    kind = type(model).__name__
    if kind not in BUILDERS:
        raise ValueError(
            f"{kind} is not a checkpointable model type "
            f"(expected one of {sorted(BUILDERS)})"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "kind": kind,
            "config": dict(config),
            "meta": dict(meta or {}),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def load_model(path: str) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model saved at ``path``; return ``(model, meta)``.

    The rebuild is strict: the stored config must produce a module whose
    parameters and buffers match the stored tensors exactly, so a checkpoint
    that no longer matches the code fails here rather than producing a
    plausible-looking wrong field.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    fmt = blob.get("format")
    if fmt != FORMAT:
        raise ValueError(f"unsupported checkpoint format {fmt!r} (expected {FORMAT})")
    kind = blob["kind"]
    if kind not in BUILDERS:
        raise ValueError(f"unknown model type {kind!r} in {path}")

    model = BUILDERS[kind](**blob["config"])
    model.load_state_dict(blob["state_dict"])  # strict: shapes and names must match
    model.eval()
    return model, dict(blob["meta"])
