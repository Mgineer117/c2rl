"""Shared divergence guard used by every env family (classic + Isaac).

A poor policy (random init, or an RL algorithm's early exploration) can drive
the dynamics to a non-finite (NaN/Inf) state. Every environment in this repo
must survive that without ending the rollout: contraction metrics (AUC,
overshoot, contraction rate) are fit over a full, fixed-length trajectory, so
an early termination on divergence would truncate exactly the data the
analysis needs. The fix used everywhere is the same: replace any non-finite
element with the corresponding element of the previous (last known-finite)
state, and keep stepping.
"""
from __future__ import annotations

import numpy as np
import torch


def carry_forward_nonfinite(current, previous):
    """Element-wise: replace non-finite entries of ``current`` with ``previous``.

    ``current``/``previous`` must be the same shape — either numpy arrays
    (classic envs) or torch tensors (Isaac envs, batched over ``num_envs``).
    """
    if isinstance(current, torch.Tensor):
        return torch.where(torch.isfinite(current), current, previous)
    return np.where(np.isfinite(current), current, previous)
