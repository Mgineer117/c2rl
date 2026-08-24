"""Project-side observation preprocessors for the ``[x, xref, uref]``
path-tracking layout used throughout this repo.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import compute_space_size


class RunningRewardScaler(nn.Module):
    """Non-biasing reward normalizer for PPO's ``rewards_shaper`` hook.

    Divides the reward by a running estimate of its standard deviation (never
    subtracts a mean), so the reward fed to GAE is ~unit-variance regardless of
    the metric-bound-dependent scale of the Mahalanobis reward
    ``tracking_scaler·(eᵀM e − next_eᵀM next_e)`` (``M = W⁻¹`` has eigenvalues in
    ``[1/w_ub, 1/w_lb]``, so its magnitude — and its heavy tail — grows as
    ``1/w_lb``; see ``env_base.get_rewards`` and c2rl.py). This tames both that
    scale and the early-training transient before the cumulative ``value_norm``
    (``RunningStandardScaler`` on the value) can catch up, which it lags because
    the frozen-CMG reward is non-stationary as the policy improves.

    Why ``/std`` and no centering (the "does not bias the problem" property):
    multiplying every reward by a positive scalar ``c`` scales all returns and
    advantages by ``c``, leaving ``argmax_π`` unchanged. Subtracting a mean
    shifts returns by a state/time-dependent constant and can change the
    advantage structure — so only the positive-scalar divide is used.

    Reuses skrl's ``RunningStandardScaler`` purely for its parallel-variance
    tracking; only ``sqrt(running_variance)`` is consumed (running_mean is
    ignored). By default tracks the variance of the raw per-step reward
    (``gamma=0.0``), which needs no episode-reset/``done`` signal — the
    ``rewards_shaper`` hook does not receive ``done`` (see ppo.py
    ``record_transition``). Setting ``gamma>0`` switches to the SB3
    ``VecNormalize`` variant (std of the discounted return ``R = γ·R + r``); with
    no ``done`` reset this leaks across episode boundaries, so it is only
    appropriate at small ``gamma`` (this repo's ``discount_factor`` is ~0.01,
    where return ≈ reward and the two variants coincide anyway).
    """

    def __init__(
        self,
        *,
        gamma: float = 0.0,
        scale: float = 1.0,
        epsilon: float = 1e-8,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.epsilon = float(epsilon)
        self._scaler = RunningStandardScaler(size=1, epsilon=epsilon, device=device)
        # per-env discounted-return accumulator (lazily sized on first call)
        self._returns: torch.Tensor | None = None

    def forward(self, rewards: torch.Tensor, timestep=None, timesteps=None) -> torch.Tensor:
        # rewards: (num_envs, 1). Match skrl's rewards_shaper(rewards, timestep, timesteps).
        with torch.no_grad():
            if self.gamma > 0.0:
                if self._returns is None or self._returns.shape != rewards.shape:
                    self._returns = torch.zeros_like(rewards)
                self._returns = self.gamma * self._returns + rewards
                tracked = self._returns
            else:
                tracked = rewards
            # update running variance, then divide reward by std (no centering).
            self._scaler(tracked, train=True)
            std = torch.sqrt(self._scaler.running_variance.float()) + self.epsilon
        return self.scale * rewards / std


def _window_length(size, x_dim: int, u_dim: int) -> int:
    """Window length from the observation space — read from the declared Dict
    when there is one, else inferred from the flat width."""
    import gymnasium as gym
    if isinstance(size, gym.spaces.Dict) and "xrefs" in size.spaces:
        return int(size["xrefs"].shape[0])
    obs_dim = compute_space_size(size, occupied_size=True)
    if (obs_dim - x_dim) % (x_dim + u_dim):
        raise ValueError(
            f"PathTrackingObservationScaler: obs_dim={obs_dim} is not x_dim + "
            f"L*(x_dim + u_dim) for x_dim={x_dim}, u_dim={u_dim}.")
    return (obs_dim - x_dim) // (x_dim + u_dim)


class PathTrackingObservationScaler(nn.Module):
    """``RunningStandardScaler`` restricted to the ``x``/``xrefs`` portion of a
    ``{x, xrefs, urefs}`` observation, further excluding ``angle_idx`` columns.

    Two things must stay raw for this repo's residual/embedding math to be
    correct (see ``c2rl.py``'s module docstring and ``angle_utils.py``):

      - ``urefs``: the residual backbones (``control``/``mlp``, squashed or
        not) take ``urefs[0]`` straight out of the observation and add it to the
        network's feedback — for the squashed backbones, that add happens
        after tanh-squashing (see ``models.py``'s ``_TanhSquashMixin``).
        Normalizing ``uref`` would make the applied control law
        ``uref_norm + feedback`` instead of ``uref + feedback``, distorting
        the reference-tracking residual.
      - ``angle_idx`` columns of ``x``/``xrefs``: ``ref_window.Feats`` replaces each
        with ``(cos, sin)`` via ``embed_angles`` so the network sees a
        continuous, periodic input. Standardizing the raw angle first
        (``(theta - mean) / std``) breaks that periodicity — ``cos``/``sin``
        of a shifted-and-rescaled angle is not ``2*pi``-periodic in ``theta``.

    Everything else in ``x``/``xrefs`` (non-angle physical states) is
    standardized exactly like the stock ``RunningStandardScaler``, using its
    own running mean/std fit only over that normalized subset.
    """

    def __init__(
        self,
        size,
        *,
        x_dim: int,
        u_dim: int,
        angle_idx: Sequence[int] = (),
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        from .ref_window import RefWindow

        # The layout is whatever the space declares; build the mask off the same
        # flat ordering RefWindow.split uses (sorted keys: urefs, x, xrefs).
        window = RefWindow(x_dim=int(x_dim), u_dim=int(u_dim),
                           length=_window_length(size, x_dim, u_dim))
        obs_dim = compute_space_size(size, occupied_size=True)
        if obs_dim != window.flat_dim:
            raise ValueError(
                "PathTrackingObservationScaler: observation width does not match the "
                f"{{x, xrefs, urefs}} layout — got obs_dim={obs_dim}, expected "
                f"{window.flat_dim} (x_dim={x_dim}, u_dim={u_dim}, length={window.length})."
            )
        angle_set = {int(i) for i in angle_idx}
        L = window.length
        normalize = torch.ones(obs_dim, dtype=torch.bool)
        normalize[: L * u_dim] = False          # urefs block: never normalized
        x_start = L * u_dim
        xrefs_start = x_start + x_dim
        for i in angle_set:
            normalize[x_start + i] = False      # x block angle column
            for k in range(L):                  # every xrefs point's angle column
                normalize[xrefs_start + k * x_dim + i] = False
        norm_idx = normalize.nonzero(as_tuple=True)[0]
        self.register_buffer("_normalize_idx", norm_idx)
        self._scaler = RunningStandardScaler(
            size=int(norm_idx.numel()), epsilon=epsilon, clip_threshold=clip_threshold, device=device
        )

    def forward(
        self, x: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        if x is None:
            return None
        idx = self._normalize_idx
        scaled = self._scaler(x.index_select(-1, idx), train=train, inverse=inverse, no_grad=no_grad)
        return x.index_copy(-1, idx, scaled)
