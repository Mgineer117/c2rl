"""Unified contraction/tracking metrics + wandb logging — single source of truth.

Every algorithm (PPO / SAC / C3M / C2RL / LQR / SD-LQR) and every path-tracking
environment reports the same four stability metrics, computed the same
memory-efficient *streaming* way — from per-env running accumulators only
(e0, e_last, e_max, running sum of error norms, step count) — never storing the
full ``(num_envs, T)`` error tensor:

  * ``auc``                — area under the normalized error curve e(t)/e(0),
                             dt-weighted trapezoidal rule.  Lower is better.
  * ``contraction_rate``   — empirical exponential rate ``lambda`` from the
                             endpoints: ``e(T) = e(0)·exp(-lambda·T·dt)`` ⇒
                             ``lambda = -ln(e_T / e_0) / (T·dt)`` (clamped ≥ 0).
                             Logged both as ``contraction_rate`` and the
                             user-facing alias ``lambda``.
  * ``running_lambda``     — whole-Episode rate, averaged over episodes that
                             contracted: per episode, ``ln(C_0/C_T)/T`` on the
                             normalized cost ``C = e/e(0)`` if the cost
                             decreased end-to-end; episodes that did not are
                             dropped from the average, not counted as 0. Only
                             from :class:`StatManagerEnvWrapper`, which has
                             the full per-step curve rather than just the
                             endpoints.
  * ``overshoot``          — peak normalized error ``e_max / e(0)`` (≥ 1 in
                             theory; a pure overshoot factor).
  * ``contraction_score``  — ``lambda / overshoot`` (higher = fast contraction
                             with little overshoot).
  * ``peak``               — distribution of per-episode peaks (``max_t
                             e(t)/e(0)``) across the env population, reported
                             as ``peak_mean``/``peak_median``/``peak_p95`` —
                             unlike ``overshoot`` (a single worst-curve
                             envelope constant ``C``), this is the actual
                             spread of how bad the worst moment gets. Only
                             from :class:`StatManagerEnvWrapper`.

``auc`` additionally reports order statistics across the env population —
``auc_median``/``auc_p05``/``auc_p25``/``auc_p75``/``auc_p95``/``auc_max``.
``auc_mean`` alone cannot separate "the whole population got worse" from "two
envs blew up": these envs never terminate, so a diverged env is pinned at the
position bound (``env_base.step``) while its reference drives away, and its
error — hence its AUC — grows for the remainder of the episode. One such env
lands orders of magnitude above a healthy ``~1.2``, which makes the mean a
de-facto count of blow-ups over the ``num_envs_for_eval`` slots. A flat median
under a thrashing mean is precisely that signature, and says the instability is
a rare-failure-rate problem rather than a degradation of typical tracking.

Each per-env quantity is reduced across the env population to a mean and a 95%
CI half-width (``1.96·SEM``, see :func:`mean_confidence_interval`); ``peak``
and the ``auc`` percentiles above are the exceptions.

Action volatility (see :meth:`StatManagerEnvWrapper._action_volatility_summary`)
is a separate smoothness diagnostic on the action stream, not a stability/
contraction metric — it is logged under ``Episode/*``, not ``Stability/*``.

The streaming AUC is exactly the trapezoid, not an approximation of it: with
``err_sum = Σ_{k=0}^{T-1} e_k`` at times ``t_k = k·dt``,

    ∫ e dt ≈ dt·(e_0/2 + e_1 + … + e_{T-1}/2) = dt·(err_sum − 0.5·e_0 − 0.5·e_last)

so the normalized AUC needs only ``err_sum``, ``e_0``, ``e_last`` — no curve.
Both endpoints get half weight; subtracting only ``e_0``, or worse adding
``e_last``, is the trapezoid-rule error earlier per-algorithm copies had.

Trajectory plots (``{prefix}/path_tracking``) show the policy rollout against
the whole reference path:

  * the rollout is recorded per step and ends when the episode does;
  * the reference is captured once per episode from the env itself
    (``get_reference_trajectory()``, present on both env families), not
    accumulated from the observation window.

The window is subsampled at ``ref_offset`` and truncated to the horizon, so a
reference rebuilt from it is an offset-dependent subsample of the real path —
the reference-window layout must never reach the logs. Consequently the two
curves have different lengths and each gets its own time axis.

The flat window layout is ``[urefs | x | xrefs]`` (``RefWindow.flatten``), so
the ``x`` block starts at ``length*u_dim`` — nonzero even for a length-1
window. ``_record`` slices past it and asserts the observation width matches
``ref_window.flat_dim``, because a layout drift here yields plots and metrics
that are wrong but entirely plausible-looking.
"""

from __future__ import annotations

import io
import math
import sys

import numpy as np
import torch

from c2rl.tasks.direct.common.eval_metrics import mean_confidence_interval

# The four stability metrics every path-tracking algorithm/env must log.
METRIC_NAMES = ("auc", "contraction_rate", "overshoot", "contraction_score")


# ─────────────────────────────────────────────────────────────────────────── #
# Streaming metric math
# ─────────────────────────────────────────────────────────────────────────── #

def per_env_metrics(
    *,
    e0: torch.Tensor,
    e_last: torch.Tensor,
    e_max: torch.Tensor,
    err_sum: torch.Tensor,
    steps: torch.Tensor,
    dt: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Per-env metric tensors from streaming accumulators (see module docstring).

    All accumulators are ``(N,)`` or ``(N, 1)`` tensors on the same device;
    ``dt`` is the physical step time [s]. ``err_sum`` is ``Σ_k e_k`` over the
    episode's recorded steps, ``steps`` is that episode's length ``T`` (≥ 1).
    """
    e0c = e0.clamp(min=eps)
    eTc = e_last.clamp(min=eps)
    # T recorded samples (t = 0, dt, …, (T-1)·dt) span (T-1) intervals, i.e.
    # (T-1)·dt of elapsed time — the same interval count the trapezoid AUC uses.
    # clamp(min=1): the T==1 case has e_last==e0 so the rate is 0 regardless, and
    # this just avoids a 0·dt divisor.
    elapsed = (steps - 1).clamp(min=1) * dt
    # Normalized dt-weighted trapezoidal AUC (both endpoints half-weighted).
    auc = ((dt / e0c) * (err_sum - 0.5 * e0 - 0.5 * e_last)).clamp(min=0.0)
    # Empirical contraction rate from the endpoints; ≥ 0 (a negative raw value
    # just means the error grew — no contraction observed — not a valid rate).
    contraction_rate = (-(torch.log(eTc) - torch.log(e0c)) / elapsed).clamp(min=0.0)
    overshoot = (e_max.clamp(min=eps) / e0c).clamp(min=1e-6)
    contraction_score = contraction_rate / overshoot
    return {
        "auc": auc,
        "contraction_rate": contraction_rate,
        "overshoot": overshoot,
        "contraction_score": contraction_score,
    }


def summarize(
    per_env: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Reduce per-env metric tensors to ``{name}_mean`` / ``{name}_ci95`` floats.

    ``mask`` (bool, ``(N,)``) restricts the reduction to the envs that carry
    valid information (e.g. finished their episode); ``None`` uses all envs.
    """
    out: dict[str, float] = {}
    for name, vals in per_env.items():
        v = vals.reshape(-1)
        if mask is not None:
            v = v[mask.reshape(-1)]
        arr = v.detach().cpu().numpy()
        m, ci = mean_confidence_interval(arr, confidence)
        out[f"{name}_mean"] = float(m)
        out[f"{name}_ci95"] = float(ci)
    return out


class StreamingErrorStats:
    """Per-env streaming accumulators for the four contraction metrics.

    Feed one env step at a time via :meth:`update`, then call :meth:`summary`
    (mean/ci95 dict) or :meth:`metrics` (per-env tensors).  ``active`` freezes
    the accumulators of envs that already finished their (first) episode, so a
    non-terminating eval loop over ``max_episode_length`` steps measures exactly
    one episode per env.
    """

    def __init__(self, num_envs: int, device) -> None:
        self._device = device
        z = lambda: torch.zeros((num_envs, 1), device=device)
        self.e0 = z()
        self.e_last = z()
        self.e_max = z()
        self.err_sum = z()
        self.steps = z()

    def update(self, error: torch.Tensor, active: torch.Tensor | None = None) -> None:
        error = error.reshape(-1, 1).float()
        active = torch.ones_like(error) if active is None else active.reshape(-1, 1).float()
        is_first = (self.steps == 0) & (active > 0)
        self.e0 = torch.where(is_first, error, self.e0)
        self.e_last = torch.where(active > 0, error, self.e_last)
        self.e_max = torch.where(active > 0, torch.maximum(self.e_max, error), self.e_max)
        self.err_sum = self.err_sum + error * active
        self.steps = self.steps + active

    def metrics(self, dt: float) -> dict[str, torch.Tensor]:
        return per_env_metrics(
            e0=self.e0, e_last=self.e_last, e_max=self.e_max,
            err_sum=self.err_sum, steps=self.steps, dt=dt,
        )

    def summary(self, dt: float, mask: torch.Tensor | None = None) -> dict[str, float]:
        return summarize(self.metrics(dt), mask)


class StatManagerEnvWrapper:
    """Env wrapper that computes paper-style batched C and lambda globally across a sampled
    subset of environments by storing the full trajectories (first-come, first-eval).
    """

    def __init__(self, env, *, num_envs_for_eval: int = 64):
        self.env = env
        self._num_envs_for_eval = num_envs_for_eval

        self._initialized = False
        self._x_dim: int | None = None
        self._pos_dim: int | None = None
        # Column where the `x` block starts in the flat observation. The
        # ref_window layout is [urefs | x | xrefs] (RefWindow.flatten), so this
        # is length*u_dim — nonzero even for a length-1 window. 0 only for a
        # non-window env (plain Isaac cartpole), which has no urefs block.
        self._x_offset: int = 0
        self._expected_obs_width: int | None = None
        self._dt: float | None = None
        self._max_ep_len: int | None = None

        self._recent_auc_mean: float = 1e2
        self._recent_auc_ci95: float = 0.0
        self._recent_lambda_mean: float = 0.0
        self._recent_lambda_ci95: float = 0.0
        self._recent_running_lambda_mean: float = 0.0
        self._recent_running_lambda_ci95: float = 0.0
        self._recent_C: float = 1e2
        self._recent_score_mean: float = 0.0
        self._recent_score_ci95: float = 0.0
        self._recent_peak_mean: float = 0.0
        self._recent_auc_median: float = 1e2
        self._recent_auc_p05: float = 1e2
        self._recent_auc_p25: float = 1e2
        self._recent_auc_p75: float = 1e2
        self._recent_auc_p95: float = 1e2
        self._recent_auc_max: float = 1e2
        self._recent_peak_median: float = 0.0
        self._recent_peak_p95: float = 0.0

        # Mahalanobis-error twins of the above (Stability_maha/* — C2RL only).
        self._recent_maha_auc_mean: float = 1e2
        self._recent_maha_auc_ci95: float = 0.0
        self._recent_maha_lambda_mean: float = 0.0
        self._recent_maha_lambda_ci95: float = 0.0
        self._recent_maha_running_lambda_mean: float = 0.0
        self._recent_maha_running_lambda_ci95: float = 0.0
        self._recent_maha_C: float = 1e2
        self._recent_maha_score_mean: float = 0.0
        self._recent_maha_score_ci95: float = 0.0
        self._recent_maha_peak_mean: float = 0.0
        self._recent_maha_auc_median: float = 1e2
        self._recent_maha_auc_p05: float = 1e2
        self._recent_maha_auc_p25: float = 1e2
        self._recent_maha_auc_p75: float = 1e2
        self._recent_maha_auc_p95: float = 1e2
        self._recent_maha_auc_max: float = 1e2
        self._recent_maha_peak_median: float = 0.0
        self._recent_maha_peak_p95: float = 0.0
        # Bumped each time a full eval buffer is reduced to metrics — lets
        # callers (e.g. C3M's eval loop) detect that a rollout actually
        # produced fresh numbers instead of silently re-reading stale ones.
        self._compute_count: int = 0

        # Buffer states
        self._eval_buffer = None
        self._time_buffer = None
        self._tracking_env_ids = None
        self._tracking_steps = None
        self._completed_slots = None
        # Slots whose episode ended short of the horizon (env_base's
        # terminate_out_of_box). AUC/lambda/C are functionals of the full
        # normalized error curve, and a curve cut at step k is a different
        # quantity — the short-slot padding below would otherwise report a
        # fabricated flat tail as if the policy had held there. Excluded from
        # every reduction instead; see _compute_batched_metrics.
        self._invalid_slots = None
        self._recent_early_end_frac: float = 0.0
        self._recent_valid_n: int = 0
        self._e0 = None

        self._traj_x_buf = None
        self._traj_xref_buf = None
        self._recent_trajs = ({}, {}, {})

        # Parallel Mahalanobis normalized-error curve (C2RL only — filled from
        # info["maha_tracking_error"] when the env supplies it). Reuses the same
        # slot/step bookkeeping as the Euclidean buffer above; only the raw value
        # and its e(0) anchor differ. Empty for envs that emit no maha error.
        self._eval_buffer_maha = None
        self._e0_maha = None
        self._maha_seen = False
        self._recent_maha_err: dict = {}

        # Action volatility (see _track_action_volatility). Lazily allocated on
        # the first step, from the action tensor's own shape/device — the action
        # dimension is not among the attributes _ensure_stats can discover.
        self._prev_action = None
        self._prev_done = None
        self._vol_sum = None
        self._vol_count = None
        self._episode_volatility = None
        self._episode_volatility_seen = None

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _first_attr(self, *names, default=None):
        for n in names:
            # Direct access first (skrl wrappers forward attributes; also try the
            # raw unwrapped env for Isaac Lab, whose skrl wrapper may not forward).
            for target in (self.env, getattr(self.env, "unwrapped", None)):
                if target is not None and hasattr(target, n):
                    return getattr(target, n)
            # Fallback to get_attr for standard Gymnasium VectorEnvs
            try:
                v = self.env.get_attr(n)
                return v[0] if isinstance(v, (list, tuple)) else v
            except Exception:
                continue
        return default

    def set_discount_factor(self, gamma: float) -> None:
        """Tell the wrapper which gamma to discount with (see
        :meth:`_track_discounted_return`). Called by the runner once the agent
        config is final, so a swept discount_factor is the one used."""
        self._gamma = float(gamma)

    def _track_discounted_return(self, reward, terminated, truncated) -> None:
        """Accumulate the episodic DISCOUNTED return G = sum_t gamma^t r_t.

        This is the objective the agent is actually given, and the only quantity
        whose maximizer is pi*_gamma. The undiscounted `total_reward_*` already
        logged is a different functional, and tuning against it would select the
        policy that happens to track well rather than the best approximation of
        pi*_gamma -- which is what the theory is about.

        gamma^t is carried as a running factor and BOTH accumulators reset on
        episode end, so the exponent is the step index WITHIN the episode. A
        global step counter would make gamma^t underflow to 0 within a few
        hundred steps at gamma=0.99 and silently report G == 0 forever after.
        """
        if getattr(self, "_gamma", None) is None:
            return
        r = torch.as_tensor(reward, device=self._device()).reshape(-1).float()
        n = r.numel()
        if getattr(self, "_disc_return", None) is None or self._disc_return.numel() != n:
            self._disc_return = torch.zeros(n, device=r.device)
            self._disc_w = torch.ones(n, device=r.device)
            self._disc_done: list[float] = []
        self._disc_return += self._disc_w * r
        self._disc_w *= self._gamma

        def _flat(x):
            if x is None:
                return torch.zeros(n, dtype=torch.bool, device=r.device)
            t = torch.as_tensor(x, device=r.device).reshape(-1).bool()
            return t if t.numel() == n else torch.zeros(n, dtype=torch.bool, device=r.device)

        done = _flat(terminated) | _flat(truncated)
        if done.any():
            self._disc_done.extend(self._disc_return[done].detach().cpu().tolist())
            del self._disc_done[:-4096]      # bound the buffer
            self._disc_return[done] = 0.0
            self._disc_w[done] = 1.0

    def discounted_return_summary(self) -> dict[str, float]:
        """``discounted_return_{mean,ci95,median,min,max}`` over completed episodes.

        Empty until at least one episode has finished, so the key is ABSENT
        rather than a misleading 0.0 -- the same rule the stability metrics use,
        and it matters here because this is the sweep's objective: a 0.0 written
        before any episode ended would look like a real (very bad) score.
        """
        v = np.asarray(getattr(self, "_disc_done", []), dtype=float)
        if v.size == 0:
            return {}
        m, ci = mean_confidence_interval(v)
        return {
            "discounted_return_mean": float(m),
            "discounted_return_ci95": float(ci),
            "discounted_return_median": float(np.median(v)),
            "discounted_return_min": float(v.min()),
            "discounted_return_max": float(v.max()),
            "discounted_return_n": float(v.size),
        }


    def _device(self):
        return getattr(self.env, "device", "cpu")

    def _ensure_stats(self) -> bool:
        if self._initialized:
            return True
        # "num_dim_x" is the classic BaseEnv name; "x_dim" is the Isaac
        # path-tracking property (path_tracking_base.py). Both env families
        # share the ref_window's [urefs | x | xrefs] flat layout (ref_window.py
        # RefWindow.flatten) — x starts after the urefs block, not at column 0.
        x_dim = self._first_attr("num_dim_x", "x_dim")
        dt = self._first_attr("step_dt", "dt")
        ep = self._first_attr("max_episode_length", "max_episode_len")
        ref_window = self._first_attr("ref_window")
        self._x_offset = int(ref_window.length * ref_window.u_dim) if ref_window is not None else 0
        self._expected_obs_width = int(ref_window.flat_dim) if ref_window is not None else None
        if x_dim is None or dt is None:
            return False

        self._x_dim = int(x_dim)
        pd = self._first_attr("pos_dimension")
        self._pos_dim = int(pd) if pd is not None else min(3, int(x_dim))
        self._dt = float(dt)
        self._max_ep_len = int(ep) if ep is not None else 1000

        num_envs = int(getattr(self.env, "num_envs", 1))
        self._num_envs_for_eval = min(num_envs, self._num_envs_for_eval)
        N = self._num_envs_for_eval
        T = self._max_ep_len
        dev = self._device()

        self._eval_buffer = torch.zeros((N, T), dtype=torch.float32, device=dev)
        self._eval_buffer_maha = torch.zeros((N, T), dtype=torch.float32, device=dev)
        self._e0_maha = torch.zeros(N, dtype=torch.float32, device=dev)
        self._time_buffer = torch.zeros((N, T), dtype=torch.float32, device=dev)
        self._tracking_env_ids = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._tracking_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._completed_slots = torch.zeros(N, dtype=torch.bool, device=dev)
        self._invalid_slots = torch.zeros(N, dtype=torch.bool, device=dev)
        self._e0 = torch.zeros(N, dtype=torch.float32, device=dev)

        self._traj_x_buf = [[] for _ in range(N)]
        self._traj_xref_buf = [[] for _ in range(N)]

        self._initialized = True
        return True

    @staticmethod
    def _early_end_flags(info):
        """Per-env "this episode is ending short of the horizon", or None.

        ``env_base.step`` emits ``episode_ended_early`` only when its
        terminate_out_of_box box is armed, so None means "not checking" rather
        than "no excursion". Read at the same step the env auto-resets, which is
        exactly when ``_record`` completes the slot it belongs to.
        """
        if not isinstance(info, dict):
            return None
        ee = info.get("episode_ended_early")
        return ee.reshape(-1).bool().detach().cpu() if torch.is_tensor(ee) else None

    def _init_flags(self):
        """Per-env bool: episode counter == 0, i.e. the env just (auto-)reset.

        Both env families reset done envs inside step() and return the fresh
        episode's first observation, so counter == 0 marks exactly the obs a
        new slot must anchor e0 on. "time_steps" is the classic BaseEnv
        counter; "episode_length_buf" is Isaac Lab's.
        """
        try:
            ts = self._first_attr("time_steps", "episode_length_buf")
            if isinstance(ts, torch.Tensor):
                return (ts == 0).reshape(-1).to(self._device())
            if ts is not None:
                return torch.tensor([int(t) == 0 for t in ts], device=self._device())
        except Exception:
            pass
        return None

    def _metric_set(self, errs: torch.Tensor, rate_divisor: float = 1.0,
                    valid: torch.Tensor | None = None) -> dict[str, float]:
        """Reduce a normalized per-slot error buffer (rows already ÷ e(0)) to the
        stability metric summary. Shared by the Euclidean ``_eval_buffer`` and
        the Mahalanobis ``_eval_buffer_maha`` — the time base (``_time_buffer``)
        and per-slot lengths (``_tracking_steps``) are common to both, only the
        error rows differ. Returns python floats keyed
        ``{auc,lambda,running_lambda,score}_{mean,ci95}`` + shared ``C``.

        ``rate_divisor`` is the exponent order of the contraction envelope the
        buffer decays under: 1 for the Euclidean ‖e‖/‖e₀‖ (envelope e^{-λt}), 2
        for the squared Mahalanobis Lyapunov V/V₀ (envelope e^{-2λt}, since the
        CCM certificate is V̇ ≤ -2λV). The raw curve's decay rate is divided by
        it so the reported λ is the true contraction rate in both cases —
        comparable to each other and to the synthesis `lbd`. Overshoot ``C`` and
        AUC are left as measured on the curve itself (no divisor).

        ``valid`` (per-slot bool) drops rows before any reduction. The time base
        and the per-slot lengths are indexed by the same mask — slicing only
        ``errs`` would silently pair row i's curve with row i's neighbour's
        clock."""
        if valid is not None:
            errs = errs[valid]
            time_buffer = self._time_buffer[valid]
            tracking_steps = self._tracking_steps[valid]
        else:
            time_buffer = self._time_buffer
            tracking_steps = self._tracking_steps
        # 1. AUC per env (trapezoid over the true per-slot time base)
        dt_array = time_buffer[:, 1:] - time_buffer[:, :-1]
        auc_vec = torch.sum(dt_array * 0.5 * (errs[:, :-1] + errs[:, 1:]), dim=1)
        # Order statistics of the per-env AUC. auc_mean alone cannot distinguish
        # "the whole population got worse" from "a couple of envs blew up": with
        # non-terminating envs a diverged one is pinned at the position bound
        # (env_base.step) and its error grows for the rest of the episode, so its
        # AUC lands orders of magnitude above a healthy ~1.2 and the mean becomes
        # a de-facto count of blow-ups. A flat median against a thrashing mean is
        # exactly that signature.
        auc_np = auc_vec.detach().cpu().numpy()
        auc_median = float(np.median(auc_np))
        auc_p05 = float(np.percentile(auc_np, 5))
        auc_p25 = float(np.percentile(auc_np, 25))
        auc_p75 = float(np.percentile(auc_np, 75))
        auc_p95 = float(np.percentile(auc_np, 95))
        auc_max = float(np.max(auc_np))

        # 2. Find curve with highest overshoot
        max_overshoots = torch.max(errs, dim=1).values
        # Distribution of per-episode peaks (max normalized error over the
        # episode) — mean/median/95th-percentile, as opposed to `C` above
        # (a single worst-curve envelope constant).
        peaks_np = max_overshoots.detach().cpu().numpy()
        peak_mean = float(np.mean(peaks_np))
        peak_median = float(np.median(peaks_np))
        peak_p95 = float(np.percentile(peaks_np, 95))
        worst_idx = torch.argmax(max_overshoots)
        x_worst = errs[worst_idx]
        t_worst = time_buffer[worst_idx]

        # 3. Find optimal C for worst curve: C(lambda) = max_t x(t)·e^{lambda·t}
        #    over lambda in (0, 10], keeping the C whose envelope AUC is minimal.
        lambdas = torch.linspace(0.01, 10.0, steps=1000, device=self._device())
        exp_term = torch.exp(lambdas.unsqueeze(1) * t_worst.unsqueeze(0))
        C_lambdas = torch.max(x_worst.unsqueeze(0) * exp_term, dim=1).values
        C_lambdas = torch.clamp(C_lambdas, min=1.0)

        T_max = t_worst[-1]
        auc_bounds = (C_lambdas / lambdas) * (1.0 - torch.exp(-lambdas * T_max))
        best_idx = torch.argmin(auc_bounds)
        best_C = C_lambdas[best_idx]

        # 4. With C fixed, per-env lambda = min_t (ln C - ln x(t)) / t  (t > 0)
        t_pos = torch.clamp(time_buffer[:, 1:], min=1e-8)  # (N, T-1)
        x_pos = errs[:, 1:]                                      # (N, T-1)

        lambda_vals = (torch.log(best_C) - torch.log(x_pos)) / t_pos
        min_lambdas = torch.min(lambda_vals, dim=1).values
        # /rate_divisor: the curve decays as e^{-(order·λ)t}, so its measured
        # rate is order·λ — undo it to report the true λ (see the docstring).
        lambda_vec = torch.clamp(min_lambdas / rate_divisor, min=0.0, max=10.0)
        score_vec = lambda_vec / torch.clamp(best_C, min=1e-6)

        # 5. Running-mean lambda — the whole-Episode contraction rate, averaged
        #    over episodes that actually contracted.  For each episode slot,
        #    compare the normalized cost at the end of the episode to its
        #    start (C = 1): if it decreased, the rate that makes the endpoint
        #    contraction condition
        #        C(T) = e^{-lambda·T}·C(0)
        #    hold with equality is  lambda = ln(C(0) / C(T)) / T.  Episodes
        #    that did not decrease are dropped from the average entirely (not
        #    counted as a 0) — this is a rate over contracting episodes only,
        #    not a score that also penalizes non-contracting ones.
        lengths = tracking_steps.clamp(max=self._max_ep_len).reshape(-1, 1)
        end_idx = (lengths - 1).clamp(min=0)  # (N, 1), last real recorded index
        err_end = torch.gather(errs, 1, end_idx).squeeze(1)
        t_end = torch.gather(time_buffer, 1, end_idx).squeeze(1).clamp(min=1e-8)
        err_start = errs[:, 0]
        episode_lambda = (torch.log(err_start) - torch.log(err_end)) / (t_end * rate_divisor)
        decreased = err_end < err_start
        running_lambda_vec = episode_lambda[decreased]

        auc_m, auc_ci = mean_confidence_interval(auc_vec.detach().cpu().numpy(), 0.95)
        lambda_m, lambda_ci = mean_confidence_interval(lambda_vec.detach().cpu().numpy(), 0.95)
        if running_lambda_vec.numel() > 0:
            run_m, run_ci = mean_confidence_interval(running_lambda_vec.detach().cpu().numpy(), 0.95)
        else:
            run_m, run_ci = 0.0, 0.0
        score_m, score_ci = mean_confidence_interval(score_vec.detach().cpu().numpy(), 0.95)
        return {
            "auc_mean": float(auc_m), "auc_ci95": float(auc_ci),
            "lambda_mean": float(lambda_m), "lambda_ci95": float(lambda_ci),
            "running_lambda_mean": float(run_m), "running_lambda_ci95": float(run_ci),
            "score_mean": float(score_m), "score_ci95": float(score_ci),
            "C": float(best_C.item()),
            "peak_mean": peak_mean, "peak_median": peak_median, "peak_p95": peak_p95,
            "auc_median": auc_median, "auc_p05": auc_p05, "auc_p25": auc_p25,
            "auc_p75": auc_p75, "auc_p95": auc_p95, "auc_max": auc_max,
            # The unreduced per-env vectors behind the four scalars above, for
            # log_metric_distributions. Every mean here is over a population
            # whose spread is the actual finding — a healthy median under a
            # thrashing mean is a rare-blow-up signature, and no scalar shows
            # that. Underscored because this dict is the one float-valued
            # contract in this module that now carries an array: it is read only
            # by _compute_batched_metrics, which copies named scalars out, so it
            # never reaches stability_summary() or the wandb scalar loggers.
            # running_lambda is shorter than the others (contracting episodes
            # only) and may be empty — the plotter guards on that.
            "_dist": {
                "auc": auc_np,
                "lambda": lambda_vec.detach().cpu().numpy(),
                "running_lambda": running_lambda_vec.detach().cpu().numpy(),
                "peak": peaks_np,
            },
        }

    def _compute_batched_metrics(self):
        N = self._num_envs_for_eval
        # Slots whose episode was cut short (terminate_out_of_box) carry a
        # Padded tail, not a measured one — see _invalid_slots. Reduce over the
        # survivors only, and report what fraction was dropped: with early
        # termination on, that fraction is the failure rate, and it is the one
        # number a mean-AUC over survivors cannot show.
        valid = ~self._invalid_slots
        self._recent_valid_n = int(valid.sum().item())
        self._recent_early_end_frac = float((~valid).float().mean().item())
        if self._recent_valid_n == 0:
            # Nothing measurable this round. Leave the _recent_* scalars alone
            # and let stability_summary() report the metrics as absent — the
            # house rule (see the agent_patches note on sentinel blending) is
            # that no datapoint beats a wrong one. Still bump _compute_count so
            # callers can tell a round completed.
            #
            # The per-env distributions and trajectory curves must be cleared,
            # not left alone: _compute_count is what the plot wrappers watch to
            # decide a fresh round exists (wandb_plot_wrapper), so keeping the
            # previous round's arrays here would re-publish stale curves under a
            # new step as though they had just been measured.
            self._recent_distributions = {}
            self._recent_trajs = ({}, {}, {})
            self._recent_maha_err = {}
            self._compute_count += 1
            return
        # Clamp once: a stored error of exactly 0 would otherwise produce
        # 0 * inf = NaN in the C search (exp(lambda*t) overflows to inf in
        # float32 once lambda*t ≳ 88) and -inf in the log for lambda.
        m = self._metric_set(torch.clamp(self._eval_buffer, min=1e-8), valid=valid)
        self._recent_auc_mean = m["auc_mean"]
        self._recent_auc_ci95 = m["auc_ci95"]
        self._recent_lambda_mean = m["lambda_mean"]
        self._recent_lambda_ci95 = m["lambda_ci95"]
        self._recent_running_lambda_mean = m["running_lambda_mean"]
        self._recent_running_lambda_ci95 = m["running_lambda_ci95"]
        self._recent_score_mean = m["score_mean"]
        self._recent_score_ci95 = m["score_ci95"]
        self._recent_C = m["C"]
        self._recent_peak_mean = m["peak_mean"]
        self._recent_auc_median = m["auc_median"]
        self._recent_auc_p05 = m["auc_p05"]
        self._recent_auc_p25 = m["auc_p25"]
        self._recent_auc_p75 = m["auc_p75"]
        self._recent_auc_p95 = m["auc_p95"]
        self._recent_auc_max = m["auc_max"]
        self._recent_peak_median = m["peak_median"]
        self._recent_peak_p95 = m["peak_p95"]
        self._recent_distributions = m["_dist"]

        # Same reduction on the squared Mahalanobis Lyapunov V(t)/V(0) =
        # ‖e(t)‖²_M/‖e(0)‖²_M — the quantity the CCM certificate contracts
        # (V̇ ≤ -2λV), hence rate_divisor=2 so the reported λ is the true rate.
        # Only when the env supplied it.
        if self._maha_seen:
            mm = self._metric_set(torch.clamp(self._eval_buffer_maha, min=1e-8),
                                  rate_divisor=2.0, valid=valid)
            self._recent_maha_auc_mean = mm["auc_mean"]
            self._recent_maha_auc_ci95 = mm["auc_ci95"]
            self._recent_maha_lambda_mean = mm["lambda_mean"]
            self._recent_maha_lambda_ci95 = mm["lambda_ci95"]
            self._recent_maha_running_lambda_mean = mm["running_lambda_mean"]
            self._recent_maha_running_lambda_ci95 = mm["running_lambda_ci95"]
            self._recent_maha_score_mean = mm["score_mean"]
            self._recent_maha_score_ci95 = mm["score_ci95"]
            self._recent_maha_C = mm["C"]
            self._recent_maha_peak_mean = mm["peak_mean"]
            self._recent_maha_auc_median = mm["auc_median"]
            self._recent_maha_auc_p05 = mm["auc_p05"]
            self._recent_maha_auc_p25 = mm["auc_p25"]
            self._recent_maha_auc_p75 = mm["auc_p75"]
            self._recent_maha_auc_p95 = mm["auc_p95"]
            self._recent_maha_auc_max = mm["auc_max"]
            self._recent_maha_peak_median = mm["peak_median"]
            self._recent_maha_peak_p95 = mm["peak_p95"]

        self._compute_count += 1

        # Save trajectories + per-slot normalized error curves (consumed by
        # log_tracking_plots via trajectories()). Keyed by slot index — an env
        # can legitimately own two slots in one buffer round (early termination
        # + re-init), so env-id keys would silently collide.
        err_rows = self._eval_buffer.detach().cpu()
        maha_rows = self._eval_buffer_maha.detach().cpu()
        res_x, res_xref, res_err, res_maha = {}, {}, {}, {}
        for j in range(N):
            res_x[j] = self._traj_x_buf[j]
            res_xref[j] = self._traj_xref_buf[j]
            res_err[j] = err_rows[j].tolist()
            if self._maha_seen:
                res_maha[j] = maha_rows[j].tolist()
        self._recent_trajs = (res_x, res_xref, res_err)
        self._recent_maha_err = res_maha

    def _record(self, obs: torch.Tensor, info: dict | None = None) -> None:
        if not self._ensure_stats():
            return

        xd, pd, xo = self._x_dim, self._pos_dim, self._x_offset

        # The slicing below hard-codes the ref_window layout. If the observation
        # is not the width that layout implies, every slice is silently off and
        # the plots/metrics come out wrong-but-plausible (this exact drift
        # shipped once: the window landed but this wrapper still assumed the old
        # [x, xref, uref] order). Fail loudly instead.
        if self._expected_obs_width is not None and obs.shape[-1] != self._expected_obs_width:
            raise RuntimeError(
                f"StatManagerEnvWrapper: observation is {obs.shape[-1]}-wide but the env's "
                f"ref_window (x_dim={xd}, length*u_dim={xo}) implies "
                f"{self._expected_obs_width}. The [urefs | x | xrefs] slicing this wrapper "
                f"uses no longer matches the observation layout.")

        unwrapped = getattr(self.env, "unwrapped", self.env)
        if isinstance(info, dict) and "tracking_error" in info:
            # Classic BaseEnv (env_base.py): "tracking_error" is the true
            # per-env squared error computed before reset_idx() overwrites
            # x_t/xref for any env whose episode just ended — same ordering
            # fix as the reward computation. Reading it from `obs` instead
            # would, for an auto-reset env, measure the fresh reset (new
            # episode) state against the old episode's e0/window, spiking the
            # normalized error at the episode boundary.
            error = torch.sqrt(torch.clamp(info["tracking_error"].reshape(-1).to(self._device()), min=0.0))
        elif hasattr(unwrapped, "get_tracking_error"):
            # Isaac path-tracking envs: angle-wrapped ||x - x_ref|| per env.
            err = unwrapped.get_tracking_error()
            if isinstance(err, torch.Tensor):
                error = err.reshape(-1, 1).to(self._device())
            else:
                error = torch.tensor(err, dtype=torch.float32, device=self._device()).reshape(-1, 1)
            error = error.reshape(-1)
        else:
            diff = obs[:, xo:xo + xd] - obs[:, xo + xd:xo + 2 * xd]
            angle_idx = self._first_attr("angle_idx")
            if angle_idx:
                for idx in angle_idx:
                    diff[:, idx] = (diff[:, idx] + math.pi) % (2 * math.pi) - math.pi
            error = torch.norm(diff, dim=-1, keepdim=True)
            error = error.reshape(-1)
        init_flags = self._init_flags()
        if init_flags is None:
            init_flags = torch.zeros(error.shape[0], dtype=torch.bool, device=self._device())

        early_end = self._early_end_flags(info)

        # Extract values for tracking
        err_vals = error.detach()

        # At an auto-reset step the classic BaseEnv's info["tracking_error"] is
        # the old episode's terminal error (computed before reset_idx), while
        # obs/counters are already the new episode's. A slot opened here would
        # anchor e0 on a near-zero stale value, inflating every normalized
        # error in the window. Substitute the env's post-reset initial error
        # (squared norm, hence sqrt) for the freshly reset envs.
        if init_flags.any():
            init_err_sq = self._first_attr("init_tracking_error")
            if isinstance(init_err_sq, torch.Tensor):
                fresh = torch.sqrt(torch.clamp(
                    init_err_sq.reshape(-1).to(err_vals), min=0.0))
                err_vals = torch.where(init_flags.reshape(-1), fresh, err_vals)

        # Mahalanobis error √(eᵀM(x)e) — same construction as err_vals above but
        # metric-weighted. None when the env emits no maha error (non-C2RL runs),
        # in which case the parallel buffer stays empty and no curve is plotted.
        maha_err_vals = None
        if isinstance(info, dict) and "maha_tracking_error" in info:
            maha_err_vals = torch.sqrt(torch.clamp(
                info["maha_tracking_error"].reshape(-1).to(self._device()), min=0.0)).detach()
            if init_flags.any():
                init_maha_sq = self._first_attr("init_maha_error")
                if isinstance(init_maha_sq, torch.Tensor):
                    fresh_m = torch.sqrt(torch.clamp(
                        init_maha_sq.reshape(-1).to(maha_err_vals), min=0.0))
                    maha_err_vals = torch.where(init_flags.reshape(-1), fresh_m, maha_err_vals)
            self._maha_seen = True

        obs_x = obs[:, xo:xo + pd].detach().cpu().numpy()

        # The reference is taken as the env's whole trajectory, captured once
        # when a slot opens — not accumulated from the observation window. The
        # window is subsampled at ref_offset and truncated to the horizon, so a
        # reference rebuilt from it is an offset-dependent subsample of the real
        # path; the plot must show the full reference against the policy
        # rollout. get_reference_trajectory() exists on both env families.
        ref_traj = self._first_attr("get_reference_trajectory")
        full_xref = None
        if callable(ref_traj):
            full_xref = ref_traj()
            if torch.is_tensor(full_xref):
                full_xref = full_xref[..., :pd].detach().cpu().numpy()

        # For each env that is initializing, complete its old slot if any, and start a new one
        init_indices = torch.nonzero(init_flags, as_tuple=True)[0]
        for env_idx in init_indices:
            # If this env was already being tracked, complete its old slot first!
            old_slots = torch.nonzero(self._tracking_env_ids == env_idx, as_tuple=True)[0]
            for old_slot in old_slots:
                if not self._completed_slots[old_slot]:
                    self._completed_slots[old_slot] = True
                    # Early termination: extend the final recorded error to the
                    # end of the horizon so every slot has a full-length curve.
                    # The slot keeps its env id — a completed episode must stay
                    # in the buffer until the whole buffer is reduced; freeing
                    # it here would let the very next init reuse and overwrite
                    # it, and (ids != -1).all() below could then never fire.
                    # Ended short of the horizon by the env's own termination
                    # box (not by the time limit)? Then the padding below is a
                    # fabrication, so the slot is excluded from every metric
                    # rather than padded into a plausible-looking full curve.
                    if early_end is not None and bool(early_end[int(env_idx)]):
                        self._invalid_slots[old_slot] = True
                    step = self._tracking_steps[old_slot]
                    if step < self._max_ep_len:
                        if self._invalid_slots[old_slot]:
                            # NaN, not the held last value: the curve simply has
                            # no data past the cut, and NaN is what makes the
                            # plotted line stop there instead of running flat.
                            self._eval_buffer[old_slot, step:] = float("nan")
                            self._eval_buffer_maha[old_slot, step:] = float("nan")
                        else:
                            last_val = self._eval_buffer[old_slot, step-1] if step > 0 else 1.0
                            self._eval_buffer[old_slot, step:] = last_val
                            last_maha = self._eval_buffer_maha[old_slot, step-1] if step > 0 else 1.0
                            self._eval_buffer_maha[old_slot, step:] = last_maha
                        time_steps_pad = torch.arange(step, self._max_ep_len, device=self._device(), dtype=torch.float32)
                        self._time_buffer[old_slot, step:] = time_steps_pad * self._dt

            # Now assign a new slot (first come, first eval — no free slot means
            # this episode simply isn't tracked this round)
            empty_slots = torch.nonzero(self._tracking_env_ids == -1, as_tuple=True)[0]
            if len(empty_slots) > 0:
                slot = empty_slots[0]
                self._tracking_env_ids[slot] = env_idx
                self._tracking_steps[slot] = 0
                self._completed_slots[slot] = False
                self._invalid_slots[slot] = False
                self._traj_x_buf[slot] = []
                # Whole reference path for this episode, captured now (it is
                # resampled per episode by reset_idx), not grown per step.
                self._traj_xref_buf[slot] = (
                    [] if full_xref is None else list(full_xref[int(env_idx)]))

        # Update active slots — batched. This runs on every env step, and the
        # old per-slot Python loop paid a device sync per slot per step (`if
        # step == 0` on a GPU tensor is a blocking .item()): 64 slots x 2 syncs
        # x every step. The scatter below is one pass, with a single .tolist()
        # sync left for the trajectory buffer (a list of numpy rows, which has
        # no tensor form to scatter into).
        active_slots = torch.nonzero((self._tracking_env_ids != -1) & (~self._completed_slots), as_tuple=True)[0]

        if active_slots.numel():
            env_ids_a = self._tracking_env_ids[active_slots]
            steps_a = self._tracking_steps[active_slots]

            # e(0) anchor for slots opening this step. Written before it is read
            # below, exactly as the sequential version did.
            fresh = steps_a == 0
            if fresh.any():
                fs, fe = active_slots[fresh], env_ids_a[fresh]
                self._e0[fs] = err_vals[fe].clamp(min=1e-8)
                if maha_err_vals is not None:
                    self._e0_maha[fs] = maha_err_vals[fe].clamp(min=1e-8)

            in_range = steps_a < self._max_ep_len
            if in_range.any():
                slot, env_id, step = (active_slots[in_range], env_ids_a[in_range],
                                      steps_a[in_range])
                self._eval_buffer[slot, step] = err_vals[env_id] / self._e0[slot]
                if maha_err_vals is not None:
                    # Squared normalized Mahalanobis error V(t)/V(0) =
                    # ‖e(t)‖²_M/‖e(0)‖²_M — the Lyapunov the CCM certificate is
                    # written on (V̇ ≤ -2λV ⇒ V(t) ≤ V(0)e^{-2λt}). The extra 2
                    # in the exponent is undone in _metric_set(rate_divisor=2)
                    # so the reported λ matches the synthesis rate `lbd`.
                    self._eval_buffer_maha[slot, step] = (maha_err_vals[env_id] / self._e0_maha[slot]) ** 2
                self._time_buffer[slot, step] = step.to(self._time_buffer.dtype) * self._dt

                # Rollout only — the reference was captured whole at slot open.
                # The one unavoidable Python loop: _traj_x_buf is a list of numpy
                # rows per slot, so there is nothing to scatter into. Two
                # .tolist() calls instead of a sync per slot.
                for s_i, e_i in zip(slot.tolist(), env_id.tolist()):
                    self._traj_x_buf[s_i].append(obs_x[e_i])

            nxt = steps_a + 1
            self._tracking_steps[active_slots] = nxt
            # Only active slots are touched, so this can never un-complete a slot
            # that finished in an earlier round.
            self._completed_slots[active_slots] = nxt >= self._max_ep_len

        # Check if all slots are completed
        if (self._tracking_env_ids != -1).all() and self._completed_slots.all():
            self._compute_batched_metrics()
            # Clear slots
            self._tracking_env_ids.fill_(-1)
            self._completed_slots.fill_(False)
            self._invalid_slots.fill_(False)

    @staticmethod
    def _obs_tensor(obs):
        """Batched obs tensor from a step/reset return (Isaac wrappers may
        return an observation dict keyed by group, e.g. {"policy": ...})."""
        if isinstance(obs, dict):
            obs = obs.get("policy")
        return obs if torch.is_tensor(obs) else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # An external reset (e.g. C3M's per-eval reset) starts a fresh window:
        # every in-flight episode is invalidated, so drop all slots. Without
        # this, leftover partial slots would be "completed" by padding at the
        # next init detection and reduced into garbage metrics.
        if self._initialized:
            self._tracking_env_ids.fill_(-1)
            self._completed_slots.fill_(False)
            self._invalid_slots.fill_(False)
        self._reset_action_volatility()
        o = self._obs_tensor(obs)
        if o is not None:
            self._record(o, info if isinstance(info, dict) else None)
        return obs, info

    def _reset_action_volatility(self) -> None:
        """Drop every in-flight action-volatility accumulator.

        Mirrors what reset() does to the eval slots: an external reset breaks
        the action sequence, so any partially accumulated episode would splice
        pre- and post-reset actions into one average. Completed per-episode
        values are kept — they are already-finished measurements, and the
        summary would otherwise go empty (and the key silently vanish from
        wandb) after every external reset.
        """
        self._prev_action = None
        self._prev_done = None
        if self._vol_sum is not None:
            self._vol_sum.zero_()
            self._vol_count.zero_()

    def _track_action_volatility(self, action, terminated, truncated) -> None:
        """Accumulate per-step ``||u_t - u_{t-1}||_2`` into per-episode means.

        A measurement of the deployed action sequence, not a penalty: the
        executed action, exploration noise included, whereas CAPS' temporal term
        regularizes the policy mean. Deliberately different quantities — this is
        what the actuator actually sees, so it stays meaningful for algorithms
        with no CAPS term and is what to read for physical deployability.

        Autoreset: both families reset done envs inside ``step()`` and return the
        new episode's first obs, so the action after a done comes from an
        unrelated initial state. Differencing across that boundary reports a
        spurious jump sized by the reset distribution, not the policy. Pairs are
        skipped when the previous step was done — hence ``_prev_done`` rather
        than the current step's flags.

        Units are raw action units per step, not per second: dividing by dt would
        make the number depend on the integrator step. Compare across configs of
        one env, never across envs with different dt.
        """
        if not torch.is_tensor(action):
            return
        a = action.detach().reshape(action.shape[0], -1).float()
        n = a.shape[0]

        done = (terminated | truncated) if torch.is_tensor(terminated) else None
        done = torch.zeros(n, dtype=torch.bool, device=a.device) if done is None \
            else done.detach().reshape(-1).to(a.device)

        if self._vol_sum is None or self._vol_sum.shape[0] != n:
            z = torch.zeros(n, dtype=torch.float32, device=a.device)
            self._vol_sum, self._vol_count = z.clone(), z.clone()
            self._episode_volatility = z.clone()
            self._episode_volatility_seen = torch.zeros(n, dtype=torch.bool, device=a.device)
            self._prev_action, self._prev_done = None, None

        if self._prev_action is not None and self._prev_action.shape == a.shape:
            valid = ~self._prev_done                      # skip pairs across a reset
            step_delta = (a - self._prev_action).norm(dim=-1)
            self._vol_sum += torch.where(valid, step_delta, torch.zeros_like(step_delta))
            self._vol_count += valid.float()

        # Finalize the episodes that ended on this step, then clear their
        # accumulators so the next episode starts from zero.
        if bool(done.any()):
            finished = done & (self._vol_count > 0)
            if bool(finished.any()):
                mean_delta = self._vol_sum / self._vol_count.clamp(min=1.0)
                self._episode_volatility = torch.where(
                    finished, mean_delta, self._episode_volatility)
                self._episode_volatility_seen |= finished
            self._vol_sum = torch.where(done, torch.zeros_like(self._vol_sum), self._vol_sum)
            self._vol_count = torch.where(done, torch.zeros_like(self._vol_count), self._vol_count)

        self._prev_action = a.clone()
        self._prev_done = done

    def _action_volatility_summary(self) -> dict[str, float]:
        """``action_volatility_{mean,ci95,max}`` over envs with a finished episode.

        Returns {} until at least one episode has completed, so the key is
        absent rather than zero — same rule the step() gate applies to the
        other stability metrics, and for the same reason: skrl's write_interval
        averages its window, so a placeholder would blend into the real value.
        """
        if self._episode_volatility_seen is None or not bool(self._episode_volatility_seen.any()):
            return {}
        v = self._episode_volatility[self._episode_volatility_seen].detach().cpu().numpy()
        m, ci = mean_confidence_interval(v)
        return {
            "action_volatility_mean": m,
            "action_volatility_ci95": ci,
            "action_volatility_max": float(np.max(v)),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._track_discounted_return(reward, terminated, truncated)
        self._track_action_volatility(action, terminated, truncated)
        o = self._obs_tensor(obs)
        if o is not None:
            self._record(o, info if isinstance(info, dict) else None)

            # Inject metrics into info["log"] so skrl's trainer (cfg
            # `environment_info: log`, scalar tensors only) tracks them.
            #
            # Gated on _compute_count: before the first buffer round completes,
            # stability_summary() returns the constructor sentinels (auc_mean
            # and C = 1e2), and the buffer only completes at the end of an
            # episode. Logging them from step 0 meant ~498 of a 500-step
            # episode reported 1e2, and skrl's write_interval averages its
            # window — so the value that reached wandb (and the sweep) was a
            # Blend of sentinel and truth, not the truth:
            #     Stability/auc_mean ≈ 60 + 0.4·true_auc
            # (measured, classic-cartpole-v0 + lqr: true 7.52 → logged 63.01;
            # overshoot 3.95 → logged 61.58, the same 60/40 mix). Worse, the
            # blend ratio depends on where buffer completion lands inside the
            # write window, so it varied run to run — noise of the same order
            # as the real differences the sweeps are trying to resolve.
            # Emitting nothing until there is a real value keeps the key
            # absent instead of wrong; skrl simply has no datapoint to average.
            # ... and only when the value has actually changed, i.e. once per
            # buffer round rather than once per step. stability_summary() is
            # recomputed by _compute_batched_metrics; between rounds it returns
            # the identical numbers, so emitting every step wrote thousands of
            # duplicate rows.
            #
            # This is not a cosmetic saving. wandb's local transaction log holds
            # one record per logged scalar, and the summary grew from ~19 keys to
            # ~54 when the full quantile set was added -- a single active run's
            # run-*.wandb reached 3.4 GB, 15 concurrent workers put the home
            # filesystem at 94 GB of a 103 GB quota, and the previous quota
            # exhaustion killed the whole search.
            _fresh = self._initialized and self._compute_count > 0 and (
                self._compute_count != getattr(self, "_last_logged_compute_count", None))
            if _fresh and isinstance(info, dict):
                self._last_logged_compute_count = self._compute_count
                if "log" not in info or not isinstance(info["log"], dict):
                    info["log"] = {}
                info["log"].update(stability_log_dict(self.stability_summary(), self._device()))
                # C2RL: the same metrics on the Mahalanobis error, under
                # Stability_maha/*. Empty dict (no keys) for non-maha envs.
                maha_summary = self.stability_maha_summary()
                if maha_summary:
                    info["log"].update(
                        stability_log_dict(maha_summary, self._device(), tab="Stability_maha"))
                # Action volatility isn't a stability/contraction metric — it's
                # a smoothness diagnostic on the action stream — so it gets
                # its own Episode/* tab rather than living under Stability/*.
                volatility = self._action_volatility_summary()
                if volatility:
                    info["log"].update(
                        stability_log_dict(volatility, self._device(), tab="Episode"))

        # Discounted return G = sum_t gamma^t r_t. Emitted only when it CHANGED
        # -- it moves only as episodes complete, so logging it every step wrote
        # ~500 identical copies per episode into wandb's transaction log.
        _n_done = len(getattr(self, "_disc_done", ()))
        if isinstance(info, dict) and _n_done != getattr(self, "_last_logged_disc_n", None):
            disc = self.discounted_return_summary()
            if disc:
                self._last_logged_disc_n = _n_done
                if "log" not in info or not isinstance(info["log"], dict):
                    info["log"] = {}
                info["log"].update(stability_log_dict(disc, self._device(), tab="Reward"))

        return obs, reward, terminated, truncated, info

    def stability_summary(self) -> dict[str, float]:
        if not self._initialized:
            return {}
        if self._compute_count > 0 and self._recent_valid_n == 0:
            # Every episode in the round was cut short, so AUC/lambda/C are
            # Unavailable, not zero and not the previous round's numbers.
            # Reporting only the fraction keeps the failure visible without
            # publishing a stale value under a fresh timestamp.
            return {"early_end_frac": self._recent_early_end_frac}
        # Every metric carries the "{name}_mean"/"{name}_ci95" key shape that
        # track_stability_summary documents and patch_auc_checkpoint
        # (agent_patches.py) looks up. C is a single shared scalar, so its
        # ci95 is 0 by construction. peak_{mean,median,p95} have no ci95 twin
        # (they're a one-shot distribution summary, not a mean/CI pair).
        return {
            "auc_mean": self._recent_auc_mean,
            "auc_ci95": self._recent_auc_ci95,
            "contraction_rate_mean": self._recent_lambda_mean,
            "contraction_rate_ci95": self._recent_lambda_ci95,
            "running_lambda_mean": self._recent_running_lambda_mean,
            "running_lambda_ci95": self._recent_running_lambda_ci95,
            "overshoot_mean": self._recent_C,
            "overshoot_ci95": 0.0,
            "contraction_score_mean": self._recent_score_mean,
            "contraction_score_ci95": self._recent_score_ci95,
            "peak_mean": self._recent_peak_mean,
            "auc_median": self._recent_auc_median,
            "auc_p05": self._recent_auc_p05,
            "auc_p25": self._recent_auc_p25,
            "auc_p75": self._recent_auc_p75,
            "auc_p95": self._recent_auc_p95,
            "auc_max": self._recent_auc_max,
            "peak_median": self._recent_peak_median,
            "peak_p95": self._recent_peak_p95,
            # Fraction of tracked episodes excluded because they ended short.
            # 0.0 whenever terminate_out_of_box is off, so the key is always
            # present and always means the same thing. With it on, this is the
            # failure rate — the statistic a mean over survivors cannot show.
            "early_end_frac": self._recent_early_end_frac,
        }

    def stability_maha_summary(self) -> dict[str, float]:
        """Same metric shape as :meth:`stability_summary`, but computed on the
        squared Mahalanobis Lyapunov V(t)/V(0) = ‖e(t)‖²_M/‖e(0)‖²_M (λ extracted
        against its e^{-2λt} envelope, so it is the true contraction rate). Empty
        until an env has supplied a maha error and a buffer round has completed —
        so the keys are absent (not sentinels) for non-C2RL runs, same rule as
        the action volatility metrics."""
        if not self._initialized or not self._maha_seen:
            return {}
        if self._compute_count > 0 and self._recent_valid_n == 0:
            return {}  # unavailable — same rule as stability_summary
        return {
            "auc_mean": self._recent_maha_auc_mean,
            "auc_ci95": self._recent_maha_auc_ci95,
            "contraction_rate_mean": self._recent_maha_lambda_mean,
            "contraction_rate_ci95": self._recent_maha_lambda_ci95,
            "running_lambda_mean": self._recent_maha_running_lambda_mean,
            "running_lambda_ci95": self._recent_maha_running_lambda_ci95,
            "overshoot_mean": self._recent_maha_C,
            "overshoot_ci95": 0.0,
            "contraction_score_mean": self._recent_maha_score_mean,
            "contraction_score_ci95": self._recent_maha_score_ci95,
            "peak_mean": self._recent_maha_peak_mean,
            "auc_median": self._recent_maha_auc_median,
            "auc_p05": self._recent_maha_auc_p05,
            "auc_p25": self._recent_maha_auc_p25,
            "auc_p75": self._recent_maha_auc_p75,
            "auc_p95": self._recent_maha_auc_p95,
            "auc_max": self._recent_maha_auc_max,
            "peak_median": self._recent_maha_peak_median,
            "peak_p95": self._recent_maha_peak_p95,
        }

    def distributions(self):
        """Per-env vectors behind the Stability scalars: ``auc``, ``lambda``,
        ``running_lambda``, ``peak``. Empty dict before the first buffer
        completion. Consumed by :func:`log_metric_distributions`."""
        return getattr(self, "_recent_distributions", None) or {}

    def trajectories(self):
        return self._recent_trajs

    def maha_trajectories(self):
        """Per-slot squared Mahalanobis normalized-error curves
        (V(t)/V(0) = ‖e(t)‖²_M/‖e(0)‖²_M), keyed by buffer slot. Empty dict for
        envs that emit no maha error."""
        return self._recent_maha_err

    def all_finished(self) -> bool:
        # We can just return False or True depending on usage.
        # This was previously used by ContractionRunner's eval loop.
        # In this new logic, the buffer fills up and resets automatically.
        return False


def reward_summary(
    total_reward: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """``total_reward_{max,min,mean,ci95}`` of per-env episodic total reward."""
    v = total_reward.reshape(-1)
    if mask is not None:
        v = v[mask.reshape(-1)]
    arr = v.detach().cpu().numpy()
    if arr.size == 0:
        return {f"total_reward_{s}": 0.0 for s in ("max", "min", "mean", "ci95")}
    m, ci = mean_confidence_interval(arr)
    return {
        "total_reward_max": float(np.max(arr)),
        "total_reward_min": float(np.min(arr)),
        "total_reward_mean": m,
        "total_reward_ci95": ci,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Unified wandb logging keys
# ─────────────────────────────────────────────────────────────────────────── #

def track_stability_summary(agent, summary: dict[str, float], *, tab: str = "Stability") -> None:
    """Push a :func:`summarize` dict onto ``agent`` under ``{tab}/...``.

    Emits ``{tab}/{name}_mean`` and ``{tab}/{name}_ci95`` for every metric —
    no bare ``{tab}/{name}`` alias, since that would be a byte-for-byte
    duplicate of ``{name}_mean``.  ``contraction_rate`` is additionally
    mirrored to the user-facing name ``lambda`` (single ``{tab}/lambda`` +
    ``{tab}/lambda_ci95``, not also ``lambda_mean`` — same reasoning).
    """
    for k, v in summary.items():
        agent.track_data(f"{tab}/{k}", float(v))
    if "contraction_rate_mean" in summary:
        agent.track_data(f"{tab}/lambda", float(summary["contraction_rate_mean"]))
        if "contraction_rate_ci95" in summary:
            agent.track_data(f"{tab}/lambda_ci95", float(summary["contraction_rate_ci95"]))


def track_reward_summary(agent, summary: dict[str, float], *, tab: str = "Reward") -> None:
    """Push a :func:`reward_summary` dict onto ``agent`` under ``{tab}/...``."""
    for k, v in summary.items():
        agent.track_data(f"{tab}/{k}", float(v))


def reward_log_dict(summary: dict[str, float], device, *, tab: str = "Reward") -> dict:
    """``extras['log']``-style dict of scalar tensors mirroring
    :func:`track_reward_summary` (``{tab}/total_reward_{max,min,mean,ci95}``).

    For environments that log per-episode reward through Isaac Lab's
    ``extras['log']`` (surfaced to skrl as scalar tensors via
    ``environment_info: log``), so PPO/SAC/C2RL land on the same
    ``Reward/total_reward_mean`` key — at the same per-episode-reset cadence as
    :func:`stability_log_dict` — that C3M's eval loop emits via
    :func:`track_reward_summary`. Without this, that key was only ever written
    once, by the post-training evaluator in train.py.
    """
    return {f"{tab}/{k}": torch.tensor(float(v), device=device) for k, v in summary.items()}


def stability_log_dict(summary: dict[str, float], device, *, tab: str = "Stability") -> dict:
    """``extras['log']``-style dict of scalar tensors mirroring
    :func:`track_stability_summary` (``{tab}/{name}_mean``, ``_ci95`` and the
    ``lambda`` alias — no bare ``{tab}/{name}`` duplicate, see that function).

    For environments that log per-episode metrics through Isaac Lab's
    ``extras['log']`` (surfaced to skrl as scalar tensors), so PPO/SAC/LQR/SD-LQR
    land on the exact same wandb keys the contraction trainers emit via
    :func:`track_stability_summary`.
    """
    out: dict = {}
    for k, v in summary.items():
        out[f"{tab}/{k}"] = torch.tensor(float(v), device=device)
    if "contraction_rate_mean" in summary:
        out[f"{tab}/lambda"] = torch.tensor(float(summary["contraction_rate_mean"]), device=device)
        if "contraction_rate_ci95" in summary:
            out[f"{tab}/lambda_ci95"] = torch.tensor(float(summary["contraction_rate_ci95"]), device=device)
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Unified trajectory / normalized-error plots
# ─────────────────────────────────────────────────────────────────────────── #

def _wandb_run():
    """Return the active wandb run (or None) without importing wandb eagerly."""
    if "wandb" not in sys.modules:
        return None
    return getattr(sys.modules["wandb"], "run", None)


def log_raw_config(raw_cfg: dict | None) -> None:
    """Push the pre-filter agent/trainer/models YAML dict to wandb.config.

    skrl's own ``Agent.init()`` already logs ``dataclasses.asdict(self.cfg)``
    (i.e. only fields the algorithm's Cfg dataclass declares) — any YAML key
    ``_filter_cfg_fields`` silently dropped (typo'd sweep param, stray key
    left over from copy-pasting another algorithm's config, ...) never makes
    it into that log. This logs the *complete* dict as loaded from YAML
    (post CLI overrides, pre dataclass-filtering) under ``raw_yaml/*`` so a
    dropped key is still visible on the run instead of silently vanishing.
    No-op if wandb is inactive.
    """
    run = _wandb_run()
    if run is None or not raw_cfg:
        return
    run.config.update({"raw_yaml": raw_cfg}, allow_val_change=True)


def log_metric_distributions(
    dists: dict,
    *,
    prefix: str = "train",
    step: int | None = None,
    title: str | None = None,
    key: str = "metric_distributions",
) -> None:
    """Push one figure of per-env metric histograms to ``{prefix}/{key}``.

    ``dists`` is :meth:`StatManagerEnvWrapper.distributions` — the unreduced
    per-env vectors behind the ``Stability/*`` scalars. Four panels in a single
    figure: AUC, contraction rate ``lambda``, running ``lambda``, peak
    overshoot.

    These exist because the scalars cannot express the shape that matters here.
    A non-terminating env that diverges is pinned at the position bound while
    its reference drives away, so its AUC grows without bound and ``auc_mean``
    degenerates into a count of blow-ups — a tight cluster plus two far-right
    outliers and a uniformly-worse population produce the same mean. The
    histogram separates them at a glance.

    Each panel draws the mean (dashed) and median (dotted). AUC and peak
    overshoot switch to a log x-axis when ``max/median > 20``, since one
    diverged env otherwise compresses the entire population into the first bin.
    ``running_lambda`` covers contracting episodes only, so its n is reported
    separately and an empty vector renders as an explicit "no contracting
    episodes" panel rather than an empty axis.

    No-op when wandb is inactive or ``dists`` is empty.
    """
    if _wandb_run() is None or not dists:
        return
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image

    PANELS = (
        ("auc", "AUC", "∫ e(t)/e(0) dt"),
        ("lambda", "contraction rate λ", "λ"),
        ("running_lambda", "running λ (contracting episodes)", "λ"),
        ("peak", "peak overshoot", "max_t e(t)/e(0)"),
    )
    if not any(np.asarray(dists.get(k, [])).size for k, _, _ in PANELS):
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    for ax, (name, panel_title, xlabel) in zip(axes.ravel(), PANELS):
        v = np.asarray(dists.get(name, []), dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.text(0.5, 0.5, "no contracting episodes" if name == "running_lambda"
                    else "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel_title)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        mean, median = float(np.mean(v)), float(np.median(v))
        # A heavy right tail (one diverged env) would otherwise put the whole
        # population in bin 0. Log bins need a strictly positive floor.
        heavy = name in ("auc", "peak") and median > 0 and v.max() / median > 20.0
        if heavy:
            lo = max(float(v[v > 0].min()) if np.any(v > 0) else 1e-8, 1e-8)
            bins = np.logspace(np.log10(lo), np.log10(max(float(v.max()), lo * 10)), 40)
            ax.set_xscale("log")
        else:
            bins = min(40, max(10, int(np.sqrt(v.size)) * 2))

        ax.hist(v, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(mean, color="#C44E52", ls="--", lw=1.5, label=f"mean {mean:.3g}")
        ax.axvline(median, color="#55A868", ls=":", lw=1.8, label=f"median {median:.3g}")
        ax.set_title(f"{panel_title}   (n={v.size})")
        ax.set_xlabel(xlabel + ("  [log]" if heavy else ""))
        ax.set_ylabel("envs")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title or prefix} — per-env metric distributions")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    payload = {f"{prefix}/{key}": wandb.Image(Image.open(buf))}
    if step is not None:
        payload["global_step"] = step
    wandb.log(payload)


def log_tracking_plots(
    traj_x: dict,
    traj_xref: dict,
    traj_error: dict,
    *,
    dt: float,
    prefix: str = "train",
    step: int | None = None,
    title: str | None = None,
    traj_maha_error: dict | None = None,
) -> None:
    """Push ``{prefix}/normalized_error`` and ``{prefix}/path_tracking`` to wandb.

    ``traj_x`` / ``traj_xref`` map an env index → list of per-step position
    arrays; ``traj_error`` maps env index → list of per-step scalar error norms.
    Dimensionality of the position vectors selects a 1-D (vs time), 2-D or 3-D
    trajectory plot.  No-op when wandb is inactive or nothing was collected.
    ``prefix`` is the full leading key path — e.g. ``"train"`` for single-policy
    algorithms, ``"train/con"`` / ``"train/opt"`` for C2RL's two policies.
    """
    if _wandb_run() is None:
        return
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image

    label = title or prefix

    def _push(fig, key):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        payload = {f"{prefix}/{key}": wandb.Image(Image.open(buf))}
        if step is not None:
            payload["global_step"] = step
        wandb.log(payload)

    # All curves for a given wandb key go into one figure laid out as a grid of
    # subplots — PER_SUBPLOT envs per subplot, ceil(n / PER_SUBPLOT) subplots —
    # so every eval env is logged (not a random handful) while each panel stays
    # readable. The grid is sized to the number of envs actually passed in, so
    # normalized_error / normalized_maha_error / path_tracking share the same
    # env set and ordering (the caller supplies a single consistent key list).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    PER_SUBPLOT = 5

    def _grid_dims(n_sub: int) -> tuple[int, int]:
        ncols = int(math.ceil(math.sqrt(n_sub)))
        nrows = int(math.ceil(n_sub / ncols))
        return nrows, ncols

    def _plot_error_grid(traj: dict, key: str, plot_title: str, ylabel: str) -> None:
        items = [(i, np.asarray(errs, dtype=np.float64))
                 for i, errs in traj.items() if errs is not None and len(errs) > 0]
        if not items:
            return
        # One shared axes, all curves overlaid thin/translucent — with dozens
        # of eval envs a per-env legend/grid is unreadable; a thin overlay
        # instead shows the trend (and its spread) at a glance.
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for i, errs_arr in items:
            e0 = max(float(errs_arr[0]), 1e-8)
            norm = errs_arr / e0
            ax.plot(norm, linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"{label} {plot_title} ({len(items)} envs)")
        fig.tight_layout()
        _push(fig, key)

    # ── Normalized error / squared-Mahalanobis error curve grids ──────────── #
    _plot_error_grid(traj_error, "normalized_error", "Normalized Error", "Normalized Error")
    if traj_maha_error:
        _plot_error_grid(traj_maha_error, "normalized_maha_error",
                         "Normalized Mahalanobis Error (squared, V/V₀)", "V(t)/V(0)")

    # ── Position trajectory vs reference grid ─────────────────────────────── #
    pos_items = []
    for i in traj_x:
        # `x if x else []` (or `x or []`) would raise on a numpy array — these
        # dicts hold lists today but are fed by callers, so test emptiness by
        # length, never by truthiness.
        xs = traj_x.get(i)
        refs = traj_xref.get(i)
        if xs is None or refs is None or len(xs) == 0 or len(refs) == 0:
            continue
        tx = np.asarray(xs, dtype=np.float64)
        txref = np.asarray(refs, dtype=np.float64)
        d = min(tx.shape[-1], txref.shape[-1])
        if d < 1:
            continue
        pos_items.append((i, tx, txref, d))
    if pos_items:
        d0 = pos_items[0][3]  # env family is homogeneous — set projection once
        n_sub = int(math.ceil(len(pos_items) / PER_SUBPLOT))
        nrows, ncols = _grid_dims(n_sub)
        subplot_kw = {"projection": "3d"} if d0 >= 3 else {}
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                                 squeeze=False, subplot_kw=subplot_kw)
        flat = axes.flatten()
        for s in range(n_sub):
            ax = flat[s]
            for i, tx, txref, _d in pos_items[s * PER_SUBPLOT:(s + 1) * PER_SUBPLOT]:
                # Separate time axes: tx is the policy rollout (ends early when
                # the episode terminates), txref is the whole reference path.
                # Sharing one axis would truncate the reference to the rollout.
                t = np.arange(len(tx))
                tref = np.arange(len(txref))
                if d0 == 1:
                    ax.scatter(t, tx[:, 0], c=t, cmap="viridis", s=8, label=f"x (env {i})")
                    ax.plot(tref, txref[:, 0], "--", label=f"x_ref (env {i})")
                    ax.set_xlabel("Step"); ax.set_ylabel("Position")
                elif d0 == 2:
                    ax.scatter(tx[:, 0], tx[:, 1], c=t, cmap="viridis", s=8, label=f"x (env {i})")
                    ax.plot(txref[:, 0], txref[:, 1], "--", label=f"x_ref (env {i})")
                    ax.set_xlabel("X"); ax.set_ylabel("Y")
                else:
                    ax.scatter(tx[:, 0], tx[:, 1], tx[:, 2], c=t, cmap="viridis", s=8, label=f"x (env {i})")
                    ax.plot(txref[:, 0], txref[:, 1], txref[:, 2], "--", label=f"x_ref (env {i})")
                    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
        for s in range(n_sub, len(flat)):
            flat[s].axis("off")
        fig.suptitle(f"{label} Path Tracking")
        fig.tight_layout()
        _push(fig, "path_tracking")
