"""Base class for path-tracking environments.

A path-tracking environment gives the agent:
    obs = [x_current, x_ref, u_ref]

and rewards it with the quadratic contraction cost:
    r = -||x_current - x_ref||_I^2   (identity weighting)

Subclasses must implement:
    _setup_scene(), _apply_action(),
    _get_physical_state() -> (N, state_dim),
    _get_dones() -> (terminated, time_out),
    _set_robot_state_from_ref(env_ids, x_ref_init)  — reset robot to match ref[0]
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from c2rl.agents.skrl.angle_utils import wrap_diff
from c2rl.agents.skrl.ref_window import RefWindow

from .eval_metrics import fit_exponential_envelope
from .state_guard import carry_forward_nonfinite
from .termination_box import TerminationBoxMixin
from .traj_buffer import TrajectoryBuffer

# wandb is optional; only used when a run is active
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

# matplotlib is optional; only used to render the PathTracking wandb figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
except ImportError:
    _plt = None

_WANDB_PLOT_INTERVAL = 20   # log trajectory plot every N completed episodes (env 0)
_VIZ_MAX_ENVS = 1         # cap on how many envs' trajectories go into the plot


class PathTrackingBase(TerminationBoxMixin, DirectRLEnv):
    """Abstract base for path-tracking environments.

    Subclass must define:
        cfg.traj_path   : str — path to .npz reference trajectory file
        cfg.action_space, observation_space

    Subclass may override:
        angle_idx : list[int] — indices within the (raw, per-block) physical
            state that hold a wrapping angle (e.g. yaw). Default () — most
            path-tracking states (proj_gravity + joint pos/vel) have no raw
            angle. Networks embed these continuously ((cos, sin), see
            agents/skrl/angle_utils.py); the tracking error computed here stays
            in raw coordinates but has these dims shortest-angle wrapped before
            any norm/reward/metric — otherwise a wraparound (e.g. a yaw U-turn)
            would spike the error by ~2*pi.
    """
    angle_idx: Sequence[int] = ()

    # One name per physical state dim (see agents/skrl/state_symmetry.py for the
    # vocabulary). Declaring names rather than raw indices is what lets the
    # networks quotient out the symmetry of the tracking problem: which dims are
    # pure translation directions, which is the planar heading, and which co-
    # rotate under a yaw rotation. Crucially the frame suffix matters --
    # root_lin_vel_b / root_ang_vel_b / projected_gravity_b are body frame and so
    # already yaw invariant, whereas a world-frame velocity would have to be
    # co-rotated with the position pair. Left empty -> angle_idx below is used as
    # before and no quotient is applied.
    #
    # Subclasses whose width depends on the robot (joint count) should override
    # the ``state_names`` property instead of this attribute.
    _state_names: Sequence[str] = ()

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self._state_names)

    @property
    def pos_dimension(self) -> int:
        """Number of pure translation directions, derived from state_names."""
        names = self.state_names
        if not names:
            return 0
        from c2rl.agents.skrl.state_symmetry import StateSymmetry
        return StateSymmetry.from_names(names).pos_dimension

    # ── Divergence guard ────────────────────────────────────────────────── #
    # An initially unstable policy (random init, or C2RL's policy early on)
    # can drive the sim to a non-finite / exploded physical state. Rather than
    # terminating the episode to recover (which would truncate the very
    # trajectory the contraction metrics are fit over), any non-finite element
    # is replaced by the same element from the last known-finite state and the
    # episode keeps running — see carry_forward_nonfinite(). This env family
    # Never terminates on divergence (see _get_dones() in each subclass);
    # falling (terminate_on_fall) is the only termination path, and it is off
    # by default for path tracking.
    _ERROR_CLAMP = 1.0e3              # cap on ||error|| feeding reward/AUC/metrics

    def _sanitize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Replace non-finite elements of the physical state with the same
        element from the last known-finite state (carry-forward), so a
        diverging rollout can't propagate NaN/Inf downstream — without ever
        needing to reset/terminate the env to do so.

        Emits a throttled warning (once per physics step, regardless of how many
        call sites sanitize) whenever it actually has to scrub a non-finite
        state — this should be rare, so a silent guard would hide a real problem.
        """
        nonfinite = ~torch.isfinite(x)
        if nonfinite.any():
            step = int(getattr(self, "common_step_counter", 0))
            if step != getattr(self, "_last_nonfinite_warn_step", -1):
                self._last_nonfinite_warn_step = step
                n_bad = int(nonfinite.any(dim=-1).sum())
                print(
                    f"[PathTracking] WARNING: {n_bad}/{x.shape[0]} env(s) produced a "
                    f"non-finite (NaN/Inf) physical state at step {step} — carrying "
                    f"the last finite value forward (divergence guard).",
                    flush=True,
                )
        if self._last_valid_state is None:
            self._last_valid_state = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        sanitized = carry_forward_nonfinite(x, self._last_valid_state)
        self._last_valid_state = sanitized
        return sanitized

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._traj_buf = TrajectoryBuffer(cfg.traj_path, self.device)

        # Reference window — the observation is s = {x, xrefs, urefs}, identical
        # in layout to the classic envs (see agents/skrl/ref_window.py and
        # classic/common/env_base.py). configure_ref_window() resizes it before
        # the models are built.
        self.ref_window = RefWindow(
            x_dim=self._traj_buf.state_dim,
            u_dim=self._traj_buf.action_dim,
            length=int(getattr(cfg, "ref_length", 1) or 1),
            offset=int(getattr(cfg, "ref_offset", 1) or 1),
        )
        self._rebuild_observation_space()

        n = self.num_envs
        self._traj_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        # See set_fix_ref_trajectories. Off by default: every reset redraws.
        self.fix_ref_trajectories = False
        self._ref_fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._x_ref = torch.zeros(n, self._traj_buf.state_dim, device=self.device)
        self._u_ref = torch.zeros(n, self._traj_buf.action_dim, device=self.device)
        self._actions = torch.zeros(n, self.action_space.shape[0], device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # last known-finite physical state, for the divergence guard (see
        # _sanitize_state) — populated lazily on first use.
        self._last_valid_state: torch.Tensor | None = None

        # ── Early-termination box — the Isaac twin of BaseEnv's ──────────── #
        # Same name, same semantics, same default (on), so
        # set_terminate_out_of_box / terminate_out_of_box / X_TERMINATION_*
        # resolve identically on both families (tests/test_isaac_parity.py's
        # rule). Two real differences, both from Isaac Sim rather than choice:
        #
        #   * The box defaults to _state_bounds(), which is ±inf for most Isaac
        #     envs (they have no physical state box the way car/segway do). An
        #     all-infinite box can never be crossed, so the feature is inert
        #     there until a subclass declares finite bounds — announced once at
        #     construction rather than left to look armed.
        #   * There is no clamp to fire at. The classic pathology is a diverged
        #     env silently pinned at its box; here it is a diverged env running
        #     on under the carry-forward guard. A non-finite state therefore
        #     counts as out-of-box: it is the same event this is meant to end.
        _bounds_lo, _bounds_hi = self._state_bounds()
        self._init_termination_box(
            getattr(self.cfg, "x_termination_min", None) or _bounds_lo,
            getattr(self.cfg, "x_termination_max", None) or _bounds_hi,
            clamp_box=None,                       # no clamp on this side
            armed=bool(getattr(self.cfg, "terminate_out_of_box", True)),
            terminal=bool(getattr(self.cfg, "x_termination_terminal", False)),
            tag="PathTracking")
        # Every subclass owns its own _get_dones (fall detection, task-specific
        # limits), so the box is folded in by wrapping the bound method once
        # here rather than editing each subclass — the same post-construction
        # patching idiom agent_patches.py uses for skrl, and for the same
        # reason: one place to change, and no subclass can forget to call it.
        self._get_dones = self._with_termination_box(self._get_dones)

        # --- episode-level eval metric accumulators ---
        # Streaming (memory-efficient) accumulators for the unified contraction
        # metrics — running sum of error norms (_episode_auc), first/last/peak
        # error and step count — so auc/contraction_rate/overshoot/
        # contraction_score are computed for every finished env without storing
        # the full per-step error curve (see contraction_metrics.per_env_metrics;
        # the same streaming math C3M/C2RL eval use).
        self._episode_auc = torch.zeros(n, device=self.device)
        self._episode_e0 = torch.zeros(n, device=self.device)
        self._episode_emax = torch.zeros(n, device=self.device)
        self._episode_contraction_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._prev_error_norm = torch.full((n,), float("inf"), device=self.device)
        self._episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        # per-episode undiscounted return, for the Reward/total_reward_* tab
        # (same cadence/keys as Stability, see _reset_idx).
        self._episode_reward = torch.zeros(n, device=self.device)

        # --- PathTracking wandb figure: position + error^2 across up to
        # _VIZ_MAX_ENVS envs (CPU, pre-allocated). "live" buffers accumulate the
        # in-progress episode per viz env; "hist" holds each viz env's most
        # recently completed episode (snapshotted at that env's own reset,
        # since viz envs can terminate/reset at different times) — the plot
        # renders from "hist", not "live".
        self._viz_n = min(_VIZ_MAX_ENVS, n)
        self._viz_error_live = np.zeros((self._viz_n, self.max_episode_length), dtype=np.float32)
        self._viz_pos_live = np.zeros((self._viz_n, self.max_episode_length, 3), dtype=np.float32)
        self._viz_state_live = np.zeros((self._viz_n, self.max_episode_length, self._traj_buf.state_dim), dtype=np.float32)
        self._viz_error_hist = np.zeros((self._viz_n, self.max_episode_length), dtype=np.float32)
        self._viz_pos_hist = np.zeros((self._viz_n, self.max_episode_length, 3), dtype=np.float32)
        self._viz_state_hist = np.zeros((self._viz_n, self.max_episode_length, self._traj_buf.state_dim), dtype=np.float32)
        self._viz_len_hist = np.zeros((self._viz_n,), dtype=np.int64)
        self._env0_episode_count = 0

        # Contraction interface — dynamics model injected by ContractionRunner
        self._dynamics_model = None
        # Frozen CMG (contraction metric generator) — injected via set_ccm()
        # for C2RL's Phase B. None for PPO/SAC/LQR/SD-LQR, which never call
        # set_ccm and so keep the plain -(error_norm**2) reward in
        # _get_rewards unchanged. Defaults match that plain reward's implicit
        # weights (identity tracking weight, no control penalty) so enabling
        # the CCM branch doesn't also silently reweight the objective.
        self.ccm_gen = None
        self.tracking_scaler = 1.0
        self.control_scaler = 0.0
        # Certified/target contraction rate + metric-conditioning inputs for
        # the theoretical exponential bound on the PathTracking figure. All
        # None by default (curve omitted) — set via set_contraction_certificate().
        self.target_lambda: float | None = None
        self._rl_discount_factor: float | None = None
        self._static_metric_bounds: tuple[float, float] | None = None
        self._cmg_bounds_fn: Callable[[torch.Tensor], tuple[float, float]] | None = None

    # ------------------------------------------------------------------ #
    # Contraction algorithm interface (C3M / LQR / SD-LQR / C2RL)
    # ------------------------------------------------------------------ #

    def set_dynamics_model(self, model) -> None:
        """Inject a NeuralDynamics model for get_f_and_B (required for Isaac envs)."""
        self._dynamics_model = model

    def _metric_from_cmg(self, x):
        """M(x) from the CMG, inverting only when the head emits W.

        Mirrors classic env_base._metric_from_cmg exactly — see
        tests/test_isaac_parity.py, which requires both families to convert the
        CMG output the same way or their Mahalanobis rewards stop being
        comparable. cmg_method="cvstem" builds the CMG with outputs_metric=True
        and its forward already returns M (so this is a pass-through and the
        per-step SPD inverse disappears); "ccm" emits W and is inverted here.
        """
        from c2rl.agents.skrl.math_utils import bound_W, spd_inverse
        raw, _ = self.ccm_gen(x)
        out = bound_W(raw, self.w_lb, x.shape[-1],
                      getattr(self.ccm_gen, "bounded", False))
        if getattr(self.ccm_gen, "outputs_metric", False):
            return out
        return spd_inverse(out)

    def set_ccm(self, ccm_gen, w_lb, device, tracking_scaler=None, control_scaler=None,
                reward_euclidean=False, reward_level=False,
                residual_anchor_scale=0.0, cvstem_r_scaler=1.0) -> None:
        """Inject the frozen CMG for C2RL's Phase B Mahalanobis reward.

        Mirrors classic/common/env_base.py's BaseEnv.set_ccm exactly (same
        signature) so C2RLSkrlTrainer.train() can call it identically for
        both env families — see _get_rewards for the reward this enables.

        The scalers default to None = keep the plain-reward weights set in
        __init__ (1.0 / 0.0).

        The last four are classic-only reward variants (residual-base RL over
        CV-STEM-LQR); _get_rewards here has no branch for them, so a non-default
        value raises rather than silently training on the Mahalanobis reward.
        """
        unsupported = [n for n, v in (("reward_euclidean", reward_euclidean),
                                      ("reward_level", reward_level),
                                      ("residual_anchor_scale", residual_anchor_scale),
                                      ("cvstem_r_scaler", cvstem_r_scaler))
                       if v not in (False, 0.0) and not (n == "cvstem_r_scaler" and v == 1.0)]
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__}._get_rewards implements only the Mahalanobis "
                f"reward; classic-only knobs requested: {', '.join(unsupported)}."
            )
        self.ccm_gen = ccm_gen
        self.w_lb = w_lb
        self.ccm_device = device
        if tracking_scaler is not None:
            self.tracking_scaler = float(tracking_scaler)
        if control_scaler is not None:
            self.control_scaler = float(control_scaler)

    def set_contraction_certificate(
        self,
        lbd: float | None,
        *,
        discount_factor: float | None = None,
        static_metric_bounds: tuple[float, float] | None = None,
        cmg_bounds_fn: Callable[[torch.Tensor], tuple[float, float]] | None = None,
    ) -> None:
        """Configure the two exponential bounds drawn on the PathTracking
        figure, both of the same shape:

            e(t) <= sqrt(m_bar / m_underbar) * [1/(1-gamma)] * e(0) * exp(-lbd*t)

        differing only in where (m_bar, m_underbar) come from:
          - Theoretical: the CMG's configured hard limits (w_lb/w_ub), i.e.
            what the metric is *guaranteed* to satisfy by construction,
            regardless of what the network has actually learned so far.
          - Empirical: the current metric's eigenvalue extremes, *measured* on
            the actually-visited states — reflects the real (possibly looser
            or tighter) network today, not the worst-case design limit.

        Called by ContractionRunner for algorithms that have a certified rate
        (C3M/C2RL's cfg.lbd); left at defaults (both curves omitted) for
        PPO/SAC/LQR/SD-LQR, which have neither a CMG nor a certified rate.

        Args:
            lbd: certified/target contraction rate (e.g. cfg.lbd).
            discount_factor: RL discount gamma of the policy actually being
                deployed/plotted — inflates both bounds by 1/(1-gamma) since a
                *discounted* objective enforces the certificate on average
                rather than as a hard per-step constraint. None (or 0) for
                C3M, which has no discounting at all — gamma > 0 only for an
                RL-trained policy (C2RL's deployed policy uses its
                discount_factor — i.e. the policy whose rollout the figure
                is actually showing).
            static_metric_bounds: (m_bar, m_underbar) from the CMG's configured
                w_lb/w_ub (m_bar=1/w_lb, m_underbar=1/w_ub) — the theoretical
                bound's fixed conditioning factor.
            cmg_bounds_fn: callable (x_batch) -> (m_bar, m_underbar), evaluating
                the current contraction metric's eigenvalue extremes on a batch
                of states — the empirical bound's measured conditioning factor.
        """
        self.target_lambda = None if lbd is None else float(lbd)
        self._rl_discount_factor = discount_factor
        self._static_metric_bounds = static_metric_bounds
        self._cmg_bounds_fn = cmg_bounds_fn

    def _get_visualization_position(self) -> torch.Tensor:
        """Returns (N, 3) world-frame position for the PathTracking figure's
        position subplot. Default: the robot's root position — meaningful for
        locomotion (humanoid/quadruped), but degenerates to a single fixed
        point for a fixed-base arm (manipulator), since its root never moves.
        Override in a subclass (e.g. to return end-effector position) for a
        more informative plot there.
        """
        robot = getattr(self, "_robot", None)
        if robot is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return robot.data.root_pos_w

    def _log_pathtracking_figure(self) -> None:
        """Render the "PathTracking" wandb figure from the viz envs' most
        recently completed episodes:
          left  — attempted-trajectory position (world xy) per env
          right — ||error||_I^2 per step, plus three reference curves (each
                  only drawn if its inputs are available):
            1. fitted envelope — fit_exponential_envelope (CAC-dev style),
               a pure data fit of the observed error trajectories: no CMG,
               no gamma, just the tightest C*exp(-lambda*t) that bounds them.
            2. theoretical bound — sqrt(m_bar/m_underbar) * [1/(1-gamma)] *
               e(0) * exp(-lbd*t), where m_bar/m_underbar come from the CMG's
               configured hard limits (w_lb/w_ub) — the worst-case guarantee
               by construction, regardless of what the network currently does.
            3. empirical bound — same shape, but m_bar/m_underbar are
               measured (max/min eigenvalues of the current CMG evaluated on
               the actually-visited states) — reflects the real network today.
          Both (2)/(3) use set_contraction_certificate()'s lbd/gamma; gamma=0
          (no inflation) for non-discounted certificates like C3M, gamma>0 for
          an RL-trained policy (C2RL).
        """
        valid = np.nonzero(self._viz_len_hist > 0)[0]
        if len(valid) == 0:
            return
        dt = self.step_dt

        # (i+1)*dt indexing, matching fit_exponential_envelope's own convention
        error_traces = [self._viz_error_hist[i, : self._viz_len_hist[i]] for i in valid]
        pos_traces = [self._viz_pos_hist[i, : self._viz_len_hist[i]] for i in valid]

        # Plot 1: Attempted trajectories
        fig_pos, ax_pos = _plt.subplots(figsize=(6, 5))
        for p in pos_traces:
            ax_pos.plot(p[:, 0], p[:, 1], linewidth=0.8, alpha=0.6)
        ax_pos.set_xlabel("x [m]")
        ax_pos.set_ylabel("y [m]")
        ax_pos.set_title(f"Attempted trajectories (n={len(valid)})")
        ax_pos.set_aspect("equal", adjustable="datalim")
        fig_pos.tight_layout()
        _wandb.log({"train/tracking_trajectory": _wandb.Image(fig_pos), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
        _plt.close(fig_pos)

        # Plot 2: Tracking error with contraction bounds
        fig_err, ax_err = _plt.subplots(figsize=(6, 5))
        max_len = max(len(e) for e in error_traces)
        t_axis = np.arange(1, max_len + 1) * dt
        for e in error_traces:
            t = np.arange(1, len(e) + 1) * dt
            ax_err.plot(t, e ** 2, linewidth=0.6, alpha=0.5, color="tab:blue")

        # 1. fitted envelope — data only, no CMG/gamma involved.
        C_fit, lbds_fit = fit_exponential_envelope(error_traces, dt)
        pos_lbds = lbds_fit[lbds_fit > 0]
        lambda_fit = float(pos_lbds.mean()) if pos_lbds.size else 0.0
        if lambda_fit > 0:
            e0_mean = float(np.mean([e[0] for e in error_traces if len(e) and e[0] > 0]))
            ax_err.plot(
                t_axis, (e0_mean * C_fit * np.exp(-lambda_fit * t_axis)) ** 2, "r--", linewidth=2,
                label=f"fitted envelope (C={C_fit:.2g}·e₀, λ̄={lambda_fit:.2g})",
            )

        if self.target_lambda is not None:
            e0_mean = float(np.mean([e[0] for e in error_traces]))
            gamma_factor = 1.0
            if self._rl_discount_factor is not None:
                gamma_factor = 1.0 / max(1e-6, 1.0 - self._rl_discount_factor)

            # 2. theoretical — CMG's configured hard limits (w_lb/w_ub).
            if self._static_metric_bounds is not None:
                m_bar, m_underbar = self._static_metric_bounds
                cond = float(np.sqrt(max(m_bar, 1e-12) / max(m_underbar, 1e-12)))
                C_theory = cond * gamma_factor * e0_mean
                ax_err.plot(
                    t_axis, (C_theory * np.exp(-self.target_lambda * t_axis)) ** 2, "g:", linewidth=2,
                    label=f"theoretical bound (C={C_theory:.2g}, λ={self.target_lambda:.2g})",
                )

            # 3. empirical — current CMG's eigenvalues, measured on visited states.
            if self._cmg_bounds_fn is not None:
                state_traces = [self._viz_state_hist[i, : self._viz_len_hist[i]] for i in valid]
                states = np.concatenate(state_traces, axis=0)
                if len(states) > 2000:
                    states = states[np.random.choice(len(states), 2000, replace=False)]
                x_batch = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                m_bar, m_underbar = self._cmg_bounds_fn(x_batch)
                cond = float(np.sqrt(max(m_bar, 1e-12) / max(m_underbar, 1e-12)))
                C_emp = cond * gamma_factor * e0_mean
                ax_err.plot(
                    t_axis, (C_emp * np.exp(-self.target_lambda * t_axis)) ** 2, "m-.", linewidth=2,
                    label=f"empirical bound (C={C_emp:.2g}, λ={self.target_lambda:.2g})",
                )

        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel(r"$\|e\|_I^2$")
        ax_err.set_title("Tracking error² with contraction bounds")
        ax_err.set_yscale("log")
        ax_err.legend(fontsize=8)
        fig_err.tight_layout()
        _wandb.log({"train/normalized_error": _wandb.Image(fig_err), "global_step": int(getattr(self, "common_step_counter", 0))})  # type: ignore[attr-defined]
        _plt.close(fig_err)

    def get_f_and_B(self, x, *, need_null: bool = True):
        """Return (f, B, B_null) for contraction agents.

        Delegates to the injected NeuralDynamics model — call
        ``set_dynamics_model(model)`` before using C3M/LQR/SDLQR/C2RL.

        ``need_null`` exists for signature parity with ``BaseEnv.get_f_and_B``
        (a contraction agent calls whichever it was handed). It is accepted and
        ignored here: the NeuralDynamics forward pass produces all three heads
        together, so there is nothing to skip.
        """
        del need_null
        if self._dynamics_model is None:
            raise RuntimeError(
                "get_f_and_B requires a NeuralDynamics model — Isaac Sim envs have no "
                "analytical dynamics. C3M/C2RL build and pretrain one themselves; "
                "lqr/sdlqr/cvstem-lqr need one injected via set_dynamics_model(), which "
                "ContractionRunner does from --dynamics_checkpoint (a dynamics.pt saved "
                "by a previous C3M/C2RL run)."
            )
        return self._dynamics_model.get_f_and_B(x)

    def get_rollout(self, buffer_size: int, mode: str, num_control_per_state: int | None = None) -> dict:
        """Sample random (x, xref, uref) triples for C3M-style contraction synthesis.

        Samples reference states from the trajectory buffer, adding bounded
        Gaussian noise to produce actual states that deviate from the reference.
        Returns numpy float32 arrays (required by mjrl's to_tensor()).

        ``num_control_per_state`` is a classic-env-only knob (see
        ``env_base.get_rollout``) accepted here purely so callers can pass it
        uniformly across both env families — Isaac's dynamics rollout draws
        real consecutive-step transitions from the trajectory buffer, so there
        is no "controls per state" tiling to configure; the argument is ignored.
        """
        if mode == "dynamics":
            return self._get_dynamics_rollout(buffer_size)
        if mode != "c3m":
            raise ValueError(f"PathTrackingBase.get_rollout: unsupported mode '{mode}'")

        buf = self._traj_buf
        # Random trajectories and random timesteps within them
        traj_ids = torch.randint(0, buf.num_trajs, (buffer_size,), device=self.device)
        steps = torch.randint(0, buf.traj_len, (buffer_size,), device=self.device)
        xref, uref = buf.get(traj_ids, steps)  # (N, state_dim), (N, action_dim)

        # Small Gaussian noise to create x ≠ xref (contraction error)
        noise = torch.randn_like(xref) * 0.05
        x = xref + noise

        return {
            "x":    x.cpu().numpy().astype(np.float32),
            "xref": xref.cpu().numpy().astype(np.float32),
            "uref": uref.cpu().numpy().astype(np.float32),
        }

    def _get_dynamics_rollout(self, buffer_size: int) -> dict:
        """Sample (x, u, x_dot) pairs for NeuralDynamics training.

        Uses consecutive (t, t+1) pairs from the reference trajectory buffer
        and approximates x_dot via finite differences: (x_{t+1} - x_t) / step_dt.
        This avoids requiring additional env interaction beyond what's in the buffer.
        """
        buf = self._traj_buf
        traj_ids = torch.randint(0, buf.num_trajs, (buffer_size,), device=self.device)
        # Avoid last step so t+1 is always valid
        steps = torch.randint(0, buf.traj_len - 1, (buffer_size,), device=self.device)
        x_t, u_t = buf.get(traj_ids, steps)
        x_next, _ = buf.get(traj_ids, steps + 1)
        x_dot = (x_next - x_t) / self.step_dt
        return {
            "x":     x_t.cpu().numpy().astype(np.float32),
            "u":     u_t.cpu().numpy().astype(np.float32),
            "x_dot": x_dot.cpu().numpy().astype(np.float32),
        }

    def get_tracking_error(self) -> torch.Tensor:
        """Current ||x - x_ref|| per env, (N,).

        Shared across all path-tracking envs (quadruped/humanoid/manipulator) —
        used by the post-training evaluator to fit the exponential contraction
        envelope C * exp(-lambda * k * dt) bounding the error curve. angle_idx
        dims (e.g. yaw) are shortest-angle wrapped before the norm — see
        _get_rewards for why an unwrapped difference would be wrong.
        """
        return torch.norm(wrap_diff(self._get_physical_state() - self._x_ref, self.angle_idx), dim=-1)

    def set_fix_ref_trajectories(self, flag: bool) -> None:
        """Freeze (or unfreeze) the per-env reference trajectories.

        The classic twin is ``BaseEnv.set_fix_ref_trajectories``; both exist so
        the training entry point can set this uniformly (see
        tests/test_isaac_parity.py). Isaac references come from a fixed dataset,
        so "fixing" here means pinning each env slot to one sampled traj id for
        the whole run rather than redrawing every episode. Turning it on clears
        the assignment so the next reset re-mints it.
        """
        self.fix_ref_trajectories = bool(flag)
        if not hasattr(self, "_ref_fixed"):
            self._ref_fixed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ref_fixed[:] = False
        print(f"[PathTrackingBase] fix_ref_trajectories={self.fix_ref_trajectories}"
              + (f" ({self.num_envs} fixed reference trajectories)"
                 if self.fix_ref_trajectories else " (resampled every episode)"))

    def get_reference_trajectory(self) -> torch.Tensor:
        """whole reference path per env: ``(num_envs, T, x_dim)``.

        The classic twin is ``BaseEnv.get_reference_trajectory``; both exist so
        logging code plots the complete reference against the policy rollout
        instead of rebuilding a reference from the (offset-subsampled,
        horizon-truncated) observation window. See tests/test_isaac_parity.py.
        """
        return self._traj_buf.full(self._traj_ids)

    @property
    def x_dim(self) -> int:
        return self._traj_buf.state_dim

    @property
    def u_dim(self) -> int:
        return self._traj_buf.action_dim

    # ------------------------------------------------------------------ #
    # Interface to implement in subclasses
    # ------------------------------------------------------------------ #

    def _get_physical_state(self) -> torch.Tensor:
        """Returns (N, state_dim) current robot physical state."""
        raise NotImplementedError

    def _set_robot_state_from_ref(
        self, env_ids: torch.Tensor, x_ref_init: torch.Tensor
    ) -> None:
        """Set robot state (joints + base vel) to match x_ref_init."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # PathTracking logic (shared)
    # ------------------------------------------------------------------ #

    def _rebuild_observation_space(self) -> None:
        """Declare ``{x, xrefs, urefs}`` as the observation space.

        Isaac's DirectRLEnv takes an integer width from the env cfg, so the
        flat width is what it is told; the Dict is exposed alongside it for
        ``RefWindow.from_space``. Both agree by construction (``flat_dim``).
        """
        lo, hi = self._state_bounds()
        u_lo, u_hi = self._control_bounds()
        self.observation_space = self.ref_window.space(lo, hi, u_lo, u_hi)
        self.cfg.observation_space = self.ref_window.flat_dim
        self.num_observations = self.ref_window.flat_dim

    def _with_termination_box(self, get_dones):
        """Wrap a subclass ``_get_dones`` so the box folds into its result.

        Returns ``(terminated, time_out)`` exactly as Isaac Lab expects; the
        excursion joins ``time_out`` (truncation) unless x_termination_terminal.
        Also publishes ``episode_ended_early`` on ``extras`` so
        StatManagerEnvWrapper can drop those episodes from AUC/lambda instead of
        measuring a curve that was cut — which is precisely the objection that
        kept this env family from terminating on divergence before.
        """
        def wrapped():
            terminated, time_out = get_dones()
            left_box = self._left_termination_box(self._get_physical_state())
            if left_box is None:
                return terminated, time_out
            self.extras["episode_ended_early"] = left_box.clone()
            if self.x_termination_terminal:
                return terminated | left_box, time_out
            return terminated, time_out | left_box
        return wrapped

    def _state_bounds(self):
        """Per-dim (low, high) for the physical state. Unbounded by default —
        the classic envs have real boxes, Isaac ones generally do not, and the
        bounds are only used to declare the space."""
        n = self.ref_window.x_dim
        return (np.full(n, -np.inf, np.float32), np.full(n, np.inf, np.float32))

    def _control_bounds(self):
        lo = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        return lo, hi

    def configure_ref_window(self, length: int, offset: int = 1,
                             gamma: float | None = None) -> None:
        """Resize the reference window and rebuild ``observation_space`` —
        the Isaac twin of ``BaseEnv.configure_ref_window``. Must be called
        before the agent/memory are built. Idempotent."""
        self.ref_window = RefWindow(x_dim=self.ref_window.x_dim, u_dim=self.ref_window.u_dim,
                                    length=int(length), offset=int(offset))
        self._rebuild_observation_space()
        print(f"[PathTrackingBase] reference window: length={self.ref_window.length} "
              f"offset={self.ref_window.offset} (spans "
              f"{(self.ref_window.length - 1) * self.ref_window.offset} steps) -> "
              f"obs {self.ref_window.flat_dim}-wide")
        if gamma is not None:
            warn = self.ref_window.check_markov(float(gamma), int(self.max_episode_length))
            if warn:
                print(warn)

    def _get_observations(self) -> dict:
        step = self.episode_length_buf.long()
        x_refs, u_refs = self._traj_buf.get_window(self._traj_ids, step, self.ref_window)
        # Slot 0 is the current reference — what the reward and every metric use.
        self._x_ref, self._u_ref = x_refs[:, 0], u_refs[:, 0]
        # sanitize so the policy never receives NaN/Inf from a diverging env
        x = self._sanitize_state(self._get_physical_state())
        # Flat, in skrl's sorted-key Dict order (RefWindow.flatten) — Isaac's
        # DirectRLEnv passes the "policy" tensor straight through without a
        # Dict-space tensorize step, so flattening happens here.
        return {"policy": self.ref_window.flatten(x, x_refs, u_refs)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh _x_ref/_u_ref for this step here rather than relying on
        # _get_observations() to have set them: DirectRLEnv.step() calls
        # _get_rewards() before _get_observations() (after incrementing
        # episode_length_buf), so without this, _x_ref/_u_ref would still hold
        # the previous step's values — pairing the post-transition physical
        # state x below against a one-step-stale reference. _traj_buf.get is a
        # pure lookup (no side effects), so recomputing it again in
        # _get_observations() with the same episode_length_buf is harmless.
        step = self.episode_length_buf.long()
        self._x_ref, self._u_ref = self._traj_buf.get(self._traj_ids, step)

        # sanitize + clamp so a diverging env yields a large-but-Finite penalty
        # (never NaN/Inf) into the reward, AUC, and Stability curves — the
        # episode is never reset/terminated because of it (see _sanitize_state).
        x = self._sanitize_state(self._get_physical_state())
        # angle_idx dims (e.g. yaw) are wrapped to the shortest-angle difference
        # Before any norm/reward/metric — a raw difference would spike by ~2*pi
        # whenever the physical angle wraps (e.g. a yaw U-turn). Non-angle dims
        # are unaffected (wrap_diff is a no-op there, and a no-op entirely when
        # angle_idx is empty).
        error = wrap_diff(x - self._x_ref, self.angle_idx)
        error_norm = torch.norm(error, dim=-1).clamp(max=self._ERROR_CLAMP)   # (N,)

        # accumulate streaming metric state: running error sum (AUC integrand),
        # initial error e0 (on the episode's first step), and peak error e_max.
        self._episode_auc += error_norm
        is_first = (self._episode_steps == 0)
        self._episode_e0 = torch.where(is_first, error_norm, self._episode_e0)
        self._episode_emax = torch.maximum(self._episode_emax, error_norm)

        # contraction flag: count steps where error strictly decreased (skip step 0)
        contracting = (~is_first) & (error_norm < self._prev_error_norm)
        self._episode_contraction_steps += contracting.long()

        self._prev_error_norm = error_norm.clone()
        self._episode_steps += 1

        # collect the first _viz_n envs' trajectories for the PathTracking figure
        viz_n = self._viz_n
        viz_steps = self._episode_steps[:viz_n]  # already incremented above; use step-1 as index
        in_range = viz_steps <= self.max_episode_length
        if in_range.any():
            idx = (viz_steps[in_range] - 1).cpu().numpy()
            rows = in_range.nonzero(as_tuple=True)[0].cpu().numpy()
            self._viz_error_live[rows, idx] = error_norm[:viz_n][in_range].detach().cpu().numpy()
            pos = self._get_visualization_position()[:viz_n]
            self._viz_pos_live[rows, idx] = pos[in_range].detach().cpu().numpy()
            self._viz_state_live[rows, idx] = x[:viz_n][in_range].detach().cpu().numpy()

        if self.ccm_gen is not None:
            # C2RL Phase B: Lyapunov-decrease reward tracking_scaler*(V - next_V)
            # - control_scaler*||u||^2, V = error^T M error, M = the frozen
            # CMG's metric at the corresponding state — mirrors classic
            # env_base.py's get_rewards exactly. _get_rewards only ever sees
            # the post-transition state (no separate "prev x" lookup buffer
            # like the reference trajectory has), so the previous step's
            # error vector + metric are cached explicitly in
            # _prev_error/_prev_M (seeded at _reset_idx) rather than
            # recomputed from a stored trajectory.
            with torch.no_grad():
                next_M = self._metric_from_cmg(x)

                prev_err_t = self._prev_error.unsqueeze(-1)
                next_err_t = error.unsqueeze(-1)
                V = torch.bmm(torch.bmm(prev_err_t.transpose(1, 2), self._prev_M), prev_err_t).squeeze(-1).squeeze(-1)
                next_V = torch.bmm(torch.bmm(next_err_t.transpose(1, 2), next_M), next_err_t).squeeze(-1).squeeze(-1)

                control_effort = torch.norm(self._actions, p=2, dim=-1) ** 2
                reward = self.tracking_scaler * (V - next_V) - self.control_scaler * control_effort

                self._prev_error = error.clone()
                self._prev_M = next_M
        else:
            # -||error||_I^2, using the sanitized+clamped norm so a diverged step
            # can't emit a huge/NaN reward that wrecks the value fn / normalizers.
            reward = -(error_norm * error_norm)
        self._episode_reward += reward
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # --- log episode metrics for envs that actually ran ---
        finished = env_ids[self._episode_steps[env_ids] > 0]
        env0_finished = bool((finished == 0).any()) if len(finished) > 0 else False
        if len(finished) > 0:
            self.extras.setdefault("log", {})

            dt = self.step_dt

            # --- contraction flag (fraction of steps the error decreased) ---
            steps = self._episode_steps[finished].float()
            valid_steps = (steps - 1).clamp(min=1)
            contraction_flag = self._episode_contraction_steps[finished].float() / valid_steps
            self.extras["log"]["Stability/contraction_flag"] = contraction_flag.mean()

            # --- Unified streaming contraction metrics for every finished env ---
            # auc (normalized dt-weighted trapezoid), contraction_rate (lambda),
            # overshoot (e_max/e0) and contraction_score — each mean + 95% CI,
            # using the same streaming computation and wandb keys as C3M/C2RL
            # (contraction_metrics), so PPO/SAC/LQR/SD-LQR share the same
            # Stability/* tab, formula and CI as the contraction algorithms.
            # Computed over all finished envs (not just the few viz envs the old
            # fit_exponential_envelope path used), for tighter statistics.
            from c2rl.agents.skrl.contraction_metrics import (
                per_env_metrics,
                stability_log_dict,
                summarize,
            )
            per_env = per_env_metrics(
                e0=self._episode_e0[finished],
                e_last=self._prev_error_norm[finished],
                e_max=self._episode_emax[finished],
                err_sum=self._episode_auc[finished],
                steps=steps,
                dt=dt,
            )
            self.extras["log"].update(stability_log_dict(summarize(per_env), self.device))

            # --- Reward/total_reward_* — same keys/cadence as the Stability
            # tab above, and as C3M's eval loop (track_reward_summary), so
            # PPO/SAC/C2RL don't rely on a once-only post-training eval point.
            from c2rl.agents.skrl.contraction_metrics import reward_log_dict, reward_summary
            self.extras["log"].update(
                reward_log_dict(reward_summary(self._episode_reward[finished]), self.device)
            )

        # --- PathTracking figure: snapshot completed episodes for viz envs ---
        # (env_ids may include indices >= _viz_n, which we simply don't track)
        env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.asarray(env_ids)
        viz_ids = env_ids_np[env_ids_np < self._viz_n]
        if len(viz_ids) > 0:
            ep_lens = self._episode_steps[torch.as_tensor(viz_ids, device=self.device)].cpu().numpy()
            for local_i, ep_len in zip(viz_ids, ep_lens):
                if ep_len > 0:
                    self._viz_error_hist[local_i] = self._viz_error_live[local_i]
                    self._viz_pos_hist[local_i] = self._viz_pos_live[local_i]
                    self._viz_state_hist[local_i] = self._viz_state_live[local_i]
                    self._viz_len_hist[local_i] = ep_len
            self._viz_error_live[viz_ids] = 0.0
            self._viz_pos_live[viz_ids] = 0.0
            self._viz_state_live[viz_ids] = 0.0

        # --- wandb PathTracking figure (position + error^2 with bounds) ---
        # Cadence still keyed off env 0's reset — a simple periodic heartbeat,
        # not a synchronization requirement (viz envs snapshot independently
        # above, so the figure renders whatever's most recently completed for
        # each of them, which may span slightly different wall-clock episodes).
        if env0_finished:
            self._env0_episode_count += 1
            if (
                _wandb is not None and _plt is not None
                and self._env0_episode_count % _WANDB_PLOT_INTERVAL == 0
                and getattr(_wandb, "run", None) is not None
                and self._viz_len_hist.max() > 0
            ):
                self._log_pathtracking_figure()

        # --- reset per-env buffers ---
        self._episode_auc[env_ids] = 0.0
        self._episode_e0[env_ids] = 0.0
        self._episode_emax[env_ids] = 0.0
        self._episode_contraction_steps[env_ids] = 0
        self._prev_error_norm[env_ids] = float("inf")
        self._episode_steps[env_ids] = 0
        self._episode_reward[env_ids] = 0.0

        # sample reference trajectories — unless fix_ref_trajectories pinned a
        # permanent one per env slot on its first reset (see
        # set_fix_ref_trajectories); then only the not-yet-assigned slots draw.
        if getattr(self, "fix_ref_trajectories", False):
            fresh = ~self._ref_fixed[env_ids]
            if bool(fresh.any()):
                new_ids = env_ids[fresh]
                self._traj_ids[new_ids] = self._traj_buf.sample_traj_ids(len(new_ids))
                self._ref_fixed[new_ids] = True
        else:
            self._traj_ids[env_ids] = self._traj_buf.sample_traj_ids(len(env_ids))

        # initialise robot to match x_ref at step 0
        x_ref_init = self._traj_buf.initial_state(self._traj_ids[env_ids])  # (n, state_dim)
        self._set_robot_state_from_ref(env_ids, x_ref_init)

        # resync the divergence guard's baseline to the freshly-reset physical
        # state, so it doesn't carry forward a stale pre-reset value into the
        # new episode.
        if self._last_valid_state is not None:
            self._last_valid_state[env_ids] = self._get_physical_state()[env_ids]

        # seed the CCM Lyapunov cache (_get_rewards) from the freshly-reset
        # state, so the first step of the new episode compares against a
        # meaningful V rather than a stale value from the env's previous
        # episode — mirrors classic env_base.py's reset_idx M-seeding.
        if self.ccm_gen is not None:
            if not hasattr(self, "_prev_M"):
                state_dim = self._traj_buf.state_dim
                self._prev_error = torch.zeros(self.num_envs, state_dim, device=self.device)
                self._prev_M = torch.zeros(self.num_envs, state_dim, state_dim, device=self.device)
            with torch.no_grad():
                x0 = self._get_physical_state()[env_ids]
                error0 = wrap_diff(x0 - x_ref_init, self.angle_idx)
                self._prev_error[env_ids] = error0
                self._prev_M[env_ids] = self._metric_from_cmg(x0)
