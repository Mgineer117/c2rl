"""Batched Torch BaseEnv for classic analytical tracking environments."""

from __future__ import annotations

import math
from abc import abstractmethod

import gymnasium as gym
import numpy as np
import torch

from c2rl.agents.skrl.ref_window import RefWindow
from c2rl.tasks.direct.common.state_guard import carry_forward_nonfinite
from c2rl.tasks.direct.common.termination_box import TerminationBoxMixin


class BaseEnv(TerminationBoxMixin, gym.Env):
    def __init__(self, env_config: dict, num_envs: int = 1, device: str = "cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.X_MIN = torch.tensor(env_config["x_min"], device=self.device, dtype=torch.float32).flatten()
        self.X_MAX = torch.tensor(env_config["x_max"], device=self.device, dtype=torch.float32).flatten()
        self.XREF_INIT_MIN = torch.tensor(env_config["xref_init_min"], device=self.device, dtype=torch.float32).flatten()
        self.XREF_INIT_MAX = torch.tensor(env_config["xref_init_max"], device=self.device, dtype=torch.float32).flatten()
        self.XE_INIT_MIN = torch.tensor(env_config["xe_init_min"], device=self.device, dtype=torch.float32).flatten()
        self.XE_INIT_MAX = torch.tensor(env_config["xe_init_max"], device=self.device, dtype=torch.float32).flatten()
        # Dims on which xref_0 is drawn bimodally: the [min,max] box is sampled
        # as usual, then one global sign per env flips all of them together, so
        # the distribution is the box plus its exact mirror image.
        #
        # Needed because a low-lbd region is generally not a box. lbd is slow
        # where |pitch| is large -- two lobes straddling zero -- and any single
        # box covering both contains the fast center between them. Worse, the
        # lobes lie on a diagonal (segway is slow only when pitch and pitch_rate
        # have opposite signs), so flipping each dim independently would land
        # half the mass on the fast diagonal. One shared sign maps the box to
        # the correct opposite lobe and keeps the distribution symmetric, which
        # a one-sided box would not be.
        #
        # Empty (the default) reproduces the previous behavior exactly.
        self.XREF_INIT_SIGN_DIMS = list(env_config.get("xref_init_sign_dims", []) or [])
        # ── Direct initial-state box (optional) ──────────────────────────── #
        # When set, x_0 is drawn from [X_INIT_MIN, X_INIT_MAX] directly and the
        # perturbation is back-solved as xe_0 = x_0 - xref_0, instead of x_0
        # being composed as xref_0 + xe_0. Saying "start in region R" is then
        # one box, rather than an XE_INIT box that has to be eroded by
        # XREF_INIT and re-checked against X_MIN/X_MAX to land where you meant.
        #
        # xe_0 is still derived (not dropped) because e(0) = x_0 - xref_0
        # anchors every normalized metric -- AUC = sum(e)/e0 divides by it.
        #
        # X_INIT_SIGN_DIMS applies the same shared-sign mirroring as
        # XREF_INIT_SIGN_DIMS, for the same reason: a low-lbd region is two
        # lobes on a diagonal, which no single box can cover without also
        # covering the fast center between them.
        #
        # None (the default) keeps the xref_0 + xe_0 composition exactly.
        _xi_lo = env_config.get("x_init_min")
        _xi_hi = env_config.get("x_init_max")
        self.X_INIT_MIN = (None if _xi_lo is None else torch.tensor(
            _xi_lo, device=self.device, dtype=torch.float32).flatten())
        self.X_INIT_MAX = (None if _xi_hi is None else torch.tensor(
            _xi_hi, device=self.device, dtype=torch.float32).flatten())
        if (self.X_INIT_MIN is None) != (self.X_INIT_MAX is None):
            raise ValueError(
                "[BaseEnv] x_init_min and x_init_max must be set together — "
                "one without the other silently falls back to xref_0 + xe_0."
            )
        self.X_INIT_SIGN_DIMS = list(env_config.get("x_init_sign_dims", []) or [])
        # ── Early-termination box (on by default) ────────────────────────── #
        # The episode ends the first step x leaves [X_TERMINATION_MIN,
        # X_TERMINATION_MAX]. Without it a diverged env is silently pinned at the
        # state box by the clamp in step() and keeps emitting off-distribution
        # transitions for the rest of the episode — on segway, 500 steps of
        # already-fallen data per failure, which is what makes the rollout batch
        # (and hence the seed) decide what PPO fits.
        #
        # The box defaults to the state box itself in every env module, i.e. it
        # fires exactly where the clamp already silently activates — the same
        # event, reported instead of hidden. Tighten it per env to end episodes
        # sooner.
        #
        # On by default. Note this shortens episodes, so numbers are not directly
        # comparable with runs made before this default flipped; pass
        # --no_terminate_out_of_box to reproduce those.
        # angle_idx is resolved further down, but _left_termination_box only
        # reads it per step — set the default now so the mixin never sees a
        # missing attribute if that ordering ever changes.
        self.angle_idx = getattr(self, "angle_idx", [])
        self._init_termination_box(
            env_config.get("x_termination_min"),
            env_config.get("x_termination_max"),
            clamp_box=(self.X_MIN, self.X_MAX),   # step() clamps into this box
            armed=bool(env_config.get("terminate_out_of_box", True)),
            terminal=bool(env_config.get("x_termination_terminal", False)),
            tag="BaseEnv")
        self.XE_MIN = torch.tensor(env_config["xe_min"], device=self.device, dtype=torch.float32).flatten()
        self.XE_MAX = torch.tensor(env_config["xe_max"], device=self.device, dtype=torch.float32).flatten()
        self.UREF_MIN = torch.tensor(env_config["uref_min"], device=self.device, dtype=torch.float32).flatten()
        self.UREF_MAX = torch.tensor(env_config["uref_max"], device=self.device, dtype=torch.float32).flatten()
        # Physical actuator limits enforced in step(): 2x the uref box, leaving
        # headroom for the feedback term of uref+feedback controllers (C2RL/C3M
        # policies legitimately exceed the declared uref action space; clamping
        # to the uref box itself breaks the contraction certificate — see the
        # c2rl yaml's clip_actions note).
        self.U_MIN = 2.0 * self.UREF_MIN
        self.U_MAX = 2.0 * self.UREF_MAX

        self.num_dim_x = env_config["num_dim_x"]
        self.num_dim_control = env_config["num_dim_control"]
        # state_names is the single source of truth for the state layout: one
        # name per dim (see agents/skrl/state_symmetry.py for the vocabulary).
        # angle_idx (which dims wrap) and pos_dimension (which dims are pure
        # translation directions) are derived from it, so they can no longer
        # disagree with each other or with the physics. The explicit keys are
        # still honoured for any env that has not been renamed yet.
        self.state_names = tuple(env_config.get("state_names") or ())
        if self.state_names:
            if len(self.state_names) != self.num_dim_x:
                raise ValueError(
                    f"state_names has {len(self.state_names)} entries but "
                    f"num_dim_x={self.num_dim_x}: {self.state_names}")
            from c2rl.agents.skrl.state_symmetry import StateSymmetry
            sym = StateSymmetry.from_names(self.state_names)
            self.angle_idx = list(sym.angle_idx)
            self.pos_dimension = sym.pos_dimension
        else:
            self.pos_dimension = env_config["pos_dimension"]
            self.angle_idx = env_config.get("angle_idx", [])

        self.time_bound = env_config["time_bound"]
        self.dt = env_config["dt"]
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = self.max_episode_len
        self.t = torch.arange(0, self.time_bound, self.dt, device=self.device, dtype=torch.float32)

        self.tracking_scaler = env_config["q"]
        self.control_scaler = env_config["r"]
        self.use_learned_dynamics = False
        # State/control sampling distribution for get_rollout's "dynamics" mode.
        # _build_cfg() puts this in the config dict; it must be read out here or
        # get_rollout raises AttributeError on the first learned-dynamics draw.
        self.sample_mode = env_config.get("sample_mode", "uniform")

        ref_unit_min = torch.cat([self.X_MIN, self.UREF_MIN])
        ref_unit_max = torch.cat([self.X_MAX, self.UREF_MAX])
        # Kept for get_rollout's uniform state sampling, which still draws a
        # single [x, xref, uref] unit — not the observation layout any more.
        self.STATE_MIN = torch.cat([self.X_MIN, ref_unit_min])
        self.STATE_MAX = torch.cat([self.X_MAX, ref_unit_max])

        # ── Reference window: the observation is s = {x, xrefs, urefs} ───── #
        # xrefs[k] = xref[t + k*ref_offset], k = 0..ref_length-1 (k=0 is the
        # Current reference, i.e. the old `xref`/`uref`). Indices past the end
        # of the episode clamp to the last one. See agents/skrl/ref_window.py.
        self.ref_window = RefWindow(
            x_dim=self.num_dim_x,
            u_dim=self.num_dim_control,
            length=int(env_config.get("ref_length", 1) or 1),
            offset=int(env_config.get("ref_offset", 1) or 1),
        )
        self.observation_space = self.ref_window.space(
            self.X_MIN, self.X_MAX, self.UREF_MIN, self.UREF_MAX)
        self.action_space = gym.spaces.Box(
            low=self.UREF_MIN.cpu().numpy(),
            high=self.UREF_MAX.cpu().numpy(),
            dtype=np.float32,
        )

        # Buffers
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.x_t = torch.zeros(self.num_envs, self.num_dim_x, dtype=torch.float32, device=self.device)
        # Reference is generated PAST the episode end, so the window never has to
        # clamp — see _size_reference_buffers.
        self._size_reference_buffers()
        self.init_tracking_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # fix_ref_trajectories: mint one episode per env slot on its first reset
        # — reference trajectory and initial state — and replay it for every
        # later episode, so the policy trains on a fixed set of num_envs tasks
        # instead of a fresh draw each time. Off by default. See reset_idx.
        self.fix_ref_trajectories = bool(env_config.get("fix_ref_trajectories", False))
        self._ref_fixed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fixed_xref = torch.zeros_like(self.xref)
        self._fixed_uref = torch.zeros_like(self.uref)
        self._fixed_x0 = torch.zeros_like(self.x_t)

        # How many episodes one reference integration serves. See
        # _pooled_system_reset: system_reset's cost is a loop over the episode's
        # max_episode_len timesteps that is vectorized over the batch, so drawing
        # ref_pool_size references costs about what drawing one costs. 1 disables
        # pooling and restores a fresh integration per reset.
        self._ref_pool_size = int(env_config.get("ref_pool_size", 64) or 1)
        self._ref_pool = None

        # The privileged critic-only `states` channel is gone: the critic now
        # reads the same {x, xrefs, urefs} observation as the actor and gets its
        # independence from its own architecture (phi/psi/combine — see
        # models.RefWindowValueModel) rather than from a second env channel.
        # skrl still threads `states` through, so leave it explicitly unset.
        self.state_space = None

        self.reset()

        # Markov check (see RefWindow.check_markov): the reward is Markov by
        # construction, but the value is only Markov if the window spans the
        # discount's effective horizon. Warn as early as possible — this is the
        # exact POMDP the old reference preview existed to fix.
        _g = env_config.get("discount_factor")
        if _g is not None:
            _warn = self.ref_window.check_markov(float(_g), self.max_episode_len)
            if _warn:
                print(_warn)

    @staticmethod
    def _build_cfg(env_config: dict, *, sample_mode: str = "uniform", time_bound: float | None = None,
                   dt: float | None = None, **extra) -> dict:
        cfg = dict(env_config)
        cfg["sample_mode"] = sample_mode
        if time_bound is not None: cfg["time_bound"] = time_bound
        if dt is not None: cfg["dt"] = dt
        # Constructor passthrough for the per-env knobs read in __init__
        # (ref_length / ref_offset / discount_factor). None means "not given" —
        # never overwrite an ENV_CONFIG default with it.
        cfg.update({k: v for k, v in extra.items() if v is not None})
        return cfg

    # Same names PathTrackingBase exposes, so ContractionRunner reads the state/
    # control dimensions identically for both env families (it looks for
    # x_dim/u_dim; without these, a classic env building a NeuralDynamics —
    # use_empirical_dynamics=true — got None for both).
    def set_fix_ref_trajectories(self, flag: bool) -> None:
        """Freeze (or unfreeze) the per-env episodes: reference and initial state.

        Each env slot replays one fixed (xref, uref, x_0) for the whole run, so
        the policy sees exactly ``num_envs`` distinct tasks. The Isaac twin is
        ``PathTrackingBase.set_fix_ref_trajectories`` (there the initial state
        is derived from the reference's first point, so pinning the trajectory
        id already pins the start). Turning it on drops any previously stored
        episodes, so the next reset re-mints them — otherwise a mid-run toggle
        would silently resurrect episodes sampled under different settings.
        """
        self.fix_ref_trajectories = bool(flag)
        self._ref_fixed[:] = False
        print(f"[BaseEnv] fix_ref_trajectories={self.fix_ref_trajectories} "
              f"({self.num_envs} fixed reference trajectories)"
              if self.fix_ref_trajectories else
              "[BaseEnv] fix_ref_trajectories=False (resampled every episode)")

    def get_reference_trajectory(self) -> torch.Tensor:
        """whole reference path per env: ``(num_envs, max_episode_len, x_dim)``.

        The Isaac twin is ``PathTrackingBase.get_reference_trajectory``; both
        exist so logging code plots the complete reference against the policy
        rollout instead of rebuilding a reference from the (offset-subsampled,
        horizon-truncated) observation window. See tests/test_isaac_parity.py.
        """
        return self.xref

    @property
    def x_dim(self) -> int:
        return self.num_dim_x

    @property
    def u_dim(self) -> int:
        return self.num_dim_control

    def get_horizon_matched_gamma(self, scale: float = 1.0):
        scale = max(1e-3, min(scale, 1.0))
        return round(1 - (1 / (scale * self.max_episode_len)), 3)

    def _apply_shared_sign(self, v: torch.Tensor, dims: list[int]) -> torch.Tensor:
        """Mirror ``v`` on ``dims`` with one sign per row (not one per dim).

        A per-dim sign would scatter the samples across all sign combinations;
        the slow region lies on a single diagonal, so only the shared sign maps
        the box onto the correct opposite lobe. See XREF_INIT_SIGN_DIMS.
        """
        if not dims:
            return v
        s = torch.where(torch.rand(v.shape[0], 1, device=self.device) < 0.5,
                        -1.0, 1.0).to(torch.float32)
        idx = torch.as_tensor(dims, device=self.device, dtype=torch.long)
        v[:, idx] = v[:, idx] * s
        return v

    def define_initial_state(self, env_ids: torch.Tensor):
        n = len(env_ids)
        rand_xref = torch.rand(n, self.num_dim_x, device=self.device, dtype=torch.float32)
        xref_0 = self.XREF_INIT_MIN + rand_xref * (self.XREF_INIT_MAX - self.XREF_INIT_MIN)
        xref_0 = self._apply_shared_sign(xref_0, self.XREF_INIT_SIGN_DIMS)

        if self.X_INIT_MIN is not None:
            # Direct initial-state box: x_0 is the primary object and xe_0 is
            # back-solved, so e(0) stays exactly x_0 - xref_0. See X_INIT_MIN.
            rand_x = torch.rand(n, self.num_dim_x, device=self.device, dtype=torch.float32)
            x_0 = self.X_INIT_MIN + rand_x * (self.X_INIT_MAX - self.X_INIT_MIN)
            x_0 = self._apply_shared_sign(x_0, self.X_INIT_SIGN_DIMS)
            return xref_0, x_0 - xref_0, x_0

        rand_xe = torch.rand(n, self.num_dim_x, device=self.device, dtype=torch.float32)
        xe_0 = self.XE_INIT_MIN + rand_xe * (self.XE_INIT_MAX - self.XE_INIT_MIN)

        return xref_0, xe_0, xref_0 + xe_0

    @abstractmethod
    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        ...

    def _rollout_reference(self, xref_0: torch.Tensor, freqs, weights) -> tuple[torch.Tensor, torch.Tensor, int]:
        xref_list = [xref_0]
        xref_wrapped_list = [xref_0]
        uref_list = []

        for i, _t in enumerate(self.t):
            # "xref_t" is the current reference state, not just the initial one.
            # A gravity-loaded plant (ball_and_beam, two_link_arm, pvtol) needs a
            # state-dependent trim in uref -- hold the ball, hold the arm up,
            # hover -- or the reference free-falls into the state box, gets
            # clamped, and the stored (xref, uref) stops being a trajectory of
            # the plant. Every u = uref + feedback controller then chases an
            # unreachable reference: two_link_arm saturated 75-100% of the time
            # and settled at a fixed error no matter how small e(0) was.
            uref_t = self.sample_reference_controls(
                freqs, weights, _t, {"xref_0": xref_0, "xref_t": xref_list[-1]})
            xref_prev = xref_list[-1]
            f_x, B_x, _ = self.get_f_and_B(xref_prev, need_null=False)
            x_dot = f_x + torch.bmm(B_x, uref_t.unsqueeze(-1)).squeeze(-1)
            next_x = xref_prev + self.dt * x_dot

            next_x_wrapped = self.wrap_angles(next_x)
            next_x_wrapped = torch.clamp(next_x_wrapped, self.X_MIN, self.X_MAX)

            xref_list.append(next_x_wrapped)
            xref_wrapped_list.append(next_x_wrapped)
            uref_list.append(uref_t)

        return torch.stack(xref_wrapped_list[:-1], dim=1), torch.stack(uref_list, dim=1), i + 1

    @abstractmethod
    def system_reset(self, env_ids: torch.Tensor):
        ...

    def _metric_from_cmg(self, x):
        """M(x) from the CMG, inverting only when the head emits W.

        cmg_method="cvstem" builds the CMG with outputs_metric=True, so its
        forward already returns M and this is a pass-through — which removes a
        batched SPD inverse from every env step. cmg_method="ccm" emits W (its
        C1/C2 losses are written in W), so M = W^-1 is formed here.
        """
        from c2rl.agents.skrl.math_utils import bound_W, spd_inverse
        raw, _ = self.ccm_gen(x)
        out = bound_W(raw, self.w_lb, self.num_dim_x,
                      getattr(self.ccm_gen, "bounded", False))
        if getattr(self.ccm_gen, "outputs_metric", False):
            return out
        return spd_inverse(out)

    def set_ccm(self, ccm_gen, w_lb, device, tracking_scaler=None, control_scaler=None,
                reward_euclidean=False, reward_level=False,
                residual_anchor_scale=0.0, cvstem_r_scaler=1.0):
        """Inject the frozen CMG (and, optionally, the reward weights) for C2RL.

        ``tracking_scaler``/``control_scaler`` default to None = keep whatever
        the env config's ``q``/``r`` set. C2RL passes its own cfg values so the
        ``cm.tracking_scaler``/``cm.control_scaler`` keys actually reach the
        reward — they were declared on the agent cfg but read by nobody, which
        made every sweep over them a no-op.
        """
        self.ccm_gen = ccm_gen
        self.w_lb = w_lb
        self.ccm_device = device
        # When True, get_rewards uses the raw Euclidean error decrement (M = I)
        # instead of the Mahalanobis one — the AUC-aligned reward for residual RL
        # over the CV-STEM-LQR baseline (the CMG is still kept, for that baseline).
        self.reward_euclidean = bool(reward_euclidean)
        # Level vs decrement euclidean reward (only when reward_euclidean). Level
        # (r = -‖e‖) is the tightest AUC alignment; see get_rewards.
        self.reward_level = bool(reward_level)
        # Residual trust anchor: penalize ‖u - u_base‖² (u_base = the CV-STEM-LQR
        # action from this same frozen CMG), so PPO deviates from the certified
        # analytic base only when it strictly helps tracking — the base already
        # beats CV-STEM-LQR, and the unanchored residual degrades it. See get_rewards.
        self.residual_anchor_scale = float(residual_anchor_scale)
        self.cvstem_r = float(cvstem_r_scaler)
        if tracking_scaler is not None:
            self.tracking_scaler = float(tracking_scaler)
        if control_scaler is not None:
            self.control_scaler = float(control_scaler)

    def set_eig_reshape(self, target_cond: float | None) -> None:
        """Reward-side ablation: reshape M's eigenvalue spread to ``target_cond``
        while keeping its eigenvectors and geometric-mean scale fixed.

        Isolates "is it conditioning or is it what the fit converged to" —
        applying this to a wide-envelope fit's M answers whether clamping cond
        alone (without refitting) recovers a tight-envelope-like reward, and
        applying it to a tight-envelope fit's M answers the converse (does
        widening only the spread, same fit otherwise, reproduce the wide
        envelope's degradation). See visualization/bound_sweep.py's docstring
        for why cond(M) — not the fit itself — was the open question.
        """
        self.eig_reshape_cond = target_cond

    def _apply_eig_reshape(self, M: torch.Tensor) -> torch.Tensor:
        target = getattr(self, "eig_reshape_cond", None)
        if target is None:
            return M
        eigvals, eigvecs = torch.linalg.eigh(M)  # ascending
        log_gm = eigvals.clamp_min(1e-12).log().mean(dim=-1, keepdim=True)
        n = eigvals.shape[-1]
        t = torch.linspace(0.0, 1.0, n, device=M.device, dtype=M.dtype)
        log_span = t * float(np.log(target))
        new_log = log_gm - 0.5 * float(np.log(target)) + log_span
        new_eigvals = new_log.exp()
        return eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2)

    def get_rewards(self, x, u, next_x, env_ids):
        t_idx = self.time_steps[env_ids]
        xref_prev = self.xref[env_ids, torch.clamp(t_idx - 1, min=0)]
        xref_curr = self.xref[env_ids, torch.clamp(t_idx, max=self.max_episode_len - 1)]

        error = self.wrap_angles(x - xref_prev)
        next_error = self.wrap_angles(next_x - xref_curr)

        tracking_error = torch.norm(next_error, p=2, dim=-1) ** 2
        control_effort = torch.norm(u, p=2, dim=-1) ** 2

        if getattr(self, "reward_euclidean", False):
            # AUC-aligned reward: the learned policy minimizes the true tracking
            # error the eval AUC measures, not the frozen-CMG Mahalanobis proxy.
            if getattr(self, "reward_level", False):
                # Level form: r = -‖e‖. AUC = ∫‖e‖/‖e0‖ dt, so the discounted sum
                # of -‖e‖ is (minus) the error integral — the tightest possible
                # alignment. The decrement form below telescopes to the endpoint
                # error e0²-eT², which a dawdle-then-settle policy games while
                # keeping AUC high (measured plateau at 0.96). Linear norm (not
                # squared) matches AUC's ‖e‖ weighting exactly.
                err_norm = torch.norm(next_error, p=2, dim=-1)
                reward = -self.tracking_scaler * err_norm \
                    - self.control_scaler * control_effort
            else:
                # Decrement form: raw Euclidean error decrement ‖e_prev‖²-‖e_next‖²
                # (M = I) — same telescoping shape as the Mahalanobis reward.
                prev_sq = torch.norm(error, p=2, dim=-1) ** 2
                reward = self.tracking_scaler * (prev_sq - tracking_error) \
                    - self.control_scaler * control_effort
            maha_tracking_error = None
        elif getattr(self, "ccm_gen", None) is not None:
            with torch.no_grad():
                if not hasattr(self, "M"):
                    self.M = torch.zeros(self.num_envs, self.num_dim_x, self.num_dim_x, device=self.device)
                M = self.M[env_ids]
                next_M = self._metric_from_cmg(next_x)
                self.M[env_ids] = next_M

                M = self._apply_eig_reshape(M)
                next_M = self._apply_eig_reshape(next_M)

                err_t = error.unsqueeze(-1)
                next_err_t = next_error.unsqueeze(-1)

                V = torch.bmm(torch.bmm(err_t.transpose(1, 2), M), err_t).squeeze(-1).squeeze(-1)
                next_V = torch.bmm(torch.bmm(next_err_t.transpose(1, 2), next_M), next_err_t).squeeze(-1).squeeze(-1)

                if getattr(self, "reward_level", False):
                    # Level form: R(t) = -‖e(t+1)‖²_M. Same rationale as the
                    # euclidean reward_level branch above — the decrement form
                    # V - next_V telescopes to the endpoint error V_0 - V_T
                    # (policy-independent up to that constant only at γ=1;
                    # at γ<1 it still vanishes almost everywhere near the
                    # optimum, starving PPO's advantage of signal exactly
                    # where this policy lives — see project memory on the
                    # single-update collapse). The level form's per-step
                    # signal never vanishes.
                    reward = -self.tracking_scaler * next_V - self.control_scaler * control_effort
                else:
                    reward = self.tracking_scaler * (V - next_V) - self.control_scaler * control_effort
                # Mahalanobis tracking error V = eᵀM(x)e (squared, like
                # "tracking_error" above) — the metric-weighted analog of the
                # plain Euclidean error, so StatManagerEnvWrapper can plot a
                # "normalized_maha_error" curve alongside "normalized_error".
                maha_tracking_error = next_V
        else:
            reward = -self.tracking_scaler * tracking_error - self.control_scaler * control_effort
            maha_tracking_error = None

        # Residual trust anchor (residual RL over CV-STEM-LQR): penalize the
        # applied action's deviation from the analytic base action u_base =
        # uref - (1/r)Bᵀ W⁻¹ (x - xref) built from this env's frozen CMG. The base
        # already beats CV-STEM-LQR; the unanchored residual degrades it (PPO games
        # the decrement reward), so this keeps the policy at the base unless a
        # deviation strictly helps tracking. See set_ccm.
        if getattr(self, "residual_anchor_scale", 0.0) > 0 and getattr(self, "ccm_gen", None) is not None:
            with torch.no_grad():
                r = self.cvstem_r + 1e-5
                uref_curr = self.uref[env_ids, torch.clamp(t_idx, max=self.max_episode_len - 1)]
                _f, B, _ = self.get_f_and_B(x)
                Mb = self._metric_from_cmg(x)
                Kb = (1.0 / r) * torch.bmm(B.transpose(1, 2).to(torch.float32), Mb)
                e_b = self.wrap_angles(x - xref_curr).unsqueeze(-1)
                u_base = uref_curr - torch.bmm(Kb, e_b).squeeze(-1)
                # ``u`` reaching get_rewards has already been clamped to
                # [U_MIN, U_MAX] by step(), so u_base must be too or the two
                # sides of this penalty live in different spaces. It is not a
                # small discrepancy: at cvstem_r_scaler=0.01 the analytic gain
                # is ‖K‖₂ ≈ 53, putting ~86% of u_base's components outside the
                # box (measured on the cached car CM dataset). Unclamped, the
                # anchor charges a large, irreducible penalty no matter what the
                # policy does, and the only gradient it supplies pushes the
                # action hard against the saturation boundary — the opposite of
                # the intended "stay near the certified base unless deviating
                # strictly helps" trust region. Clamping makes both sides the
                # control the plant actually applies.
                u_base = torch.clamp(u_base, self.U_MIN, self.U_MAX)
            reward = reward - self.residual_anchor_scale * torch.norm(u - u_base, p=2, dim=-1) ** 2

        infos = {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }
        if maha_tracking_error is not None:
            infos["maha_tracking_error"] = maha_tracking_error
        return reward, infos

    def reset(self, seed=None, options=None):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        info = {"x": self.x_t.clone(), "tracking_error": self.init_tracking_error.clone()}
        # Anchor the maha error curve's e(0) at reset, same as tracking_error
        # above — without this the StatManager records the step-0 slot with no
        # maha value and its e0_maha stays 0 (later ÷0). Only present for C2RL.
        if getattr(self, "ccm_gen", None) is not None and hasattr(self, "init_maha_error"):
            info["maha_tracking_error"] = self.init_maha_error.clone()
        return self.construct_state(self.x_t), info

    def _pooled_system_reset(self, env_ids: torch.Tensor):
        """``system_reset`` draws that share one reference integration.

        ``system_reset``'s cost is dominated by ``_rollout_reference``, a loop
        over the episode's ``max_episode_len`` timesteps that is vectorized over
        the batch — so drawing 64 references costs about what drawing 1 costs.
        Measured on car: one reset is 100.8 ms against 0.204 ms for a step, i.e.
        493 steps' worth. That is affordable once per 500-step episode, but
        ``terminate_out_of_box`` ends episodes long before ``max_episode_len``,
        so the whole integration was being paid every few steps — training ran
        53x slower with the box armed (0.27 it/s vs 14.4 it/s on car).

        The draw is unchanged, only batched: ``define_initial_state`` and the
        reference-weight draw read ``len(env_ids)`` and nothing else, so pooled
        samples are the same iid samples. No classic env's ``system_reset``
        indexes by ``env_ids``.

        Two paths do depend on draw position and fall through to a direct call:
        ``_held_out_weights`` indexes its bank by ``arange(n) % bank_size``, and
        ``fix_ref_trajectories`` pins one episode per slot on its first reset.
        """
        n = len(env_ids)
        if (self._ref_pool_size <= 1
                or self.fix_ref_trajectories
                or getattr(self, "_held_out_weights", None) is not None):
            return self.system_reset(env_ids)

        pool = self._ref_pool
        if pool is None or pool["cursor"] + n > pool["x0"].shape[0]:
            batch = max(self._ref_pool_size, n)
            # Only the length is read; see the docstring.
            x0, xref, uref, length = self.system_reset(env_ids.new_zeros(batch))
            pool = {"x0": x0, "xref": xref, "uref": uref,
                    "length": length, "cursor": 0}
            self._ref_pool = pool

        c = pool["cursor"]
        pool["cursor"] = c + n
        return (pool["x0"][c:c + n], pool["xref"][c:c + n],
                pool["uref"][c:c + n], pool["length"])

    def reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        self.time_steps[env_ids] = 0
        self.episode_reward[env_ids] = 0.0

        x_0, xref_arr, uref_arr, _ = self._pooled_system_reset(env_ids)
        xref_arr = torch.clamp(xref_arr, self.X_MIN, self.X_MAX)
        # x_0 = xref_0 + xe_0 must respect the box too. The reference is clamped on
        # the line above; the initial STATE was not, and xe_0 is drawn from XE_INIT
        # independently of where xref sits, so the sum routinely left the box.
        # Measured on car: xref's velocity is pinned at 1.5 while xe_0 draws
        # U[-1,1], giving x_0 velocities across [0.5, 2.5] against a [1.0, 2.0]
        # box — 44% of episodes began outside it.
        #
        # Not a cosmetic bound. The CMG is regressed over the box only, so outside
        # it M(x) is an extrapolation and the Mahalanobis reward is computed from a
        # metric that certifies nothing there. Every Stability/* metric is also
        # normalized by the error at x_0, so out-of-box starts move the very
        # denominator the runs are compared on.
        x_0 = torch.clamp(x_0, self.X_MIN, self.X_MAX)
        if self.fix_ref_trajectories:
            # First reset of a slot mints its permanent episode — reference and
            # initial state; later resets discard the freshly sampled ones and
            # restore the stored pair. Pinning the reference alone would leave
            # x_0 = xref[0] + xe_0 redrawing xe_0 every episode, so the task
            # would still vary through its initial condition.
            fresh = ~self._ref_fixed[env_ids]
            if bool(fresh.any()):
                new_ids = env_ids[fresh]
                self._fixed_xref[new_ids] = xref_arr[fresh]
                self._fixed_uref[new_ids] = uref_arr[fresh]
                self._fixed_x0[new_ids] = x_0[fresh]
                self._ref_fixed[new_ids] = True
            xref_arr = self._fixed_xref[env_ids]
            uref_arr = self._fixed_uref[env_ids]
            x_0 = self._fixed_x0[env_ids]
        self.xref[env_ids] = xref_arr
        self.uref[env_ids] = uref_arr
        self.x_t[env_ids] = x_0

        self.init_tracking_error[env_ids] = torch.norm(x_0 - self.xref[env_ids, 0], p=2, dim=-1) ** 2

        if getattr(self, "ccm_gen", None) is not None:
            if not hasattr(self, "M"):
                self.M = torch.zeros(self.num_envs, self.num_dim_x, self.num_dim_x, device=self.device)
            if not hasattr(self, "init_maha_error"):
                self.init_maha_error = torch.zeros(self.num_envs, device=self.device)
            with torch.no_grad():
                M0 = self._metric_from_cmg(x_0)
                self.M[env_ids] = M0
                # Squared Mahalanobis error e0ᵀM(x0)e0 for e0/e(0) normalization
                # of the maha error curve — mirrors init_tracking_error above,
                # but angle-wrapped since M-weighting an unwrapped angle error
                # would be meaningless.
                e0 = self.wrap_angles(x_0 - self.xref[env_ids, 0]).unsqueeze(-1)
                self.init_maha_error[env_ids] = torch.bmm(
                    torch.bmm(e0.transpose(1, 2), M0), e0).squeeze(-1).squeeze(-1)

    def step(self, u: torch.Tensor):
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
        u = torch.nan_to_num(u)
        u = torch.clamp(u, self.U_MIN, self.U_MAX)
        self.time_steps += 1

        f_x, B_x, _ = self.get_f_and_B(self.x_t, need_null=False)
        x_dot = f_x + torch.bmm(B_x, u.unsqueeze(-1)).squeeze(-1)
        next_x = self.x_t + self.dt * x_dot

        next_x = carry_forward_nonfinite(next_x, self.x_t)

        # Measured on the raw integrated state, before the position freeze and
        # the X-box clamp below — both erase the excursion, so a check placed
        # after them could never fire. None when the feature is off.
        left_box = self._left_termination_box(next_x)

        pos_min = self.X_MIN[:self.pos_dimension]
        pos_max = self.X_MAX[:self.pos_dimension]
        out_of_bounds = (next_x[:, :self.pos_dimension] < pos_min) | (next_x[:, :self.pos_dimension] > pos_max)
        invalid_mask = out_of_bounds.any(dim=-1)
        next_x[invalid_mask, :self.pos_dimension] = self.x_t[invalid_mask, :self.pos_dimension]

        next_x_wrapped = self.wrap_angles(next_x)
        next_x_wrapped = torch.clamp(next_x_wrapped, self.X_MIN, self.X_MAX)

        reward, infos = self.get_rewards(self.x_t, u, next_x_wrapped, torch.arange(self.num_envs, device=self.device))
        self.episode_reward += reward

        self.x_t = next_x_wrapped
        state = self.construct_state(self.x_t)

        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncation = self.time_steps >= self.episode_len
        if left_box is not None:
            # Truncation, not termination, unless explicitly asked otherwise —
            # see x_termination_terminal in __init__ for why (zeroed bootstrap
            # = suicide bonus on a cost reward).
            if self.x_termination_terminal:
                termination = termination | left_box
            else:
                truncation = truncation | left_box
        dones = termination | truncation

        info_dict = {
            "x": self.x_t.clone(),
            "tracking_error": infos["tracking_error"],
            "control_effort": infos["control_effort"],
            "relative_tracking_error": infos["tracking_error"] / torch.clamp(self.init_tracking_error, min=1e-8),
        }
        # Forward the Mahalanobis error (C2RL only) so StatManagerEnvWrapper's
        # _record can fill the maha buffer every step — not just at reset. Without
        # this the maha curve/Stability_maha metrics stay pinned at their step-0
        # value (flat) because get_rewards' infos are rebuilt into this dict here.
        if "maha_tracking_error" in infos:
            info_dict["maha_tracking_error"] = infos["maha_tracking_error"]

        # Which envs are ending short of the horizon. StatManagerEnvWrapper
        # invalidates those slots: AUC/lambda/C are defined on the full-length
        # normalized error curve, and a curve cut at step k is not the same
        # quantity — padding it (which is what the wrapper does for a short slot)
        # would report a fabricated flat tail as if the policy had held there.
        if left_box is not None:
            info_dict["episode_ended_early"] = left_box.clone()

        if dones.any():
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)

            # Stability/* is computed centrally by StatManagerEnvWrapper (the
            # paper-style batched C/lambda metrics) — the env only reports the
            # episodic reward. Must be a scalar torch tensor: skrl's trainer
            # (`environment_info: log`) silently drops plain Python floats.
            info_dict["log"] = {
                "Reward/total_reward_mean": self.episode_reward[done_idx].mean().detach().clone(),
            }

            # Per-key slice: the observation is a Dict now, so this cannot index
            # the whole thing at once.
            info_dict["final_observation"] = {k: v[done_idx].clone() for k, v in state.items()}
            info_dict["_final_observation"] = dones.clone()

            self.reset_idx(done_idx)
            # Reconstruct state after reset for done envs (IsaacLab standard)
            state = self.construct_state(self.x_t)

        return state, reward, termination, truncation, info_dict

    def get_f_and_B(self, x: torch.Tensor, *, need_null: bool = True):
        """``(f, B, B_null)``. ``need_null=False`` returns ``None`` for B_null and
        skips computing it — the annihilator is only used by the contraction
        SDPs, while the reference rollout integrates 500 steps per episode reset
        and discarded it every time (20% of that rollout's cost, and with early
        termination the rollout runs on every early reset, not once per 500
        steps). Same signature on PathTrackingBase — see tests' parity rule."""
        if getattr(self, "use_learned_dynamics", False):
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(self.wrap_angles(x))
            return f_x, B_x, Bbot_x

        return (self._f_logic(x), self._B_logic(x),
                self._B_null_logic(x) if need_null else None)

    def get_rollout(self, buffer_size: int, mode: str, num_control_per_state: int | None = None):
        if mode == "c3m":
            xref = (self.X_MAX - self.X_MIN) * torch.rand(buffer_size, self.num_dim_x, device=self.device) + self.X_MIN
            uref = (self.UREF_MAX - self.UREF_MIN) * torch.rand(buffer_size, self.num_dim_control, device=self.device) + self.UREF_MIN
            xe = (self.XE_MAX - self.XE_MIN) * torch.rand(buffer_size, self.num_dim_x, device=self.device) + self.XE_MIN
            x = torch.clamp(xe + xref, self.X_MIN, self.X_MAX)
            return {
                "x": x,
                "xref": xref,
                "uref": uref,
            }

        n_control_per_x = num_control_per_state if num_control_per_state is not None else 3
        batch_size = math.ceil(buffer_size / n_control_per_x)
        if self.sample_mode == "Gaussian":
            x_mean = (self.X_MAX + self.X_MIN) / 2.0
            x_std = (self.X_MAX - self.X_MIN) / 6.0
            u_mean = (self.UREF_MAX + self.UREF_MIN) / 2.0
            u_std = (self.UREF_MAX - self.UREF_MIN) / 6.0
            x = torch.normal(x_mean.expand(batch_size, -1), x_std.expand(batch_size, -1))
            u = torch.normal(u_mean.expand(batch_size, -1), u_std.expand(batch_size, -1))
        else:
            x = self.X_MIN + torch.rand(batch_size, self.num_dim_x, device=self.device) * (self.X_MAX - self.X_MIN)
            u = self.UREF_MIN + torch.rand(batch_size, self.num_dim_control, device=self.device) * (self.UREF_MAX - self.UREF_MIN)

        x = x.repeat(n_control_per_x, 1)
        u_list = [u[torch.randperm(batch_size)] for _ in range(n_control_per_x)]
        u = torch.cat(u_list, dim=0)

        f, B, _ = self.get_f_and_B(x)
        x_dot = f + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        # "x_dot", not "x_next": every consumer of a "dynamics" rollout
        # (dynamics_pretrain.pretrain_dynamics, C3M/C2RL._train_dynamics) fits
        # NeuralDynamics against ẋ directly, matching PathTrackingBase's
        # _get_dynamics_rollout on the Isaac side.
        return {
            "x": x,
            "u": u,
            "x_dot": x_dot,
        }

    def wrap_angles(self, x: torch.Tensor):
        """Wrap the ``angle_idx`` columns into (-pi, pi]; others pass through.

        Same formula as before, but one vectorized ``torch.where`` against a
        cached mask instead of a full clone plus a Python loop with an in-place
        write per angle dim. This is on the per-step path (step, reward,
        reference rollout — ~190k calls in a 700-step 32-env rollout), and the
        no-angle case now returns ``x`` untouched instead of cloning it.
        """
        if not self.angle_idx:
            return x
        mask = getattr(self, "_angle_mask", None)
        if mask is None or mask.shape[-1] != x.shape[-1]:
            mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
            mask[list(self.angle_idx)] = True
            self._angle_mask = mask
        return torch.where(mask, torch.remainder(x + math.pi, 2 * math.pi) - math.pi, x)

    def configure_ref_window(self, length: int, offset: int = 1,
                             gamma: float | None = None) -> None:
        """Resize the reference window and rebuild ``observation_space``.

        Must be called before the agent/memory are built off
        ``observation_space``. Used by the training entry point (``--ref_length``
        / ``--ref_offset``) and to make a fresh eval env mirror the training
        env's exact layout. Idempotent."""
        self.ref_window = RefWindow(x_dim=self.num_dim_x, u_dim=self.num_dim_control,
                                    length=int(length), offset=int(offset))
        self.observation_space = self.ref_window.space(
            self.X_MIN, self.X_MAX, self.UREF_MIN, self.UREF_MAX)
        # The window length just changed, so the reference has to be regenerated at
        # the new span -- otherwise the tail of every window would clamp onto
        # xref[-1] again, which is the padding this sizing exists to remove.
        # Safe here because this method is documented as pre-agent-construction.
        self._size_reference_buffers()
        self._fixed_xref = torch.zeros_like(self.xref)
        self._fixed_uref = torch.zeros_like(self.uref)
        if hasattr(self, "_ref_fixed"):
            self._ref_fixed[:] = False
        self._ref_pool = None
        print(f"[BaseEnv] reference generated to {self.ref_gen_len} steps for a "
              f"{self.max_episode_len}-step episode "
              f"({self.ref_gen_len - self.max_episode_len} beyond the end, so the "
              f"window never clamps)")
        print(f"[BaseEnv] reference window: length={self.ref_window.length} "
              f"offset={self.ref_window.offset} (spans "
              f"{(self.ref_window.length - 1) * self.ref_window.offset} steps) -> "
              f"obs {self.ref_window.flat_dim}-wide")
        if gamma is not None:
            warn = self.ref_window.check_markov(float(gamma), self.max_episode_len)
            if warn:
                print(warn)

    # ── Superseded by the {x, xrefs, urefs} reference window above ───────── #
    # The old flat layout was [x, xref, uref] plus an optional "preview" tail of
    # future rows, whose width/content was inferred by every consumer rather than
    # declared. Kept here (commented) for reference while the new window beds in.
    #
    # def _preview_offsets(self, num_points: int, gamma: float) -> list[int]:
    #     """Geometrically-spaced future-step offsets spanning the effective horizon.
#
    #     The window extent is the discount's effective horizon ``H = 1/(1-gamma)``
    #     (in steps) — how far the value function actually looks — clamped to the
    #     episode. Within ``[1, H]`` it places ``num_points`` unique offsets on a
    #     Geometric ladder: near-term references (where the error is being crushed
    #     now) get dense coverage, the far horizon a few anchors, matching how the
    #     discount weights them. At a low gamma ``H`` collapses to ~1, so preview
    #     is ~just the next uref — exactly when look-ahead is not needed anyway.
    #     """
    #     if not num_points or gamma is None or not (0.0 < gamma < 1.0):
    #         return []
    #     H = min(self.max_episode_len - 1, max(1, int(round(1.0 / (1.0 - gamma)))))
    #     if num_points >= H:
    #         return list(range(1, H + 1))
    #     geom = torch.logspace(0.0, math.log10(float(H)), num_points)
    #     offs = sorted({int(round(v)) for v in geom.tolist() if 1 <= round(v) <= H})
    #     return offs
#
    # def set_preview_offsets(self, offsets: Sequence[int], include_xref: bool = False,
    #                         include_uref: bool = True) -> None:
    #     """Set the preview window to an explicit list of step offsets and widen
    #     the observation space to match. Used to make a fresh eval env mirror the
    #     training env's exact layout (copy its ``preview_offsets``).
#
    #     ``include_xref=False`` (default) reproduces the historic uref-only
    #     preview exactly. ``include_xref=True`` additionally appends, for each
    #     offset, the future reference position relative to the current state
    #     (``wrap_angles(xref_future - x)`` — see ``construct_state``): a
    #     translation-invariant "where is the path heading from here" signal,
    #     one ``x_dim``-wide block per offset, ordered nearest-offset-first like
    #     the uref block it follows. Not full SE(2)-invariant (no rotation into
    #     the current heading frame) — a deliberate simplification.
#
    #     ``include_uref=False`` drops the future-uref block from the preview
    #     tail entirely (the tail becomes xref-only) — needs ``include_xref=True``
    #     or the preview would carry nothing. Default True reproduces the
    #     historic layout exactly.
    #     """
    #     self.preview_offsets = [int(o) for o in offsets]
    #     self._preview_include_xref = bool(include_xref) and bool(self.preview_offsets)
    #     self._preview_include_uref = bool(include_uref) and bool(self.preview_offsets)
    #     if not self.preview_offsets:
    #         self._preview_offsets_t = None
    #         self.observation_space = gym.spaces.Box(
    #             low=self.STATE_MIN.cpu().numpy(), high=self.STATE_MAX.cpu().numpy(),
    #             dtype=np.float32)
    #         return
    #     self._preview_offsets_t = torch.tensor(
    #         self.preview_offsets, device=self.device, dtype=torch.long)
    #     p = len(self.preview_offsets)
    #     lo, hi = self.STATE_MIN, self.STATE_MAX
    #     if self._preview_include_uref:
    #         lo = torch.cat([lo] + [self.UREF_MIN] * p)
    #         hi = torch.cat([hi] + [self.UREF_MAX] * p)
    #     if self._preview_include_xref:
    #         # Symmetric superset bound for a relative-position difference —
    #         # X_MIN/X_MAX are x_dim-wide (STATE_MIN/max are the full [x,xref,
    #         # uref] base width; using those here would over-widen by (u_dim +
    #         # x_dim) per point instead of x_dim).
    #         rel_lo = self.X_MIN - self.X_MAX
    #         rel_hi = self.X_MAX - self.X_MIN
    #         lo = torch.cat([lo] + [rel_lo] * p)
    #         hi = torch.cat([hi] + [rel_hi] * p)
    #     self.observation_space = gym.spaces.Box(
    #         low=lo.cpu().numpy(), high=hi.cpu().numpy(), dtype=np.float32)
#
    # def _full_trajectory_offsets(self) -> list[int]:
    #     """every future step offset from the current time to episode end
    #     (dense, not the geometric ladder — see ``_preview_offsets``). Meant to
    #     pair with a sequence gate encoder (``film_gate_encoder in ("gru",
    #     "attn")``): rather than hand-picking a small window sized off the
    #     discount's effective horizon, hand the encoder the whole remaining
    #     reference and let it learn what to attend to / how much to forget.
    #     Cost warning: construct_state recomputes this every step, so the
    #     gate encoder's forward pass is O(max_episode_len) per step instead of
    #     O(num_preview) — an O(max_episode_len^2) cost per episode overall.
    #     Intended for GPU (cluster) runs, not the CPU-only local sweeps."""
    #     return list(range(1, self.max_episode_len))
#
    # def configure_preview(self, num_points: int, gamma: float, include_xref: bool = False,
    #                       full_trajectory: bool = False, include_uref: bool = True) -> None:
    #     """Enable/resize reference preview and widen the observation space to
    #     match. Must be called before the agent/memory are built off
    #     ``observation_space``. Idempotent; ``num_points<=0`` disables preview.
    #     ``include_xref`` — see ``set_preview_offsets`` — defaults to False,
    #     reproducing the historic uref-only preview exactly. ``full_trajectory``
    #     (see ``_full_trajectory_offsets``) replaces the geometric ``num_points``/
    #     ``gamma`` ladder with every future offset up to episode end; ``gamma``
    #     is unused in that case (kept in the signature so callers don't branch).
    #     ``include_uref=False`` drops the future-uref block, testing an
    #     xref-only preview (needs ``include_xref=True`` or there is nothing
    #     left in the tail)."""
    #     offsets = self._full_trajectory_offsets() if full_trajectory else self._preview_offsets(int(num_points), gamma)
    #     self.set_preview_offsets(offsets, include_xref=include_xref, include_uref=include_uref)
    #     if self.preview_offsets:
    #         if self._preview_include_uref and self._preview_include_xref:
    #             _what = "future-uref + relative-future-xref"
    #         elif self._preview_include_xref:
    #             _what = "relative-future-xref (uref excluded)"
    #         else:
    #             _what = "future-uref"
    #         print(f"[BaseEnv] reference preview: {len(self.preview_offsets)} {_what} "
    #               f"points at step offsets {self.preview_offsets} (gamma={gamma}) -> "
    #               f"obs_dim {self.observation_space.shape[0]}")
#
    # def configure_value_state(self, num_points: int, gamma: float = 0.99,
    #                           full_trajectory: bool = False) -> None:
    #     """Enable the privileged critic-only state channel: ``[x, future-xref
    #     relative to x]`` at ``num_points`` offsets (or every remaining step if
    #     ``full_trajectory``), completely independent of whatever preview the
    #     Actor's own observation carries (see the ``_value_state_offsets_t``
    #     docstring in ``__init__``). Must be called before the agent/memory are
    #     built off ``state_space``, mirroring ``configure_preview``. Idempotent;
    #     ``num_points<=0`` (and not full_trajectory) disables it (``state()``
    #     then returns ``None``, ``state_space`` stays ``None``)."""
    #     offsets = self._full_trajectory_offsets() if full_trajectory else self._preview_offsets(int(num_points), gamma)
    #     if not offsets:
    #         self._value_state_offsets_t = None
    #         self.state_space = None
    #         return
    #     self._value_state_offsets_t = torch.tensor(offsets, device=self.device, dtype=torch.long)
    #     p = len(offsets)
    #     # Same symmetric superset bound construct_state uses for its own
    #     # relative-xref preview block (see set_preview_offsets).
    #     rel_lo = self.X_MIN - self.X_MAX
    #     rel_hi = self.X_MAX - self.X_MIN
    #     lo = torch.cat([self.X_MIN] + [rel_lo] * p)
    #     hi = torch.cat([self.X_MAX] + [rel_hi] * p)
    #     self.state_space = gym.spaces.Box(low=lo.cpu().numpy(), high=hi.cpu().numpy(), dtype=np.float32)
    #     print(f"[BaseEnv] privileged critic state: {p} relative-future-xref points "
    #           f"at step offsets {offsets} (gamma={gamma}) -> state_dim {self.state_space.shape[0]}")
#
    # def state(self) -> torch.Tensor | None:
    #     """The privileged critic-only state ``[x, future-xref relative to x]``
    #     — see ``configure_value_state``. ``None`` when not configured (skrl's
    #     ``GymnasiumWrapper.state()`` catches the resulting exception from
    #     indexing a ``None`` offsets tensor and returns ``None`` itself, but we
    #     short-circuit explicitly here for clarity)."""
    #     if self._value_state_offsets_t is None:
    #         return None
    #     idx = torch.clamp(self.time_steps, max=self.max_episode_len - 1)
    #     env_idx = torch.arange(self.num_envs, device=self.device)
    #     pidx = torch.clamp(idx.unsqueeze(-1) + self._value_state_offsets_t.unsqueeze(0),
    #                        max=self.max_episode_len - 1)                    # (N, P)
    #     xref_fut = self.xref[env_idx.unsqueeze(-1), pidx]                   # (N, P, x_dim)
    #     p = xref_fut.shape[1]
    #     raw_rel = (xref_fut - self.x_t.unsqueeze(1)).reshape(self.num_envs * p, -1)
    #     xref_rel = self.wrap_angles(raw_rel).reshape(self.num_envs, p, -1)  # relative to current x
    #     return torch.cat([self.x_t, xref_rel.reshape(self.num_envs, -1)], dim=-1)
#
    def _size_reference_buffers(self) -> None:
        """Allocate ``xref``/``uref`` long enough that the window never clamps.

        The episode runs ``max_episode_len`` steps, but the window at the last of
        them still reaches ``max_episode_len - 1 + (length-1)*offset``. Generating
        only ``max_episode_len`` points forces every index past the end to clamp
        onto ``xref[-1]``, which pads the observation with a synthetic constant
        tail whose length grows as the episode advances. Generating to
        ``ref_gen_len`` instead makes every window entry real reference data.

        Why that matters beyond tidiness: the padded tail is not reference the
        plant ever tracks, so the actor's window encoder spends capacity on a
        signal that carries no information about the task, and its content
        changes with ``t`` even when the underlying reference does not.

        What it does NOT change is the Markov requirement. The episodic return
        from ``t`` touches ``xref[t .. max_episode_len-1]``, so the window still
        has to be at least ``max_episode_len`` long for the value to be a
        function of the observation -- generating further ahead removes the
        padding, it does not license a shorter window. ``check_markov`` is still
        asked about ``max_episode_len`` for exactly that reason.

        Strictly, the observation process is no longer closed under the shift:
        ``xrefs_{t+1}`` needs ``xref[t + length*offset]``, which is not in
        ``s_t``. That index is beyond the episode end, so it never enters any
        reward -- it is reward-irrelevant nuisance rather than hidden state the
        value depends on, which is the opposite of truncating the window inside
        the horizon.
        """
        L, off = self.ref_window.length, self.ref_window.offset
        self.ref_gen_len = int(self.max_episode_len + (L - 1) * off)
        # The generation loop iterates over self.t, so it has to cover the same span.
        self.t = torch.arange(self.ref_gen_len, device=self.device,
                              dtype=torch.float32) * self.dt
        self.xref = torch.zeros(self.num_envs, self.ref_gen_len, self.num_dim_x,
                                dtype=torch.float32, device=self.device)
        self.uref = torch.zeros(self.num_envs, self.ref_gen_len, self.num_dim_control,
                                dtype=torch.float32, device=self.device)

    def construct_state(self, x: torch.Tensor):
        """The observation ``s = {x, xrefs, urefs}``, flattened in skrl's
        sorted-key order (see ``RefWindow.split`` — the one place that layout
        is encoded).

        ``xrefs``/``urefs`` are the reference at steps ``t + k*offset``,
        ``k = 0..length-1``, clamped at the episode end so the reference holds
        its terminal setpoint rather than running off the buffer. Both are raw
        world-frame reference points: the relative-position / wrapped-angle
        transform is a network-input concern and lives in ``ref_window.Feats``,
        so every consumer (actor, critic, the analytic controllers) applies the
        identical map instead of each re-deriving it from a pre-baked tail.

        Returns the dict, not a flat tensor: skrl's wrapper tensorizes and
        flattens it against ``observation_space`` on the way in, which is what
        makes ``RefWindow.split``'s sorted-key ordering correct by construction
        rather than by convention. ``construct_state_flat`` is the tensor form
        for callers that want what the models actually receive.
        """
        idx = torch.clamp(self.time_steps, max=self.max_episode_len - 1)
        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(-1)
        # ref_gen_len, not max_episode_len: the reference extends past the episode
        # end precisely so these indices are real data instead of clamped padding.
        widx = self.ref_window.window_indices(idx, self.ref_gen_len)  # (N, L)
        return {
            "x": x,
            "xrefs": self.xref[env_idx, widx],                            # (N, L, x_dim)
            "urefs": self.uref[env_idx, widx],                            # (N, L, u_dim)
        }

    def construct_state_flat(self, x: torch.Tensor) -> torch.Tensor:
        """``construct_state`` flattened exactly as skrl flattens it — the
        tensor the models receive in ``inputs["observations"]``."""
        d = self.construct_state(x)
        return self.ref_window.flatten(d["x"], d["xrefs"], d["urefs"])

    @staticmethod
    def _zeros(shape, x):
        return torch.zeros(shape, device=x.device, dtype=x.dtype)

    @abstractmethod
    def _f_logic(self, x):
        ...

    @abstractmethod
    def _B_logic(self, x):
        ...

    def _B_null_logic(self, x):
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)
        n = x.shape[0]
        Bbot = torch.cat(
            (torch.eye(eye_dims, device=x.device, dtype=x.dtype),
             torch.zeros(zero_dims, device=x.device, dtype=x.dtype)),
            dim=0,
        )
        return Bbot.repeat(n, 1, 1)

