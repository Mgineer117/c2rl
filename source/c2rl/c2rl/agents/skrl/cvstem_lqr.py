"""CV-STEM-LQR — Tsukamoto's CV-STEM/NCM controller (``AstroHiro/ncm``, ``classncm``).

Analytical, no learnable RL parameters (``update()`` is a no-op), like
``sdlqr.py``'s LQR/SD-LQR. The deployed law is his ``simulation``'s control
branch::

    u = u_ref - K(x)·(x - x_ref),   K(x) = R⁻¹·B(x)ᵀ·M(x)

with the same ``R = r_scaler·I`` the metric SDP's Riccati term uses, so ``K`` is
the certified CV-STEM gain.

The pipeline, offline, once at construction — his ``train``:

1. ``sample_state_box`` — ``cm_samples`` states i.i.d. uniform over the env's
   state box (his ``xlims`` draw).
2. ``cvstem_joint`` — one SDP over all of them: per-state ``W̄_k``, a single
   shared ``ν`` and ``χ``, his ``(W̄-I)/dt`` term, objective ``J = χ/λ + ν``.
3. ``M_to_cholvec`` + ``regress_cholm`` — labels ``chol(W_k⁻¹)``, MSE-fit
   ``nn_modules.CholMetric``. That network is the metric from then on;
   ``M = RᵀR`` is SPD by construction with no eigenvalue bound anywhere.

λ selection is his ``linesearch`` (walk λ up, take ``argmin J``) when
``lbd_linesearch: [lo, hi, da]`` is set; otherwise the configured ``lbd`` is used
directly, his ``cvstem0`` at a fixed α.

Nothing here deviates from the reference: the Ẇ term is always his ``(W̄-I)/dt``,
the samples are always i.i.d. uniform over the box, and the network is fit to
exactly the states the joint program certified. ``scripts/find_uniform_lambda.py``
runs this same program at a smaller sample count to pick ``lbd``.

The one thing his code leaves implicit is that ``dt`` is the CV-STEM sampling
period, not the integrator step — his cart-pole notebook synthesizes at ``dt=1``
while the simulation runs at ``0.1``. ``cm_dt`` is that knob here.

An infeasible joint SDP aborts the run — there is no per-state λ-backoff and no
partial-feasibility rate, because one program covers every sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from skrl.agents.torch.base import Agent, AgentCfg

from .angle_utils import wrap_diff
from .nn_modules import CholMetric
from .ref_window import RefWindow
from .rl_glue import filter_cfg_fields

# Printed verbatim on the abort path. search/sweep_runner.py greps the child's
# output for this exact string, so the two must stay in sync — it is the signal
# that turns an infeasible trial into a recorded bad datapoint.
INFEASIBLE_MARKER = "CVSTEM-LQR INFEASIBLE"


class CVSTEMInfeasibleError(RuntimeError):
    """Raised when the joint CV-STEM SDP has no solution at this (λ, ε).

    A distinct type rather than a bare RuntimeError so a caller can tell "this
    parameter point is not solvable" apart from an ordinary crash.
    """


# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CVSTEMLQRCfg(AgentCfg):
    # R = r_scaler·I, in both the SDP's Riccati term and the deployed gain.
    # Inert: ν is a free variable and the LMI sees only ν/r, so the solved ν
    # absorbs r and K is unchanged (measured on the car: ν/r, χ and ‖K‖₂ all
    # identical for r from 1e-2 to 1e4). λ, ε and cm_dt are the only knobs that
    # move the gain — nu_weight/chi_weight are inert too (ν and χ come out
    # jointly minimal, so the objective weights never activate a trade-off).
    r_scaler: float = 1.0

    # ── The joint CV-STEM SDP (yaml `cm:` block) ───────────────────────────── #
    lbd: float = 0.5              # contraction rate λ (his alpha)
    cm_eps: float = 0.01          # strict-definiteness margin (his epsilon)
    # The dt of his ``(W̄-I)/dt`` term — the CV-STEM sampling period, which is a
    # free hyperparameter of the synthesis and not the integrator step. His
    # cart-pole notebook sets it to 1 while the simulation integrates at 0.1 and
    # RK4-substeps at 0.01. ``None`` falls back to the env's dt, which is a 33x
    # harsher Ẇ bound on the classic envs and inflates ν by the same factor.
    cm_dt: float | None = None
    # Optional deployment envelope w_lb·I ⪯ W ⪯ w_ub·I on the deployed w = W̄/ν,
    # the two scalar caps solve_cm_metric applies: ν ≤ 1/w_lb, χ ≤ ν·w_ub. Both
    # None (the default) is Tsukamoto's program exactly — ν and χ free, no
    # envelope. w_lb is the direct gain cap, ‖K‖₂ ≤ ‖B‖₂/(r·w_lb), and is the
    # knob r only imitates: ν absorbs r whenever χ is pinned, w_lb never is.
    cm_w_lb: float | None = None
    cm_w_ub: float | None = None
    cm_solver: str = "MOSEK"      # cvxpy SDP solver (SCS | CLARABEL | MOSEK)
    # Uniform state samples in the one joint program (his Nx=1000). A solver-Size
    # knob: each sample adds an x_dim² PSD block and two LMIs to one problem.
    cm_samples: int = 1000
    # J = chi_weight·χ + nu_weight·ν (his d₁b̄/α and d₂). None → 1/lbd.
    chi_weight: float | None = None
    nu_weight: float = 1.0
    cm_seed: int = 0              # so a re-run certifies the same sample draw
    # His ``linesearch``: [lbd_lo, lbd_hi, da]. Walks λ up and takes argmin J (the
    # steady-state-error bound) — his λ selection. Absent (default) = use ``lbd``
    # above directly, his cvstem0 at a fixed α. Neither criterion looks at the
    # actuator box — whether the certified gain fits is an empirical question the
    # rollout answers, not one the SDP is asked.
    lbd_linesearch: tuple | None = field(default=None)
    # Cache the joint solve here. Unlike C2RL — which has always cached — this
    # agent re-solved from scratch on every run, so cm_samples was capped by what
    # was tolerable per run rather than by what the metric fit needs: at
    # cm_samples=10000 the program takes ~13 h (measured T ∝ N^1.95), which is a
    # fine one-time offline cost and an impossible per-run one. Keyed on every
    # knob that enters the program (see ncm_synthesis.cvstem_cache_path /
    # load_cvstem_dataset), so a config change re-solves instead of silently
    # deploying a metric certifying a different rate. Empty string disables it.
    cm_data_path: str = ""

    # ── The metric network (yaml `cmg:` block) ─────────────────────────────── #
    cmg_hidden_dims: tuple = (100, 100, 100)   # his 3x100 ReLU MLP
    # Sized for >=100k gradient steps, which is what the fit actually needs:
    # measured on the car, relative metric error 0.55 -> 0.23 -> 0.06 -> 0.03 at
    # 0.5k/5k/25k/100k steps. The old 100 epochs x 32 batch was ~3k steps.
    cmg_regress_epochs: int = 300
    cmg_regress_lr: float = 1.0e-3
    cmg_regress_batch_size: int = 256
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict | None = field(default=None)
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 30


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class CVSTEMLQRAgent(Agent):
    """CV-STEM/NCM contraction controller wrapped as a native skrl Agent.

    Extra constructor kwargs:
      ``get_f_and_B``: ``(x) -> (f, B, Bbot)`` (analytical env dynamics or a
        loaded NeuralDynamics).
      ``x_lo``/``x_hi``: the env's own state box — his ``xlims``. Required.
      ``dt``: fallback for the SDP's ``Ẇ`` term when ``cfg.cm_dt`` is unset.
        Required only in that case — see ``CVSTEMLQRCfg.cm_dt``, which is his
        CV-STEM sampling period and takes precedence.
    """

    def __init__(
        self,
        *,
        cfg: CVSTEMLQRCfg | dict,
        models: dict | None = None,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_f_and_B: Callable,
        x_lo=None,
        x_hi=None,
        dt: float | None = None,
        x_dim: int | None = None,
        u_dim: int | None = None,
        angle_idx: list | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = CVSTEMLQRCfg(**filter_cfg_fields(cfg, CVSTEMLQRCfg, context="CVSTEMLQRAgent"))
        super().__init__(
            cfg=cfg,
            models=models or {},
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        # The observation space declares its layout ({x, xrefs, urefs}), so
        # x_dim/u_dim are read, never guessed from obs_dim.
        self._window = RefWindow.from_space(observation_space)
        self._x_dim = self._window.x_dim if x_dim is None else x_dim
        self._u_dim = self._window.u_dim if u_dim is None else u_dim
        self._angle_idx = angle_idx or []
        self._cfg = cfg
        self._get_f_and_B = get_f_and_B
        # cfg.cm_dt wins over the env's dt: his dt is the CV-STEM sampling period,
        # a synthesis hyperparameter, not the integrator step (see CVSTEMLQRCfg).
        # The env's dt is only the fallback for configs that don't state one.
        dt = cfg.cm_dt if cfg.cm_dt is not None else dt
        # The box is a property of the env and has no sane default — a guessed
        # box certifies the wrong region — and a missing dt silently rescales Ẇ.
        if x_lo is None or x_hi is None or dt is None:
            raise ValueError(
                "CVSTEMLQRAgent needs the env's state box (x_lo/x_hi) and a dt: the "
                "joint CV-STEM SDP samples uniformly over that box and its (W̄-I)/dt "
                "term is scaled by that dt. Set `cm_dt` in the yaml's cm: block, or "
                "pass dt= to fall back to the env step."
            )
        self._x_lo, self._x_hi, self._dt = x_lo, x_hi, float(dt)

        self._metric = CholMetric(self._x_dim, list(cfg.cmg_hidden_dims)).to(device)
        self._synthesize()

    # ── Phase A: his train(), once at construction ─────────────────────────── #

    def _synthesize(self) -> None:
        """Uniform samples → one joint SDP → MSE-fit ``CholMetric``, then freeze."""
        from .ncm_synthesis import (
            cvstem_cache_path,
            cvstem_metric_dataset,
            load_cvstem_dataset,
            regress_cholm,
            save_cvstem_dataset,
        )

        cfg = self._cfg
        tag = "[CVSTEM-LQR]"
        # Cache key = every knob the joint program reads. lbd_linesearch is in it
        # because it overrides lbd (the solve picks its own λ), so two configs
        # differing only there must not share a file.
        cache_cfg = dict(
            lbd=cfg.lbd, r_scaler=cfg.r_scaler, w_lb=cfg.cm_w_lb, w_ub=cfg.cm_w_ub,
            eps=cfg.cm_eps, dt=self._dt, n_samples=cfg.cm_samples,
            solver=cfg.cm_solver, seed=cfg.cm_seed, nu_weight=cfg.nu_weight,
            chi_weight=cfg.chi_weight,
            linesearch=("none" if not cfg.lbd_linesearch
                        else ",".join(f"{float(v):g}" for v in cfg.lbd_linesearch)),
        )
        cache = cvstem_cache_path(
            cfg.cm_data_path, lbd=cfg.lbd, r_scaler=cfg.r_scaler, w_lb=cfg.cm_w_lb,
            w_ub=cfg.cm_w_ub, eps=cfg.cm_eps, dt=self._dt, n_samples=cfg.cm_samples,
        ) if cfg.cm_data_path else None

        dataset = load_cvstem_dataset(cache, tag=tag, **cache_cfg) if cache else None
        if dataset is None:
            dataset = cvstem_metric_dataset(
                self._get_f_and_B, self._x_lo, self._x_hi,
                n_samples=cfg.cm_samples, lbd=cfg.lbd, eps=cfg.cm_eps, dt=self._dt,
                solver=cfg.cm_solver, r_scaler=cfg.r_scaler,
                chi_weight=cfg.chi_weight, nu_weight=cfg.nu_weight,
                w_lb=cfg.cm_w_lb, w_ub=cfg.cm_w_ub,
                seed=cfg.cm_seed, device=self.device, tag=tag,
                linesearch=(tuple(cfg.lbd_linesearch) if cfg.lbd_linesearch else None),
            )
            if cache is not None and dataset is not None:
                save_cvstem_dataset(cache, dataset, **cache_cfg)
        if dataset is None:
            # No metric at all — every downstream number would describe an
            # uncertified controller, so this is fatal rather than a fallback.
            msg = (
                f"{INFEASIBLE_MARKER}: the joint CV-STEM SDP over {cfg.cm_samples} "
                f"uniform state-box samples is infeasible at lbd={cfg.lbd}, "
                f"cm_eps={cfg.cm_eps}, dt={self._dt}. LOWER lbd — "
                f"scripts/find_uniform_lambda.py reports the largest rate this "
                f"same program certifies."
            )
            print(msg, flush=True)
            raise CVSTEMInfeasibleError(msg)
        regress_cholm(
            self._metric, dataset,
            epochs=cfg.cmg_regress_epochs, lr=cfg.cmg_regress_lr,
            batch_size=cfg.cmg_regress_batch_size,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            device=self.device, tag=tag,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
        )
        # W̄ ∈ [I, χI] and W = W̄/ν, so the deployed metric's eigenvalues are
        # M ∈ [ν/χ, ν] — the envelope the SDP chose. Read by the runner for the
        # PathTracking contraction certificate.
        self.metric_nu = float(dataset["nu"])
        self.metric_chi = float(dataset["chi"])
        for p in self._metric.parameters():
            p.requires_grad_(False)
        self._metric.eval()
        # lbd from the dataset, not from cfg — the linesearch may have chosen it.
        self.metric_lbd = float(dataset["lbd"])
        print(f"{tag} Phase A complete — metric frozen (nu={self.metric_nu:.4g}, "
              f"chi={self.metric_chi:.4g}, J={dataset['J']:.4g}, lbd={self.metric_lbd:.4g}).")

    # ── action computation ─────────────────────────────────────────────────── #

    def _compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """``M(x) = RᵀR`` from the frozen network, ``K = R⁻¹BᵀM``, ``u = uref - K·e``.

        No metric inverse at deploy: the SDP's ``W`` was inverted once, offline,
        to build the labels, and the network predicts ``M`` directly.
        """
        r = self._cfg.r_scaler + 1e-5  # strictly positive — mirrors sdlqr.py's guard
        obs = obs.to(self.device)
        # RefWindow.split, never a hand-slice. skrl flattens the Dict observation
        # in sorted-key order — [urefs | x | xrefs], not [x | xref | uref] — and
        # the two have the same total width at ref_length=1, so a hand-slice
        # mis-reads every block without ever tripping a shape check.
        x, xrefs, urefs = self._window.split(obs)
        x, xref, uref = x.float(), xrefs[:, 0].float(), urefs[:, 0].float()

        with torch.no_grad():
            _f, B, _Bbot = self._get_f_and_B(x)
            M = self._metric(x)                                    # (b, x, x)
            K = (1.0 / r) * torch.bmm(B.to(torch.float32).transpose(1, 2), M)
            e = wrap_diff(x - xref, self._angle_idx).unsqueeze(-1)
            return uref - torch.bmm(K, e).squeeze(-1)               # (b, u_dim)

    def act(self, observations, states, *, timestep: int, timesteps: int):
        orig_device = observations.device
        actions = self._compute_action(observations).detach().to(orig_device)
        log_prob = torch.zeros(actions.shape[0], 1, device=orig_device)
        return actions, {"log_prob": log_prob}

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def record_transition(
        self, *, observations, states, actions, rewards, next_observations,
        next_states, terminated, truncated, infos, timestep, timesteps,
    ) -> None:
        super().record_transition(
            observations=observations, states=states, actions=actions,
            rewards=rewards, next_observations=next_observations,
            next_states=next_states, terminated=terminated, truncated=truncated,
            infos=infos, timestep=timestep, timesteps=timesteps,
        )

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass  # analytical — no gradient updates
