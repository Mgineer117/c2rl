"""Shared utilities for the standalone visualization scripts (classic envs only).

Everything here is read-Only with respect to the main codebase: it imports the
classic envs and the agents' network/math modules but never modifies them, and
none of the training/eval code imports this package back.

Concepts
--------
Scenario        one frozen episode: initial state x0, reference trajectory
                (xref, uref), dt, horizon — extracted from a seeded env reset so
                every script/policy sees the identical episode.
Metric provider M(x) — the contraction metric conditioning the "normalized
                error". Two sources:
                  * CCMMetric   — a trained CMG network loaded from a checkpoint
                                  (c3m.pt / c2rl_ppo.pt "cmg" entry); batched.
                  * CVSTEMMetric — Tsukamoto CV-STEM SDP solved online per
                                  state (ncm_synthesis.solve_cm_metric); one
                                  solve per queried state, so geometry code
                                  reuses one M per timestep (metric.batched
                                  is False).
Policy          a deterministic callable obs(1, 2*x_dim+u_dim) -> u(1, u_dim).
                c3m / c2rl_ppo are rebuilt from checkpoints + their yaml config
                (visualization/policies/<env>/, falling back to the task's
                default agent yaml for architecture); lqr / cvstem_lqr are
                analytical and need no checkpoint.

Normalized error (the quantity every plot shows):
    r(t) = sqrt( e_tᵀ M(x_t) e_t / e_0ᵀ M(x_0) e_0 ),   e_t = wrap(x_t - xref_t)

Scope — u_dim <= 2 only (car/turtlebot 2, cartpole/segway 1). A deliberate
restriction, not a limitation to work around: at one or two inputs the full
control space is plottable, so the landscape is complete, with no projection and
no information loss (u_dim 1 -> a t x u surface; u_dim 2 -> a u0 x u1 surface
animated over t).

Projecting down to a scalar axis (an earlier design) is provably worse: no
continuous injection R^m -> R^1 exists for m >= 2, so a bijection like
digit-interleaving is necessarily discontinuous and destroys exactly the
neighborhood structure the plot's shape depends on. For m > 2 the principled
reduction is a linear projection onto span(B^T(2Me + w)), the one direction the
contraction rate depends on to first order — but at m <= 2 none is needed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

# ── repo imports (read-only) ───────────────────────────────────────────────── #
_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_VIZ_DIR)
_SRC = os.path.join(_REPO_ROOT, "source", "c2rl")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import c2rl.tasks.direct.classic  # noqa: F401,E402  registers classic-*-v0
import gymnasium as gym  # noqa: E402
from c2rl.agents.skrl.angle_utils import wrap_diff  # noqa: E402
from c2rl.agents.skrl.math_utils import (  # noqa: E402
    b_jacobian,
    bound_W,
    jacobian,
    spd_inverse,
)
from c2rl.agents.skrl.ncm_synthesis import cvstem_joint  # noqa: E402
from c2rl.agents.skrl.nn_modules import BoundedCCM_Generator, CCM_Generator  # noqa: E402

# Only envs with u_dim <= 2 — the full control space is plottable without any
# projection (see the module docstring). quadrotor (u_dim=4) is deliberately out.
CLASSIC_ENVS = ("car", "cartpole", "turtlebot", "segway")
U_DIM = {"cartpole": 1, "segway": 1, "car": 2, "turtlebot": 2}
POLICIES_DIR = os.path.join(_VIZ_DIR, "policies")
OUTPUT_DIR = os.path.join(_VIZ_DIR, "output")

# Fixed categorical hue order from the reference palette. Verified mutually
# separable for normal vision and simulated deuteranopia/protanopia (OKLab ΔE),
# so identity never rests on a hue a CVD reader can't resolve. The surface uses
# viridis, which overlaps these hues, so every overlay additionally carries a
# white halo/edge for figure-ground separation.
SERIES_COLORS = {
    # metric sources (error_geometry.py) and policies (policy_overlay.py) never
    # appear in the same figure, so the same validated hues serve both sets.
    "ccm": "#2a78d6",                # blue
    "cvstem_pretrained": "#e87ba4",  # magenta
    "cvstem_online": "#008300",      # green
    # Deliberately gray: random is the untrained baseline, not a fourth
    # competing metric, and gray says so. It is the one entry that intentionally
    # fails the palette chroma floor (chroma 0 "reads as gray" — here that is the
    # message), so like uref it carries secondary encoding (dashed) rather than
    # relying on hue. #6e6e6e specifically: it clears CVD separation against the
    # three hues (worst ΔE 10.1 deutan vs green) where a lighter #8a8a8a fails
    # against magenta (ΔE 4.0 protan), and a darker #4a4a4a leaves the band.
    "random": "#6e6e6e",
    # The unconditioned M = I baseline (bound_sweep.py). Gray for the same reason
    # random is, and the same gray deliberately: both are reference baselines, not
    # competing series, and no figure shows the two together (random is an
    # untrained CMG, so it only appears where CMG objectives are compared, while
    # euclidean only appears where they are all held fixed). If one ever does, this
    # needs its own validated gray — #6e6e6e is at the edge of the CVD band.
    "euclidean (M = I)": "#6e6e6e",
    "c3m": "#2a78d6",        # blue
    "c2rl_ppo": "#008300",   # green
    "cvstem_lqr": "#e87ba4",  # magenta
    "lqr": "#eda100",        # yellow
    # Purple, validated against the other four policies (worst all-pairs ΔE 13.0
    # protan / 16.3 normal). Not orange, the intuitive "next" hue: #eb6834 fails
    # hard against both green (ΔE 3.2 protan) and magenta (12.9 normal vision).
    "sd_lqr": "#4a3aa7",
    # Near-black, not mid-gray: a mid-gray collides with green under simulated
    # deuteranopia (ΔE 2.3). uref is the reference, never a competing series, so
    # it also stays dashed/X-marked (secondary encoding).
    "uref": "#0b0b0b",
}

# The metric sources the geometry can be conditioned on (see make_metric).
# "random" is the untrained control baseline, not a metric source proper.
METRIC_KINDS = ("ccm", "cvstem_pretrained", "cvstem_online", "random")


# ─────────────────────────────────────────────────────────────────────────── #
# Environment / scenario
# ─────────────────────────────────────────────────────────────────────────── #


def _env_id(name: str) -> str:
    """Short name -> gym id, via the one map the classic package owns."""
    from c2rl.tasks.direct.classic import env_id
    return env_id(name)

def make_env(env_name: str, seed: int, time_bound: float | None = None):
    """Instantiate a classic env (num_envs=1, cpu) with a seeded reset.

    Seeding happens before construction: BaseEnv.__init__ calls reset(), and
    reset draws x0/xref/uref through torch.rand — so the seed fully determines
    the scenario.
    """
    if env_name not in CLASSIC_ENVS:
        raise ValueError(f"env must be one of {CLASSIC_ENVS}, got {env_name!r}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(
        _env_id(env_name),
        num_envs=1,
        device="cpu",
        disable_env_checker=True,
        **({"time_bound": time_bound} if time_bound is not None else {}),
    ).unwrapped
    return env


@dataclass
class Scenario:
    """One frozen episode extracted from a seeded env reset."""
    x0: torch.Tensor      # (x_dim,)
    xref: torch.Tensor    # (T, x_dim)
    uref: torch.Tensor    # (T, u_dim)
    t: torch.Tensor       # (T,)
    dt: float
    T: int
    x_dim: int
    u_dim: int
    angle_idx: list = field(default_factory=list)

    @property
    def e0(self) -> torch.Tensor:
        return wrap_diff((self.x0 - self.xref[0]).unsqueeze(0), self.angle_idx)[0]


def get_scenario(env) -> Scenario:
    return Scenario(
        x0=env.x_t[0].clone(),
        xref=env.xref[0].clone(),
        uref=env.uref[0].clone(),
        t=env.t.clone(),
        dt=float(env.dt),
        T=int(env.max_episode_len),
        x_dim=int(env.num_dim_x),
        u_dim=int(env.num_dim_control),
        angle_idx=list(env.angle_idx),
    )


def step_dynamics(env, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One Euler step of the env dynamics — mirrors BaseEnv.step()'s state
    update (actuator clamp to the physical U box, position carry-forward,
    angle wrap, state clamp) without its reward/reset machinery."""
    u = torch.clamp(torch.nan_to_num(u), env.U_MIN, env.U_MAX)
    f, B, _ = env.get_f_and_B(x)
    x_dot = f + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
    next_x = x + env.dt * x_dot

    p = env.pos_dimension
    oob = ((next_x[:, :p] < env.X_MIN[:p]) | (next_x[:, :p] > env.X_MAX[:p])).any(dim=-1)
    next_x[oob, :p] = x[oob, :p]
    return torch.clamp(env.wrap_angles(next_x), env.X_MIN, env.X_MAX)


def control_grid(env, scen: Scenario, num_chunks: int, u_range: str = "physical") -> torch.Tensor:
    """Per-dimension control levels, ``(u_dim, num_chunks)``.

    "physical": the actuator box step() actually enforces (2x the uref box —
    uref+feedback controllers legitimately exceed the declared action space).
    "uref": the declared uref action-space box.

    Spanning the real actuator box matters: the one-step-optimal control
    satisfies B·u ≈ -e/dt, i.e. ‖u*‖ ~ ‖e‖/dt, so early in an episode the
    optimum lies outside the box (the landscape is a monotone wall — more
    feedback is always better) and only comes inside once ‖e‖ has decayed
    enough. A sweep window narrower than the box hides that transition.
    """
    lo, hi = (env.U_MIN, env.U_MAX) if u_range == "physical" else (env.UREF_MIN, env.UREF_MAX)
    return torch.stack(
        [torch.linspace(float(lo[j]), float(hi[j]), num_chunks) for j in range(scen.u_dim)]
    )


def frame_indices(K: int, num_frames: int) -> np.ndarray:
    """Evenly spaced timestep indices to render as video frames."""
    return np.unique(np.linspace(0, K - 1, min(num_frames, K)).astype(int))


# ─────────────────────────────────────────────────────────────────────────── #
# Metric providers
# ─────────────────────────────────────────────────────────────────────────── #

class CCMMetric:
    """M(x) from a CMG network ("cmg" entry of c3m.pt / c2rl_ppo.pt). Batched —
    one forward pass covers any number of states, so geometry code evaluates M
    at every candidate next-state.

    ``random_init`` builds the same architecture with the same w_lb/w_ub bounds
    from the same config, then simply does not load the trained weights: the
    untrained-CMG control baseline. Holding everything but the weights fixed is
    what makes the comparison attributable to training rather than to
    architecture or bounds. Note the baseline is not "no metric" — a random CMG
    is still a bounded SPD field, so any structure its landscape shows is
    structure the architecture and bounds impose for free, before learning. That
    is the point: it is the floor a trained CCM has to beat.
    """

    batched = True

    def __init__(self, ckpt_path: str, cfg: dict, scen: Scenario, *,
                 random_init: bool = False, seed: int = 0):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "cmg" not in state:
            raise KeyError(f"{ckpt_path} has no 'cmg' entry (keys: {list(state)})")
        sd = {k.removeprefix("ccm_gen."): v for k, v in state["cmg"].items()
              if k.startswith("ccm_gen.")}

        # Bounded vs raw generator is detectable from the weight names —
        # BoundedCCM_Generator: model./mu/logstd; CCM_Generator: backbone./mu_head.
        # Read off the checkpoint even when random_init discards the weights, so
        # the baseline mirrors the trained net's architecture exactly.
        bounded = any(k.startswith("model.") for k in sd)
        net_cfg = (cfg.get("models", {}).get("cmg", {}).get("network") or [{}])[0]
        hidden = list(net_cfg.get("layers", [128, 128]))
        activation = net_cfg.get("activations", "tanh")
        self.w_lb, self.w_ub = _metric_bounds(cfg)

        cls = BoundedCCM_Generator if bounded else CCM_Generator
        kwargs = dict(x_dim=scen.x_dim, hidden_dim=hidden, activation=activation,
                      mode="deterministic", device="cpu", angle_idx=scen.angle_idx)
        if bounded:
            kwargs.update(w_lb=self.w_lb, w_ub=self.w_ub)
        if random_init:
            # Seeded locally: a fresh generator's draw must be reproducible
            # without perturbing the global RNG the scenario/env were seeded from.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.gen = cls(**kwargs)
        else:
            self.gen = cls(**kwargs)
            self.gen.load_state_dict(sd)
        self.gen.eval()
        for p in self.gen.parameters():
            p.requires_grad_(False)
        self._x_dim = scen.x_dim
        # Contraction rate the CMG was trained against — for the e^{-λt} guide.
        # Kept for the random baseline too: it is the rate the trained net targets,
        # drawn as the same reference line, not a claim the baseline achieves it.
        self.lbd = float(cfg.get("cm", {}).get("lbd", cfg.get("agent", {}).get("lbd", 0.5)))
        self.name = (f"random (untrained, seed {seed}, arch of "
                     f"{os.path.basename(ckpt_path)})" if random_init
                     else f"ccm ({os.path.basename(ckpt_path)})")

    @torch.no_grad()
    def M(self, x: torch.Tensor) -> torch.Tensor:
        raw_W, _ = self.gen(x)
        W = bound_W(raw_W, self.w_lb, self._x_dim, getattr(self.gen, "bounded", False))
        return spd_inverse(W)


class CVSTEMMetric:
    """M(x) from ONE joint CV-STEM SDP over the queried states.

    Not a per-state solve. An independent SDP per state carries its own nu/chi and
    no Wdot term, so it is a per-state feasibility statement rather than a
    contraction certificate -- the same objection that made build_cm_dataset a
    single joint program and removed the per-step "online" metric. This solves the
    batch jointly, with nu/chi shared, which is also what ships.

    Infeasibility is reported, never substituted. The previous version fell back to
    the IDENTITY for any state the pointwise SDP missed, which drew an infeasible
    region as a perfectly conditioned Euclidean one."""

    batched = False

    def __init__(self, env, cfg: dict, solver: str | None = None):
        cm = cfg.get("cm", {})
        self.env = env
        self.lbd = float(cm.get("lbd", 0.5))
        self.w_lb = float(cm.get("w_lb", 0.1))
        self.w_ub = float(cm.get("w_ub", 10.0))
        self.eps = float(cm.get("cm_eps", 0.01))
        self.solver = solver or cm.get("cm_solver", "SCS")
        self.r_scaler = float(cfg.get("agent", {}).get("r_scaler",
                              cm.get("cvstem_r_scaler", 1.0)))
        # 1.0, matching build_cm_dataset's dt=(cm_wdot_dt or temporal_dt or 1.0).
        # The env dt would inflate the (W-I)/dt term ~33x — see docs/cluster.md.
        self.cm_dt = float(cm.get("cm_dt", 1.0) or 1.0)
        self.infeasible = 0
        self.solves = 0
        self.name = f"cvstem (joint SDP, {self.solver})"

    def _drift_jacobian_and_B(self, x: torch.Tensor):
        x = x.detach().clone().requires_grad_()
        with torch.enable_grad():
            f, B, _ = self.env.get_f_and_B(x)
            A = jacobian(f, x, create_graph=False)
        return A.detach(), B.detach()

    def M(self, x: torch.Tensor) -> torch.Tensor:
        A, B = self._drift_jacobian_and_B(x)
        d = x.shape[-1]
        self.solves += 1
        sol = cvstem_joint(
            A.numpy().astype(np.float64), B.numpy().astype(np.float64),
            lbd=self.lbd, eps=self.eps, dt=self.cm_dt, solver=self.solver,
            r_scaler=self.r_scaler, w_lb=self.w_lb, w_ub=self.w_ub,
        )
        if sol is None:
            # One program over the whole batch, so infeasibility is a statement
            # about this state set, not about one point in it. Raise: silently
            # returning the identity would draw an uncertifiable region as a
            # perfectly conditioned Euclidean one.
            self.infeasible += 1
            raise RuntimeError(
                f"joint CV-STEM SDP INFEASIBLE over the {x.shape[0]} queried states "
                f"at lbd={self.lbd:g}, eps={self.eps:g}, w=[{self.w_lb:g},{self.w_ub:g}], "
                f"r={self.r_scaler:g}. No metric family certifies that rate here, so "
                f"there is no geometry to plot.")
        W = np.asarray(sol["W"], dtype=np.float64)
        return torch.as_tensor(np.linalg.inv(W), dtype=torch.float32).reshape(-1, d, d)


class CVSTEMPretrainedMetric:
    """M(x) from a frozen CMG regressed onto per-state CV-STEM SDP solutions —
    Tsukamoto's NCM, i.e. exactly ``cvstem_lqr.py``'s ``metric_source="pretrained"``
    (``build_cm_dataset`` + ``regress_cmg``).

    Batched: once fitted, M(x) is a network forward pass, so the metric is
    evaluated at every candidate next-state like the CCM one. The certificate
    is only as tight as the regression fit — which is precisely what makes this
    worth showing next to the online SDP metric: the gap between the two
    geometries is the regression error.

    The per-state SDP solve is the expensive part, so the {x, W*} dataset is
    cached (keyed on the full SDP config — any knob change re-solves).
    """

    batched = True

    def __init__(self, env, scen: Scenario, cfg: dict, *, num_samples: int,
                 cache_path: str, solver: str | None = None, device: str = "cpu"):
        from pathlib import Path

        from c2rl.agents.skrl.ncm_synthesis import (
            build_cm_dataset,
            load_cached_cm_dataset,
            regress_cmg,
            save_cm_dataset,
        )

        cm = cfg.get("cm", {})
        cmg = cfg.get("cmg", {})
        agent = cfg.get("agent", {})
        self.lbd = float(cm.get("lbd", 0.5))
        self.w_lb = float(cm.get("w_lb", 0.1))
        self.w_ub = float(cm.get("w_ub", 10.0))
        self.eps = float(cm.get("cm_eps", 0.01))
        self.solver = solver or cm.get("cm_solver", "SCS")
        self.r_scaler = float(agent.get("r_scaler", cm.get("cvstem_r_scaler", 1.0)))
        # 1.0, matching build_cm_dataset's dt=(cm_wdot_dt or temporal_dt or 1.0).
        # The env dt would inflate the (W-I)/dt term ~33x — see docs/cluster.md.
        self.cm_dt = float(cm.get("cm_dt", 1.0) or 1.0)
        tag = "[viz-cvstem-pretrain]"

        # Cache identity must mirror save/load exactly, or a config change would
        # silently reuse stale W* targets (see load_cached_cm_dataset).
        cache_kwargs = dict(
            lbd=self.lbd, w_lb=self.w_lb, w_ub=self.w_ub, eps=self.eps,
            solver=self.solver, num_samples=num_samples, tag=tag,
            r_scaler=self.r_scaler, chi_weight=None, nu_weight=1.0,
            wdot_dt=0.0, random_ratio=0.0, wdot_trajectory=False, temporal_dt=0.0,
        )
        path = Path(cache_path)
        dataset = load_cached_cm_dataset(path, **cache_kwargs)
        if dataset is None:
            print(f"{tag} solving {num_samples} CV-STEM SDPs ({self.solver}) — "
                  f"cached to {path} for later runs")
            dataset = build_cm_dataset(
                env.get_rollout, env.get_f_and_B,
                x_dim=scen.x_dim, lbd=self.lbd, w_lb=self.w_lb, w_ub=self.w_ub,
                eps=self.eps, num_samples=num_samples, solver=self.solver,
                device=device, tag=tag,
                min_feasibility_rate=float(cm.get("min_feasibility_rate", 0.0)),
                r_scaler=self.r_scaler, max_lambda_reductions=self.max_red,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            save_cm_dataset(path, dataset, **cache_kwargs)
        self.feasibility_rate = float(dataset.get("feasibility_rate", float("nan")))

        net = (cfg.get("models", {}).get("cmg", {}).get("network") or [{}])[0]
        self.gen = BoundedCCM_Generator(
            x_dim=scen.x_dim, hidden_dim=list(net.get("layers", [128, 128])),
            activation=net.get("activations", "tanh"), mode="deterministic",
            w_lb=self.w_lb, w_ub=self.w_ub, device=device, angle_idx=scen.angle_idx,
        )
        regress_cmg(
            self.gen, dataset, w_lb=self.w_lb, x_dim=scen.x_dim, bounded=True,
            epochs=int(cmg.get("cmg_regress_epochs", 10)),
            lr=float(cmg.get("cmg_regress_lr", 1e-3)),
            batch_size=int(cmg.get("cmg_regress_batch_size", 1024)),
            lr_scheduler=cmg.get("cmg_regress_lr_scheduler", ""),
            lr_scheduler_kwargs=cmg.get("cmg_regress_lr_scheduler_kwargs"),
            device=device, tag=tag,
            val_frac=float(cmg.get("cmg_val_frac", 0.1)),
            early_stop_patience=int(cmg.get("cmg_early_stop_patience", 10)),
        )
        for p in self.gen.parameters():
            p.requires_grad_(False)
        self.gen.eval()
        self._x_dim = scen.x_dim
        self.name = f"cvstem pretrained (frozen CMG, {self.feasibility_rate:.0%} feasible)"

    @torch.no_grad()
    def M(self, x: torch.Tensor) -> torch.Tensor:
        raw_W, _ = self.gen(x)
        return spd_inverse(bound_W(raw_W, self.w_lb, self._x_dim, True))


class CCMTrainedMetric:
    """M(x) from a CMG trained to minimize the C1/C2 contraction losses —
    ``ncm_synthesis.train_cmg_ccm``, i.e. c2rl's ``cmg_method="ccm"`` path: the
    Manchester CCM conditions enforced pointwise by gradient descent, with no
    per-state SDP and no MSE regression.

    This is the exact counterpart of ``CVSTEMPretrainedMetric``: same network,
    same bounds, same state sampling — the only difference is the objective the
    CMG was fitted with (C1/C2 contraction losses here vs MSE onto CV-STEM SDP
    solutions there). That is what makes the two geometries a clean comparison
    of the two synthesis formulations rather than of two arbitrary checkpoints.

    Deliberately not loaded from a checkpoint. c3m.pt's CMG is trained jointly
    with its controller on ``pd_loss + c1_loss + c2_loss (+ os_loss)``, so it is
    co-adapted to that controller and is not "the C1/C2 metric"; c2rl_ppo.pt's
    CMG is a genuine C1/C2 fit, but only when its config says cmg_method=ccm,
    and it would still tie this geometry to one training run. Training here from
    the config makes the panel mean exactly what its name says. Pass
    ``--metric-ckpt`` to override with a stored CMG anyway.

    The fit is the expensive part, so the trained weights are cached, keyed on
    every knob that affects them.
    """

    batched = True

    def __init__(self, env, scen: Scenario, cfg: dict, *, num_samples: int | None = None,
                 cache_path: str, device: str = "cpu",
                 random_init: bool = False, seed: int = 0):
        import json
        from pathlib import Path

        from c2rl.agents.skrl.ncm_synthesis import train_cmg_ccm

        cm = cfg.get("cm", {})
        cmg = cfg.get("cmg", {})
        self.lbd = float(cm.get("lbd", 0.5))
        self.w_lb = float(cm.get("w_lb", 0.1))
        self.w_ub = float(cm.get("w_ub", 10.0))
        self.eps = float(cm.get("cm_eps", 0.01))
        tag = "[viz-ccm-c1c2]"

        net = (cfg.get("models", {}).get("cmg", {}).get("network") or [{}])[0]
        hidden = list(net.get("layers", [128, 128]))
        activation = net.get("activations", "tanh")
        epochs = int(cmg.get("cmg_regress_epochs", 10))
        lr = float(cmg.get("cmg_regress_lr", 1e-3))
        batch_size = int(cmg.get("cmg_regress_batch_size", 1024))
        random_ratio = float(cmg.get("cmg_random_ratio", 0.0))
        # Default to the config's own budget (what c2rl trains this CMG with),
        # not to --cmg-samples: that flag sizes the CV-STEM SDP dataset, where
        # every sample costs a solve, whereas the C1/C2 path has no SDP and is
        # cheap per sample. Undersizing it here silently starves the fit —
        # cmg_memory_size/batch_size * epochs is the gradient-step budget, so
        # e.g. 2048 samples at 10 epochs is 20 steps and trains nothing.
        if num_samples is None:
            num_samples = int(cmg.get("cmg_memory_size", 8192))

        gen_kwargs = dict(
            x_dim=scen.x_dim, hidden_dim=hidden, activation=activation,
            mode="deterministic", w_lb=self.w_lb, w_ub=self.w_ub,
            device=device, angle_idx=scen.angle_idx,
        )

        if random_init:
            # The untrained control baseline: identical network, bounds and
            # config to the C1/C2 fit above — the only difference is that
            # train_cmg_ccm never runs. Seeded locally so a fresh draw cannot
            # perturb the global RNG the scenario/env were seeded from.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.gen = BoundedCCM_Generator(**gen_kwargs)
            for p in self.gen.parameters():
                p.requires_grad_(False)
            self.gen.eval()
            self._x_dim = scen.x_dim
            self.name = f"random (untrained CMG, seed {seed})"
            return

        self.gen = BoundedCCM_Generator(**gen_kwargs)

        # Cache key = every knob that changes the fitted weights. Anything
        # missing here would silently serve a stale CMG after a config edit.
        key = dict(lbd=self.lbd, w_lb=self.w_lb, w_ub=self.w_ub, eps=self.eps,
                   epochs=epochs, lr=lr, batch_size=batch_size,
                   num_samples=num_samples, random_ratio=random_ratio,
                   hidden=hidden, activation=activation, x_dim=scen.x_dim,
                   lr_scheduler=cmg.get("cmg_regress_lr_scheduler", ""),
                   val_frac=float(cmg.get("cmg_val_frac", 0.1)),
                   early_stop=int(cmg.get("cmg_early_stop_patience", 10)))
        path = Path(cache_path)
        blob = torch.load(path, map_location=device, weights_only=False) if path.exists() else None
        if blob is not None and blob.get("key") == json.dumps(key, sort_keys=True):
            self.gen.load_state_dict(blob["state_dict"])
            print(f"{tag} reusing cached C1/C2-trained CMG from {path}")
        else:
            if blob is not None:
                print(f"{tag} cached CMG is stale (config changed) — retraining")
            print(f"{tag} training CMG on C1/C2 losses over {num_samples} states "
                  f"({epochs} epochs) — cached to {path} for later runs")
            train_cmg_ccm(
                self.gen, env.get_f_and_B, env.get_rollout,
                x_dim=scen.x_dim, u_dim=scen.u_dim, lbd=self.lbd,
                w_lb=self.w_lb, w_ub=self.w_ub, eps=self.eps,
                epochs=epochs, lr=lr, batch_size=batch_size,
                num_samples=num_samples,
                lr_scheduler=cmg.get("cmg_regress_lr_scheduler", ""),
                lr_scheduler_kwargs=cmg.get("cmg_regress_lr_scheduler_kwargs"),
                device=device, tag=tag,
                val_frac=float(cmg.get("cmg_val_frac", 0.1)),
                early_stop_patience=int(cmg.get("cmg_early_stop_patience", 10)),
                random_ratio=random_ratio,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": json.dumps(key, sort_keys=True),
                        "state_dict": self.gen.state_dict()}, path)
        for p in self.gen.parameters():
            p.requires_grad_(False)
        self.gen.eval()
        self._x_dim = scen.x_dim
        self.name = "ccm (CMG trained on C1/C2 losses)"

    @torch.no_grad()
    def M(self, x: torch.Tensor) -> torch.Tensor:
        raw_W, _ = self.gen(x)
        return spd_inverse(bound_W(raw_W, self.w_lb, self._x_dim, True))


def make_metric(kind: str, env, scen: Scenario, *, metric_ckpt: str | None = None,
                metric_cfg: dict | None = None, solver: str | None = None,
                cmg_samples: int = 16384, ccm_samples: int | None = None,
                cache_dir: str | None = None, random_seed: int = 0):
    """Build a conditioning metric. Each kind is defined by the objective its CMG
    minimizes, not by whichever checkpoint happens to be on disk — that is what
    makes the geometries a comparison of synthesis formulations:

      * "ccm"              — CMG trained to minimize the C1/C2 contraction losses
                             (train_cmg_ccm; no SDP, no regression). Batched.
      * "cvstem_pretrained"— CMG trained to minimize MSE regression loss onto
                             CV-STEM SDP solutions (build_cm_dataset +
                             regress_cmg). Batched.
      * "cvstem_online"    — no CMG at all: the CV-STEM SDP re-solved at every
                             state. Not batched.
      * "random"           — untrained CMG: "ccm"'s architecture, config and
                             w_lb/w_ub bounds with no training at all. The
                             control baseline: structure the trained metrics show
                             is only creditable to their objective insofar as it
                             exceeds this.

    ``metric_ckpt`` overrides "ccm" with a stored CMG instead. Use it knowingly:
    c3m.pt's CMG is trained jointly with its controller on
    ``pd_loss + c1_loss + c2_loss (+ os_loss)``, so it is co-adapted to that
    controller and is not a pure C1/C2 metric; c2rl_ppo.pt's is a real C1/C2 fit
    only when its config sets cmg_method=ccm.
    """
    env_name = getattr(env, "task", None)
    cache_root = cache_dir or os.path.join(_VIZ_DIR, "cache")
    if kind == "cvstem_pretrained":
        cfg = metric_cfg or load_algo_cfg(env_name, "cvstem_lqr")
        cache = os.path.join(cache_root, f"{env_name}_cvstem_cm_data.npz")
        return CVSTEMPretrainedMetric(env, scen, cfg, num_samples=cmg_samples,
                                      cache_path=cache, solver=solver)
    if kind in ("ccm", "random"):
        if metric_ckpt is not None:
            if not os.path.exists(metric_ckpt):
                raise FileNotFoundError(f"--metric-ckpt {metric_ckpt} does not exist")
            cfg = metric_cfg or load_algo_cfg(env_name, _algo_of_ckpt(metric_ckpt))
            return CCMMetric(metric_ckpt, cfg, scen,
                             random_init=(kind == "random"), seed=random_seed)
        # No checkpoint needed: the C1/C2 objective and the architecture both
        # come from the config, so the panel means what its name says regardless
        # of which runs happen to have been checkpointed into policies/.
        cfg = metric_cfg or load_algo_cfg(env_name, "c2rl_ppo")
        cache = os.path.join(cache_root, f"{env_name}_ccm_c1c2_cmg.pt")
        return CCMTrainedMetric(env, scen, cfg, num_samples=ccm_samples,
                                cache_path=cache,
                                random_init=(kind == "random"), seed=random_seed)
    if kind == "cvstem_online":
        cfg = metric_cfg or load_algo_cfg(env_name, "cvstem_lqr")
        return CVSTEMMetric(env, cfg, solver=solver)
    raise ValueError(f"metric must be one of {METRIC_KINDS}, got {kind!r}")


def _algo_of_ckpt(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base if base in ("c3m", "c2rl_ppo", "c2rl_sac") else "c3m"


def _metric_bounds(cfg: dict) -> tuple[float, float]:
    """w_lb/w_ub live under agent: for C3M configs and under cm: for C2RL ones."""
    agent, cm = cfg.get("agent", {}), cfg.get("cm", {})
    return (float(cm.get("w_lb", agent.get("w_lb", 0.1))),
            float(cm.get("w_ub", agent.get("w_ub", 10.0))))


# ─────────────────────────────────────────────────────────────────────────── #
# Config / checkpoint resolution
# ─────────────────────────────────────────────────────────────────────────── #

_CFG_FILENAMES = {
    "c3m": "skrl_c3m_cfg.yaml",
    "c2rl_ppo": "skrl_c2rl_ppo_cfg.yaml",
    "c2rl_sac": "skrl_c2rl_sac_cfg.yaml",
    "cvstem_lqr": "skrl_cvstem_lqr_cfg.yaml",
    "lqr": "skrl_lqr_cfg.yaml",
    "sdlqr": "skrl_sdlqr_cfg.yaml",
}


def load_algo_cfg(env_name: str, algo: str) -> dict:
    """Resolve an algorithm's yaml config: visualization/policies/<env>/<algo>.yaml
    first (the config stored next to the checkpoint — authoritative for actor
    dims), then the task's default agent yaml."""
    local = os.path.join(POLICIES_DIR, env_name, f"{algo}.yaml")
    if os.path.exists(local):
        with open(local) as f:
            return yaml.safe_load(f)
    task_yaml = os.path.join(
        _SRC, "c2rl", "tasks", "direct", "classic", env_name, "agents",
        _CFG_FILENAMES[algo],
    )
    if not os.path.exists(task_yaml) and algo == "cvstem_lqr":
        # Not every env ships a cvstem_lqr yaml; the c2rl_ppo config's cm: block
        # carries the same CV-STEM LMI knobs (lbd/w_lb/w_ub/cm_eps/cm_solver/
        # cvstem_r_scaler), which is all CVSTEMMetric reads.
        print(f"[viz] no {_CFG_FILENAMES[algo]} for {env_name}; using the "
              f"c2rl_ppo config's cm: block for the CV-STEM knobs")
        return load_algo_cfg(env_name, "c2rl_ppo")
    with open(task_yaml) as f:
        return yaml.safe_load(f)


def find_checkpoint(env_name: str, algo: str) -> str | None:
    p = os.path.join(POLICIES_DIR, env_name, f"{algo}.pt")
    return p if os.path.exists(p) else None


# ─────────────────────────────────────────────────────────────────────────── #
# Policies
# ─────────────────────────────────────────────────────────────────────────── #

def _build_actor_from_cfg(cfg: dict, env, scen: Scenario):
    """Instantiate the actor class the training run used — same backbone→class
    dispatch as the runners, reusing models.py classes so checkpoint keys match
    exactly. Returns (model, deterministic_action_fn)."""
    from c2rl.agents.skrl.models import (
        CLActorModel,
        CLDeterministicActorModel,
        MLPResidualActorModel,
        SquashedCLActorModel,
        SquashedCLDeterministicActorModel,
        SquashedMLPActorModel,
    )

    pol = cfg.get("models", {}).get("policy", {})
    klass = pol.get("class", "DeterministicMixin")
    backbone = pol.get("backbone", "control")
    net = (pol.get("network") or [{}])[0]
    kwargs = dict(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
        hidden_dim=list(net.get("layers", [128, 128])),
        activation=net.get("activations", "tanh"),
        x_dim=scen.x_dim,
        angle_idx=scen.angle_idx,
    )

    if klass == "DeterministicMixin":
        cls = (SquashedCLDeterministicActorModel if backbone == "control-squashed"
               else CLDeterministicActorModel)
        model = cls(**kwargs)
        def act(obs):
            return model.compute({"observations": obs})[0]
        return model, act

    # GaussianMixin — deterministic evaluation takes the mean action.
    if backbone == "control":
        kwargs.pop("x_dim")  # CLActorModel infers it; doesn't take the kwarg
        model = CLActorModel(**kwargs)
        def act(obs):
            return model.compute({"observations": obs})[0]
        return model, act
    if backbone == "mlp":
        kwargs.pop("x_dim")
        model = MLPResidualActorModel(**kwargs)
        def act(obs):
            return model.compute({"observations": obs})[0]
        return model, act
    if backbone in ("control-squashed", "mlp-squashed"):
        cls = SquashedCLActorModel if backbone == "control-squashed" else SquashedMLPActorModel
        model = cls(**kwargs)
        def act(obs):
            # act() applies the tanh-squash + post-squash uref residual;
            # mean_actions is its deterministic (noise-free) action.
            _, out = model.act({"observations": obs}, role="policy")
            return out["mean_actions"]
        return model, act
    raise ValueError(f"Unsupported policy backbone {backbone!r} (class {klass})")


def load_checkpoint_policy(env, scen: Scenario, env_name: str, algo: str):
    """Rebuild a trained deterministic policy (c3m / c2rl_ppo / c2rl_sac) from
    visualization/policies/<env>/<algo>.pt + its config."""
    ckpt_path = find_checkpoint(env_name, algo)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No {algo}.pt under {os.path.join(POLICIES_DIR, env_name)}/ — "
            f"copy a trained checkpoint there (see visualization/README.md).")
    cfg = load_algo_cfg(env_name, algo)
    model, act = _build_actor_from_cfg(cfg, env, scen)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(state["policy"])
    except RuntimeError as e:
        print(f"[viz] strict load of {algo}.pt policy failed ({e}); retrying strict=False")
        model.load_state_dict(state["policy"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        return act(obs)

    return policy_fn


def make_lqr_policy(env, scen: Scenario, env_name: str, *, at_state: bool = False):
    """LQR-family control law: A = ∂f/∂x|_z + Σⱼ urefⱼ·∂Bⱼ/∂x|_z, care gain,
    u = uref - K·e.

    ``at_state`` picks the linearisation point z, which is the only difference
    between the two agents this mirrors:
      * False → z = xref: classic LQR (sdlqr.LQRAgent). One fixed gain per
        reference point; the gain does not know where the state actually is.
      * True  → z = x: state-dependent LQR (sdlqr.SDLQRAgent). The gain is
        re-solved at the current state, so it stays valid far from the
        reference where the xref-linearisation has stopped describing the
        dynamics.
    """
    from scipy.linalg import solve_continuous_are

    cfg = load_algo_cfg(env_name, "sdlqr" if at_state else "lqr").get("agent", {})
    Q_s, R_s = float(cfg.get("Q_scaler", 1.0)), float(cfg.get("R_scaler", 0.0))
    d, m = scen.x_dim, scen.u_dim
    Q = (Q_s + 1e-5) * np.eye(d)
    R = (R_s + 1e-5) * np.eye(m)

    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        x = obs[:, :d]
        xref = obs[:, d:2 * d]
        uref = obs[:, 2 * d:2 * d + m]
        z = (x if at_state else xref).clone().requires_grad_()
        with torch.enable_grad():
            f, B, _ = env.get_f_and_B(z)
            DfDx = jacobian(f, z, create_graph=False)
            DBDx = b_jacobian(B, z, m, create_graph=False)
        A = (DfDx + torch.einsum("bxyu,bu->bxy", DBDx, uref)).detach()
        B = B.detach()
        e = wrap_diff(x - xref, scen.angle_idx)
        u = torch.empty(obs.shape[0], m)
        for i in range(obs.shape[0]):
            try:
                P = solve_continuous_are(A[i].numpy(), B[i].numpy(), Q, R)
                K = np.linalg.solve(R, B[i].numpy().T @ P)
            except (np.linalg.LinAlgError, ValueError):
                K = np.zeros((m, d))  # non-stabilizable here → zero feedback
            u[i] = uref[i] - torch.as_tensor(K, dtype=torch.float32) @ e[i]
        return u

    return policy_fn


def make_cvstem_lqr_policy(env, scen: Scenario, env_name: str, solver: str | None = None):
    """CV-STEM-LQR (online): u = uref - (1/r)·B(x)ᵀ·M(x)·e with M from the
    per-state CV-STEM SDP — mirrors cvstem_lqr.CVSTEMLQRAgent's online path.
    No checkpoint needed."""
    cfg = load_algo_cfg(env_name, "cvstem_lqr")
    metric = CVSTEMMetric(env, cfg, solver=solver)
    r = metric.r_scaler + 1e-5
    d, m = scen.x_dim, scen.u_dim

    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        x = obs[:, :d]
        xref = obs[:, d:2 * d]
        uref = obs[:, 2 * d:2 * d + m]
        _, B, _ = env.get_f_and_B(x)
        M = metric.M(x)
        K = (1.0 / r) * torch.bmm(B.transpose(1, 2), M)
        e = wrap_diff(x - xref, scen.angle_idx).unsqueeze(-1)
        return uref - torch.bmm(K, e).squeeze(-1)

    policy_fn.metric = metric  # expose solve stats
    return policy_fn


def make_policy(name: str, env, scen: Scenario, env_name: str, solver: str | None = None):
    if name in ("c3m", "c2rl_ppo", "c2rl_sac"):
        return load_checkpoint_policy(env, scen, env_name, name)
    if name == "lqr":
        return make_lqr_policy(env, scen, env_name, at_state=False)
    if name == "sd_lqr":
        return make_lqr_policy(env, scen, env_name, at_state=True)
    if name == "cvstem_lqr":
        return make_cvstem_lqr_policy(env, scen, env_name, solver=solver)
    if name == "uref":
        return lambda obs: obs[:, 2 * scen.x_dim: 2 * scen.x_dim + scen.u_dim]
    raise ValueError(f"Unknown policy {name!r}")


# ─────────────────────────────────────────────────────────────────────────── #
# Rollout + error geometry
# ─────────────────────────────────────────────────────────────────────────── #

def _metric_error(e: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """eᵀMe for batched e (n, d) and M (n, d, d) → (n,)."""
    return torch.einsum("ni,nij,nj->n", e, M, e)


TRUNK_MODES = ("uref", "greedy", "cvstem_lqr", "lqr", "sd_lqr")


def trunk_states(env, scen: Scenario, mode: str = "cvstem_lqr", *, env_name: str | None = None,
                 metric=None, levels: torch.Tensor | None = None,
                 horizon: int = 1, solver: str | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """The state/control sequence the error geometry is built along ("the trunk").

    The trunk fixes where the landscape is sampled; the candidate grid swept at
    each trunk state is a throwaway branch that never feeds back. A real
    modelling choice, since the trunk decides which region you end up looking at:

      * ``uref`` — zero feedback. The only fully algorithm- And
        metric-independent mode, so all metrics are compared at identical states.
        Not the default because the error grows unchecked into a badly-tracked
        region no working controller would visit.
      * ``greedy`` — the grid control minimising the ``horizon``-step-ahead
        normalized error, i.e. the best a grid-confined controller could do under
        ``metric``. Metric-Dependent, so one designated metric must drive the
        trunk for all panels (error_geometry.py --trunk-metric) or the panels sit
        on different trajectories and differences stop being attributable to it.
      * ``cvstem_lqr`` (default) / ``lqr`` / ``sd_lqr`` — follow that analytical
        controller. Metric-independent, so panels still share states, and the
        trunk stays where a real controller goes. cvstem_lqr costs one SDP/step.

    Returns (x (T, x_dim), u (T-1, u_dim)); u[k] takes x[k] to x[k+1].
    """
    if mode not in TRUNK_MODES:
        raise ValueError(f"Unknown trunk mode {mode!r}; choose from {TRUNK_MODES}")
    if mode == "uref":
        xs = [scen.x0.unsqueeze(0)]
        for k in range(scen.T - 1):
            xs.append(step_dynamics(env, xs[-1], scen.uref[k].unsqueeze(0)))
        return torch.cat(xs, dim=0), scen.uref[: scen.T - 1].clone()

    if mode == "greedy":
        if metric is None or levels is None:
            raise ValueError("greedy trunk needs both `metric` and `levels`")
        cands = _candidate_controls(levels)
        V0 = _V0(scen, metric)   # same normalizer the landscapes use
        xs, us = [scen.x0.unsqueeze(0)], []
        for k in range(scen.T - 1):
            # Look `h` steps ahead but commit only one step: receding-horizon,
            # so the trunk is a real trajectory rather than a chain of stale
            # h-step plans.
            h = min(horizon, scen.T - 1 - k)
            r = _landscape_step(env, scen, metric, xs[-1].squeeze(0),
                                scen.uref[k], cands, k, V0, h)
            u_best = cands[int(torch.argmin(r))].unsqueeze(0)
            us.append(u_best)
            xs.append(step_dynamics(env, xs[-1], u_best))
        return torch.cat(xs, dim=0), torch.cat(us, dim=0)

    if env_name is None:
        raise ValueError(f"trunk mode {mode!r} needs `env_name` to load its config")
    policy_fn = make_policy(mode, env, scen, env_name, solver=solver)
    xs, us = [scen.x0.unsqueeze(0)], []
    for k in range(scen.T - 1):
        obs = torch.cat([xs[-1], scen.xref[k:k + 1], scen.uref[k:k + 1]], dim=-1)
        u = policy_fn(obs).detach()
        us.append(u)
        xs.append(step_dynamics(env, xs[-1], u))
    return torch.cat(xs, dim=0), torch.cat(us, dim=0)


def normalized_error(scen: Scenario, metric, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(r, r_euc) — normalized metric error and its Euclidean counterpart along x."""
    e = wrap_diff(x - scen.xref[: x.shape[0]], scen.angle_idx)
    M = metric.M(x) if metric.batched else torch.cat(
        [metric.M(x[k:k + 1]) for k in range(x.shape[0])], dim=0)
    V = _metric_error(e, M).clamp_min(1e-12)
    return torch.sqrt(V / V[0]), e.norm(dim=-1) / e[0].norm().clamp_min(1e-8)


def rollout(env, scen: Scenario, policy_fn, metric) -> dict:
    """Deterministic rollout of one policy along the frozen scenario.

    Returns x (T, x_dim), u (T-1, u_dim), r (T,) normalized metric error,
    r_euc (T,) normalized Euclidean error.
    """
    xs = [scen.x0.unsqueeze(0)]
    us = []
    for k in range(scen.T - 1):
        obs = torch.cat([xs[-1], scen.xref[k].unsqueeze(0), scen.uref[k].unsqueeze(0)], dim=-1)
        u = policy_fn(obs)
        us.append(u)
        xs.append(step_dynamics(env, xs[-1], u))
    x = torch.cat(xs, dim=0)                      # (T, x_dim)
    u = torch.cat(us, dim=0)                      # (T-1, u_dim)

    e = wrap_diff(x - scen.xref, scen.angle_idx)  # (T, x_dim)
    M = metric.M(x) if metric.batched else torch.cat(
        [metric.M(x[k:k + 1]) for k in range(scen.T)], dim=0)
    V = _metric_error(e, M).clamp_min(1e-12)
    r = torch.sqrt(V / V[0])
    r_euc = e.norm(dim=-1) / e[0].norm().clamp_min(1e-8)
    return {"x": x, "u": u, "r": r, "r_euc": r_euc}


def _candidate_controls(levels: torch.Tensor) -> torch.Tensor:
    """Per-dimension levels ``(u_dim, C)`` → the flat candidate set ``(n, u_dim)``:
    the C levels for u_dim==1, their full C×C cartesian product for u_dim==2.
    Shared by the landscapes and the greedy trunk so both search the same set.
    """
    if levels.shape[0] == 1:
        return levels[0].unsqueeze(-1)
    g0, g1 = torch.meshgrid(levels[0], levels[1], indexing="ij")
    return torch.stack([g0.reshape(-1), g1.reshape(-1)], dim=-1)


def _V0(scen: Scenario, metric) -> float:
    """e0ᵀM(x0)e0 — the normalizer shared by every landscape and rollout."""
    M0 = metric.M(scen.x0.unsqueeze(0))
    return float(_metric_error(scen.e0.unsqueeze(0), M0).clamp_min(1e-12))


def _propagate(env, x: torch.Tensor, u: torch.Tensor, horizon: int) -> torch.Tensor:
    """Hold ``u`` constant for ``horizon`` steps from ``x``."""
    for _ in range(horizon):
        x = step_dynamics(env, x, u)
    return x


def _landscape_step(env, scen: Scenario, metric, x_k: torch.Tensor,
                    u_applied_k: torch.Tensor, cands: torch.Tensor,
                    k: int, V0: float, horizon: int) -> torch.Tensor:
    """Normalized error after holding each candidate control for ``horizon``
    steps from ``x_k``, measured against ``xref[k+horizon]``.

    ``horizon`` exists because a one-step lookahead cannot show a basin here.
    The error-minimizing control over a lookahead of ``H`` steps must satisfy
    ``B·u ≈ -e/(H·dt)``, i.e. ``‖u*‖ ~ ‖e‖/(H·dt)``. At H=1 and dt=0.03 that is
    ~33‖e‖, far outside the actuator box for any realistic error, so "more
    feedback is always better" and every slice is a monotone wall — the control
    simply has no time to act. Raising H shrinks the required ‖u*‖ into the box
    and the true interior optimum becomes visible, without reintroducing any
    dependence on a controller.

    M is evaluated at each candidate's end-state when the metric is batched
    (CCM / pretrained CMG); for the per-state-SDP CV-STEM metric one M solved
    at the reference control's end-state is reused across that step's
    candidates (one cvxpy solve per timestep instead of per candidate).
    """
    n = cands.shape[0]
    x_end = _propagate(env, x_k.expand(n, -1), cands, horizon)
    e_end = wrap_diff(x_end - scen.xref[k + horizon], scen.angle_idx)
    if metric.batched:
        M = metric.M(x_end)
    else:
        base_end = _propagate(env, x_k.unsqueeze(0), u_applied_k.unsqueeze(0), horizon)
        M = metric.M(base_end).expand(n, -1, -1)
    return torch.sqrt(_metric_error(e_end, M).clamp_min(1e-12) / V0)


def landscape_steps(scen: Scenario, horizon: int) -> int:
    """Number of timesteps a landscape with this lookahead is defined over."""
    return max(1, scen.T - horizon)


def compute_landscape_1d(env, scen: Scenario, metric, x: torch.Tensor,
                         u: torch.Tensor, levels: torch.Tensor,
                         horizon: int = 1) -> np.ndarray:
    """u_dim == 1: the complete t × u error landscape → ``(C, K)``.

    ``heat[c, k]`` = normalized error after holding the single control
    ``levels[0, c]`` for ``horizon`` steps from the visited state ``x[k]``.
    With one input the swept axis is the whole control space, so this loses
    nothing.
    """
    C = levels.shape[1]
    K = min(u.shape[0], landscape_steps(scen, horizon))
    V0 = _V0(scen, metric)
    cands = _candidate_controls(levels)                  # (C, 1)
    heat = np.empty((C, K), dtype=np.float32)
    for k in range(K):
        heat[:, k] = _landscape_step(env, scen, metric, x[k], u[k], cands,
                                     k, V0, horizon).numpy()
    return heat


def compute_landscape_2d(env, scen: Scenario, metric, x: torch.Tensor,
                         u: torch.Tensor, levels: torch.Tensor,
                         frames: np.ndarray, horizon: int = 1) -> np.ndarray:
    """u_dim == 2: the complete u0 × u1 error landscape per frame → ``(C, C, F)``.

    ``heat[i, j, f]`` = normalized error after holding
    ``(levels[0, i], levels[1, j])`` for ``horizon`` steps from the visited
    state ``x[frames[f]]``. The full cartesian product of both inputs is
    evaluated, so each frame is the entire control plane at that timestep —
    again no projection, no loss.
    """
    C = levels.shape[1]
    V0 = _V0(scen, metric)
    cands = _candidate_controls(levels)                             # (C*C, 2)
    heat = np.empty((C, C, len(frames)), dtype=np.float32)
    for f, k in enumerate(frames):
        vals = _landscape_step(env, scen, metric, x[k], u[k], cands,
                               int(k), V0, horizon)
        heat[:, :, f] = vals.reshape(C, C).numpy()
    return heat
