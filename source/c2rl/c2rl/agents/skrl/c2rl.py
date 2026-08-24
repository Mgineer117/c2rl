"""C2RL — single-policy contraction-metric RL against a Neural Contraction
Metric (NCM) reward.

C2RLAgent trains one real skrl PPO/SAC sub-agent (per ``base_algorithm``)
against a Mahalanobis tracking reward

    ``tracking_scaler·(V_t - V_{t+1}) - control_scaler·||u||²``,   V = eᵀM(x)e

with ``e = x - xref``. The tracking term is the per-step decrease of ``V`` — a
contraction signal, not a level penalty — computed identically by both env
families. The control term is disabled (``control_scaler = 0``) in every
shipped config.

The env computes that reward from the frozen Phase-A CMG,
since the chosen object is injected into the env and called from its
``get_rewards()``:

  * "cmg" (default) — a CMG network ``W(x)`` synthesized in Phase A then frozen
    for the whole run (Tsukamoto's NCM recipe). Mandatory ``models["cmg"]``.


``cmg_method="ccm"`` trains the CMG directly with C1/C2 losses (no SDP),
so the pair is three real configurations, not four.

CMG training (``cmg_method``), both in ``ncm_synthesis.py``:

  * "cvstem" — sample ``cmg_memory_size`` states, solve one SDP per state for
    ``W*(x)`` (``build_cm_dataset``), MSE-regress the CMG onto the feasible
    ``{x -> W*}`` (``regress_cmg``). No differentiable certificate loss.
  * "ccm" — train the CMG directly on Manchester's C1 (contraction) and C2
    (killing) losses (``train_cmg_ccm``). No SDP, no regression.

Both require a ``BoundedCCM_Generator`` (hard eigenvalue bounds in the forward
pass, not a soft penalty); ``__init__`` raises otherwise, and ContractionRunner
always passes ``constrain_eigenvalues=True`` regardless of yaml.

States are drawn from the classic env's analytic state space (``get_rollout``,
unlimited) or, with ``dynamics_pretrain_data_path`` set, from that offline
``dynamics_data.npz`` (capped to its size, warning if it's short).

Phase b is identical across all three configurations: each hands off one object
with the same ``W, _ = metric(x)`` contract. The reward is never overwritten
agent-side — ``_inject_ccm`` puts that object in the env via ``set_ccm``, and
the env's own ``get_rewards()`` returns the Mahalanobis reward natively. So what
reaches ``record_transition`` (and thus PPO's GAE / SAC's replayed critic
target) is already correct, with no per-algorithm reward plumbing to keep in
sync. ``_inject_ccm`` raises if no env accepted it — silently missing it would
mean training on the plain baseline reward.

Normalization: the metric and reward always use raw observations. ``M(x)`` and
``e = x - xref`` are defined in physical coordinates, and per-dimension
normalization would scale ``x`` and ``xref`` independently, distorting ``e``.
``uref`` and ``angle_idx`` columns are likewise excluded (see
``rl_glue.make_base_rl_cfg``). ``use_state_norm`` is False in every config.

Learned dynamics (Isaac / ``use_empirical_dynamics``): a ``NeuralDynamics``
model supplies f/B/B_null and ∂f/∂x, pretrained once before Phase A. Under
"cmg" that is its only consumer (synthesis needs them; Phase B never touches it
again, the CMG being frozen). Under "online" the opposite — queried every step.
Classic envs use analytical ``get_f_and_B`` and skip pretraining.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tqdm as _tqdm
from skrl.agents.torch.base import Agent, AgentCfg
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.base import Trainer, TrainerCfg

from .math_utils import build_lr_scheduler
from .ref_window import RefWindow
from .rl_glue import filter_cfg_fields, make_base_rl_cfg

# ─────────────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────────────── #

# NOTE: base_algorithm is not a field on either cfg below — it's an explicit
# C2RLAgent constructor kwarg (see ContractionRunner._setup_c2rl), set by which
# entry point you use (skrl_c2rl_ppo_cfg.yaml / skrl_c2rl_sac_cfg.yaml). Each
# cfg's base-algorithm fields mirror the real skrl PPO_CFG/SAC_CFG field names
# 1:1, so a c2rl-sac.yaml actually validates against SAC's own parameter names.
# make_base_rl_cfg() still reads from the raw yaml dict (self._raw_cfg) and
# filters against whichever of PPO_CFG/SAC_CFG applies, so any valid field works
# from yaml even if not declared below.

@dataclass
class C2RLPPOCfg(AgentCfg):
    """C2RL config for base_algorithm="PPO". PPO fields mirror skrl's PPO_CFG."""
    # PPO shared config (see skrl.agents.torch.ppo.PPO_CFG) — the deployed
    # policy's own PPO sub-agent is built from this.
    rollouts: int = 16
    learning_epochs: int = 8
    mini_batches: int = 2
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 2.5
    kl_threshold: float = 0.0
    grad_norm_clip: float = 0.5
    time_limit_bootstrap: bool = False
    use_state_norm: bool = False  # off by default — see module docstring / rl_glue.make_base_rl_cfg
    use_value_norm: bool = True
    use_reward_norm: bool = False  # non-biasing running-std reward normalizer (r/std) — see rl_glue.make_base_rl_cfg
    rewards_shaper_scale: float = 1.0  # yaml convenience for PPO_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    # True (default): log_std follows the fixed schedule in
    # std_dev_annealing_kwargs and is not learned (CLActor builds it with
    # requires_grad=False and the patch overwrites it every step).
    # False: PPO learns log_std through the policy loss, as in stock PPO. The
    # schedule then does not apply, and entropy_loss_scale is left as configured
    # (the annealing patch zeroes it, since a bonus would fight the schedule).
    # Off: PPO learns log_std (the False branch in __init__ flips requires_grad
    # so it is a real parameter, not a frozen constant). Annealing drives sigma
    # down a fixed schedule regardless of what the policy needs, which fights
    # the exploration a swept discount factor requires -- a gamma=0.999 run and
    # a gamma=0.01 run want very different exploration at the same step count.
    std_dev_annealing: bool = False
    # CAPS action-smoothness regularization on the policy loss (both 0 = off).
    # Not a reward term: it leaves the MDP, the observation and the dynamics
    # untouched, so the offline CV-STEM synthesis stays valid. See
    # agent_patches.patch_caps_regularizer.
    caps_temporal_scale: float = 0.0  # weight on ||pi(s_t) - pi(s_{t+1})||^2
    caps_spatial_scale: float = 0.0   # weight on ||pi(s) - pi(s_bar)||^2, s_bar ~ N(s, sigma^2)
    caps_spatial_std: float = 0.05    # sigma, in raw observation units (see patch_caps_regularizer)
    # Set by ContractionRunner from the yaml `memory:` block's memory_size, not
    # read from `agent:` directly; declared purely so filter_cfg_fields()
    # recognizes it instead of warning.
    memory_size: int = -1
    # Deployed policy's discount factor — a single policy trained against the
    # Mahalanobis reward, so there is no con/opt duality here.
    discount_factor: float = 0.99
    # ── Metric source: Where the env's Mahalanobis reward gets M(x) ──────── #
    # "cmg" (default): a CMG network synthesized in Phase A (cmg_method selects
    # how) and frozen for the whole run. "online": no Phase A at all — solve the
    # CV-STEM SDP per env per step at the visited states
    # feasible metric instead of a regression of one, at the cost of num_envs
    # SDP solves per step. cmg_method="ccm" forces "cmg" (there is no per-state
    # SDP to solve online under the C1/C2 pipeline).
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2  # contraction rate λ — used by both cmg_method's synthesis loss
    # ── SDP contraction metric ("cvstem" cmg_method only) — see ncm_synthesis.py ── #
    cm_eps: float = 1e-2   # strict-definiteness margin on the contraction LMI (both methods)
    cm_solver: str = "SCS"  # cvxpy SDP solver ("cvstem" only)
    # "ccm" — C1/C2 loss minimization (train_cmg_ccm): Manchester-style,
    # eliminates B via the annihilator, existence-only certificate, no SDP, pure
    # gradient descent on the pointwise LMI. "cvstem" (default) — CV-STEM
    # regression (build_cm_dataset + regress_cmg): solves one joint SDP that
    # keeps B via a Riccati BR⁻¹Bᵀ term, then MSE-regresses the CMG onto the
    # solutions. See ncm_synthesis.py module docstring for the LMIs and module
    # docstring above for the two pipelines.
    cmg_method: str = "cvstem"
    # R = cvstem_r_scaler·I in the BR⁻¹Bᵀ Riccati term (mirrors sdlqr.py's
    # R_scaler); "cvstem" method only. See ncm_synthesis.cvstem_joint — control
    # enters the LMI only through this penalty, not a bounded control box.
    cvstem_r_scaler: float = 1.0
    # Residual RL over the analytic controller. When True, the actor mean is
    # u = CV-STEM-LQR(K=R⁻¹BᵀW⁻¹)·baseline + learned bilinear residual (residual
    # zero-initialized) — the policy starts at the certified contraction controller
    # (the same frozen CMG from Phase A) and PPO learns only the correction. Needs
    # the 'control' backbone (a CLActor policy). See nn_modules.CVSTEMLQRBase.
    cvstem_residual_base: bool = False
    # Inert since the actor architecture was fixed to W2(xrefs)·tanh(W1(x)·e).
    # Kept so an older yaml carrying these keys still parses (a key the dataclass
    # does not declare is silently dropped — see rl_glue.filter_cfg_fields).
    # ── superseded ──
    # π-network variant: make the residual a pure anticipatory feedforward head
    # (an MLP of the reference preview [uref, future-uref] only) instead of the
    # bilinear e-feedback. Lq-preview-optimal control = feedback(−K·e) [the base]
    # + feedforward(future reference) [this head]. Being independent of e, it can
    # Add the anticipation the myopic CV-STEM-LQR base lacks without reshaping (and
    # corrupting) its certified feedback — the failure mode of the bilinear
    # residual (∝ e), which PPO drifts to AUC 1.19. Needs cvstem_residual_base +
    # a reference preview (--num_preview>0). Trained on the Mahalanobis reward at a
    # high discount_factor so anticipation is actually rewarded. See
    # models.CLActorModel.ff_head / compute.
    residual_feedforward: bool = False
    # π-network architecture for residual RL on the CV-STEM-LQR base. Selects how
    # the learned residual is generated from the current state x, the reference
    # preview P, and the invariant error e (all zero-init → start at the base):
    #   "bilinear"    — original CLActor W₂(x,xref,P)·tanh(W₁(x,xref,P)·e) (blended)
    #   "feedforward" — ff(P) only, independent of e (== residual_feedforward)
    #   "decoupled"   — W₂(P)·tanh(W₁(x)·e)                     (physical/geometric split)
    #   "latent_bias" — W₂(x)·tanh(W₁(x)·e + f_prev(P))         (recommended)
    #   "film"        — (W₂(x)⊙γ₂(P))·tanh((W₁(x)⊙γ₁(P))·e)     (gain scheduling)
    # See nn_modules DecoupledResidual / LatentPreviewResidual / FiLMResidual.
    residual_variant: str = "bilinear"
    # Contraction pretraining of π before PPO (the fix for u = uref + π, which
    # otherwise can't learn a stabilizing feedback from scratch — it diverges on
    # high-dim systems). Minimizes the C3M closed-loop contraction condition
    # Cu = Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0 (the C3M pd_loss) over π only, against the
    # Frozen Phase-A CMG metric M(x) — a contraction certificate learned from the
    # metric geometry, not from CV-STEM-LQR. Puts PPO in the basin of a stabilizing
    # controller so it refines rather than has to discover one. Preview is zeroed
    # during pretraining (certify the stabilizing feedback; PPO adds anticipation).
    # See C2RLAgent._pretrain_residual_contraction.
    residual_contraction_pretrain: bool = False
    residual_pretrain_epochs: int = 200
    residual_pretrain_lr: float = 1.0e-3
    residual_pretrain_batch: int = 1024
    residual_pretrain_samples: int = 65536
    # Which objective _pretrain_residual_contraction uses: "contraction" (default,
    # legacy) minimizes the C3M Cu≺0 SDP violation above; "cvstemlqr" instead
    # directly MSE-regresses u = uref + π(s) onto the analytic CV-STEM-LQR control
    # law CVSTEMLQRBase(x,xref,uref) computed from the same frozen Phase-A metric.
    # This is plain supervised regression against a fixed, deterministic target —
    # well-posed (drives to ~0 MSE) unlike the SDP violation loss, which plateaus
    # well above 0 (observed ~2.6-3.3 across seeds) and correlates with how much
    # residual overshoot/AUC variance each seed ends up with. The analytic base is
    # Never attached to the policy here (policy.base_controller stays unset) — the
    # deployed control law is always u = uref + π, exactly as in "contraction"
    # mode; CVSTEMLQRBase is used only to generate a regression target during this
    # pretraining phase. See C2RLAgent._pretrain_residual_cvstemlqr.
    residual_pretrain_method: str = "contraction"
    # Clamp the "cvstemlqr" regression target to the box the env actually
    # applies, before the MSE.
    #
    # At cvstem_r_scaler=0.01, K = R⁻¹BᵀM gives ‖K‖₂ ≈ 53 (measured, cached car
    # CM dataset), so over the ±1 XE_INIT box the feedback term has median
    # magnitude ≈ 23/component while env_base.step clamps to ±6. 86% of target
    # components and 98% of target samples land outside the box.
    #
    # Regressing π onto that unclamped target costs twice: (1) most capacity and
    # residual MSE live in a range the plant can't realize, which is why MSE
    # plateaus well above 0 at any epoch count; (2) the pretrained policy
    # deploys deep in saturation, so ∂(applied u)/∂π = 0 almost everywhere —
    # PPO's task gradient vanishes while CAPS and the residual anchor (acting on
    # π's raw o(20) output) do not. The annealed std (exp(-3) ≈ 0.05) can't
    # escape an O(20) mean, so each seed's saturation pattern is frozen in from
    # pretraining. A direct mechanism for seed-wise AUC spread.
    #
    # Clamping makes the regression well-posed against the executable control
    # law without re-solving the SDP (unlike raising cvstem_r_scaler, which
    # changes the metric and invalidates the cached dataset). Pretraining only;
    # the deployed law is unchanged.
    residual_pretrain_clamp_target: bool = False
    # ── Phase-0 single-update-collapse diagnostics/ablations ──────────────── #
    # See project memory on the single-update PPO collapse: flat for ~2000
    # steps, then one destructive update, then permanent oscillation, even
    # though a contraction/cvstem-lqr pretrain already left π at (near-)optimal
    # mean actions. Three suspects, three knobs:
    #
    # Ablation A — exploration std is not pretrained (_pretrain_residual_* excludes
    # log_std from the trained params by design — the certificate is about the
    # deterministic control law), so Phase B can start with an init std comparable
    # to the full control range (car's control box is lim=1.0, default
    # initial_log_std=0.0 => std=1.0) even though the mean is already optimal —
    # the rollout data PPO trains its first updates on is then generated by a
    # near-random controller. Set this to override log_std_parameter to a small
    # value right after pretraining (see C2RLAgent._maybe_attach_residual_base);
    # None (default) leaves whatever the yaml's initial_log_std says. Needs
    # patch_ppo_std_annealing's lazy initial_log_std capture (see agent_patches.py)
    # to not be immediately overwritten on the first post_interaction call.
    residual_pretrain_init_log_std: float | None = None
    # Ablation B — critic cold-start: the critic starts random while the
    # near-optimal policy's true advantages are ~0, and GAE normalization divides
    # by that near-zero std, manufacturing a full-scale, confident gradient out
    # of noise. >0: after Phase A/residual-pretrain, roll out the frozen
    # pretrained actor in the real env for this many steps and MSE-regress the
    # critic onto the resulting (short-horizon, since discount_factor is small
    # here) Monte-Carlo returns before Phase B's PPO updates start. 0 (default)
    # = off (critic starts from its random init, as before). See
    # C2RLAgent.pretrain_critic / C2RLSkrlTrainer.train.
    pretrain_critic_steps: int = 0
    pretrain_critic_epochs: int = 200
    pretrain_critic_lr: float = 1.0e-3
    # Ablation C — PPO's GAE advantage normalization ((a - mean)/(std+eps))
    # amplifies exactly the near-zero-variance batches Ablation B's premise
    # describes into full-scale gradients. True: skip that normalization (train
    # on raw GAE advantages). See agent_patches.patch_ppo_diagnostics.
    disable_advantage_norm: bool = False
    # AUC-aligned reward: raw Euclidean error decrement (M = I) instead of the
    # frozen-CMG Mahalanobis one. Meant to be paired with cvstem_residual_base —
    # the contraction certificate is provided by the baseline, so the residual
    # optimizes the true tracking objective. See env_base.get_rewards.
    reward_euclidean: bool = False
    # With reward_euclidean: Level form r=-‖e‖ (tightest AUC alignment) vs the
    # default decrement form. See env_base.get_rewards.
    reward_level: bool = False
    # Supervised warm-start of the residual so the actor reproduces the online
    # CV-STEM-LQR controller (per-state SDP metric W_online, which the frozen CMG
    # only approximates), then PPO fine-tunes from there. Reuses the Phase-A SDP
    # solutions — no new solves. Needs cvstem_residual_base + cmg_method=cvstem.
    # See C2RLAgent._distill_residual_from_online.
    cvstem_residual_distill: bool = False
    residual_distill_epochs: int = 300
    residual_frozen: bool = False  # ablation: keep residual at 0 (pure base)
    residual_anchor_scale: float = 0.0  # penalize ‖u-u_base‖² in reward (residual trust anchor)
    # Weights of the CV-STEM objective J = cm_chi_weight·χ + cm_nu_weight·ν, which
    # cvstem_joint always minimizes (Tsukamoto's classncm.cvstem0). χ and ν are
    # the metric's condition number and scale, and they are decision variables:
    # W̄ ⪰ I, W̄ ⪯ χI, deployed W = W̄/ν. "cvstem" method only.
    cm_chi_weight: float | None = None  # None → 1/lbd, mirroring Tsukamoto's chi/alp
    cm_nu_weight: float = 1.0           # his d2_over
    # If > 0, include Tsukamoto's Ẇ ≈ (W̄ - I)/dt proxy for the material derivative
    # (classncm.cvstem0 puts the integration step here). 0 = omit it, which is what
    # the pointwise-per-state design otherwise forces (no neighbouring sample to
    # difference against — see ncm_synthesis.py's module docstring). "cvstem" method
    # only; superseded by cm_wdot_trajectory when that's on.
    cm_wdot_dt: float = 0.0
    # Real Ẇ from offline reference trajectories ("cvstem" method only): instead of
    # dropping Ẇ or using the static cm_wdot_dt proxy above, sample states as
    # trajectory-ordered chunks from dynamics_pretrain_data_path (which must be
    # set — raises otherwise) and difference each state's solved normalized W̄
    # against the actual previous state's along that same reference trajectory —
    # Ẇ ≈ (W̄_t − W̄_{t−1})/cm_temporal_dt, the real material derivative rather
    # than an approximation (see ncm_synthesis.build_cm_dataset's
    # traj_x/traj_lengths/temporal_dt and dynamics_pretrain.load_offline_trajectories).
    # Incompatible with cmg_random_ratio>0 (mixing in i.i.d. random states would
    # break trajectory continuity — ignored when this is on) and with
    # cmg_method="ccm" (train_cmg_ccm has no per-state SDP to add Ẇ to; raises).
    cm_wdot_trajectory: bool = False
    # Integration step between consecutive states in the offline trajectory data —
    # Not auto-derived from the env (dynamics_data.npz doesn't record it); set it
    # to the same dt used to generate that file (scripts/skrl/train.py's
    # _generate_ref_trajs: env_cfg.sim.dt * env_cfg.decimation). Only read when
    # cm_wdot_trajectory=True.
    cm_temporal_dt: float = 0.05
    # On SDP infeasibility at a state, retry that state alone with λ halved,
    # up to this many times, before giving up on it (0 = old behavior, drop
    # Guards build_cm_dataset against silently regressing the CMG onto a small,
    # likely-biased subset of states — raises before regression if the SDP's
    # feasible fraction falls below this (0.0 = old behavior, only guards
    # against 0% feasible; see ncm_synthesis.build_cm_dataset). "cvstem" method only.
    # Cache path for the synthesized {x, W} CM dataset (build_cm_dataset's
    # expensive per-state SDP solve) — see synthesize_cmg. Loaded instead of
    # re-solving when it exists and matches lbd/w_lb/w_ub/cm_eps/cm_solver/
    # cmg_memory_size exactly; written after a fresh solve otherwise. Defaults
    # to a `cm_data.npz` next to dynamics_pretrain_data_path when unset (Isaac
    # envs); classic envs with no data_path need this set explicitly to get
    # caching at all (there's no offline dynamics file to derive a path from).
    # "cvstem" method only.
    cm_data_path: str = ""
    # ── Offline CMG synthesis (Phase A, always runs before Phase B) ─────────── #
    # Sample cmg_memory_size states — uniformly from the classic env's analytic
    # state space (get_rollout) or, when dynamics_pretrain_data_path is set,
    # uniformly from that offline dynamics_data.npz (capped + warned if
    # cmg_memory_size exceeds the data on disk; see synthesize_cmg). "cvstem":
    # solve ONE joint SDP over all samples (cvstem_joint, reusing lbd/w_lb/w_ub/cm_eps/
    # cm_solver above), then MSE-regress the CMG network onto {x -> W*} for
    # cmg_regress_epochs (build_cm_dataset / regress_cmg). "ccm": train the CMG
    # directly with C1/C2 losses for cmg_regress_epochs, no SDP (train_cmg_ccm).
    # Either way the CMG is frozen (freeze_cmg) before Phase B.
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 1000
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    # Held out from cmg_memory_size as a validation split never regressed on;
    # regress_cmg stops once its MSE hasn't improved for cmg_early_stop_patience
    # consecutive epochs, restoring the best-val-epoch CMG weights instead of
    # whatever cmg_regress_epochs happens to land on (see ncm_synthesis.regress_cmg
    # / math_utils.EarlyStopper). <=0 disables both (always regress the full budget).
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Fraction (0..1) of the CMG-dataset states drawn from the broad/off-reference
    # distribution (states an early chaotic policy actually visits) rather than the
    # reference-trajectory tube — the rest are reference states. 0 = old behavior
    # (all reference, or all of the offline pool). Random states come from the
    # offline dynamics-pretrain pool if configured, else get_rollout("dynamics")
    # (uniform state-space coverage). See ncm_synthesis._sample_cm_states.
    cmg_random_ratio: float = 0.0
    # Dynamics — learned NeuralDynamics (ẋ = f(x) + B(x)·u) unless
    # use_empirical_dynamics=True (classic envs only). Feeds Phase A's CMG
    # synthesis (SDP dataset for "cvstem", C1/C2 gradient computation for "ccm").
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""
    # Fixed pretraining buffer size — sampled once (offline-data subsample when
    # dynamics_pretrain_data_path is set, else a fresh get_rollout draw), then
    # multi-epoch trained over, mirroring cmg_memory_size's role in CMG
    # synthesis (see dynamics_pretrain.pretrain_dynamics). Classic envs can
    # feasibly use any size (synthetic analytic sampling); Isaac envs with a
    # data_path are capped (+ warned) to the offline data actually on disk.
    emp_dynamics_memory_size: int = 8192
    # Classic envs only: how many distinct control vectors get paired with each
    # sampled state in a "dynamics" rollout (env_base.get_rollout); replaces
    # the old hardcoded 3.
    num_controls_per_state: int = 3
    # Held out from emp_dynamics_memory_size as a validation split never
    # trained on; pretrain_dynamics stops once its MSE hasn't improved for
    # dynamics_early_stop_patience consecutive epochs, restoring the
    # best-val-epoch NeuralDynamics weights instead of whatever
    # dynamics_pretrain_epochs happens to land on (see
    # dynamics_pretrain.pretrain_dynamics / math_utils.EarlyStopper). <=0
    # disables both (always pretrain the full budget).
    dynamics_val_frac: float = 0.1
    dynamics_early_stop_patience: int = 10


@dataclass
class C2RLSACCfg(AgentCfg):
    """C2RL config for base_algorithm="SAC". SAC fields mirror skrl's SAC_CFG."""
    # Not a real SAC_CFG field, and not the replay buffer size (that's
    # memory_size): it's how many env steps C2RLSkrlTrainer takes per outer
    # iteration, i.e. the cadence of the outer agent's checkpoint/flush pass.
    # SAC still updates every single step within that chunk.
    rollouts: int = 16
    gradient_steps: int = 1
    batch_size: int = 64
    polyak: float = 0.005
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict = field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.0
    learn_entropy: bool = True
    initial_entropy_value: float = 0.2
    use_state_norm: bool = False  # off by default — see module docstring / rl_glue.make_base_rl_cfg
    use_reward_norm: bool = False  # non-biasing running-std reward normalizer (r/std) — see rl_glue.make_base_rl_cfg
    rewards_shaper_scale: float = 1.0  # yaml convenience for SAC_CFG's rewards_shaper — see rl_glue.make_base_rl_cfg
    std_dev_annealing_kwargs: dict | None = None  # forwarded to patch_ppo_std_annealing()
    # CAPS action-smoothness regularization — see C2RLPPOCfg / patch_caps_regularizer.
    caps_temporal_scale: float = 0.0
    caps_spatial_scale: float = 0.0
    caps_spatial_std: float = 0.05
    memory_size: int = -1
    discount_factor: float = 0.99
    # ── Metric source — "cmg" frozen CMG network (cmg_method selects how it's
    # trained). See C2RLPPOCfg for the shared cm/cmg knobs. ─────────────────── #
    w_ub: float = 10.0
    w_lb: float = 0.1
    tracking_scaler: float = 1.0
    control_scaler: float = 0.0
    lbd: float = 1e-2
    # ── SDP contraction metric ("cvstem" method only) — see ncm_synthesis.py ── #
    cm_eps: float = 1e-2
    cm_solver: str = "SCS"
    cmg_method: str = "cvstem"  # "ccm" (C1/C2 minimization) | "cvstem" (SDP regression, default) — see module docstring
    cvstem_r_scaler: float = 1.0
    cvstem_residual_base: bool = False  # actor = CV-STEM-LQR baseline + learned residual (see C2RLPPOCfg)
    reward_euclidean: bool = False       # AUC-aligned Euclidean-decrement reward (see C2RLPPOCfg)
    reward_level: bool = False           # Level (r=-‖e‖) vs decrement euclidean reward (see C2RLPPOCfg)
    cvstem_residual_distill: bool = False # warm-start residual to online CV-STEM-LQR (see C2RLPPOCfg)
    residual_distill_epochs: int = 300
    # Weights of the CV-STEM objective J (always minimized) — see
    # C2RLPPOCfg.cm_chi_weight above and ncm_synthesis.cvstem_joint.
    cm_chi_weight: float | None = None
    cm_nu_weight: float = 1.0
    cm_wdot_dt: float = 0.0  # superseded by cm_wdot_trajectory when that's on
    # Real Ẇ from offline reference trajectories — see C2RLPPOCfg.cm_wdot_trajectory
    # / cm_temporal_dt above.
    cm_wdot_trajectory: bool = False
    cm_temporal_dt: float = 0.05
    cm_data_path: str = ""
    # ── Offline CMG synthesis (Phase A, always runs before Phase B) ─────────── #
    cmg_memory_size: int = 8192
    cmg_regress_epochs: int = 1000
    cmg_regress_lr: float = 1e-3
    cmg_regress_lr_scheduler: str = ""
    cmg_regress_lr_scheduler_kwargs: dict = field(default_factory=dict)
    cmg_regress_batch_size: int = 1024
    cmg_val_frac: float = 0.1
    cmg_early_stop_patience: int = 10
    # Random/off-reference state fraction for the CMG dataset — see
    # C2RLPPOCfg.cmg_random_ratio above / ncm_synthesis._sample_cm_states.
    cmg_random_ratio: float = 0.0
    # Dynamics
    use_empirical_dynamics: bool = False
    dynamics_lr: float = 1e-3
    dynamics_lr_scheduler: str = ""
    dynamics_lr_scheduler_kwargs: dict = field(default_factory=dict)
    dynamics_batch_size: int = 4096
    dynamics_pretrain_epochs: int = 5
    dynamics_pretrain_data_path: str = ""
    emp_dynamics_memory_size: int = 8192
    num_controls_per_state: int = 3
    dynamics_val_frac: float = 0.1
    dynamics_early_stop_patience: int = 10


@dataclass
class C2RLTrainerCfg(TrainerCfg):
    timesteps: int = 300000  # deployed-policy (RL) env steps; the offline CMG-synthesis
                             # phase (Phase A) runs once before this loop


def cm_dataset_target(cfg) -> tuple[Path | None, dict]:
    """Where this config's offline ``{x → W*(x)}`` dataset lives, and its key.

    The single source of truth for both sides of the contract: the agent
    (``_synthesize_cmg_cvstem``) loads from here, and
    ``scripts/build_cm_dataset.py`` writes here. Deriving the path or the
    ``cache_kwargs`` twice is how a generator silently produces a file the agent
    then key-misses and refuses — the two must be computed by one function.

    Returns ``(path, cache_kwargs)``; ``path`` is None when the config names no
    dataset location at all, which the caller must treat as an error.
    """
    from .ncm_synthesis import cm_dataset_cache_path, cm_dataset_filename

    data_path = getattr(cfg, "dynamics_pretrain_data_path", "") or None
    explicit_cache_path = getattr(cfg, "cm_data_path", "") or None
    # cm_wdot_trajectory (see C2RLPPOCfg's docstring): real Ẇ from offline
    # Reference trajectories instead of dropping it or Tsukamoto's static
    # cm_wdot_dt proxy. random_ratio is meaningless there (the whole point is
    # Not mixing in i.i.d. states) so it is forced to 0 in the cache key too.
    wdot_trajectory = bool(getattr(cfg, "cm_wdot_trajectory", False))
    temporal_dt = cfg.cm_temporal_dt if wdot_trajectory else 0.0
    random_ratio = 0.0 if wdot_trajectory else getattr(cfg, "cmg_random_ratio", 0.0)

    if explicit_cache_path:
        # An explicit cm_data_path is treated as a base name: the swept SDP
        # knobs are appended (cm_dataset_filename) so different lbd/w_lb/w_ub
        # runs never clobber or wrongly reuse each other's cache.
        base = Path(explicit_cache_path)
        cache_path = base.with_name(cm_dataset_filename(
            cfg.lbd, cfg.w_lb, cfg.w_ub, cfg.cvstem_r_scaler, stem=base.stem))
    elif data_path:
        cache_path = cm_dataset_cache_path(
            data_path, lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub,
            r_scaler=cfg.cvstem_r_scaler)
    else:
        cache_path = None

    cache_kwargs = dict(
        lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
        solver=cfg.cm_solver, num_samples=cfg.cmg_memory_size, tag="[C2RL]",
        r_scaler=cfg.cvstem_r_scaler,
        chi_weight=cfg.cm_chi_weight,
        nu_weight=cfg.cm_nu_weight, wdot_dt=cfg.cm_wdot_dt,
        random_ratio=random_ratio,
        wdot_trajectory=wdot_trajectory, temporal_dt=temporal_dt,
    )
    return cache_path, cache_kwargs


# ─────────────────────────────────────────────────────────────────────────── #
# Agent
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLAgent(Agent):
    """C2RL agent — native skrl Agent, single deployed policy.

    Models in ``models`` dict:
      ``"policy"``   — the real deployed SAC/PPO policy.
      ``"value"`` (PPO) or ``"critic_1"``/``"critic_2"``/``"target_critic_1"``/
        ``"target_critic_2"`` (SAC) — the deployed policy's own critic(s).
      ``"dynamics"`` — optional NeuralDynamics (use_empirical_dynamics=True).
      ``"cmg"``      — required. MetricModel (``BoundedCCM_Generator``, i.e.
        ``constrain_eigenvalues=True``) synthesized offline (see synthesize_cmg)
        and frozen before Phase B; read for the Mahalanobis reward.

    Extra constructor kwargs: ``get_rollout``, ``get_f_and_B``, ``x_dim``,
    ``u_dim``, ``num_envs``, ``angle_idx``, ``base_algorithm``.
    """

    def __init__(
        self,
        *,
        cfg: C2RLPPOCfg | C2RLSACCfg | dict,
        models: dict,
        memory=None,
        observation_space,
        state_space=None,
        action_space,
        device,
        get_rollout: Callable,
        get_f_and_B: Callable | None = None,
        base_algorithm: str = "PPO",
        x_dim: int | None = None,
        u_dim: int | None = None,
        num_envs: int = 1,
        angle_idx: list | None = None,
    ) -> None:
        self._angle_idx = list(angle_idx or [])
        CfgCls = C2RLSACCfg if base_algorithm.upper() == "SAC" else C2RLPPOCfg
        if isinstance(cfg, dict):
            self._raw_cfg = cfg.copy()
            parsed_cfg = CfgCls(**filter_cfg_fields(cfg, CfgCls, context="C2RLAgent"))
        else:
            self._raw_cfg = cfg.__dict__.copy()
            parsed_cfg = cfg

        super().__init__(
            cfg=parsed_cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        # The observation space declares its layout ({x, xrefs, urefs}), so
        # x_dim/u_dim are read, never guessed from obs_dim (the old
        # (obs_dim - u_dim)//2 parity rule silently mis-split any other layout).
        self._window = RefWindow.from_space(observation_space)


        if u_dim is None:
            u_dim = self._window.u_dim
        if x_dim is None:
            x_dim = self._window.x_dim

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._device = device
        self._cfg = parsed_cfg
        self._base_algorithm = base_algorithm.upper()
        self._num_envs = num_envs
        self._get_rollout = get_rollout

        # ── Metric source setup ──────────────────────────────────────────── #
        # This agent always owns the (optional) learned dynamics directly — Phase
        # A's CMG synthesis needs f/B/∂f/∂x (SDP dataset for "cvstem", C1/C2
        # gradient computation for "ccm"). The CMG network (models["cmg"]) is
        # fit offline (see C2RLSkrlTrainer.train / synthesize_cmg) and frozen
        # before Phase B — see module docstring.
        self._setup_dynamics(parsed_cfg, models, get_f_and_B, x_dim, u_dim, device)
        if "cmg" not in models:
            raise ValueError(
                "[C2RL] models['cmg'] is required — C2RL always synthesizes a CMG "
                "network offline before Phase B (see module docstring)."
            )
        self._ccm_gen = models["cmg"].ccm_gen
        # Kept so Phase A can attach the CV-STEM-LQR residual baseline to it
        # (cvstem_residual_base). Same object as self._rl_agent.models["policy"].
        self._policy_model = models["policy"]
        if not getattr(self._ccm_gen, "bounded", False):
            raise ValueError(
                "[C2RL] models['cmg'] must be a BoundedCCM_Generator "
                "(constrain_eigenvalues=True) — C2RL always hard-bounds the CMG's "
                "eigenvalues, regardless of cmg_method. Set "
                "models.cmg.network.constrain_eigenvalues: true in the yaml, or "
                "let ContractionRunner build it (it forces this)."
            )
        if bool(getattr(parsed_cfg, "cm_wdot_trajectory", False)):
            if parsed_cfg.cmg_method != "cvstem":
                raise ValueError(
                    "[C2RL] cm_wdot_trajectory=True needs cmg_method='cvstem' — "
                    "'ccm' (train_cmg_ccm) has no per-state SDP to add a Ẇ term to."
                )
            if not getattr(parsed_cfg, "dynamics_pretrain_data_path", ""):
                raise ValueError(
                    "[C2RL] cm_wdot_trajectory=True needs dynamics_pretrain_data_path "
                    "set to a trajectory-structured dynamics_data.npz (see "
                    "dynamics_pretrain.load_offline_trajectories) — there is no other "
                    "source of REAL trajectory order to difference Ẇ against."
                )

        # ── Phase B: a real skrl PPO/SAC agent for the deployed policy ───── #
        # PPO's memory holds exactly one on-policy rollout chunk. SAC's is a
        # persistent replay buffer, sized per-parallel-env (memory_size rows ×
        # num_envs) so Isaac Sim's 1000+ envs don't multiply skrl's usual ~1M
        # default up to an OOM; memory_size=-1 falls back to a modest default.
        _DEFAULT_SAC_BUFFER_ROWS = 10000
        if self._base_algorithm == "PPO":
            mem_size = parsed_cfg.rollouts
        else:
            mem_size = parsed_cfg.memory_size
            if mem_size == -1:
                mem_size = _DEFAULT_SAC_BUFFER_ROWS
        memory = RandomMemory(memory_size=mem_size, num_envs=num_envs, device=device)
        self._memory = memory

        rl_kwargs = dict(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        base_cfg = make_base_rl_cfg(
            self._raw_cfg,
            base_algorithm=self._base_algorithm,
            gamma=parsed_cfg.discount_factor,
            name="policy",
            experiment_dir=self.experiment_dir,
            device=device,
            observation_space=observation_space,
            angle_idx=self._angle_idx,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        if self._base_algorithm == "PPO":
            from skrl.agents.torch.ppo import PPO as BaseRLAgent
            rl_models = {"policy": models["policy"], "value": models["value"]}
        elif self._base_algorithm == "SAC":
            from skrl.agents.torch.sac import SAC as BaseRLAgent
            rl_models = {
                "policy": models["policy"],
                "critic_1": models["critic_1"], "critic_2": models["critic_2"],
                "target_critic_1": models["target_critic_1"], "target_critic_2": models["target_critic_2"],
            }
        else:
            raise ValueError(f"[C2RL] Unsupported base_algorithm: {self._base_algorithm}")

        self._rl_agent = BaseRLAgent(cfg=base_cfg, models=rl_models, memory=memory, **rl_kwargs)

        from c2rl.agents.skrl.agent_patches import (
            patch_caps_regularizer,
            patch_kl_logging,
            patch_ppo_diagnostics,
            patch_ppo_std_annealing,
            patch_sac_entropy_clamp,
        )
        patch_kl_logging(self._rl_agent)
        patch_ppo_diagnostics(
            self._rl_agent,
            disable_advantage_norm=bool(getattr(parsed_cfg, "disable_advantage_norm", False)),
        )
        patch_sac_entropy_clamp(self._rl_agent)
        # Applied to the inner PPO/SAC sub-agent: C2RL's outer agent has no
        # .policy/.scaler of its own for the patch to hook.
        patch_caps_regularizer(
            self._rl_agent,
            temporal_scale=parsed_cfg.caps_temporal_scale,
            spatial_scale=parsed_cfg.caps_spatial_scale,
            spatial_std=parsed_cfg.caps_spatial_std,
        )
        _std_dev_annealing_kwargs = parsed_cfg.std_dev_annealing_kwargs
        # Anneal for PPO unless the config opts out, regardless of the policy's
        # backbone — SAC keeps this off since it learns log_std via its own
        # automatic entropy tuning (see SquashedCLActorModel's docstring).
        _anneal = self._base_algorithm == "PPO" and parsed_cfg.std_dev_annealing
        patch_ppo_std_annealing(self._rl_agent, _anneal, _std_dev_annealing_kwargs)
        if self._base_algorithm == "PPO" and not _anneal:
            # CLActor builds logstd with requires_grad=False (it exists to be
            # annealed, see nn_modules.CLActor), and patch_ppo_std_annealing —
            # the only other thing that touches it — is now a no-op. Without
            # this flip the parameter would simply stay frozen at its initial
            # value and "learned std" would silently mean "constant std".
            lsp = getattr(self._rl_agent.policy, "log_std_parameter", None)
            if lsp is None:
                raise RuntimeError(
                    "std_dev_annealing=False asks PPO to LEARN log_std, but the policy "
                    f"({type(self._rl_agent.policy).__name__}) exposes no log_std_parameter.")
            lsp.requires_grad_(True)

        self._rl_agent.init()

        checkpoint_extra = (
            {"value": models["value"]} if self._base_algorithm == "PPO" else
            {"critic_1": models["critic_1"], "critic_2": models["critic_2"]}
        )
        self.checkpoint_modules.update({
            "policy": models["policy"],
            **checkpoint_extra,
        })
        self.checkpoint_modules["cmg"] = models["cmg"]
        if self._neural_dynamics is not None:
            self.checkpoint_modules["dynamics"] = self._neural_dynamics

    # ── Setup helpers ───────────────────────────────────────────────────── #

    def _setup_dynamics(self, cfg, models, get_f_and_B, x_dim, u_dim, device) -> None:
        """Own the (optional) learned dynamics directly (feeds Phase A's CMG synthesis).

        The CMG synthesis needs ``f``/``B``/``B_null`` and ``∂f/∂x`` at every
        state it uses (SDP dataset for "cvstem", C1/C2 gradient computation for
        "ccm"). Under analytical dynamics that comes from the env's exact
        ``get_f_and_B``; otherwise a NeuralDynamics model (pretrained before
        training by the trainer's ``pretrain_dynamics``) provides it. This
        mirrors C3M's dynamics interface expected by ``dynamics_pretrain``.
        """
        if not cfg.use_empirical_dynamics:
            if get_f_and_B is None:
                raise ValueError(
                    "C2RL: analytical dynamics (use_empirical_dynamics=False) requires a "
                    "get_f_and_B callable (classic envs only). Isaac Sim envs have no "
                    "analytical dynamics — set use_empirical_dynamics=True."
                )
            self._get_f_and_B = get_f_and_B
            self._neural_dynamics = None
            self._dynamics_optimizer = None
            self._dynamics_lr_scheduler = None
        else:
            self._neural_dynamics = models.get("dynamics", None)
            if self._neural_dynamics is None:
                raise ValueError(
                    "C2RL requires a 'dynamics' model in the models dict when "
                    "use_empirical_dynamics=True (add a models.dynamics block to the config)."
                )
            self._get_f_and_B = self._neural_dynamics.get_f_and_B
            self._dynamics_optimizer = torch.optim.Adam(
                self._neural_dynamics.parameters(), lr=cfg.dynamics_lr
            )
            self._dynamics_lr_scheduler = build_lr_scheduler(
                self._dynamics_optimizer, cfg.dynamics_lr_scheduler, cfg.dynamics_lr_scheduler_kwargs
            )

    # ── skrl Agent interface ────────────────────────────────────────────── #

    def act(self, observations, states, *, timestep: int, timesteps: int):
        """Evaluation entry point (play.py / train_utils' rollouts call this).

        Goes to the policy model rather than delegating to ``self._rl_agent.act``
        so a nonzero ``random_timesteps`` can't turn an eval rollout — which
        always passes ``timestep=0`` — into uniform random actions. The
        observation preprocessor is applied by hand for the same reason it is
        applied inside the sub-agent's own act(): skipping it would evaluate the
        policy on a different input scale than it was trained on whenever
        ``use_state_norm`` is enabled (it is off in every shipped config, so
        this is identity today).
        """
        with torch.no_grad():
            inputs = {"observations": self._rl_agent._observation_preprocessor(observations)}
            actions, outputs = self._rl_agent.models["policy"].act(inputs, role="policy")
        return actions, outputs

    # act/pre_interaction/post_interaction/update are all abstract on skrl's
    # Agent, so they must be defined even where the base behavior is all we
    # want. record_transition is not abstract and we add nothing to it, so it's
    # left inherited. C2RLSkrlTrainer drives the actual Phase B update itself
    # (see update_policy), so update() here stays a no-op.
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass

    # ── Phase B update ──────────────────────────────────────────────────── #

    def update_policy(self, *, timestep: int, timesteps: int) -> None:
        """Drive the inner RL agent's update, once per trainer rollout chunk.

        PPO consumes a whole on-policy chunk at a time, so its update belongs
        here. SAC updates every step, driven by its own ``post_interaction`` —
        nothing to do. Neither branch touches rewards: the env already returns
        the Mahalanobis reward (see the module docstring).
        """
        if self._base_algorithm == "PPO":
            # skrl's own act() puts the policy in eval() mode for rollout
            # collection (no dropout/batchnorm here, so that's harmless) and
            # never re-enables train() before calling update() itself — fine
            # for every other backbone, but a FiLM gate's GRU (film_gate_
            # encoder="gru") needs train mode for cuDNN's backward, exactly
            # like the two pretrain methods above. Vendored skrl/ code is
            # off-limits to edit, so this is patched from here instead.
            self._policy_model.train()
            self._rl_agent.update(timestep=timestep, timesteps=timesteps)

    def _train_dynamics(self, data: dict) -> float:
        """MSE training of NeuralDynamics on (x, u, x_dot) data (same as C3M).

        Called by the trainer's ``pretrain_dynamics`` when learning dynamics —
        Phase A's CMG synthesis needs f/B/∂f/∂x before it runs.
        """
        import torch.nn as nn
        dev = self._neural_dynamics.device
        x     = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        u     = torch.as_tensor(data["u"], dtype=torch.float32, device=dev)
        x_dot = torch.as_tensor(data["x_dot"], dtype=torch.float32, device=dev)

        pred = self._neural_dynamics.predict_x_dot(x, u)
        loss = nn.functional.mse_loss(pred, x_dot)

        self._dynamics_optimizer.zero_grad()
        loss.backward()
        if all(torch.isfinite(p.grad).all() for p in self._neural_dynamics.parameters() if p.grad is not None):
            torch.nn.utils.clip_grad_norm_(self._neural_dynamics.parameters(), 1.0)
            self._dynamics_optimizer.step()
        return loss.item()

    def freeze_cmg(self) -> None:
        """Freeze the synthesized CMG before Phase B."""
        for p in self._ccm_gen.parameters():
            p.requires_grad_(False)
        self._ccm_gen.eval()

    def _sample_cmg_x(self) -> np.ndarray | None:
        """Draw the ``cmg_memory_size`` states CMG synthesis will solve the SDP over.

        Uniformly subsampled from the offline ``dynamics_data.npz`` when
        ``dynamics_pretrain_data_path`` is set (capped to the data on disk, with
        a warning if ``cmg_memory_size`` asks for more), else ``None`` so
        ``build_cm_dataset`` falls back to freshly sampling the classic env's
        analytic state space via ``get_rollout``.
        """
        cfg = self._cfg
        data_path = getattr(cfg, "dynamics_pretrain_data_path", "") or None
        if not data_path:
            return None
        from .dynamics_pretrain import load_offline_dynamics_data
        x_all = load_offline_dynamics_data(data_path, tag="[C2RL]")["x"]
        n_avail = x_all.shape[0]
        n_samples = cfg.cmg_memory_size
        if n_samples > n_avail:
            print(f"[C2RL] WARNING: cmg_memory_size={n_samples} exceeds the "
                  f"{n_avail} available offline dynamics samples — using {n_avail} instead.")
            n_samples = n_avail
        idx = np.random.choice(n_avail, size=n_samples, replace=False)
        return x_all[idx]

    def _log_cmg_condition_numbers(self, x_np, *, max_states: int = 1000) -> None:
        """Post-pretraining diagnostic: condition number κ(x) = λmax/λmin of the
        frozen CMG's bounded ``W(x)`` over (a ≤``max_states`` subsample of) the
        states it was trained on. κ(W) == κ(M=W⁻¹), so this is also the
        anisotropy of the Mahalanobis reward the policy trains against; the
        w_lb/w_ub box bounds it by w_ub/w_lb. Tracks mean and 95% quantile
        (flushed by the caller's timestep=-1 write)."""
        from .math_utils import bound_W
        x_np = np.asarray(x_np, dtype=np.float32)[:, :self._x_dim]
        if x_np.shape[0] > max_states:
            idx = np.random.choice(x_np.shape[0], size=max_states, replace=False)
            x_np = x_np[idx]
        x = torch.as_tensor(x_np, device=self._device)
        with torch.no_grad():
            raw_W, _ = self._ccm_gen(x)
            W = bound_W(raw_W, self._cfg.w_lb, self._x_dim,
                        getattr(self._ccm_gen, "bounded", False))
            eig = torch.linalg.eigvalsh(W)  # ascending, (n, x_dim)
            cond = eig[:, -1] / eig[:, 0].clamp(min=1e-12)
        cond_mean = cond.mean().item()
        cond_q95 = torch.quantile(cond, 0.95).item()
        print(f"[C2RL] CMG condition number over {x.shape[0]} training states: "
              f"mean={cond_mean:.4g}, 95%={cond_q95:.4g} "
              f"(bound w_ub/w_lb={self._cfg.w_ub / self._cfg.w_lb:.4g})")
        self.track_data("Loss / C2RL/cmg/cond_mean", cond_mean)
        self.track_data("Loss / C2RL/cmg/cond_q95", cond_q95)

    def synthesize_cmg(self, *, timesteps: int = 0) -> dict:
        """Offline CMG synthesis (Phase A, always runs before Phase B) —
        dispatches to one of two pipelines depending on ``cmg_method``:

        * **``"cvstem"``** (CV-STEM): convex optimization.  Sample states, solve
          one pointwise SDP per state (``build_cm_dataset``), MSE-regress the CMG
          network onto the feasible ``{x → W*}`` targets (``regress_cmg``), then
          freeze.  The SDP results are cached to disk.

        * **``"ccm"``** (default — C1/C2 loss minimization): neural-network
          training.  Train the CMG network end-to-end with C1 (contraction) and
          C2 (killing) losses (``train_cmg_ccm``) over uniformly sampled states
          — no per-state SDP, no MSE regression.  C2 makes the metric
          ``u``-independent by construction, so no u-box vertex enumeration is
          needed.

        Called once by the trainer before Phase B — needs the dynamics already
        pretrained so the SDP / gradient computation has meaningful
        ``f``/``B``/``∂f/∂x``.

        Logs per-epoch loss/LR curves at negative timesteps so they precede
        Phase B on the ``global_step`` x-axis — same convention
        ``dynamics_pretrain.py`` uses for the NeuralDynamics fit.
        """
        cfg = self._cfg
        if cfg.cmg_method == "ccm":
            info = self._synthesize_cmg_ccm(timesteps=timesteps)
        else:
            info = self._synthesize_cmg_cvstem(timesteps=timesteps)
        self._maybe_attach_residual_base()
        self._attach_critic_potential()
        return info


    def _maybe_attach_residual_base(self) -> None:
        """After the CMG is frozen (Phase A), configure the actor's residual π(s).
        The control law is u = base + π(s): by default base = the reference
        feedforward uref (u = uref + π — π learns the whole feedback and the
        deployed policy stands on its own); with ``cvstem_residual_base`` the base
        becomes the analytic CV-STEM-LQR controller (warm-start, off by default).
        π's architecture is now fixed: ``W2(xrefs)·tanh(W1(x)·e)`` (see
        nn_modules.CLActor). The old ``residual_variant`` dispatch (bilinear /
        decoupled / latent_bias / film / feedforward) is gone — those heads are
        commented out in nn_modules."""
        policy = self._policy_model
        cl_actor = getattr(policy, "cl_actor", None)
        if cl_actor is None or not hasattr(policy, "base_controller"):
            if getattr(self._cfg, "cvstem_residual_base", False):
                print("[C2RL] cvstem_residual_base requested but the policy is not a "
                      "CLActor ('control' backbone) — skipping (actor unchanged).", flush=True)
            return

        use_base = bool(getattr(self._cfg, "cvstem_residual_base", False))

        if use_base:
            from .nn_modules import CVSTEMLQRBase
            policy.base_controller = CVSTEMLQRBase(
                self._ccm_gen, self._get_f_and_B,
                r_scaler=self._cfg.cvstem_r_scaler, w_lb=self._cfg.w_lb,
                window=self._window, angle_idx=self._angle_idx,
            )
            # Warm-start: zero W2's output so π starts at 0 and training starts
            # exactly at the analytic base.
            cl_actor.w2.zero_output()
        # else: base = urefs[0] (compute()), π keeps its standard init so the policy
        # starts with feedback and learns the full contraction law from scratch.

        _base_desc = "CV-STEM-LQR(K=R⁻¹BᵀW⁻¹)" if use_base else "urefs[0] (reference feedforward)"
        _init = "π zero-init, warm-started AT the base" if use_base else "π standard-init, learns feedback from scratch"
        print(f"[C2RL] control law: u = {_base_desc} + π  |  π = W₂(xrefs)·tanh(W₁(x)·e)"
              f"  |  {_init}.", flush=True)

        if use_base and getattr(self._cfg, "cvstem_residual_distill", False):
            self._distill_residual_from_online()
        if use_base and getattr(self._cfg, "residual_frozen", False):
            # Ablation (base only): keep π at 0 so the deployed actor is the pure
            # CV-STEM-LQR base — measures the base's own AUC. Freeze every residual
            # param; logstd stays trainable so PPO still has policy params.
            for name, p in policy.named_parameters():
                if any(k in name for k in ("cl_actor.w1.", "cl_actor.w2.")):
                    p.requires_grad_(False)
            print("[C2RL] residual FROZEN at 0 — actor = pure CV-STEM-LQR baseline.",
                  flush=True)

        # Contraction pretraining of π (the fix for u = uref + π; see cfg field).
        # Skipped when warm-started on the analytic base (already stabilizing) or
        # when the residual is frozen (ablation).
        if (getattr(self._cfg, "residual_contraction_pretrain", False)
                and not (use_base and getattr(self._cfg, "residual_frozen", False))):
            method = getattr(self._cfg, "residual_pretrain_method", "contraction")
            if method == "cvstemlqr":
                self._pretrain_residual_cvstemlqr()
            else:
                self._pretrain_residual_contraction()

        # Ablation A (Phase-0 collapse diagnostics — see
        # residual_pretrain_init_log_std's docstring in C2RLPPOCfg): pretraining
        # above leaves the mean action near-optimal but never touches
        # log_std_parameter, so Phase B would otherwise start exploring at
        # whatever the yaml's initial_log_std says — comparable to the full
        # control range at the default 0.0. Overriding it here, before Phase B's
        # rollout loop starts, is picked up correctly by
        # patch_ppo_std_annealing's lazy initial-value capture (its schedule
        # anchors to whatever log_std_parameter holds at the first real
        # post_interaction call, which is after this).
        init_std = getattr(self._cfg, "residual_pretrain_init_log_std", None)
        if init_std is not None and hasattr(policy, "log_std_parameter"):
            policy.log_std_parameter.data.fill_(float(init_std))
            print(f"[C2RL] Ablation A: log_std_parameter overridden to {init_std} "
                  f"(std={math.exp(float(init_std)):.4g}) post-pretrain.", flush=True)

    def _attach_critic_potential(self) -> None:
        """O6: hand the (now frozen) Phase-A CMG to the critic so it can evaluate
        the analytic potential ``-Phi(s) = ||e||^2_M`` and represent
        ``V(s) = f_theta(s) + ||e||^2_M`` — see models._AnalyticPotentialMixin.

        Only takes effect for a critic built with ``analytic_potential=True``
        (``models.critic.analytic_potential`` / ``--critic_analytic_potential``);
        otherwise the attribute is simply never read.
        """
        value = getattr(self._rl_agent, "value", None)
        if value is None or not getattr(value, "use_analytic_potential", False):
            return
        value._pot_ccm_gen = self._ccm_gen
        # A disabled preprocessor is skrl's ``_empty_preprocessor`` bound method;
        # an enabled one is a RunningStandardScaler, i.e. an nn.Module. Testing
        # the callable's type name is not enough (a bound method's type is
        # "method", never "function"), so the warning fired even with
        # use_value_norm=false — check for the Module instead.
        _vp = getattr(self._rl_agent, "_value_preprocessor", None)
        if isinstance(_vp, torch.nn.Module):
            print(f"[C2RL] WARNING: analytic-potential critic with a value preprocessor "
                  f"ACTIVE ({type(_vp).__name__}) — the added ||e||^2_M term is in REAL "
                  f"value units while the network output is normalized. Run this arm with "
                  f"use_value_norm=false.", flush=True)
        else:
            print(f"[C2RL] O6 scale check OK: value preprocessor is disabled "
                  f"({type(_vp).__name__}), so the analytic term and the network output "
                  f"share real value units.", flush=True)
        print("[C2RL] O6: critic parameterized as V(s) = f_theta(s) + ||e||^2_M "
              "(analytic potential from the frozen CMG).", flush=True)

    def pretrain_critic(self, env, *, num_steps: int, epochs: int, lr: float) -> None:
        """Ablation B: warm-start the critic on the frozen, already-pretrained
        actor's own Monte-Carlo returns, before Phase B's PPO updates start.

        After pretraining π's mean is already near-optimal, so the true
        advantage is ~0 almost everywhere while the critic is still at random
        init. GAE divides by that near-zero advantage std, turning noise into a
        full-scale confident gradient the moment the critic produces its first
        coherent (but wrong) value landscape. A head start on the real return
        scale removes that source of the single destructive update.

        Rolls the frozen actor out in the real env for ``num_steps`` (the CMG
        must already be injected — the Mahalanobis reward is the fit target),
        computes each step's MC return discounting to the end of the window only
        (no bootstrap past it; negligible bias at this discount_factor, and a
        longer window shrinks it further), then MSE-regresses ``value`` onto it.

        ``_value_preprocessor`` is not bypassed. Its ``inverse=True`` path (what
        compute_gae uses) clamps to ±clip_threshold (default 5) Before scaling
        by sqrt(running_variance) — never an identity map for an output fit
        directly to MC-return scale, which here runs well past ±5. Regressing
        the raw output onto ``ret_all`` (an earlier version) meant the first
        live act() truncated most of the fitted landscape at the clamp, and the
        scale mismatch against the untouched running stats fed a runaway
        compute_gae: raw advantage std grew ~2 -> ~1e16 in one run while
        grad_norm_clip kept the policy bounded and the reward curve looked
        stable. So: fit the preprocessor's stats to ``ret_all`` once up front
        (one large representative batch, better calibrated than accumulating
        from tiny per-rollout ones), then regress onto the forward-standardized
        target in that same space. Network output and preprocessor stats then
        agree, and the first live inverse call is near-identity.
        """
        rl_agent = self._rl_agent
        if getattr(rl_agent, "value", None) is None:
            print("[C2RL] Ablation B: no value model — skipping critic warmstart.", flush=True)
            return
        device = self.device
        gamma = float(self._cfg.discount_factor)
        obs_pre = getattr(rl_agent, "_observation_preprocessor", None) or (lambda t, **_unused: t)
        state_pre = getattr(rl_agent, "_state_preprocessor", None) or (lambda t, **_unused: t)
        value_pre = getattr(rl_agent, "_value_preprocessor", None)

        observations, _ = env.reset()
        states = env.state() if hasattr(env, "state") else None
        obs_buf, state_buf, rew_buf, done_buf = [], [], [], []
        with torch.no_grad():
            for _ in range(num_steps):
                actions, _ = rl_agent.act(observations, states, timestep=0, timesteps=1)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                obs_buf.append(observations)
                # The privileged critic (TrajectoryAwareValueModel, built when
                # --critic_encoder is set) reads inputs["states"], not
                # inputs["observations"] — so the warmstart has to buffer that
                # channel too, or its value.act() below raises KeyError.
                if states is not None:
                    state_buf.append(states)
                rew_buf.append(rewards)
                done_buf.append((terminated | truncated).float())
                observations = next_obs
                states = env.state() if hasattr(env, "state") else None

        returns = [None] * num_steps
        running = torch.zeros_like(rew_buf[-1])
        for t in reversed(range(num_steps)):
            running = rew_buf[t] + gamma * (1.0 - done_buf[t]) * running
            returns[t] = running

        obs_all = torch.cat(obs_buf, dim=0)
        state_all = torch.cat(state_buf, dim=0) if state_buf else None
        ret_all = torch.cat(returns, dim=0)
        N = obs_all.shape[0]
        batch = min(4096, N)

        if value_pre is not None:
            with torch.no_grad():
                value_pre(ret_all, train=True)                    # fit running stats once, from the full batch
                target_all = value_pre(ret_all, train=False).detach()  # regression target lives in the same
        else:                                                          # normalized space compute_gae's
            target_all = ret_all                                       # inverse=True will map back from

        opt = torch.optim.Adam(rl_agent.value.parameters(), lr=lr)
        print(f"[C2RL] Ablation B: critic warmstart — {epochs} epochs x {N} (s,G) pairs "
              f"from a {num_steps}-step frozen-actor rollout (gamma={gamma})…", flush=True)
        for ep in range(epochs):
            perm = torch.randperm(N, device=device)
            tot, nb = 0.0, 0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                inputs = {"observations": obs_pre(obs_all[idx], train=False)}
                if state_all is not None:
                    # Mirror skrl PPO's own act()/_update(), which always pass
                    # Both channels through their respective preprocessors.
                    inputs["states"] = state_pre(state_all[idx], train=False)
                pred, _ = rl_agent.value.act(inputs, role="value")
                loss = ((pred - target_all[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item()); nb += 1
            if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
                print(f"[C2RL] critic warmstart epoch {ep + 1}/{epochs} mse={tot / max(1, nb):.4e}",
                      flush=True)
        print("[C2RL] critic WARM-STARTED on frozen-actor Monte-Carlo returns.", flush=True)

    def _window_pool(self, n: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools of plausible (xref, uref) rows to fill the future window slots
        during pretraining — see the call sites. Empty when length == 1 (the
        window is then just the current reference and there is nothing to fill)."""
        if self._window.length <= 1:
            return None, None
        roll = self._get_rollout(n, "c3m")
        return (torch.as_tensor(roll["xref"], dtype=torch.float32, device=dev),
                torch.as_tensor(roll["uref"], dtype=torch.float32, device=dev))

    def _pretrain_residual_contraction(self) -> None:
        """Pretrain π so u = base + π contracts w.r.t. the frozen Phase-A metric
        M(x): minimize the C3M closed-loop condition Cu = Ṁ + 2·sym(M(A+BK)) + 2λM
        ≺ 0 (the C3M pd_loss) over π only. This is a contraction certificate learned
        from the metric geometry — not CV-STEM-LQR — so it respects u = uref + π
        while putting PPO in the basin of a stabilizing controller (PPO from scratch
        can't find one, esp. in high dim → divergence). When the actor has a
        preview tail, each pretrain step draws random plausible preview content
        (get_rollout's own uref distribution) rather than P=0 — certifying Cu≺0
        under a nonzero preview bias is what stops deployment-time preview from
        pushing the actor out of the certified region (P=0-only pretraining
        diverged once real preview was fed in). Reuses the exact C3M loss
        (c3m._compute_loss). CMG is already frozen."""
        from torch import matmul, transpose

        from .math_utils import (
            b_jacobian,
            bound_W,
            jacobian,
            loss_pos_matrix_random_sampling,
            spd_inverse,
            weighted_gradients,
        )
        cfg = self._cfg
        policy = self._policy_model
        # cuDNN's RNN backward requires the module to be in train mode at
        # forward time (it only saves the reserve buffers backward needs when
        # training=True) — models otherwise default to eval here, which is
        # silently fine for every other backbone but raises "cudnn RNN backward
        # can only be called in training mode" the moment a FiLM gate uses a
        # GRU (film_gate_encoder="gru") and this loop's jacobian()/backward()
        # runs through it. Plain Linear/Tanh/GRU-with-no-dropout compute
        # Identical values in train vs eval, so this has no effect beyond
        # unlocking that backward pass.
        policy.train()
        dev = self.device
        x_dim, u_dim = self._x_dim, self._u_dim
        bounded = getattr(self._ccm_gen, "bounded", False)
        # π params: everything trainable in the policy except logstd (which does
        # not enter the deterministic control u the certificate is about).
        params = [p for n, p in policy.named_parameters()
                  if p.requires_grad and "log_std" not in n]
        if not params:
            print("[C2RL] contraction pretrain: no trainable π params — skipping.", flush=True)
            return
        lr = float(getattr(cfg, "residual_pretrain_lr", 1e-3))
        epochs = int(getattr(cfg, "residual_pretrain_epochs", 200))
        batch = int(getattr(cfg, "residual_pretrain_batch", 1024))
        N = int(getattr(cfg, "residual_pretrain_samples", 65536))
        n_samp = int(getattr(cfg, "pd_loss_num_samples", 128))
        lbd = float(cfg.lbd)
        eps = float(getattr(cfg, "cm_eps", 1.0))
        opt = torch.optim.Adam(params, lr=lr)
        data = self._get_rollout(N, "c3m")                       # {x, xref, uref}
        X = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        XR = torch.as_tensor(data["xref"], dtype=torch.float32, device=dev)
        UR = torch.as_tensor(data["uref"], dtype=torch.float32, device=dev)
        I = torch.eye(x_dim, device=dev)
        # Future-window pool: the window slots k>=1 are drawn i.i.d. from the same
        # uniform reference distribution get_rollout("c3m") samples — a proxy for
        # "a plausible future reference" without needing trajectory structure.
        # Certifying Cu≺0 against random (not just zero / constant) window content
        # is what prevents the deployment-time window from pushing the actor out of
        # the certified region (zero-preview-only pretraining diverged once a real,
        # nonzero window was fed in — see project memory).
        pool_xr, pool_ur = self._window_pool(max(N, 8192), dev)
        self._ccm_gen.eval()
        print(f"[C2RL] contraction-pretraining π vs frozen CMG: {epochs} epochs × {N} "
              f"states (Cu≺0, window length={self._window.length} with RANDOM future "
              f"reference points)…", flush=True)
        for ep in range(epochs):
            perm = torch.randperm(N, device=dev)
            tot, nb = 0.0, 0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                x = X[idx].clone().requires_grad_()
                xref, uref = XR[idx], UR[idx]
                b = x.shape[0]
                # frozen metric M(x) — CMG params are frozen, so grad flows only to
                # x (needed for Ṁ), never to the metric.
                raw_W, _ = self._ccm_gen(x)
                W = bound_W(raw_W, cfg.w_lb, x_dim, bounded)
                M = spd_inverse(W)
                with torch.enable_grad():
                    f, B, _Bbot = self._get_f_and_B(x)
                f = f.to(torch.float32); B = B.to(torch.float32)
                DfDx = jacobian(f, x, create_graph=False).detach()
                DBDx = b_jacobian(B, x, u_dim, create_graph=False).detach()
                f = f.detach(); B = B.detach()
                state = self._window.synth_obs(x, xref, uref, pool_xr, pool_ur)
                # cuDNN's RNN kernel has no double-backward support at all (not a
                # train/eval-mode thing — a hard cuDNN API limitation) and this
                # loop needs exactly that: K=jacobian(u,x, create_graph=True) is
                # itself a backward through the forward below, and pd.backward()
                # below then differentiates through that. Only matters when the
                # window encoder is a GRU (--encoder gru); harmless/no-op
                # otherwise. Disabling cuDNN for this one forward call switches
                # PyTorch to its generic (non-cuDNN) RNN path, which does support
                # double-backward, at some extra compute cost for this call only.
                with torch.backends.cudnn.flags(enabled=False):
                    u = policy.compute({"observations": state}, role="policy")[0]
                K = jacobian(u, x)                               # feedback gain ∂u/∂x
                A = DfDx + torch.einsum('bxyu,bu->bxy', DBDx, u)
                dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
                dot_M = weighted_gradients(M, dot_x, x)          # Ṁ (differentiable via dot_x)
                MABK = matmul(M, A + matmul(B, K))
                Cu = dot_M + (MABK + transpose(MABK, 1, 2)) + 2 * lbd * M
                pd = loss_pos_matrix_random_sampling(-(Cu + eps * I), num_samples=n_samp)
                opt.zero_grad()
                pd.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    opt.step()
                tot += float(pd.item()); nb += 1
            if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
                print(f"[C2RL] contraction pretrain epoch {ep + 1}/{epochs} "
                      f"pd_loss(Cu≻0 violation)={tot / max(1, nb):.4e}", flush=True)
        print("[C2RL] π contraction-PRETRAINED (u = uref + π certified vs frozen CMG) — "
              "PPO now fine-tunes it on the Mahalanobis reward.", flush=True)

    def _pretrain_residual_cvstemlqr(self) -> None:
        """Pretrain π so u = uref + π(s) matches the analytic CV-STEM-LQR control
        law u_cvstemlqr(x,xref,uref) = uref - K(x)·e, K = R⁻¹BᵀW⁻¹, computed from
        the same frozen Phase-A metric M(x) as _pretrain_residual_contraction's
        certificate. Plain supervised MSE regression — no SDP, no jacobians, no
        C3M loss — so it is well-posed and actually converges near 0, unlike the
        Cu≺0 violation loss (observed to plateau ~2.6-3.3 regardless of epochs;
        that residual violation correlates with per-seed overshoot/AUC variance,
        see project memory). Reuses the CVSTEMLQRBase class the analytic baseline
        eval uses, but only to generate targets here — it is never attached to
        policy.base_controller, so the deployed law stays u = uref + π exactly as
        in "contraction" mode. Same rollout/preview-pool sampling as
        _pretrain_residual_contraction, for a fair like-for-like comparison."""
        from .nn_modules import CVSTEMLQRBase
        cfg = self._cfg
        policy = self._policy_model
        policy.train()  # cuDNN RNN backward needs train mode — see the sibling
                         # method's comment; identical values, just unlocks backward.
        dev = self.device
        x_dim, u_dim = self._x_dim, self._u_dim
        params = [p for n, p in policy.named_parameters()
                  if p.requires_grad and "log_std" not in n]
        if not params:
            print("[C2RL] cvstem-lqr regression pretrain: no trainable π params — skipping.",
                  flush=True)
            return
        base = CVSTEMLQRBase(
            self._ccm_gen, self._get_f_and_B, r_scaler=cfg.cvstem_r_scaler,
            w_lb=cfg.w_lb, window=self._window, angle_idx=self._angle_idx,
        )
        lr = float(getattr(cfg, "residual_pretrain_lr", 1e-3))
        epochs = int(getattr(cfg, "residual_pretrain_epochs", 200))
        batch = int(getattr(cfg, "residual_pretrain_batch", 1024))
        N = int(getattr(cfg, "residual_pretrain_samples", 65536))
        opt = torch.optim.Adam(params, lr=lr)
        data = self._get_rollout(N, "c3m")                       # {x, xref, uref}
        X = torch.as_tensor(data["x"], dtype=torch.float32, device=dev)
        XR = torch.as_tensor(data["xref"], dtype=torch.float32, device=dev)
        UR = torch.as_tensor(data["uref"], dtype=torch.float32, device=dev)
        pool_xr, pool_ur = self._window_pool(max(N, 8192), dev)
        # Actuator box the env actually applies. env_base.step clamps to
        # U_MIN/U_MAX = 2 * UREF_MIN/max, while self.action_space is declared at
        # the uref bounds — hence the factor 2 here, which mirrors that step()
        # rather than the declared space. See residual_pretrain_clamp_target.
        clamp_target = bool(getattr(cfg, "residual_pretrain_clamp_target", False))
        u_lo = u_hi = None
        if clamp_target:
            u_hi = 2.0 * torch.as_tensor(self.action_space.high, dtype=torch.float32, device=dev)
            u_lo = 2.0 * torch.as_tensor(self.action_space.low, dtype=torch.float32, device=dev)

        self._ccm_gen.eval()
        _clamp_desc = (f", target CLAMPED to the applied actuator box "
                       f"[{u_lo.min().item():g}, {u_hi.max().item():g}]" if clamp_target else "")
        print(f"[C2RL] cvstem-lqr regression-pretraining π vs analytic base: {epochs} "
              f"epochs × {N} states (MSE, window length={self._window.length} with RANDOM "
              f"future reference points{_clamp_desc})…", flush=True)
        for ep in range(epochs):
            perm = torch.randperm(N, device=dev)
            tot, nb = 0.0, 0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                x, xref, uref = X[idx], XR[idx], UR[idx]
                state = self._window.synth_obs(x, xref, uref, pool_xr, pool_ur)
                u = policy.compute({"observations": state}, role="policy")[0]
                target = base(state)
                if clamp_target:
                    target = torch.clamp(target, u_lo, u_hi)
                loss = torch.mean((u - target) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    opt.step()
                tot += float(loss.item()); nb += 1
            if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
                print(f"[C2RL] cvstem-lqr regression pretrain epoch {ep + 1}/{epochs} "
                      f"mse={tot / max(1, nb):.4e}", flush=True)
        print("[C2RL] π REGRESSION-PRETRAINED to match CV-STEM-LQR (u = uref + π, base "
              "NOT attached) — PPO now fine-tunes it on the Mahalanobis reward.", flush=True)

    def _distill_residual_from_online(self) -> None:
        """Supervised warm-start: regress the actor's residual so the actor
        reproduces the online CV-STEM-LQR controller — the one that uses the exact
        per-state SDP metric W_online(x), which the frozen CMG only approximates
        (the residual base's remaining AUC gap is precisely that approximation,
        amplified by W⁻¹ near w_lb). Since
            u_base   = uref - (1/r)Bᵀ W_frozen⁻¹ e,
            u_online = uref - (1/r)Bᵀ W_online⁻¹ e,
        the target residual is u_online - u_base = -(K_online - K_frozen)·e, a
        state-dependent linear map the bilinear residual regresses onto. Reuses
        the Phase-A per-state SDP solutions (self._cm_dataset) — no new solves.
        The actor then starts at online-LQR quality and PPO fine-tunes below it
        with preview-aware anticipation the myopic per-state controller lacks."""
        ds = getattr(self, "_cm_dataset", None)
        policy = self._policy_model
        cl_actor = getattr(policy, "cl_actor", None)
        if ds is None or cl_actor is None:
            print("[C2RL] residual distillation SKIPPED (needs cmg_method=cvstem "
                  "dataset + a CLActor policy).", flush=True)
            return
        from .math_utils import bound_W, spd_inverse
        dev = self.device
        r = float(self._cfg.cvstem_r_scaler) + 1e-5
        x_dim, u_dim = self._x_dim, self._u_dim
        x = torch.as_tensor(ds["x"], dtype=torch.float32, device=dev)          # (N, x_dim)
        W_online = torch.as_tensor(ds["W"], dtype=torch.float32, device=dev)   # (N, x_dim, x_dim)
        N = x.shape[0]
        with torch.no_grad():
            _f, B, _ = self._get_f_and_B(x)
            B = B.to(torch.float32)                                            # (N, x, u)
            K_online = (1.0 / r) * torch.bmm(B.transpose(1, 2), spd_inverse(W_online))
            raw_Wf, _ = self._ccm_gen(x)
            Wf = bound_W(raw_Wf, self._cfg.w_lb, x_dim,
                         getattr(self._ccm_gen, "bounded", False))
            K_frozen = (1.0 / r) * torch.bmm(B.transpose(1, 2), spd_inverse(Wf))
            dK = K_online - K_frozen                                           # (N, u, x)
        params = [p for p in cl_actor.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-3)
        epochs = int(getattr(self._cfg, "residual_distill_epochs", 300))
        batch = 4096
        for ep in range(epochs):
            perm = torch.randperm(N, device=dev)
            tot = 0.0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                xi = x[idx]
                b = xi.shape[0]
                # Errors spanning the deployment regime (~XE_INIT scale); the
                # target is linear in e, PPO refines the exact operating range.
                e = (torch.rand(b, x_dim, device=dev) * 2 - 1)
                target = -torch.bmm(dK[idx], e.unsqueeze(-1)).squeeze(-1)      # (b, u)
                # Locally-constant reference held across the window — what the
                # linear target -dK·e is defined against.
                obs = self._window.synth_obs(
                    xi, xi - e, torch.zeros(b, u_dim, device=dev))
                feedback, _ = cl_actor._feedback(obs)
                loss = torch.mean((feedback - target) ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * b
            if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
                print(f"[C2RL] residual distill epoch {ep+1}/{epochs} mse={tot / N:.4e}",
                      flush=True)
        print("[C2RL] residual DISTILLED to online CV-STEM-LQR (per-state SDP) — "
              "actor warm-started at online-LQR quality; PPO fine-tunes from here.",
              flush=True)

    def _synthesize_cmg_ccm(self, *, timesteps: int = 0) -> dict:
        """CCM path: train the CMG network directly with C1+C2 losses."""
        from .ncm_synthesis import train_cmg_ccm
        cfg = self._cfg
        has_writer = getattr(self, "writer", None) is not None
        epochs = cfg.cmg_regress_epochs

        # ~100 wandb points regardless of epochs, final epoch always flushed.
        log_every = max(1, epochs // 100)

        def _on_epoch(epoch: int, train_loss: float, lr: float, val_loss: float) -> None:
            self.track_data("Loss / C2RL/cmg/c1c2_loss", train_loss)
            self.track_data("Loss / C2RL/cmg/regress_lr", lr)
            if not np.isnan(val_loss):
                self.track_data("Loss / C2RL/cmg/c1c2_val_loss", val_loss)
            if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
                self.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        info = train_cmg_ccm(
            self._ccm_gen, self._get_f_and_B, self._get_rollout,
            x_dim=self._x_dim, u_dim=self._u_dim,
            lbd=cfg.lbd, w_lb=cfg.w_lb, w_ub=cfg.w_ub, eps=cfg.cm_eps,
            epochs=epochs, lr=cfg.cmg_regress_lr, batch_size=cfg.cmg_regress_batch_size,
            num_samples=cfg.cmg_memory_size,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            device=self._device, tag="[C2RL]",
            on_epoch=_on_epoch,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
            x_samples=self._sample_cmg_x(),
            random_ratio=getattr(cfg, "cmg_random_ratio", 0.0),
        )
        self.freeze_cmg()
        self.track_data("Loss / C2RL/cmg/c1c2_loss_best", info["final_loss"])
        if not np.isnan(info["final_val_loss"]):
            self.track_data("Loss / C2RL/cmg/c1c2_val_loss_best", info["final_val_loss"])
        self._log_cmg_condition_numbers(info["x"])
        if has_writer:
            self.write_tracking_data(timestep=-1, timesteps=timesteps)
        return {
            "feasibility_rate": 1.0,  # no SDP, no infeasibility concept
            "residual_mean": float("nan"),
            "residual_max": float("nan"),
            "lambda_reduced_rate": 0.0,
            "regress_mse": info["final_loss"],
        }

    def _synthesize_cmg_cvstem(self, *, timesteps: int = 0) -> dict:
        """CV-STEM path: Load the offline ``{x → W*(x)}`` dataset, then MSE-regress.

        The per-state SDP is not solved here. C2RL consumes a dataset generated
        ahead of time by ``scripts/build_cm_dataset.py`` and committed under
        ``data/``; a missing cache raises instead of silently re-solving.

        Solving at agent construction meant every seed, every gamma and every
        sweep worker paid for the same 100k-state solve and raced to write the
        same file — 14 minutes each, multiplied by the whole grid, for a dataset
        that depends on none of the things being swept (the cache key is
        lbd/w_lb/w_ub/r/eps/solver/N, and gamma and seed appear in none of them).
        Generating it once offline also makes the metric an input to the
        experiment rather than a per-run artifact, so two runs cannot disagree
        about what W they were certified against.
        """
        from .ncm_synthesis import load_cached_cm_dataset, regress_cmg
        cfg = self._cfg
        cache_path, cache_kwargs = cm_dataset_target(cfg)
        if cache_path is None:
            raise RuntimeError(
                "[C2RL] cmg_method='cvstem' needs an offline {x -> W*(x)} dataset, but "
                "neither cm_data_path nor dynamics_pretrain_data_path is set, so there "
                "is nowhere to load one from. Set cm.cm_data_path in the agent yaml "
                "(e.g. 'data/classic/cartpole/cm_data.npz') and generate it with "
                "scripts/build_cm_dataset.py."
            )
        dataset = load_cached_cm_dataset(cache_path, **cache_kwargs)
        if dataset is None:
            # Either the file is absent or one of the cache_kwargs disagrees with
            # it. Both mean the same thing here: the metric this run is
            # configured for has not been generated. Re-solving inline would hide
            # a config/dataset mismatch behind a 14-minute pause.
            raise FileNotFoundError(
                f"[C2RL] no offline CM dataset at {cache_path} matching this config "
                f"(lbd={cfg.lbd}, w_lb={cfg.w_lb}, w_ub={cfg.w_ub}, "
                f"r={cfg.cvstem_r_scaler}, eps={cfg.cm_eps}, solver={cfg.cm_solver}, "
                f"N={cfg.cmg_memory_size}).\nGenerate it first:\n"
                f"  python scripts/build_cm_dataset.py --task <task> --algorithm "
                f"<c2rl-ppo|c2rl-sac>\n"
                f"The SDP is no longer solved at agent construction — see "
                f"_synthesize_cmg_cvstem."
            )
        # Kept for the optional residual distillation (cvstem_residual_distill):
        # the per-state SDP solutions W_online(x) are the exact online CV-STEM-LQR
        # metric the frozen CMG only approximates. See _distill_residual_from_online.
        self._cm_dataset = dataset

        has_writer = getattr(self, "writer", None) is not None
        epochs = cfg.cmg_regress_epochs

        self.track_data("Loss / C2RL/cm_synthesis/feasibility_rate", dataset["feasibility_rate"])
        self.track_data("Loss / C2RL/cm_synthesis/residual_mean", dataset["residual_mean"])
        self.track_data("Loss / C2RL/cm_synthesis/residual_max", dataset["residual_max"])
        self.track_data("Loss / C2RL/cm_synthesis/lambda_reduced_rate", dataset.get("lambda_reduced_rate", 0.0))
        if has_writer:
            self.write_tracking_data(timestep=-epochs - 1, timesteps=timesteps)

        # ~100 wandb points regardless of epochs, final epoch always flushed —
        # same cadence as dynamics_pretrain.pretrain_dynamics.
        log_every = max(1, epochs // 100)

        def _on_epoch(epoch: int, train_mse: float, lr: float, val_mse: float) -> None:
            self.track_data("Loss / C2RL/cmg/regress_mse", train_mse)
            self.track_data("Loss / C2RL/cmg/regress_lr", lr)
            if not np.isnan(val_mse):
                self.track_data("Loss / C2RL/cmg/regress_val_mse", val_mse)
            if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
                self.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        bounded = getattr(self._ccm_gen, "bounded", False)
        info = regress_cmg(
            self._ccm_gen, dataset,
            w_lb=cfg.w_lb, x_dim=self._x_dim, bounded=bounded,
            epochs=epochs, lr=cfg.cmg_regress_lr,
            lr_scheduler=cfg.cmg_regress_lr_scheduler,
            lr_scheduler_kwargs=cfg.cmg_regress_lr_scheduler_kwargs,
            batch_size=cfg.cmg_regress_batch_size, device=self._device, tag="[C2RL]",
            on_epoch=_on_epoch,
            val_frac=cfg.cmg_val_frac, early_stop_patience=cfg.cmg_early_stop_patience,
        )
        self.freeze_cmg()
        # Single post-loop point (distinct key from the per-epoch curves above)
        # for the epoch actually restored into ccm_gen — may differ from the
        # curves' last point when training ran past its best epoch without
        # early-stopping triggering (see regress_cmg's best-epoch restore).
        self.track_data("Loss / C2RL/cmg/regress_mse_best", info["final_loss"])
        if not np.isnan(info["final_val_loss"]):
            self.track_data("Loss / C2RL/cmg/regress_val_mse_best", info["final_val_loss"])
        self._log_cmg_condition_numbers(dataset["x"])
        if has_writer:
            self.write_tracking_data(timestep=-1, timesteps=timesteps)
        return {
            "feasibility_rate": dataset["feasibility_rate"],
            "residual_mean": dataset["residual_mean"],
            "residual_max": dataset["residual_max"],
            "lambda_reduced_rate": dataset.get("lambda_reduced_rate", 0.0),
            "regress_mse": info["final_loss"],
        }

    def save_dynamics(self, path: str) -> None:
        if self._neural_dynamics is not None:
            self._neural_dynamics.save(path)
            print(f"[C2RL] Saved NeuralDynamics → {path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Trainer
# ─────────────────────────────────────────────────────────────────────────── #

class C2RLSkrlTrainer(Trainer):
    """skrl Trainer for C2RL — offline CMG synthesis (Phase A), then
    single-policy RL against the frozen CMG's Mahalanobis reward (Phase B)."""

    @staticmethod
    def _forward_env_log(agent, infos) -> None:
        """Forward the env's per-episode ``extras['log']`` (path_tracking_base's /
        classic env_base's ``Stability/*``) onto the outer agent's tracking_data,
        exactly as skrl's SequentialTrainer does for PPO/SAC. No-op if nothing
        finished this step."""
        if not isinstance(infos, dict):
            return
        log = infos.get("log")
        if not isinstance(log, dict):
            return
        for k, v in log.items():
            key = k if "/" in k else f"Info / {k}"
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    agent.track_data(key, v.item())
            elif isinstance(v, (int, float)):
                agent.track_data(key, float(v))

    @staticmethod
    def _inject_ccm(env, agent, metric=None) -> None:
        """Hand a frozen copy of the synthesized CMG to the env(s) so their own
        ``get_rewards()`` returns the Mahalanobis reward natively (see the
        module docstring — this is the only reward path).

        Raises if no env accepted it: a missing ``set_ccm`` means the run would
        silently train on the plain baseline reward, which is exactly the
        classic-vs-isaaclab parity bug this guard exists to prevent. A deepcopy,
        not the agent's own module, so the env holds a stable frozen metric even
        if the agent's ``_ccm_gen`` were ever touched again.

        ``metric`` overrides what is injected. It is injected as-is: stateless apart
        from counters, and deep-copying it would drag along whatever
        ``get_f_and_B`` is bound to — the env itself, under analytical dynamics.
        """
        import copy

        # Peel wrappers down to the concrete env: first the ._env chain
        # (BatchedGymnasiumWrapper etc.), then gymnasium's .unwrapped.
        inner = env
        while hasattr(inner, "_env") and getattr(inner, "_env") is not inner:
            inner = inner._env
        while hasattr(inner, "unwrapped") and getattr(inner, "unwrapped") is not inner:
            inner = inner.unwrapped

        device = agent.device
        ccm = metric if metric is not None else copy.deepcopy(agent._ccm_gen).to(device)
        # A classic SyncVectorEnv exposes per-env instances via .envs; an Isaac
        # env is a single batched env that takes set_ccm directly.
        targets = list(getattr(inner, "envs", [])) or [inner]
        injected = 0
        for e in targets:
            e = e.unwrapped if hasattr(e, "unwrapped") else e
            if hasattr(e, "set_ccm"):
                e.set_ccm(
                    ccm,
                    agent._cfg.w_lb,
                    device,
                    tracking_scaler=agent._cfg.tracking_scaler,
                    control_scaler=agent._cfg.control_scaler,
                    reward_euclidean=getattr(agent._cfg, "reward_euclidean", False),
                    reward_level=getattr(agent._cfg, "reward_level", False),
                    residual_anchor_scale=getattr(agent._cfg, "residual_anchor_scale", 0.0),
                    cvstem_r_scaler=getattr(agent._cfg, "cvstem_r_scaler", 1.0),
                )
                injected += 1
        if injected == 0:
            raise RuntimeError(
                "[C2RL] no env accepted set_ccm — Phase B would train on the plain "
                f"baseline reward instead of the Mahalanobis one. Env type: {type(inner).__name__}."
            )

    def train(self) -> None:
        agent: C2RLAgent = self.agents if not isinstance(self.agents, list) else self.agents[0]
        env = self.env
        timesteps = self.cfg.timesteps

        agent.init(trainer_cfg=self.cfg)
        from .contraction_metrics import log_raw_config
        log_raw_config(getattr(self, "_wandb_raw_cfg", None))
        agent.enable_training_mode(True)

        # Pretrain learned dynamics (if any) before Phase A's CMG synthesis, so it
        # has meaningful f/B/∂f/∂x (SDP dataset for "cvstem", C1/C2 gradient
        # computation for "ccm"). No-op for analytical dynamics (classic envs).
        from .dynamics_pretrain import pretrain_dynamics
        pretrain_dynamics(
            agent,
            epochs=getattr(agent._cfg, "dynamics_pretrain_epochs", 5),
            data_path=getattr(agent._cfg, "dynamics_pretrain_data_path", "") or None,
            timesteps=timesteps,
            memory_size=getattr(agent._cfg, "emp_dynamics_memory_size", None),
            num_controls_per_state=getattr(agent._cfg, "num_controls_per_state", None),
            tag="[C2RL]",
            val_frac=getattr(agent._cfg, "dynamics_val_frac", 0.1),
            early_stop_patience=getattr(agent._cfg, "dynamics_early_stop_patience", 10),
        )

        # ── Phase A: offline CMG synthesis — "cvstem" solves one SDP per sampled
        # state (build_cm_dataset) and MSE-regresses the CMG onto {x -> W*}
        # (regress_cmg); "ccm" trains the CMG directly with C1/C2 losses
        # (train_cmg_ccm). Either way the CMG is frozen before Phase B reads its
        # static metric. synthesize_cmg logs feasibility/residual/loss/LR itself. ──
        # Phase A always runs. There is no per-step alternative: a per-state SDP
        # carries its own nu/chi and no Wdot term, so it certifies nothing about
        # contraction, and solving one every step just repeats that error.
        info = agent.synthesize_cmg(timesteps=timesteps)
        print(f"[C2RL] Phase A ({agent._cfg.cmg_method}) — CMG synthesized "
              f"(feasible {info['feasibility_rate']:.1%}, λ-reduced {info['lambda_reduced_rate']:.1%}, "
              f"loss {info['regress_mse']:.4g}) and frozen.")

        # ── Phase B: rollout + train the deployed policy against the Mahalanobis
        # reward computed from the frozen CMG. ─────────────────────────────
        rl_agent = agent._rl_agent
        rl_agent.enable_training_mode(True)
        rollout_steps = agent._cfg.rollouts

        self._inject_ccm(env, agent)

        # Ablation B (Phase-0 collapse diagnostics) — see
        # C2RLAgent.pretrain_critic / C2RLPPOCfg.pretrain_critic_steps. Needs the
        # CMG already injected above (the rollout's reward must be the real
        # Mahalanobis one) and must run before the env.reset() below, which
        # starts Phase B's own rollout from a fresh episode.
        pretrain_critic_steps = int(getattr(agent._cfg, "pretrain_critic_steps", 0))
        if agent._base_algorithm == "PPO" and pretrain_critic_steps > 0:
            agent.pretrain_critic(
                env,
                num_steps=pretrain_critic_steps,
                epochs=int(getattr(agent._cfg, "pretrain_critic_epochs", 200)),
                lr=float(getattr(agent._cfg, "pretrain_critic_lr", 1.0e-3)),
            )

        observations, infos = env.reset()
        states = env.state() if hasattr(env, "state") else None
        global_step = 0
        # Coarse flush cadence for the inner rl_agent — flushing every step/chunk
        # would collapse the 100-episode reward/timestep deques to a spiky curve.
        flush_interval = max(1, timesteps // 100)
        next_flush = flush_interval

        # ── Outer-agent checkpointing ──────────────────────────────────────
        # The outer agent is stepped once per rollout chunk, so skrl's
        # `(timestep + 1) % checkpoint_interval` gate in Agent.post_interaction
        # can never line up (stride `rollout_steps` vs. interval 2000) — it
        # would silently write no checkpoints at all. Drive them off the real
        # step count instead. The best-model comparison also needs episode
        # returns, and `record_transition` runs on the inner rl_agent, so the
        # outer agent's own tracking_data never holds them: mirror the inner
        # agent's running mean across (captured per chunk, since
        # write_tracking_data clears `_track_rewards`).
        ckpt_interval = agent.checkpoint_interval if isinstance(agent.checkpoint_interval, int) else 0
        next_ckpt = global_step + ckpt_interval if ckpt_interval > 0 else None
        last_return: float | None = None

        def _write_outer_checkpoint(step: int, *, force_best: bool = False) -> None:
            improved = last_return is not None and last_return > agent.checkpoint_best_modules["reward"]
            if improved or (force_best and not agent.checkpoint_best_modules["modules"]):
                agent.checkpoint_best_modules.update({
                    "timestep": step,
                    "reward": last_return if last_return is not None else agent.checkpoint_best_modules["reward"],
                    "saved": False,
                    "modules": {
                        k: copy.deepcopy(agent._get_internal_value(v))
                        for k, v in agent.checkpoint_modules.items()
                    },
                })
            agent.write_checkpoint(timestep=step, timesteps=timesteps)

        pbar = _tqdm.tqdm(total=timesteps, desc="C2RL training (Phase B)", file=sys.stdout)
        while global_step < timesteps:
            if agent._base_algorithm == "PPO":
                rl_agent.memory.reset()
            steps_to_take = min(rollout_steps, timesteps - global_step)
            for _ in range(steps_to_take):
                rl_agent.pre_interaction(timestep=global_step, timesteps=timesteps)
                with torch.no_grad():
                    actions, _ = rl_agent.act(observations, states, timestep=global_step, timesteps=timesteps)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                next_states = env.state() if hasattr(env, "state") else None

                # The env's get_rewards() already computes the Mahalanobis reward
                # via the injected frozen CCM — use it directly.
                rl_agent.record_transition(
                    observations=observations, states=states, actions=actions,
                    rewards=rewards, next_observations=next_obs, next_states=next_states,
                    terminated=terminated, truncated=truncated, infos=infos,
                    timestep=global_step, timesteps=timesteps,
                )

                self._forward_env_log(agent, infos)
                observations = next_obs
                states = next_states

                if agent._base_algorithm == "SAC":
                    agent.update_policy(timestep=global_step, timesteps=timesteps)
                    rl_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                    if global_step % flush_interval == 0 and getattr(rl_agent, "writer", None) is not None:
                        rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

                global_step += 1
                pbar.update(1)

            # Chunk-based update for PPO
            if agent._base_algorithm == "PPO":
                agent.update_policy(timestep=global_step, timesteps=timesteps)
                rl_agent.post_interaction(timestep=global_step, timesteps=timesteps)
                if global_step >= next_flush and getattr(rl_agent, "writer", None) is not None:
                    rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)
                    next_flush = global_step + flush_interval

            # Outer agent: drives its own checkpoint cadence (checkpoint_modules)
            # and flushes whatever Stability/* logs _forward_env_log collected.
            agent.post_interaction(timestep=global_step, timesteps=timesteps)
            if getattr(agent, "writer", None) is not None:
                agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

            if getattr(rl_agent, "_track_rewards", None):
                last_return = float(np.mean(rl_agent._track_rewards))
            if next_ckpt is not None and global_step >= next_ckpt:
                _write_outer_checkpoint(global_step)
                next_ckpt = global_step + ckpt_interval

        # Final checkpoint — also guarantees a best_agent.pt exists for the
        # post-training eval when no interval boundary was crossed cleanly.
        if ckpt_interval > 0:
            if getattr(rl_agent, "_track_rewards", None):
                last_return = float(np.mean(rl_agent._track_rewards))
            _write_outer_checkpoint(global_step, force_best=True)

        if getattr(rl_agent, "writer", None) is not None:
            rl_agent.write_tracking_data(timestep=global_step, timesteps=timesteps)

        # Persist the learned dynamics for reuse/inspection (matches C3M).
        if agent._neural_dynamics is not None:
            dyn_path = os.path.join(agent.experiment_dir, "checkpoints", "dynamics.pt")
            os.makedirs(os.path.dirname(dyn_path), exist_ok=True)
            agent.save_dynamics(dyn_path)
