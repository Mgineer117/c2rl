"""Find a feasible ``(lbd, r, w_lb, w_ub)`` for cvstem-lqr over uniform state samples.

The loop, and that is all it is::

    lbd, r = 10.0, 0.1                      # w_lb = 0.01, w_ub = 100, fixed
    repeat:
        solve the joint CV-STEM SDP at (lbd, r) over N uniform state-box samples
        LMI infeasible          -> lbd /= 1.5
        >5% of controls outside -> r   *= 2  (then lbd /= 1.5 once r passes --r-max)
        otherwise               -> done

Then the agent is built at that point and rolled for one episode, so the
certificate and the AUC it actually produces are reported together.

The envelope ``w_lb·I ⪯ W ⪯ w_ub·I`` is a fixed input, never searched: if nothing
certifies inside it, the script reports INFEASIBLE rather than widening it. Note
``‖M‖₂ ≤ ν ≤ 1/w_lb``, so ``w_lb`` is a hard cap on the gain
(``‖K‖₂ ≤ ‖B‖₂/(r·w_lb)``) and therefore also a hard limit on what the ``r``
branch can rescue — which is the point of pinning it.

The SDP is ``ncm_synthesis.cvstem_joint`` — Tsukamoto's ``cvstem0``, one program
over all samples with ``ν``/``χ`` shared and his ``(W̄-I)/dt`` term. States are
drawn i.i.d. uniform from the env's state box, his ``xlims`` draw, which is the
same draw ``cvstem_lqr`` synthesizes over.

The control check — A 5% budget, not a veto
--------------------------------------------
Uniform state samples carry no reference, so the error is drawn from the env's own
reset perturbation box ``[XE_INIT_MIN, XE_INIT_MAX]`` and the feedforward from
``[UREF_MIN, UREF_MAX]``. (Not ``XE_MIN``/``XE_MAX`` -- that is C3M's flat +-1
training-perturbation box, not the tracking error an episode presents; see the
comment at the ``e_lo, e_hi`` assignment.) At every sampled state, ``--n-draws`` of each are pushed through the
control the agent would really apply, ``u = uref - K(x)·e`` with ``K = (1/r)BᵀM``,
and the check fails only when more than ``--viol-frac`` (5%) of that population
lands outside the control box.

A hard 0% veto does not work here and the reason is structural. The control box is
the uref box widened by ``--u-expansion`` (2x — see ``expand_box``), so the budget
left for feedback is exactly ``UREF_MAX``. Requiring the worst case — every error
component at its box extreme at once — to fit that budget demands
``Σ_j |K_ij|·e_j^max ≤ UREF_MAX_i``, i.e. ``‖K‖ ≲ 1.5`` on the car, where the
gain that actually tracks is ``≈ 22``. Measured: the veto returned lbd=0.1156,
r=102.4, ν pinned at its w_lb cap, ``‖K‖ ≈ 1`` and **AUC 40.4** against 1.35 for
the shipped config and 69 for open loop — barely better than no feedback. Worse,
that controller then exceeded the box anyway in the episode (``|u|max = 4.6``
against ±3), because the weak gain let the tracking error grow to ~19, twenty
times outside the error box the constraint was written over.

The configurations that perform saturate transiently and rely on it. So the only
honest form of this check is a budget on how often that happens.

``expand_box`` is sign-aware: a bound widens by multiplying when it already points
outward (negative ``lo``, positive ``hi``) and by dividing when it points inward,
since ``2 x positive_lo`` would narrow the box. Zero is a fixed point, which is
what turtlebot's ``v ∈ [0, 0.44]`` needs.

When ``r`` bites, and when it does not
---------------------------------------
``r`` is live only where ``χ`` has slack. The LMI sees ``ν`` and ``r`` only as
``ν/r``, so if the program is tight enough that ``W̄`` is pinned against ``χI``,
the solved ``ν`` scales exactly with ``r`` and ``K = R⁻¹BᵀM`` does not move.
Measured on the car, both regimes, same script::

    lbd=10   r 0.1 -> 6554 : chi frozen at 422.2, nu/r = 4423 flat, violation 278 flat
    lbd=0.26 r 0.1 -> 102.4: chi 2.27 -> 5.68,   nu/r 8.31 -> 1.39, violation 5.73 -> 0.064

So raising ``r`` buys a smaller gain by letting the solver trade into a
worse-conditioned metric — until ``χ`` saturates, after which it buys nothing.
The loop therefore doubles ``r`` up to ``--r-max`` and only then lowers ``lbd``.
Do not add a "the violation stopped moving, give up on r" shortcut: the violation
sits on plateaus that later break through (on the car at lbd=0.39 it stalls at
0.66 from r=3.2 to 6.4, and at lbd=0.26 the same plateau drops to 0.086 by
r=25.6).

Note the ``cm_dt`` dependence: at ``cm_dt = env.dt`` the ``(W̄-I)/dt`` term is 33x
harsher on the classic envs and pins the solution at the corner, which is the
regime where ``r`` looks inert. It is not inert at ``cm_dt = 1``.

Example::

    python scripts/find_uniform_lambda.py --task classic-car-v0 --cm-dt 1.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "source", "c2rl"))

from c2rl.agents.skrl.contraction_metrics import per_env_metrics  # noqa: E402
from c2rl.agents.skrl.cvstem_lqr import CVSTEMLQRAgent  # noqa: E402
from c2rl.agents.skrl.ncm_synthesis import (  # noqa: E402
    cvstem_joint,
    drift_jacobians,
    sample_state_box,
)


def eps_for_n(n: int) -> float:
    """The LMI margin ε to use at ``n`` uniform samples.

        N <=   100 -> 0.10
        N <=  1000 -> 0.05
        N >   1000 -> 0.01

    ε is a covering radius, not a stability requirement. Tsukamoto's own program
    (AstroHiro classncm.py:189) uses epsilon = 0, so the LMI need only be
    negative semidefinite at the sampled states; ε > 0 is what buys the states
    in between, since a margin ε at each sample still holds nearby when the
    field is Lipschitz. The sparser the draw, the wider the gaps, and the more
    margin each sample has to carry.

    Holding ε fixed while N grows is therefore stricter than necessary, and it
    is what made ball_and_beam and two_link_arm certify at N=100 and then fail
    the N=1000 synthesis: not a stability failure, a margin one.
    """
    if n <= 100:
        return 0.1
    if n <= 1000:
        return 0.05
    return 0.01


def expand_box(lo, hi, factor=2.0):
    """Widen a box outward by ``factor``, per-component and sign-aware.

    A bound widens by multiplying when it already points outward (negative
    ``lo``, positive ``hi``) and by dividing when it points inward (positive
    ``lo``, negative ``hi``) — ``2 × positive_lo`` would narrow the box, not
    widen it. Zero is a fixed point either way, which is what turtlebot's
    ``v ∈ [0, 0.44]`` needs (its lower bound must stay at 0).
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return (np.where(lo < 0, lo * factor, lo / factor),
            np.where(hi > 0, hi * factor, hi / factor))


def control_violation_rate(W, B, *, r_scaler, e_draws, uref_draws, u_lo, u_hi):
    """Fraction of generated controls ``u = uref - K(x)·e`` that leave the box.

    Not a worst case. ``e_draws``/``uref_draws`` are ``(n_states, n_draws, dim)``
    samples of the error and feedforward; this forms the control the agent would
    actually apply at each and counts how many land outside.

    The worst case is a box vertex — every error component at its extreme at once
    — which no episode visits: it demands ``Σ_j |K_ij|·e_j`` fit the budget where
    a typical draw needs about ``‖K_i‖₂·‖e‖₂``. Vetoing on it rejected every gain
    that actually tracks (measured on the car: it drove ν to its w_lb cap and
    ``‖K‖`` to ~1, against the ~22 that reaches AUC 1.3).

    The draws are made once by the caller and reused at every ``(lbd, r)``, so the
    accept/reject decision moves only with the metric — resampling per iteration
    would let the search accept on RNG luck near the threshold.
    """
    r = r_scaler + 1e-5
    bad = total = 0
    for k in range(W.shape[0]):
        K = (1.0 / r) * B[k].T @ np.linalg.inv(W[k])              # (u_dim, x_dim)
        u = uref_draws[k] - e_draws[k] @ K.T                      # (n_draws, u_dim)
        bad += int(((u < u_lo) | (u > u_hi)).any(axis=1).sum())
        total += u.shape[0]
    return bad / total


def evaluate(env, agent, n_envs):
    """One episode through the real agent -> the Stability/* metrics."""
    obs, _ = env.reset(seed=0)
    T = int(env.max_episode_len)
    e0 = env.wrap_angles(env.x_t - env.xref[:, 0]).norm(dim=-1)
    e_last, e_max, err_sum = e0.clone(), e0.clone(), e0.clone()
    umax = 0.0
    for t in range(T):
        flat = torch.cat([torch.as_tensor(obs[k], dtype=torch.float32).reshape(n_envs, -1)
                          for k in sorted(obs)], dim=-1)
        with torch.no_grad():
            u, _ = agent.act(flat, None, timestep=t, timesteps=T)
        umax = max(umax, float(u.abs().max()))
        obs, _, _, _, _ = env.step(u.numpy())
        e = env.wrap_angles(env.x_t - env.xref[:, min(t + 1, T - 1)]).norm(dim=-1)
        e = torch.nan_to_num(e, nan=1e6, posinf=1e6)
        e_max = torch.maximum(e_max, e)
        err_sum += e
        e_last = e
    m = per_env_metrics(e0=e0, e_last=e_last, e_max=e_max, err_sum=err_sum,
                        steps=torch.full_like(e0, T + 1), dt=float(env.dt))
    return {k: float(v.mean()) for k, v in m.items()}, umax


def main() -> int:
    p = argparse.ArgumentParser(
        description="Search (lbd, r) for cvstem-lqr over uniform state samples, "
                    "then evaluate one episode.")
    p.add_argument("--task", default="classic-car-v0")
    p.add_argument("--num-samples", "--num_samples", type=int, default=100,
                   help="Uniform state-box samples in the SEARCH loop's SDP. "
                        "Tsukamoto's Nls=100. Cost is superlinear — each sample adds "
                        "an x_dim^2 PSD block plus two LMIs to one program, so N=1000 "
                        "is ~550s per solve against ~6s at N=100.")
    p.add_argument("--eval-samples", "--eval_samples", type=int, default=10000,
                   help="Samples for the FINAL synthesis that gets rolled out. MUST match the "
                        "env config's cm_samples (10000 everywhere here) or the check is "
                        "vacuous: it verifies a program that never ships. The default was "
                        "1000 and that is exactly how ball_and_beam and two_link_arm passed "
                        "verification and then failed at deploy N. Pair with eps_for_n, which "
                        "loosens eps as N grows, so the denser program is not also held to a "
                        "sparser draw's margin. Use --no-eval for a search-only run.")
    p.add_argument("--lbd0", type=float, default=10.0, help="Starting lbd.")
    p.add_argument("--r0", type=float, default=0.1, help="Starting r_scaler.")
    p.add_argument("--lbd-factor", "--lbd_factor", type=float, default=1.5,
                   help="Divide lbd by this on LMI infeasibility.")
    p.add_argument("--r-factor", "--r_factor", type=float, default=2.0,
                   help="Multiply r by this on a control-bound violation.")
    p.add_argument("--lbd-min", "--lbd_min", type=float, default=0.01)
    p.add_argument("--r-max", "--r_max", type=float, default=1.0e4)
    p.add_argument("--viol-frac", "--viol_frac", type=float, default=0.05,
                   help="Raise r only when MORE than this fraction of generated "
                        "controls leaves the box. A hard 0%% veto rejects every gain "
                        "that actually tracks on these envs (the performant ones "
                        "saturate transiently by design).")
    p.add_argument("--n-draws", "--n_draws", type=int, default=100,
                   help="Error/feedforward draws per sampled state. num_samples x "
                        "this is the population the fraction is measured over.")
    p.add_argument("--u-expansion", "--u_expansion", type=float, default=2.0,
                   help="Control box = uref box widened by this, sign-aware "
                        "(see expand_box). 2.0 is what every env here uses.")
    p.add_argument("--w-lb", "--w_lb", type=float, default=0.01,
                   help="Deployment envelope lower bound, FIXED — the search never "
                        "moves it. If nothing certifies inside it, that is the answer.")
    p.add_argument("--w-ub", "--w_ub", type=float, default=100.0,
                   help="Deployment envelope upper bound, FIXED.")
    p.add_argument("--cm-eps", "--cm_eps", type=float, default=None,
                   help="ε, the strict-definiteness margin (Tsukamoto's epsilon, which "
                        "is 0 — see eps_for_n). Default AUTO: scheduled from the sample "
                        "count, 0.1 at N<=100, 0.05 at N<=1000, 0.01 beyond. Pass a "
                        "float to pin BOTH phases to it.")
    p.add_argument("--cm-dt", "--cm_dt", type=float, default=None,
                   help="The LMI's dt = the agent's cm_dt (his CV-STEM sampling "
                        "period, NOT the integrator step). Default: the env's dt.")
    p.add_argument("--solver", default="MOSEK")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-envs", "--eval_envs", type=int, default=16)
    p.add_argument("--no-eval", "--no_eval", action="store_true",
                   help="Stop after the search; skip the episode.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    import c2rl.tasks.direct.classic  # noqa: F401  (registers the ids)
    import gymnasium as gym
    env = gym.make(args.task, num_envs=args.eval_envs, device="cpu").unwrapped

    def _np(t):
        return t.detach().cpu().numpy().astype(np.float64)

    lmi_dt = float(env.dt) if args.cm_dt is None else float(args.cm_dt)
    uref_lo, uref_hi = _np(env.UREF_MIN), _np(env.UREF_MAX)
    # XE_INIT, not XE_MIN/XE_MAX. The check asks "of the controls this metric
    # generates, what fraction is unrealizable?", so the error draws have to be
    # the errors an episode actually presents -- XE_INIT is exactly what reset()
    # perturbs by, and it is sized per env (quadrotor's is per-dimension, from
    # measured budget consumption along the CV-STEM tube).
    #
    # XE_MIN/XE_MAX is C3M's training-perturbation box: a flat +-1 on every dim,
    # identical across all four envs, sampled by get_rollout(., "c3m") for the
    # contraction loss. It was never sized to represent tracking error, and on
    # segway it is 6.7x wider in pitch than reset ever produces -- a 1 rad (57
    # deg) tilt error, which the env's own comment calls "mid-fall, not a
    # tracking error". Measured at a fixed metric (lbd=0.0101, r=6.4), swapping
    # only this box moves the violation rate 35.20% -> 0.61%, i.e. it alone was
    # the difference between segway certifying and returning INFEASIBLE at every
    # (lbd, r, envelope) tried. Cartpole is insensitive: 3.01% -> 0.00%.
    e_lo, e_hi = _np(env.XE_INIT_MIN), _np(env.XE_INIT_MAX)
    # The control box is the uref box widened by --u-expansion, which is how every
    # env defines it (env_base.py: "the applied box is 2x this"). Derived rather
    # than read off U_MIN/U_MAX so the check states the relationship it relies on.
    u_lo, u_hi = expand_box(uref_lo, uref_hi, args.u_expansion)
    if not (np.allclose(u_lo, _np(env.U_MIN)) and np.allclose(u_hi, _np(env.U_MAX))):
        print(f"[uniform-lambda] WARNING: {args.u_expansion:g}x uref gives "
              f"[{u_lo.round(3)}, {u_hi.round(3)}] but the env's own actuator box is "
              f"[{_np(env.U_MIN).round(3)}, {_np(env.U_MAX).round(3)}] — using the "
              f"derived one, which is what --u-expansion asks for.")

    # ε per phase, scheduled from that phase's own sample count (eps_for_n).
    # The search and the final synthesis run at different N, so pinning one ε to
    # both makes the denser program strictly harder than the margin it needs.
    pinned = args.cm_eps is not None
    eps_search = args.cm_eps if pinned else eps_for_n(args.num_samples)
    eps_eval = args.cm_eps if pinned else eps_for_n(args.eval_samples)

    print(f"[uniform-lambda] task={args.task}  x_dim={int(env.num_dim_x)} "
          f"u_dim={int(env.num_dim_control)}  env dt={float(env.dt):g}")
    print(f"[uniform-lambda] {args.num_samples} states i.i.d. uniform over the box; "
          f"ONE joint SDP, nu/chi shared, (W-I)/dt at cm_dt={lmi_dt:g}, "
          f"eps={eps_search:g}, solver={args.solver}")
    if pinned:
        print(f"[uniform-lambda] cm_eps PINNED to {args.cm_eps:g} for both phases "
              f"(AUTO would give search {eps_for_n(args.num_samples):g} / "
              f"eval {eps_for_n(args.eval_samples):g}).")
    else:
        print(f"[uniform-lambda] cm_eps AUTO: search {eps_search:g} at N="
              f"{args.num_samples}, eval {eps_eval:g} at N={args.eval_samples} "
              f"(eps is a covering radius — denser draw, less margin needed).")
    print(f"[uniform-lambda] control check: {args.n_draws} draws per state of "
          f"u = uref - K*e, e ~ U[{e_lo.round(3)}, {e_hi.round(3)}], uref ~ "
          f"U[{uref_lo.round(3)}, {uref_hi.round(3)}]; FAILS when more than "
          f"{args.viol_frac:.1%} land outside [{u_lo.round(3)}, {u_hi.round(3)}]")
    w_lb, w_ub = args.w_lb, args.w_ub
    print(f"[uniform-lambda] start lbd={args.lbd0:g} r={args.r0:g}; envelope FIXED at "
          f"w_lb={w_lb:g} w_ub={w_ub:g} (never adapted)")
    print(f"[uniform-lambda] LMI infeasible       -> lbd /= {args.lbd_factor:g}")
    print(f"[uniform-lambda] >{args.viol_frac:.1%} out of box -> r *= {args.r_factor:g}, "
          f"then lbd /= {args.lbd_factor:g} once r passes {args.r_max:g}")

    x_np = sample_state_box(env.X_MIN, env.X_MAX, n=args.num_samples, seed=args.seed)
    A, B = drift_jacobians(env.get_f_and_B, x_np)
    # Drawn once and reused at every (lbd, r) — see control_violation_rate.
    rng = np.random.default_rng(args.seed)
    shape = (args.num_samples, args.n_draws)
    e_draws = rng.uniform(e_lo, e_hi, size=(*shape, e_lo.size))
    uref_draws = rng.uniform(uref_lo, uref_hi, size=(*shape, uref_lo.size))

    lbd, r, sol = args.lbd0, args.r0, None
    while True:
        if lbd < args.lbd_min:
            print(f"\n[uniform-lambda] RESULT: INFEASIBLE — nothing certifies with the "
                  f"control in box down to lbd={args.lbd_min:g} inside the fixed "
                  f"envelope w_lb={w_lb:g}, w_ub={w_ub:g}.")
            return 2
        sol = cvstem_joint(A, B, lbd=lbd, eps=eps_search, dt=lmi_dt,
                           solver=args.solver, r_scaler=r, w_lb=w_lb, w_ub=w_ub)
        state = f"lbd={lbd:8.4f} r={r:9.4g}"
        if sol is None:
            print(f"[uniform-lambda]   {state} -> LMI INFEASIBLE — lowering lbd")
            lbd /= args.lbd_factor
            r = args.r0
            continue
        frac = control_violation_rate(sol["W"], B, r_scaler=r, e_draws=e_draws,
                                      uref_draws=uref_draws, u_lo=u_lo, u_hi=u_hi)
        if frac <= args.viol_frac:
            print(f"[uniform-lambda]   {state} -> FEASIBLE, {frac:.2%} of controls out "
                  f"of box (<= {args.viol_frac:.1%}) "
                  f"(nu={sol['nu']:.4g}, chi={sol['chi']:.4g}, J={sol['J']:.4g})")
            break
        print(f"[uniform-lambda]   {state} -> {frac:.2%} of controls out of box "
              f"— raising r (nu={sol['nu']:.4g}, chi={sol['chi']:.4g})")
        r *= args.r_factor
        if r > args.r_max:
            print(f"[uniform-lambda]   r exhausted at --r-max {args.r_max:g} — "
                  f"lowering lbd and restarting r at {args.r0:g}")
            r = args.r0
            lbd /= args.lbd_factor

    print(f"\n[uniform-lambda] RESULT: lbd = {lbd:.4f}, r_scaler = {r:g}, "
          f"w_lb = {w_lb:g}, w_ub = {w_ub:g}")
    print(f"  nu = {sol['nu']:.4g} (max eig of M), chi = {sol['chi']:.4g} "
          f"(condition number), J = {sol['J']:.4g}")
    print(f"\n  Set in the env's skrl_cvstem_lqr_cfg.yaml:\n"
          f"      agent: r_scaler: {r:g}\n"
          f"      cm:    lbd: {lbd:.3f}\n"
          f"             cm_eps: {eps_search:g}  (search; the config synthesizes at {eps_eval:g})\n"
          f"             cm_dt: {lmi_dt:g}\n"
          f"             cm_w_lb: {w_lb:g}\n"
          f"             cm_w_ub: {w_ub:g}")
    if args.no_eval:
        return 0

    print(f"\n[uniform-lambda] building the agent at that (lbd, r) and rolling ONE "
          f"episode over {args.eval_envs} envs (cm_samples={args.eval_samples}) ...")
    if args.eval_samples > args.num_samples:
        print(f"[uniform-lambda] NOTE: the final program is {args.eval_samples} samples "
              f"against the {args.num_samples} the search certified — strictly harder, "
              f"so it can come back infeasible even though the search cleared.")
    agent = CVSTEMLQRAgent(
        cfg={"lbd": lbd, "r_scaler": r, "cm_eps": eps_eval, "cm_dt": lmi_dt,
             "cm_w_lb": w_lb, "cm_w_ub": w_ub,
             "cm_solver": args.solver, "cm_samples": args.eval_samples,
             "cm_seed": args.seed, "cmg_hidden_dims": [100, 100, 100],
             "cmg_regress_epochs": 3000, "cmg_regress_batch_size": 32,
             "cmg_regress_lr_scheduler": "StepLR",
             "cmg_regress_lr_scheduler_kwargs": {"step_size": 1, "gamma": 0.999},
             "cmg_early_stop_patience": 300},
        models={}, observation_space=env.observation_space,
        action_space=env.action_space, device="cpu", get_f_and_B=env.get_f_and_B,
        x_lo=env.X_MIN.numpy(), x_hi=env.X_MAX.numpy(), dt=lmi_dt,
        x_dim=int(env.num_dim_x), u_dim=int(env.num_dim_control),
        angle_idx=list(env.angle_idx))
    m, umax = evaluate(env, agent, args.eval_envs)
    print(f"\n[uniform-lambda] ONE-EPISODE EVAL: auc={m['auc']:.4f} "
          f"rate={m['contraction_rate']:.4f} overshoot={m['overshoot']:.4f} "
          f"score={m['contraction_score']:.4f}  |u|max={umax:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
