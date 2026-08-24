"""Which states cap the uniform λ, and what a subset buys you if you drop them.

``find_uniform_lambda`` answers "what λ holds over the whole box". This answers
the follow-up: **which samples are the ones holding it down**, and how much
faster the rate gets on a subset that excludes them.

Two modes, because there are two defensible ways to pick that subset and they
answer different questions::

    --mode certificate   read the binding states off the SDP, shrink away from
                         exactly them. Maximum λ gain per unit volume removed.
                         Diagnostic: tells you what the certificate is limited by.

    --mode tube          nest by tube radius around the reference. The subset the
                         closed loop actually visits, so it is not chosen to
                         flatter the number. Deployment claim.

Why both MODES use nested *Sample* Sets
---------------------------------------
λ_max over a smaller set is mathematically ≥ λ_max over a larger one — fewer
constraints, larger feasible set. Draw fresh i.i.d. samples per level and that
guarantee evaporates: each level estimates its own box's worst case from its own
finite draw, so a smaller box can draw a nastier point than its parent did and
the curve goes down. Measured on cartpole with no envelope, N=100 per level:
λ = 4.89, 8.86, 13.31, 11.02, 11.08 — the drop at level 3 is pure resampling
noise, and it is indistinguishable by eye from a real effect.

So both modes draw once at the top and filter. Level k's samples are a literal
subset of level k-1's, monotonicity holds by construction, and any increase is
signal. The cost is depth: an isotropic 0.6 shrink in d active dims keeps 0.6^d
of the points per level, so ``--num-samples`` has to be large enough that the
deepest level still has enough left (``--min-samples`` enforces it).

What "binding" means here
-------------------------
``cvstem_joint`` returns only ``{W, nu, chi, J}`` — no duals. It does not need
to: every constraint is reconstructible from the solution. For each sample::

    W̄_k = W_k·ν
    S_k  = (W̄_k - I)/dt + A_k W̄_k + W̄_k A_kᵀ + 2λW̄_k - ν(2/r)B_k B_kᵀ

and the program asked for ``S_k ⪯ -εI``. So ``-ε - λ_max(S_k) ≥ 0`` is the LMI
slack, and the samples with slack ≈ 0 are the ones the rate is resting on.

The envelope constraints ``I ⪯ W̄_k ⪯ χI`` are reported the same way, and that
distinction is the whole point: if the tight samples are tight on the LMI, the
state box is what caps λ and shrinking it will help. If they are pinned against
χI instead, the deployment envelope is the cap, and no amount of shrinking moves
the number — measured on the car, where λ_max was 2.1183 at every nesting level
of a 5-level shrink, but ≥40 with the envelope dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import c2rl.tasks.direct.classic  # noqa: F401  (registers classic-*-v0)
import gymnasium as gym
import numpy as np
import torch
import yaml
from c2rl.agents.skrl.ncm_synthesis import cvstem_joint, sample_state_box


# ───────────────────────────────────────────────────────────────────────── #
# The diagnostic: per-sample slack on every constraint of the joint program
# ───────────────────────────────────────────────────────────────────────── #
def constraint_slacks(A, B, sol, *, lbd, eps, dt, r_scaler):
    """Per-sample slack on each constraint of ``cvstem_joint``'s program.

    Returns ``{"lmi", "chi", "eye"}``, each ``(n,)`` and ``>= 0`` when the
    constraint holds. Small = that sample is holding the certificate down.

    ``lmi`` is scale-normalized by ``‖S_k‖₂`` so samples with very different
    metric magnitudes stay comparable; the raw margin is ``lmi_abs``.
    """
    nu, chi = sol["nu"], sol["chi"]
    r = r_scaler + 1e-5                    # matches cvstem_joint exactly
    eye = np.eye(A.shape[1])
    Wbar = sol["W"] * nu                   # undo the W = W̄/ν normalization
    lmi_abs, chi_s, eye_s, norms = [], [], [], []
    for k in range(A.shape[0]):
        Wb = Wbar[k]
        S = ((Wb - eye) / dt + A[k] @ Wb + Wb @ A[k].T + 2.0 * lbd * Wb
             - nu * ((2.0 / r) * (B[k] @ B[k].T)))
        S = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(S)
        wv = np.linalg.eigvalsh(0.5 * (Wb + Wb.T))
        lmi_abs.append(-eps - ev[-1])      # S ⪯ -εI  ->  slack = -ε - λmax(S)
        norms.append(max(abs(ev[0]), abs(ev[-1]), 1e-12))
        chi_s.append(chi - wv[-1])         # W̄ ⪯ χI
        eye_s.append(wv[0] - 1.0)          # W̄ ⪰ I
    lmi_abs, norms = np.array(lmi_abs), np.array(norms)
    return {"lmi": lmi_abs / norms, "lmi_abs": lmi_abs,
            "chi": np.array(chi_s), "eye": np.array(eye_s)}


def binding_mask(slacks, *, frac):
    """Samples with the least LMI slack, tightest first.

    ``frac=0`` gives just the single tightest, which is what the greedy cut
    wants; larger fractions are for reporting how concentrated the binding set is.
    """
    n_take = max(1, int(round(frac * slacks["lmi"].size)))
    return np.argsort(slacks["lmi"])[:n_take]


def what_binds(slacks, sol, *, w_lb, w_ub, tol=1e-6, rel=2e-2):
    """What the certificate is actually limited by.

    Checks the two global scalar caps first, and that ordering is the whole
    point. ``ν ≤ 1/w_lb`` and ``χ ≤ ν·w_ub`` are single constraints shared by
    every sample, so they never show up as per-sample tightness — a run can
    report "1 of 200 samples tight on the LMI" while ν sits exactly on its cap
    and the envelope is what is really holding λ down. Measured on cartpole at
    N=200: ν = 100.000 against a 1/w_lb of exactly 100.

    Only if both caps have slack does per-sample tightness decide, and only then
    does shrinking the state box have anything to bite on.

    ``rel`` is 2% rather than a tight tolerance because an interior-point solver
    lands near an active cap, not on it: measured on the car, ν came back at
    99.78% and 98.46% of ``1/w_lb`` at two different envelopes, and both were
    genuinely envelope-limited (λ tracked the cap, 11.24 -> 53.23 as the cap went
    1e4 -> 1e6). At 0.1% both were misreported as state-box-limited.
    """
    n_lmi = int((slacks["lmi_abs"] < tol).sum())
    n_chi = int((slacks["chi"] < tol).sum())
    nu_pinned = w_lb is not None and sol["nu"] >= (1.0 / w_lb) * (1.0 - rel)
    chi_pinned = w_ub is not None and sol["chi"] >= sol["nu"] * w_ub * (1.0 - rel)
    if nu_pinned:
        label = "envelope (ν at 1/w_lb)"
    elif chi_pinned:
        label = "envelope (χ at ν·w_ub)"
    elif n_chi > n_lmi:
        label = "envelope (χI per-sample)"
    else:
        label = "state box (LMI)"
    return label, n_lmi, n_chi, bool(nu_pinned or chi_pinned)


# ───────────────────────────────────────────────────────────────────────── #
# λ_max by bisection over a fixed sample set
# ───────────────────────────────────────────────────────────────────────── #
def solve_at(A, B, lbd, kw):
    return cvstem_joint(A, B, lbd=lbd, **kw)


def max_lambda(A, B, kw, *, lo=0.01, hi=60.0, tol=0.005):
    """Largest λ this sample set certifies. Returns ``(λ, solution, n_solves)``."""
    base = solve_at(A, B, lo, kw)
    if base is None:
        return 0.0, None, 1
    if (top := solve_at(A, B, hi, kw)) is not None:
        return hi, top, 2                      # ceiling hit; true max is higher
    best, calls = (lo, base), 2
    while hi - lo > tol * max(lo, 0.05):
        mid = 0.5 * (lo + hi)
        if (sol := solve_at(A, B, mid, kw)) is not None:
            lo, best = mid, (mid, sol)
        else:
            hi = mid
        calls += 1
    return best[0], best[1], calls


def active_dims_auto(env, *, n=64, seed=0, tol=1e-9):
    """Dims that ``A(x)``/``B(x)`` actually depend on, found by perturbation.

    Shrinking a dim the LMI cannot see removes volume without removing
    difficulty, which fakes a trend in the λ-vs-volume curve. Car is the worked
    example: ``f = [v cosθ, v sinθ, 0, 0]`` ignores x and y entirely, so a
    "shrink" along them is pure denominator. Detected rather than declared,
    because at quadrotor's 10 states declaring it is guesswork.
    """
    lo, hi = _np(env.X_MIN), _np(env.X_MAX)
    x = sample_state_box(env.X_MIN, env.X_MAX, n=n, seed=seed)
    A0, B0 = jacobians(env, x)
    out = []
    for d in range(x.shape[1]):
        xp = x.copy()
        step = 0.05 * (hi[d] - lo[d])
        xp[:, d] = np.clip(xp[:, d] + step, lo[d], hi[d])
        A1, B1 = jacobians(env, xp)
        if max(np.abs(A1 - A0).max(), np.abs(B1 - B0).max()) > tol:
            out.append(d)
    return np.array(out if out else range(x.shape[1]))


def jacobians(env, x_np):
    """``(A, B)`` at each state — the drift Jacobian, matching real synthesis."""
    x = torch.as_tensor(x_np, dtype=torch.float32, device=env.device)
    x = x.clone().requires_grad_(True)
    f, B, _ = env.get_f_and_B(x)
    A = torch.stack([torch.autograd.grad(f[:, i].sum(), x, retain_graph=True)[0]
                     for i in range(f.shape[1])], dim=1)
    return A.detach().cpu().numpy().astype(np.float64), B.detach().cpu().numpy().astype(np.float64)


# ───────────────────────────────────────────────────────────────────────── #
# Mode 1 — certificate-driven: shrink away from exactly what binds
# ───────────────────────────────────────────────────────────────────────── #
def run_certificate(env, x_all, kw, args, active):
    """Level k+1's box = bounding box of level k's non-binding samples.

    That is the greedy "maximum λ per unit volume removed" step: the binding
    samples are, by definition, the only ones whose removal can raise λ.
    """
    lo0, hi0 = _np(env.X_MIN), _np(env.X_MAX)
    lo, hi = lo0.copy(), hi0.copy()
    keep = np.ones(len(x_all), dtype=bool)
    rows = []
    for k in range(args.levels):
        idx = np.flatnonzero(keep)
        if idx.size < args.min_samples:
            print(f"[subsets] stopping at level {k}: {idx.size} samples left "
                  f"(< --min-samples {args.min_samples})")
            break
        A, B = jacobians(env, x_all[idx])
        t0 = time.time()
        lbd, sol, calls = max_lambda(A, B, kw, hi=args.lbd_hi, tol=args.tol)
        if sol is None:
            print(f"[subsets] level {k}: INFEASIBLE even at λ={args.lbd_min}")
            break
        slacks = constraint_slacks(A, B, sol, lbd=lbd, eps=kw["eps"],
                                   dt=kw["dt"], r_scaler=kw["r_scaler"])
        cap, n_lmi, n_chi, env_capped = what_binds(
            slacks, sol, w_lb=kw["w_lb"], w_ub=kw["w_ub"])
        vol = float(np.prod((hi[active] - lo[active]) / (hi0[active] - lo0[active])))
        rows.append(dict(level=k, lbd=float(lbd), n=int(idx.size), vol_frac=vol,
                         nu=sol["nu"], chi=sol["chi"], capped_by=cap,
                         env_capped=env_capped,
                         n_tight_lmi=n_lmi, n_tight_chi=n_chi,
                         lo=lo.tolist(), hi=hi.tolist(), secs=time.time() - t0))
        print(f"[cert] L{k} n={idx.size:5d} vol={vol:7.4f}  λ_max={lbd:8.4f}  "
              f"ν={sol['nu']:8.3f} χ={sol['chi']:8.3f}  capped by {cap} "
              f"(tight: {n_lmi} LMI / {n_chi} χ)  [{calls} solves, {time.time()-t0:.0f}s]",
              flush=True)
        warn_row(rows, "cert")
        if env_capped:
            print("[cert]      ^ envelope ACTIVE (ν/χ at a cap). This does NOT mean "
                  "shrinking is futile — measured on cartpole, λ still rose 0.0441 -> "
                  "0.1418 (3.2x) with ν pinned at 100 on every level, because the cap "
                  "and the LMI bind together. It does mean part of the ceiling is the "
                  "envelope rather than the plant.")
        if args.drop_mode == "samples":
            # Drop the tightest `drop_frac` outright, keeping no box at all.
            # This is the loosest honest subset rule and the fastest-rising
            # curve: it removes every binding state each level instead of one,
            # and is not restricted to axis-aligned cuts. The price is that the
            # retained set is a point cloud, not a region you can write down —
            # so it upper-bounds what any subset of this size could certify,
            # rather than describing a set you could deploy on. Use it to show
            # the headroom exists; use --drop-mode box to get a usable region.
            tight = idx[binding_mask(slacks, frac=args.drop_frac)]
            keep[tight] = False
            rest = x_all[np.flatnonzero(keep)]
            bb_lo, bb_hi = lo0.copy(), hi0.copy()
            if rest.size:
                bb_lo[active], bb_hi[active] = rest[:, active].min(0), rest[:, active].max(0)
            lo, hi = bb_lo, bb_hi          # reported only; membership is by index
            print(f"[cert]      dropped {tight.size} tightest "
                  f"({idx.size} -> {int(keep.sum())} samples)")
            continue
        # Cut the box so the tightest samples fall outside it.
        #
        # The obvious version — drop the tight samples, take the bounding box of
        # what is left — silently stalls: `keep` is then recomputed from the box,
        # so any tight sample that was interior is back inside it and nothing
        # changed. Measured on segway, whose binding states are interior: levels
        # 2, 3 and 4 were byte-identical (n=94, λ=6.9815) and the "curve" was an
        # artifact of the same solve repeated three times.
        #
        # So cut along one face instead, excluding only the single tightest
        # sample and scoring candidates by how few samples they cost.
        #
        # Both of those details are load-bearing. Excluding the whole tight set
        # (drop_frac = 10% of samples) demands a cut past all of them at once,
        # and when they are spread along an axis the cheapest such cut is still
        # enormous: measured on cartpole, one step went 100 -> 14 samples and
        # ended the run at level 1. And scoring by volume picks cuts through
        # dense regions, since volume says nothing about where the samples are.
        # One sample at a time, scored by sample loss, is the actual greedy step.
        # Repeat that single-sample cut until ~drop_frac of the level's samples
        # are gone, so a box level is a step comparable to a `samples` level
        # while the retained set stays a box that can be written down and
        # checked at runtime. Ranking comes from this level's solve and is not
        # refreshed between cuts inside a level — refreshing would cost one SDP
        # per cut. The ranking goes stale as the set shrinks, which makes this a
        # cheaper approximation of the greedy step, never an invalid region:
        # every reported box is still exactly the set the next λ is solved over.
        order = idx[binding_mask(slacks, frac=1.0)]      # all, tightest first
        target = max(1, int(round(args.drop_frac * idx.size)))
        cuts = []
        while idx.size - int(keep.sum()) < target:
            still_in = [t for t in order if bool(keep[t])]
            if not still_in:
                break
            tightest = still_in[0]
            best_cut = None
            for d in active:
                for side in ("lo", "hi"):
                    val = (float(x_all[tightest, d]) + 1e-9 if side == "lo"
                           else float(x_all[tightest, d]) - 1e-9)
                    if not (lo[d] < val < hi[d]):
                        continue
                    trial_lo, trial_hi = lo.copy(), hi.copy()
                    if side == "lo":
                        trial_lo[d] = val
                    else:
                        trial_hi[d] = val
                    n_left = int(np.all((x_all >= trial_lo) & (x_all <= trial_hi),
                                        axis=1).sum())
                    if n_left < int(keep.sum()) and (best_cut is None or n_left > best_cut[0]):
                        best_cut = (n_left, (d, side, val))
            if best_cut is None:
                break
            _, (d, side, val) = best_cut
            if side == "lo":
                lo[d] = val
            else:
                hi[d] = val
            keep = np.all((x_all >= lo) & (x_all <= hi), axis=1)
            cuts.append(f"{names_of(env)[d]} {side}->{val:+.3f}")
        if not cuts:
            print("[cert]      no axis-aligned cut excludes the binding samples "
                  "(they straddle the box) — stopping.")
            break
        print(f"[cert]      {len(cuts)} cut(s): {', '.join(cuts)} "
              f"({idx.size} -> {int(keep.sum())} samples)")
    return rows


def warn_row(rows, tag):
    """Two failure modes that otherwise read as findings."""
    r = rows[-1]
    if r["nu"] > 1e5:
        print(f"[{tag}]      WARNING: nu={r['nu']:.3g}. With no envelope nothing bounds "
              f"the metric scale, so K = R^-1 B^T M is ~nu and this rate is not "
              f"deployable at any gain a real actuator has. The SDP is also badly "
              f"conditioned at this scale, so treat the number as unreliable, not just "
              f"impractical. Pass --w-lb/--w-ub to bound it.")
    if len(rows) > 1 and r["lbd"] < rows[-2]["lbd"] - 1e-6:
        print(f"[{tag}]      WARNING: lambda DROPPED {rows[-2]['lbd']:.4f} -> "
              f"{r['lbd']:.4f} on a NESTED subset. Removing constraints cannot lower "
              f"lambda, so this is solver inaccuracy, not an effect. Do not read the "
              f"trend through it.")


def names_of(env):
    return list(getattr(env, "state_names", None) or
                [f"x{i}" for i in range(int(env.num_dim_x))])


# ───────────────────────────────────────────────────────────────────────── #
# Mode 2 — tube: nest by radius around the reference
# ───────────────────────────────────────────────────────────────────────── #
def run_tube(env, kw, args, active):
    """Tube of radius ρ about the reference, as a genuinely nested sample family.

    The obvious construction — ``x = xref + ρ·xe`` for shrinking ρ — is not
    nested. Scaling ρ moves every sample to a new location instead of dropping
    any, so each level is a fresh draw of its own region and λ_max is re-
    estimated from scratch. Measured on cartpole: 0.0304, 0.0441, 0.0384, 0.0347
    — the dips are resampling noise, exactly the artifact the box mode already
    taught us to avoid.

    So draw one cloud with radii ``s ~ U(0,1)`` and let level k keep the samples
    with ``s <= ρ_k``. Those points lie in the radius-ρ_k tube, each level's set
    is a literal subset of the last, and λ_max is monotone by construction.
    Sample count falls off as ρ_k, which is what ``--min-samples`` guards.

    This is the set the closed loop lives in, so the λ it certifies is the one
    that applies in deployment — provided the tube is invariant, which is the
    claim you owe.
    """
    g = torch.Generator(device=env.device).manual_seed(args.seed)
    n = args.num_samples

    def _u(lo, hi):
        return lo + torch.rand(n, lo.numel(), device=env.device, generator=g) * (hi - lo)

    xref = _u(env.X_MIN, env.X_MAX)
    xe = _u(env.XE_MIN, env.XE_MAX)
    s_rad = torch.rand(n, 1, device=env.device, generator=g)   # radius per sample
    x_all = torch.clamp(xref + s_rad * xe, env.X_MIN, env.X_MAX)
    rows = []
    for k in range(args.levels):
        rho = args.shrink ** k
        keep = (s_rad[:, 0] <= rho)
        n_k = int(keep.sum())
        if n_k < args.min_samples:
            print(f"[tube] stopping at rho={rho:.4f}: {n_k} samples inside "
                  f"(< --min-samples {args.min_samples})")
            break
        x = x_all[keep]
        A, B = jacobians(env, _np(x))
        t0 = time.time()
        lbd, sol, calls = max_lambda(A, B, kw, hi=args.lbd_hi, tol=args.tol)
        if sol is None:
            # A smaller tube is strictly easier, so one infeasible radius says
            # nothing about the next. Breaking here would throw the experiment
            # away because its widest level happened not to certify.
            print(f"[tube] L{k} rho={rho:6.4f}: INFEASIBLE at lambda>={args.lbd_min} "
                  f"— skipping to a tighter tube", flush=True)
            continue
        slacks = constraint_slacks(A, B, sol, lbd=lbd, eps=kw["eps"],
                                   dt=kw["dt"], r_scaler=kw["r_scaler"])
        cap, n_lmi, n_chi, env_capped = what_binds(
            slacks, sol, w_lb=kw["w_lb"], w_ub=kw["w_ub"])
        rows.append(dict(level=k, rho=rho, lbd=float(lbd), n=n_k, nu=sol["nu"],
                         chi=sol["chi"], capped_by=cap, env_capped=env_capped,
                         n_tight_lmi=n_lmi, n_tight_chi=n_chi,
                         secs=time.time() - t0))
        print(f"[tube] L{k} rho={rho:6.4f} n={n_k:4d}  λ_max={lbd:8.4f}  ν={sol['nu']:8.3f} "
              f"χ={sol['chi']:8.3f}  capped by {cap}  "
              f"[{calls} solves, {time.time()-t0:.0f}s]", flush=True)
        warn_row(rows, "tube")
    return rows


# ───────────────────────────────────────────────────────────────────────── #
def _np(t):
    return t.detach().cpu().numpy().astype(np.float64) if hasattr(t, "detach") \
        else np.asarray(t, dtype=np.float64)


def plot(rows, args, active, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = ([r["vol_frac"] for r in rows] if args.mode == "certificate"
          else [r["rho"] for r in rows])
    ys = [r["lbd"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(xs, ys, "o-")
    for r, x, y in zip(rows, xs, ys):
        ax[0].annotate(f"L{r['level']}", (x, y), textcoords="offset points",
                       xytext=(6, -10), fontsize=9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("volume fraction retained" if args.mode == "certificate"
                     else r"tube radius $\rho$")
    ax[0].set_ylabel(r"certified $\lambda_{max}$")
    ax[0].set_title(f"{args.task} — {args.mode} nesting\n"
                    f"nested SAMPLE sets, so this curve is monotone by construction")
    ax[0].grid(alpha=.3)

    caps = [r["capped_by"] for r in rows]
    ax[1].bar(range(len(rows)), [r["n_tight_lmi"] for r in rows], label="tight on LMI")
    ax[1].bar(range(len(rows)), [r["n_tight_chi"] for r in rows],
              bottom=[r["n_tight_lmi"] for r in rows], label=r"tight on $\chi I$")
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([f"L{r['level']}\n{c.split()[0]}" for r, c in zip(rows, caps)],
                          fontsize=8)
    ax[1].set_ylabel("samples at the constraint")
    ax[1].set_title("What actually caps λ at each level\n"
                    "(χ-dominated => shrinking the box cannot help)")
    ax[1].legend()
    ax[1].grid(alpha=.3, axis="y")
    fig.suptitle(f"Active dims: {', '.join(names[d] for d in active)}", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[subsets] wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="classic-cartpole-v0")
    p.add_argument("--mode", choices=("certificate", "tube"), default="certificate")
    p.add_argument("--config", default=None,
                   help="cvstem-lqr yaml to read λ/r/envelope/eps/dt from. "
                        "Default: the task's own skrl_cvstem_lqr_cfg.yaml.")
    p.add_argument("--num-samples", "--num_samples", type=int, default=300,
                   help="Drawn ONCE at the top; levels filter this pool. ONE joint "
                        "SDP over all of them, and measured solve time goes as "
                        "N^1.95 (6s at N=100, 571s at N=1000, 3444s at N=2500), "
                        "so a bisection of ~13 solves per level puts the practical "
                        "ceiling near 300-400, not thousands.")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--drop-frac", "--drop_frac", type=float, default=0.10,
                   help="certificate mode: fraction of tightest samples dropped per level.")
    p.add_argument("--shrink", type=float, default=0.6, help="tube mode: radius factor.")
    p.add_argument("--drop-mode", "--drop_mode", choices=("box", "samples"), default="box",
                   help="certificate mode: 'box' cuts one axis-aligned face per level "
                        "(retained set is a BOX you can deploy on, slower rise); "
                        "'samples' drops the tightest --drop-frac outright (retained "
                        "set is a point cloud, faster rise, upper bound only).")
    p.add_argument("--min-samples", "--min_samples", type=int, default=50,
                   help="Stop before a level too sparse to mean anything.")
    p.add_argument("--active-dims", "--active_dims", default=None,
                   help="Comma-separated dims the Jacobian depends on. Default: "
                        "detected by perturbation (active_dims_auto). Dims A(x)/B(x) "
                        "ignore only dilute the volume number and fake a trend.")
    p.add_argument("--free-envelope", "--free_envelope", action="store_true",
                   help="Drop w_lb/w_ub. Isolates the plant's own limit — but the "
                        "gain is then unbounded, so the λ is NOT deployable.")
    p.add_argument("--cm-eps", "--cm_eps", type=float, default=None,
                   help="Override the config's cm_eps. Needed to compare envs, since "
                        "they do not all ship the same one (segway is 0.01 where "
                        "car/cartpole/quadrotor are 0.1) and eps changes what "
                        "certifies.")
    p.add_argument("--w-lb", "--w_lb", type=float, default=None,
                   help="Override/supply the envelope lower bound. Needed for envs "
                        "whose config ships none (segway), where the free program "
                        "drives nu to ~1e7 and the solve stops being trustworthy.")
    p.add_argument("--w-ub", "--w_ub", type=float, default=None)
    p.add_argument("--lbd-hi", "--lbd_hi", type=float, default=60.0)
    p.add_argument("--lbd-min", "--lbd_min", type=float, default=0.01)
    p.add_argument("--tol", type=float, default=0.005, help="Bisection tolerance (relative).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/lambda_subsets.png")
    p.add_argument("--json", default=None)
    p.add_argument("--self-check", "--self_check", action="store_true",
                   help="Run the built-in consistency checks and exit.")
    args = p.parse_args(argv)

    if args.self_check:
        return self_check()

    short = args.task.replace("classic-", "").replace("-v0", "")
    cfg_path = args.config or (
        f"source/c2rl/c2rl/tasks/direct/classic/{short}"
        f"/agents/skrl_cvstem_lqr_cfg.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cm = cfg["cm"]
    env = gym.make(args.task, num_envs=1, device="cpu").unwrapped
    names = list(getattr(env, "state_names", None) or
                 [f"x{i}" for i in range(int(env.num_dim_x))])
    active = (np.array([int(d) for d in args.active_dims.split(",")])
              if args.active_dims else active_dims_auto(env, seed=args.seed))

    # An absent cm_w_lb/cm_w_ub is not an error: it means the reference program
    # with no envelope, which is what segway ships. cm_dt absent would silently
    # fall back to the env integrator step and solve a 30x harsher LMI, so that
    # one is called out rather than defaulted quietly.
    if "cm_dt" not in cm:
        print(f"[subsets] WARNING: {cfg_path} has no cm_dt; using env dt={env.dt:g}. "
              f"That is a much harsher (W-I)/dt than the configs' 1.0.")
    kw = dict(eps=args.cm_eps if args.cm_eps is not None else cm["cm_eps"],
              dt=cm.get("cm_dt", float(env.dt)), solver=cm.get("cm_solver", "MOSEK"),
              r_scaler=cfg["agent"]["r_scaler"],
              w_lb=None if args.free_envelope else (args.w_lb if args.w_lb is not None
                                                   else cm.get("cm_w_lb")),
              w_ub=None if args.free_envelope else (args.w_ub if args.w_ub is not None
                                                   else cm.get("cm_w_ub")))
    print(f"[subsets] {args.task}  mode={args.mode}  N={args.num_samples}  "
          f"eps={kw['eps']} dt={kw['dt']} r={kw['r_scaler']} "
          f"envelope={'FREE' if args.free_envelope else [kw['w_lb'], kw['w_ub']]}")
    print(f"[subsets] active dims: {[names[d] for d in active]}")

    if args.mode == "certificate":
        x_all = sample_state_box(env.X_MIN, env.X_MAX, n=args.num_samples, seed=args.seed)
        rows = run_certificate(env, x_all, kw, args, active)
    else:
        rows = run_tube(env, kw, args, active)

    if not rows:
        print("[subsets] nothing certified — no plot written.")
        return 2
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    plot(rows, args, active, names, args.out)
    gain = rows[-1]["lbd"] / rows[0]["lbd"] if rows[0]["lbd"] > 0 else float("nan")
    print(f"[subsets] λ {rows[0]['lbd']:.4f} -> {rows[-1]['lbd']:.4f}  ({gain:.2f}x)")
    return 0


# ───────────────────────────────────────────────────────────────────────── #
def self_check():
    """Cheap end-to-end checks on the two things that can silently be wrong."""
    rng = np.random.default_rng(0)
    n, d = 6, 3
    # Unstable drift, or lambda_max runs off to the bisection ceiling and the
    # monotonicity test below is vacuous.
    A = rng.normal(size=(n, d, d)) * 0.5 + 0.5 * np.eye(d)
    B = rng.normal(size=(n, d, 1))
    kw = dict(eps=0.05, dt=1.0, solver="CLARABEL", r_scaler=1.0, w_lb=None, w_ub=None)

    lbd, sol, _ = max_lambda(A, B, kw, hi=5.0, tol=0.02)
    assert sol is not None and lbd > 0, "self-check needs a feasible reference point"
    assert lbd < 5.0, f"lambda hit the ceiling ({lbd}) — test system is too easy"

    # 1. Every constraint reconstructed from {W, nu, chi} must hold, since the
    #    solver was asked for exactly these. Guards the W = Wbar/nu unscaling and
    #    the r = r_scaler + 1e-5 offset, both silent if wrong.
    s = constraint_slacks(A, B, sol, lbd=lbd, eps=kw["eps"], dt=kw["dt"],
                          r_scaler=kw["r_scaler"])
    assert s["lmi_abs"].min() > -1e-4, f"LMI reconstructed as violated: {s['lmi_abs'].min()}"
    assert s["eye"].min() > -1e-4, f"W >= I reconstructed as violated: {s['eye'].min()}"
    assert s["chi"].min() > -1e-4, f"W <= chiI reconstructed as violated: {s['chi'].min()}"
    # ...and at lambda_max at least one sample must be at the LMI, or the
    # bisection stopped early and "binding" means nothing.
    assert s["lmi_abs"].min() < 1e-2, f"nothing is tight at lambda_max: {s['lmi_abs'].min()}"

    # 2. The property nested sample sets exist to guarantee: dropping samples
    #    only removes constraints, so the same lambda must stay feasible. Tested
    #    directly rather than by comparing two bisections, which would fold
    #    solver noise into the assertion.
    for drop in (1, 3):
        assert solve_at(A[:n - drop], B[:n - drop], lbd, kw) is not None, (
            f"subset of {n - drop} samples infeasible at lambda={lbd} that the "
            f"full {n} certified — monotonicity is broken")

    print(f"self-check OK  (lambda_max {lbd:.4f}, tightest LMI slack "
          f"{s['lmi_abs'].min():.2e}, subsets stay feasible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
