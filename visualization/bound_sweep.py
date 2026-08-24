"""Feature 3 — error_geometry.py's control-space geometry, one panel per w_lb/w_ub pair.

This is error_geometry.py with one axis swapped. That script holds the bounds
fixed and gives a panel to each synthesis objective (ccm vs cvstem_pretrained vs
cvstem_online vs random); this one pins the objective to ``cmg_method="ccm"``
(ncm_synthesis.train_cmg_ccm — the C1/C2 contraction losses, no SDP, no
regression) and gives a panel to each ``[w_lb, w_ub]`` eigenvalue envelope.
Everything else — the trunk, the control grid, the lookahead, the normalization,
the figures — is unchanged, so the panels differ by the envelope and nothing else.

Leading them is a euclidean panel (M = I, --no-euclidean to drop it): no metric
at all, the naive geometry every conditioned panel is a departure from. It costs
one landscape and turns "is this basin the metric's doing?" from a judgement call
into a comparison on screen. Note where it sits on the axis the envelopes are
spread along: cond(M) = 1, below even the tightest envelope's ceiling of 3 — it
is the isotropy those envelopes are clamping the C1/C2 fit toward.

Each envelope gets its own C1/C2 fit from the same config, over the same states,
cached separately. The panel is therefore the metric c2rl would actually deploy
at those bounds, not a rescaling of one fit: w_lb/w_ub enter
BoundedCCM_Generator's forward pass as hard sigmoid eigenvalue bounds
(nn_modules.py), so they shape what the fit can even represent, and refitting
under each is the only faithful way to show them.

Why this exists: C2RL is very sensitive to w_lb/w_ub (works at 0.5/1.5, badly
elsewhere) while the normalized-error curves show little difference. Those curves
are √(eᵀMe / e₀ᵀMe₀), which is invariant to M → c·M — blind to the metric's scale
by construction, and CV-STEM's SDP even solves for a scale-normalized W̄ ⪰ I with
the scale parked in ν (ncm_synthesis.solve_cm_metric). The landscapes here are
built from the same normalized quantity, so read them as the geometry the bounds
produce — the shape of the basin, where its optimum sits, how sharp it is — not
as a scale comparison, which no panel in this file can show.

At each trunk state, every control in the actuator box is held for --lookahead H
steps and the resulting error measured, exactly as in error_geometry.py:

    error(u, k) = sqrt( e'^T M(x') e' / e0^T M(x0) e0 ),  e' = wrap(x' - xref_{k+H})

  * u_dim == 1 (cartpole, segway) -> static t x u x error surface per envelope (.svg)
  * u_dim == 2 (car, turtlebot)   -> MP4: u0 x u1 x error surface per envelope per
                                    timestep, with --history translucent shells

The trunk (--trunk, viz_common.trunk_states) is comma-Separated, one output file
per trunk (default: uref,cvstem_lqr,greedy), and within each file it is always
one trunk shared by every panel:

  * ``cvstem_lqr`` — that analytical controller's trajectory. Metric-independent,
    and stays in the well-tracked region a real controller occupies.
  * ``uref`` — zero feedback. The only fully algorithm- And metric-independent
    choice, at the cost of a trunk whose error grows unchecked (car: |e| 0.819 →
    9.892) into a region no working controller would visit.
  * ``greedy`` — best control on the grid, metric-Dependent, so one designated
    envelope drives the trunk for every panel (--trunk-bounds); letting each panel
    follow its own metric would put them on different trajectories and their
    differences would no longer be attributable to the envelope alone.

Measured on car (seed 42), worth knowing before reading a panel: the C1/C2 fit
wants cond(M) ≈ 45-47 under every envelope, so 0.5:1.5 sits pinned against its
box (realized 2.99 of a permitted 3) while 0.1:10 is not binding at all (46.6
inside a box of 100). The tight envelope is the one actively clamping the metric
toward isotropy — that is the difference these panels are drawing.

Classic envs only. Standalone: nothing in the training code depends on this.

Examples:
    python visualization/bound_sweep.py --env car     # 3 envelopes x 3 trunks
    python visualization/bound_sweep.py --env car --bounds 0.5:1.5,0.1:10
    python visualization/bound_sweep.py --env car --trunk uref
    python visualization/bound_sweep.py --env segway --bounds 0.1:1,0.5:1.5
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import viz_common
from viz_common import (
    CLASSIC_ENVS,
    OUTPUT_DIR,
    TRUNK_MODES,
    compute_landscape_1d,
    compute_landscape_2d,
    control_grid,
    frame_indices,
    get_scenario,
    landscape_steps,
    load_algo_cfg,
    make_env,
    make_metric,
    normalized_error,
    trunk_states,
    wrap_diff,
)

# The default sweep, ordered by envelope width — the sequential ramp in
# assign_colors encodes that order, so keep any --bounds list sorted the same way.
DEFAULT_BOUNDS = "0.1:1.0,0.5:1.5,0.1:10"

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

# Panel key for the unconditioned baseline. A str, so it never collides with the
# (w_lb, w_ub) float tuples the envelope panels are keyed by.
EUCLIDEAN = "euclidean"
EUCLIDEAN_LABEL = "euclidean (M = I)"


class EuclideanMetric:
    """M(x) = I — the naive, unconditioned baseline panel.

    Not a synthesis result and not an envelope: it is the geometry when nothing
    conditions it. √(eᵀMe / e₀ᵀMe₀) collapses to plain ‖e‖/‖e₀‖, so this panel's
    surface is the raw Euclidean error landscape and its error curve coincides
    with the r_euc every other panel already carries. That redundancy is the
    point — it puts the reference in the comparison, as a surface next to the
    conditioned surfaces, instead of a number to be recalled. A conditioned
    metric earns its shape only insofar as its basin departs from this one.

    cond(M) = 1 exactly, tighter than any envelope here permits (the tightest,
    0.5:1.5, allows 3), so it also anchors the low end of the cond axis the
    panels are spread along — and it is exactly what the tight envelopes are
    clamping the C1/C2 fit toward. Draw it first for that reason.
    """

    batched = True
    name = EUCLIDEAN_LABEL

    def __init__(self, x_dim: int):
        self.x_dim = x_dim

    def M(self, x):
        import torch
        return torch.eye(self.x_dim, dtype=x.dtype, device=x.device).expand(
            x.shape[0], -1, -1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=CLASSIC_ENVS)
    p.add_argument("--bounds", default=DEFAULT_BOUNDS,
                   help=f"comma-separated w_lb:w_ub pairs — one C1/C2 fit and one PANEL "
                        f"each (default: {DEFAULT_BOUNDS})")
    p.add_argument("--trunk", default="uref,cvstem_lqr,greedy",
                   help=f"COMMA-SEPARATED baseline controls to sample the landscape "
                        f"ALONG — one output file per trunk (default: "
                        f"uref,cvstem_lqr,greedy, the three reference points: no "
                        f"feedback / a real controller / the best on the grid). "
                        f"Choose from {TRUNK_MODES}. The metrics are fitted ONCE and "
                        f"reused across trunks. See viz_common.trunk_states.")
    p.add_argument("--no-euclidean", action="store_true",
                   help="drop the leading euclidean (M = I) baseline panel — the "
                        "unconditioned geometry every envelope panel is a "
                        "departure from (default: keep it)")
    p.add_argument("--trunk-bounds", default=None,
                   help="--trunk greedy only: the ONE w_lb:w_ub pair whose metric drives "
                        "the trunk, shared by every panel so the panels stay comparable "
                        "(default: the first of --bounds)")
    p.add_argument("--trunk-lookahead", type=int, default=1,
                   help="--trunk greedy only: steps the greedy control looks ahead "
                        "before committing ONE step (default 1 = most one-step decrement)")
    p.add_argument("--num-chunks", type=int, default=41,
                   help="control levels per dimension (default 41)")
    p.add_argument("--lookahead", type=int, default=10,
                   help="steps each candidate control is HELD for before the error "
                        "is measured (default 10). A 1-step lookahead cannot show a "
                        "basin: the optimum needs ||u|| ~ ||e||/(H*dt), which at H=1 "
                        "is far outside the actuator box, so every slice is a "
                        "monotone wall. See viz_common._landscape_step.")
    p.add_argument("--u-range", choices=["physical", "uref"], default="physical",
                   help="control box to sweep: the actuator box step() enforces "
                        "(2x the uref box, default) or the declared uref box")
    p.add_argument("--ccm-samples", type=int, default=None,
                   help="states for each C1/C2 fit (default: the config's own "
                        "cmg_memory_size). Shared by every envelope — the fits must "
                        "differ ONLY by the bounds. No SDP here, so this is cheap per "
                        "sample; shrinking it starves every fit equally")
    p.add_argument("--error-range", type=float, nargs=2, metavar=("LO", "HI"),
                   default=None,
                   help="optionally PIN the surface's normalized-error axis "
                        "(colour + 3D height). Default: retuned per frame over the "
                        "shells on screen, so their change is visible.")
    p.add_argument("--history", type=int, default=10,
                   help="u_dim==2 only: how many earlier frames' geometries to keep "
                        "on screen, drawn translucent with alpha ramping toward the "
                        "present (0 = none, 1 = just the previous one)")
    p.add_argument("--num-frames", type=int, default=150,
                   help="u_dim==2 only: video frames sampled over the episode")
    p.add_argument("--fps", type=int, default=20, help="u_dim==2 only: video frame rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-bound", type=float, default=None,
                   help="episode length in seconds; MUST divide evenly by the env's dt")
    p.add_argument("--solver", default=None,
                   help="cvxpy SDP solver override (SCS/CLARABEL/MOSEK); cvstem_lqr trunk only")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    return p.parse_args()


def parse_bounds(spec: str) -> list[tuple[float, float]]:
    out = []
    for tok in (s.strip() for s in spec.split(",")):
        if not tok:
            continue
        try:
            lo, hi = (float(v) for v in tok.split(":"))
        except ValueError:
            raise SystemExit(f"--bounds entry {tok!r} is not w_lb:w_ub (e.g. 0.5:1.5)")
        if not 0 < lo < hi:
            raise SystemExit(f"--bounds entry {tok!r} needs 0 < w_lb < w_ub")
        out.append((lo, hi))
    if not out:
        raise SystemExit("--bounds needs at least one w_lb:w_ub pair")
    return list(dict.fromkeys(out))


def bound_label(lo: float, hi: float) -> str:
    return f"w_lb={lo:g}, w_ub={hi:g}"


def panel_label(key) -> str:
    return EUCLIDEAN_LABEL if key is EUCLIDEAN else bound_label(*key)


def panel_box(key) -> str:
    """What cond(M) the panel's metric is allowed — the envelope's mechanism."""
    return "cond = 1 by construction" if key is EUCLIDEAN else f"cond box ≤ {key[1] / key[0]:g}"


def resolve_trunks(requested: str) -> list[str]:
    names = [n.strip() for n in requested.split(",") if n.strip()]
    unknown = [n for n in names if n not in TRUNK_MODES]
    if unknown:
        raise SystemExit(f"unknown trunks {unknown}; choose from {TRUNK_MODES}")
    if not names:
        raise SystemExit("--trunk needs at least one mode")
    return list(dict.fromkeys(names))


def assign_colors(labels: list[str]) -> None:
    """Register the panel labels in viz_common.SERIES_COLORS' shared table.

    viz_plot.draw_error_panel colors each series by looking its NAME up there and
    falls back to a single ink for a miss — so without this every envelope's error
    curve would come out identical black. Registering here rather than editing
    viz_common keeps the envelope labels out of a table whose entries are a fixed,
    CVD-validated categorical order for metrics and policies: these are not
    categories at all but an ordered magnitude (envelope width), which takes a
    sequential single-hue ramp light→dark so "wider" is legible without the legend.
    Started at 0.30 — the ramp's lightest steps wash out on the light surface.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    ramp = LinearSegmentedColormap.from_list("bounds", ["#dce7f5", "#1d3f6e"])
    for i, name in enumerate(labels):
        viz_common.SERIES_COLORS[name] = to_hex(
            ramp(0.30 + 0.70 * (i / max(len(labels) - 1, 1))))


def make_bounded_metric(env, scen, base_cfg, lo, hi, *, ccm_samples):
    """One ccm CMG fitted under the [lo, hi] envelope.

    The envelope is injected into a copy of the config's ``cm:`` block — where
    CCMTrainedMetric reads w_lb/w_ub from, and where c2rl reads them too — so the
    panel is the metric c2rl would deploy at those bounds.

    Each pair gets its own cache directory. CCMTrainedMetric's cache is one file
    per env keyed on a config hash that includes w_lb/w_ub — correct, but
    single-slot, so a sweep sharing it would evict and retrain every panel on
    every rerun. A directory per pair makes the sweep cache properly.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("cm", {})
    cfg["cm"]["w_lb"], cfg["cm"]["w_ub"] = lo, hi
    cache_dir = os.path.join(_VIZ_DIR, "cache", f"bounds_wlb{lo:g}_wub{hi:g}")
    metric = make_metric("ccm", env, scen, metric_cfg=cfg, ccm_samples=ccm_samples,
                         cache_dir=cache_dir)
    metric.name = f"ccm (C1/C2) — {bound_label(lo, hi)}"
    return metric


def main():
    args = parse_args()
    t0 = time.time()

    env = make_env(args.env, args.seed, time_bound=args.time_bound)
    scen = get_scenario(env)
    bounds = parse_bounds(args.bounds)
    trunks = resolve_trunks(args.trunk)
    print(f"[bounds] env={args.env} T={scen.T} dt={scen.dt} u_dim={scen.u_dim} "
          f"|e0|={scen.e0.norm():.3f} bounds={[bound_label(*b) for b in bounds]} "
          f"trunks={trunks}")

    # c2rl_ppo is the config whose cm: block these bounds belong to — the same one
    # CCMTrainedMetric defaults to, so only w_lb/w_ub are overridden here.
    base_cfg = load_algo_cfg(args.env, "c2rl_ppo")
    levels = control_grid(env, scen, args.num_chunks, args.u_range)

    # ── one C1/C2 fit per envelope, built once and reused across every trunk ── #
    # A metric is a property of the env + config, not of the trajectory it is
    # evaluated along, so fitting per trunk would repeat the expensive part for
    # identical weights.
    metrics: dict[tuple[float, float], object] = {}
    for lo, hi in bounds:
        try:
            metrics[(lo, hi)] = make_bounded_metric(env, scen, base_cfg, lo, hi,
                                                    ccm_samples=args.ccm_samples)
        except (RuntimeError, FileNotFoundError) as e:
            # One unfittable envelope must not cost the other geometries — an
            # aggressive w_lb can make the C1/C2 fit diverge on some envs, and
            # that is itself worth seeing the other panels next to.
            print(f"[bounds] SKIPPING {bound_label(lo, hi)}: {e}")
    bounds = [b for b in bounds if b in metrics]
    if not bounds:
        raise SystemExit("[bounds] no envelope could be fitted — nothing to plot.")
    # The ramp is an ordered encoding of envelope width, so it spans the envelopes
    # only; euclidean is a baseline of a different kind and takes the recessive
    # gray + dashed treatment viz_plot gives every baseline.
    assign_colors([bound_label(*b) for b in bounds])

    # Baseline first: the panels read left-to-right as departures from it.
    keys = ([] if args.no_euclidean else [EUCLIDEAN]) + bounds
    if not args.no_euclidean:
        metrics[EUCLIDEAN] = EuclideanMetric(scen.x_dim)

    outs = [render_trunk(args, env, scen, levels, keys, bounds, metrics, t) for t in trunks]
    print(f"[bounds] wrote {len(outs)} file(s) in {time.time() - t0:.1f}s:")
    for o in outs:
        print(f"[bounds]   {o}")


def render_trunk(args, env, scen, levels, keys, bounds, metrics, trunk):
    """Build the trunk, every panel's landscape along it, and render one figure.

    ``keys`` is what gets drawn (optionally euclidean, then the envelopes);
    ``bounds`` is the envelopes alone, for the choices only an envelope can make.
    """
    # ── the trunk: One trajectory, shared by every panel ─────────────────── #
    # Shared is the whole point — the panels differ only by their conditioning
    # metric's envelope, so they must be sampled at identical states. `greedy` is
    # the one metric-Dependent mode, hence --trunk-bounds driving a single trunk.
    trunk_metric = None
    if trunk == "greedy":
        tb = parse_bounds(args.trunk_bounds)[0] if args.trunk_bounds else bounds[0]
        if tb not in metrics:
            raise SystemExit(f"--trunk-bounds {bound_label(*tb)} is not in --bounds")
        print(f"[bounds] greedy trunk driven by {bound_label(*tb)} (shared by all panels)")
        trunk_metric = metrics[tb]
    x, u = trunk_states(env, scen, trunk, env_name=args.env, metric=trunk_metric,
                        levels=levels, horizon=args.trunk_lookahead, solver=args.solver)
    e_trunk = wrap_diff(x - scen.xref[: x.shape[0]], scen.angle_idx).norm(dim=-1)
    print(f"[bounds] trunk={trunk}: euclidean |e| {float(e_trunk[0]):.3f} → "
          f"{float(e_trunk[-1]):.3f}  (max {float(e_trunk.max()):.3f})")
    K = min(u.shape[0], landscape_steps(scen, args.lookahead))
    frames = frame_indices(K, args.num_frames) if scen.u_dim == 2 else None

    geo: dict = {}
    for key in keys:
        metric = metrics[key]
        r, r_euc = normalized_error(scen, metric, x)
        heat = (compute_landscape_1d(env, scen, metric, x, u, levels, args.lookahead)
                if scen.u_dim == 1 else
                compute_landscape_2d(env, scen, metric, x, u, levels, frames, args.lookahead))
        geo[key] = {"heat": heat, "r": r, "r_euc": r_euc, "metric": metric}
        # cond(M) is the envelope's whole mechanism — reported against the box it
        # is allowed, because a fit pinned at its ceiling and one sitting well
        # inside are different situations that the surface alone does not name.
        # Computed for euclidean too rather than asserted: it is a cheap check
        # that the baseline metric really is I along this trunk.
        import torch
        M = metric.M(x) if metric.batched else torch.cat(
            [metric.M(x[k:k + 1]) for k in range(x.shape[0])], dim=0)
        eig = torch.linalg.eigvalsh(M)
        cond = (eig[:, -1] / eig[:, 0].clamp_min(1e-12)).median()
        print(f"[bounds] {panel_label(key):>22}: cond(M) median {float(cond):8.3f} "
              f"({panel_box(key)})  ·  trunk final normalized error {float(r[-1]):.4f}")

    return draw(args, scen, levels, keys, bounds, geo, trunk, x, u, frames)


def draw(args, scen, levels, keys, bounds, geo, trunk, x, u, frames):
    import matplotlib.pyplot as plt
    from viz_plot import (
        INK_2,
        SAVE_FORMAT,
        add_colorbar,
        draw_error_panel,
        draw_frame_2d,
        draw_surface_1d,
        error_norm,
        save_video,
    )

    t = scen.t.numpy()
    labels = [panel_label(k) for k in keys]
    n = len(keys)
    norm = error_norm([g["heat"] for g in geo.values()], args.error_range)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.env}_bounds_{trunk}_seed{args.seed}"
    trunk_desc = {
        "uref": "open-loop u_ref propagation (algorithm- and metric-independent)",
        "greedy": f"greedy {args.trunk_lookahead}-step-ahead grid control under "
                  f"{args.trunk_bounds or bound_label(*bounds[0])} (metric-dependent; "
                  f"ONE trunk shared by all panels)",
        "cvstem_lqr": "the CV-STEM-LQR trajectory (metric-independent)",
        "lqr": "the LQR trajectory (metric-independent)",
        "sd_lqr": "the SD-LQR trajectory (metric-independent)",
    }[trunk]
    # Kept to error_geometry.py's length: the suptitle is centered on a figure
    # sized by the panel count, so a longer line overflows both edges rather than
    # wrapping. "same objective, only the envelope differs" is already carried by
    # the title's "per w_lb/w_ub envelope (ccm CMG)" and does not need restating.
    subtitle = (f"seed {args.seed} · trunk = {trunk_desc} · {args.num_chunks} "
                f"control levels per dim over the {args.u_range} box · "
                f"{args.lookahead}-step lookahead ({args.lookahead * scen.dt:.2f}s hold)")
    err_series = [(lab, geo[k]["r"].numpy()) for k, lab in zip(keys, labels)]

    if scen.u_dim == 1:
        fig = plt.figure(figsize=(11, 4.4 * n + 3.4), constrained_layout=True)
        gs = fig.add_gridspec(n + 1, 2, width_ratios=[1, 0.035],
                              height_ratios=[1.0] * n + [0.95])
        for i, (key, lab) in enumerate(zip(keys, labels)):
            ax = fig.add_subplot(gs[i, 0], projection="3d")
            draw_surface_1d(ax, t, levels[0].numpy(), geo[key]["heat"], norm,
                            uref=scen.uref[:, 0].numpy())
            ax.set_title(f"{lab}   ({panel_box(key)})",
                         fontweight="bold", color=INK_2, pad=-2)
        ax_err = fig.add_subplot(gs[n, 0])
        draw_error_panel(ax_err, t, err_series)
        ax_err.set_ylabel("open-loop normalized error")
        ax_err.set_xlabel("t  [s]")
        add_colorbar(fig, norm, fig.add_subplot(gs[:, 1]))
        fig.suptitle(f"{args.env} — complete t × u error geometry per w_lb/w_ub "
                     f"envelope (ccm CMG)\n{subtitle}", fontsize=12, color=INK_2)
        out = os.path.join(args.output_dir, f"{stem}.{SAVE_FORMAT}")
        fig.savefig(out, bbox_inches="tight", format=SAVE_FORMAT)
    else:
        u0, u1 = levels[0].numpy(), levels[1].numpy()
        fig = plt.figure(figsize=(5.6 * n + 1.4, 8.4), constrained_layout=True)
        gs = fig.add_gridspec(2, n + 1, height_ratios=[1.4, 1.0],
                              width_ratios=[1] * n + [0.045])
        axes3d = [fig.add_subplot(gs[0, i], projection="3d") for i in range(n)]
        ax_err = fig.add_subplot(gs[1, :n])
        cax = fig.add_subplot(gs[:, n])

        def update(f):
            k = int(frames[f])
            lo = max(0, f - args.history)
            # One norm per frame, shared by all panels: the envelopes stay
            # comparable within the frame while the axis retunes to the shells on
            # screen so their change is visible.
            fnorm = error_norm(
                [geo[key]["heat"][:, :, j] for key in keys for j in range(lo, f + 1)],
                args.error_range)
            add_colorbar(fig, fnorm, cax)
            for ax, key, lab in zip(axes3d, keys, labels):
                ax.clear()
                hist = [geo[key]["heat"][:, :, j] for j in range(lo, f)]
                span = (f"  ·  shells t = {t[int(frames[lo])]:.2f}→{t[k]:.2f}s"
                        if hist else "")
                draw_frame_2d(ax, u0, u1, geo[key]["heat"][:, :, f], fnorm,
                              uref_pt=scen.uref[k].numpy(),
                              title=f"{lab}\nstep {k}  ·  t = {t[k]:.2f} s{span}",
                              history=hist)
            ax_err.clear()
            draw_error_panel(ax_err, t, err_series)
            ax_err.axvline(t[k], color=INK_2, lw=1.2)
            ax_err.set_ylabel("open-loop normalized error")
            ax_err.set_xlabel("t  [s]")
            fig.suptitle(
                f"{args.env} — complete u₀ × u₁ error geometry per w_lb/w_ub envelope "
                f"(ccm CMG)     t = {t[k]:5.2f} s\n{subtitle} · × = u_ref · "
                f"faded shells = previous geometries",
                fontsize=12, color=INK_2)

        out = save_video(fig, update, len(frames),
                         os.path.join(args.output_dir, f"{stem}.mp4"), fps=args.fps)

    np.savez_compressed(
        os.path.join(args.output_dir, f"{stem}.npz"),
        t=t, u_levels=levels.numpy(), x=x.numpy(), u=u.numpy(),
        xref=scen.xref.numpy(), uref=scen.uref.numpy(),
        trunk=trunk, lookahead=args.lookahead, bounds=np.array(bounds, dtype=float),
        # `bounds` is the envelopes; `panels` is what was actually drawn, which
        # also keys the heat/r/r_euc arrays and may lead with euclidean.
        panels=np.array(labels),
        **({"frames": frames} if frames is not None else {}),
        **{f"{lab}_{q}": (geo[k][q].numpy() if hasattr(geo[k][q], "numpy") else geo[k][q])
           for k, lab in zip(keys, labels) for q in ("heat", "r", "r_euc")},
    )
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
