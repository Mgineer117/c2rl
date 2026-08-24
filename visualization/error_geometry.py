"""Feature 1 — the complete control-space error geometry, one panel per metric.

One geometry per contraction-metric source, side by side, all over the same
states so the metric is the only difference. Each is defined by the objective
its CMG minimizes, trained here from the config rather than loaded from whatever
checkpoint is on disk, so each panel means what its name says:

  * ccm               — CMG trained on the C1/C2 contraction losses
                        (train_cmg_ccm; no SDP, no regression).
  * cvstem_pretrained — CMG trained by MSE regression onto CV-STEM SDP solutions
                        (build_cm_dataset + regress_cmg; the Tsukamoto NCM).
  * cvstem_online     — no CMG: the SDP re-solved at every state.
  * random            — control baseline: ccm's architecture/config/bounds,
                        untrained.

So ccm vs cvstem_pretrained compares the two synthesis formulations on identical
networks; cvstem_online vs cvstem_pretrained is that fit's regression error; and
both vs random is what the objective bought. Random matters because a random CMG
is still a bounded SPD field, so its landscape is not featureless — only the
excess over it is creditable to the objective.

--metric-ckpt overrides ccm with a stored CMG. Use knowingly: c3m.pt's CMG is
trained jointly with its controller on pd+c1+c2(+os) loss, so it is co-adapted
and not a pure C1/C2 metric.

Trunk (--trunk, ``viz_common.trunk_states``) is the trajectory the states are
sampled along. Accepts a comma-separated list, one output file per trunk
(default uref,cvstem_lqr,greedy); within a file all panels share one trunk.

  * ``cvstem_lqr`` (default) / ``lqr`` / ``sd_lqr`` — follow that analytical
    controller. Metric-independent, and stays in the well-tracked region a real
    controller occupies. Trade: conditioned on where that controller went.
  * ``uref`` — zero feedback. The only fully algorithm- And metric-independent
    choice, at the cost of unchecked error growth into a region no working
    controller visits (car: |e| 0.819 -> 2.741, vs 0.065 under cvstem_lqr).
  * ``greedy`` — best grid control under --trunk-metric. Metric-Dependent, so
    one designated metric drives the trunk for every panel; letting each follow
    its own would put them on different trajectories and the differences would
    no longer be attributable to the metric.

At each state every control in the actuator box is held for --lookahead H steps::

    error(u, k) = sqrt( e'^T M(x') e' / e0^T M(x0) e0 ),  e' = wrap(x' - xref_{k+H})

Restricted to u_dim <= 2, where the whole control space plots directly with
nothing projected away:

  * u_dim == 1 (cartpole, segway) -> static t x u x error surface (.svg)
  * u_dim == 2 (car, turtlebot)   -> MP4, one u0 x u1 surface per timestep, each
    frame keeping the previous --history geometries as translucent shells so the
    motion is visible within a single frame.

The error axis retunes every frame over exactly the shells on screen. All panels
in a frame share one norm (so metrics are comparable within a frame); colour and
height are not comparable across frames — read motion from the shells.
--error-range pins the axis when cross-run comparability matters more.

Why h > 1 by default: the error-minimizing control over H steps satisfies
||u*|| ~ ||e||/(H*dt). At H=1 (dt=0.03) that is ~33*||e||, far outside the
actuator box, so the control has no time to act and every slice is a monotone
wall with no interior optimum. Raising H brings ||u*|| inside the box without
reintroducing any controller dependence. It matters most here precisely because
the states are open-loop: their error grows rather than decays, so it never
reaches the regime where a 1-step landscape would show structure by itself.

Classic envs only. Standalone: no training code depends on this.

Examples:
    python visualization/error_geometry.py --env car   # all metrics x 3 trunks
    python visualization/error_geometry.py --env car --metrics ccm,random
    python visualization/error_geometry.py --env car --trunk uref
    python visualization/error_geometry.py --env segway --metrics cvstem_online
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from viz_common import (
    CLASSIC_ENVS,
    METRIC_KINDS,
    OUTPUT_DIR,
    TRUNK_MODES,
    compute_landscape_1d,
    compute_landscape_2d,
    control_grid,
    find_checkpoint,
    frame_indices,
    get_scenario,
    landscape_steps,
    make_env,
    make_metric,
    normalized_error,
    trunk_states,
    wrap_diff,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=CLASSIC_ENVS)
    p.add_argument("--metrics", default=None,
                   help=f"comma-separated subset of {METRIC_KINDS} (default: all)")
    p.add_argument("--random-seed", type=int, default=0,
                   help="random only: seed for the untrained CMG's weight init. "
                        "Vary it to check whether the baseline's landscape is a "
                        "property of the architecture or an accident of one draw.")
    p.add_argument("--metric-ckpt", default=None,
                   help="ccm: OVERRIDE the C1/C2 fit with a stored CMG. Note c3m.pt's "
                        "CMG is trained jointly with its controller (pd_loss + c1 + c2), "
                        "so it is NOT a pure C1/C2 metric")
    p.add_argument("--trunk", default="uref,cvstem_lqr,greedy",
                   help=f"COMMA-SEPARATED baseline controls to sample the landscape "
                        f"ALONG — one output file per trunk (default: "
                        f"uref,cvstem_lqr,greedy, the three reference points: no "
                        f"feedback / a real controller / the best on the grid). "
                        f"Choose from {TRUNK_MODES}. The metrics are built ONCE and "
                        f"reused across trunks. See viz_common.trunk_states.")
    p.add_argument("--trunk-metric", choices=METRIC_KINDS, default=None,
                   help="--trunk greedy only: the ONE metric driving the trunk, shared "
                        "by every panel so the panels stay comparable "
                        "(default: the first of --metrics)")
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
                   help="ccm: states for the C1/C2 fit (default: the config's own "
                        "cmg_memory_size). No SDP here, so this is cheap per sample — "
                        "do NOT shrink it to --cmg-samples' size or the fit starves")
    p.add_argument("--cmg-samples", type=int, default=16384,
                   help="cvstem_pretrained: states solved for the CMG regression "
                        "(cached; default 16384)")
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
                   help="cvxpy SDP solver override (SCS/CLARABEL/MOSEK)")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    return p.parse_args()


def resolve_metrics(requested: str | None, env_name: str) -> list[str]:
    if requested:
        names = [n.strip() for n in requested.split(",") if n.strip()]
        unknown = [n for n in names if n not in METRIC_KINDS]
        if unknown:
            raise SystemExit(f"unknown metrics {unknown}; choose from {METRIC_KINDS}")
        return names
    names = []
    for n in METRIC_KINDS:
        # random discards the trained weights but still reads the checkpoint
        # for its architecture, so it has the same prerequisite as ccm.
        if n in ("ccm", "random") and not (find_checkpoint(env_name, "c3m")
                                               or find_checkpoint(env_name, "c2rl_ppo")):
            print(f"[geometry] skipping {n}: no c3m.pt / c2rl_ppo.pt for {env_name}")
            continue
        names.append(n)
    return names


def resolve_trunks(requested: str) -> list[str]:
    names = [n.strip() for n in requested.split(",") if n.strip()]
    unknown = [n for n in names if n not in TRUNK_MODES]
    if unknown:
        raise SystemExit(f"unknown trunks {unknown}; choose from {TRUNK_MODES}")
    if not names:
        raise SystemExit("--trunk needs at least one mode")
    return list(dict.fromkeys(names))   # de-dup, order preserved


def main():
    args = parse_args()
    t0 = time.time()

    env = make_env(args.env, args.seed, time_bound=args.time_bound)
    scen = get_scenario(env)
    kinds = resolve_metrics(args.metrics, args.env)
    trunks = resolve_trunks(args.trunk)
    print(f"[geometry] env={args.env} T={scen.T} dt={scen.dt} u_dim={scen.u_dim} "
          f"|e0|={scen.e0.norm():.3f} metrics={kinds} trunks={trunks}")

    levels = control_grid(env, scen, args.num_chunks, args.u_range)

    # ── metrics are built once and reused across every trunk ──────────────── #
    # A metric is a property of the env + config, not of the trajectory it is
    # evaluated along, so training/solving it per trunk would repeat the whole
    # expensive part (C1/C2 fit, SDP dataset) for identical results.
    metrics: dict[str, object] = {}
    for kind in kinds:
        try:
            metrics[kind] = make_metric(
                kind, env, scen, metric_ckpt=args.metric_ckpt, solver=args.solver,
                cmg_samples=args.cmg_samples, ccm_samples=args.ccm_samples,
                random_seed=args.random_seed)
        except (RuntimeError, FileNotFoundError) as e:
            # One unavailable metric must not cost the other geometries: the
            # CV-STEM SDP is structurally infeasible on some envs (segway,
            # turtlebot) and build_cm_dataset raises outright there.
            print(f"[geometry] SKIPPING {kind}: {e}")
    kinds = [k for k in kinds if k in metrics]
    if not kinds:
        raise SystemExit(
            "[geometry] no metric could be built — nothing to plot. For an env whose "
            "CV-STEM LMI is infeasible (segway/turtlebot), use --metrics ccm,random "
            "(both are CMG-based and need no SDP).")

    outs = []
    for trunk in trunks:
        outs.append(render_trunk(args, env, scen, levels, kinds, metrics, trunk))
    print(f"[geometry] wrote {len(outs)} file(s) in {time.time() - t0:.1f}s:")
    for o in outs:
        print(f"[geometry]   {o}")


def render_trunk(args, env, scen, levels, kinds, metrics, trunk):
    """Build the trunk, every metric's landscape along it, and render one figure."""
    # ── the trunk: One trajectory, shared by every panel ─────────────────── #
    # Shared is the whole point — the panels differ only by their conditioning
    # metric, so they must be sampled at identical states. `greedy` is the one
    # metric-Dependent mode, hence --trunk-metric driving a single trunk rather
    # than one trunk per panel.
    trunk_metric = None
    if trunk == "greedy":
        tm_kind = args.trunk_metric or kinds[0]
        print(f"[geometry] greedy trunk driven by metric={tm_kind} "
              f"(shared by all panels)")
        trunk_metric = metrics[tm_kind]
    x, u = trunk_states(env, scen, trunk, env_name=args.env,
                        metric=trunk_metric, levels=levels,
                        horizon=args.trunk_lookahead, solver=args.solver)
    e_trunk = wrap_diff(x - scen.xref[: x.shape[0]], scen.angle_idx).norm(dim=-1)
    print(f"[geometry] trunk={trunk}: euclidean |e| {float(e_trunk[0]):.3f} → "
          f"{float(e_trunk[-1]):.3f}  (max {float(e_trunk.max()):.3f})")
    K = min(u.shape[0], landscape_steps(scen, args.lookahead))
    frames = frame_indices(K, args.num_frames) if scen.u_dim == 2 else None

    geo: dict[str, dict] = {}
    for kind in kinds:
        metric = metrics[kind]
        r, r_euc = normalized_error(scen, metric, x)
        heat = (compute_landscape_1d(env, scen, metric, x, u, levels, args.lookahead)
                if scen.u_dim == 1 else
                compute_landscape_2d(env, scen, metric, x, u, levels, frames, args.lookahead))
        geo[kind] = {"heat": heat, "r": r, "r_euc": r_euc, "metric": metric}
        print(f"[geometry] {kind:>18}: {metric.name} — trunk final normalized "
              f"error {float(r[-1]):.4f}")
        if getattr(metric, "infeasible", 0):
            print(f"[geometry] {' ':>18}  WARNING: CV-STEM SDP infeasible at "
                  f"{metric.infeasible}/{metric.solves} states (identity metric there)")

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
    n = len(kinds)
    norm = error_norm([g["heat"] for g in geo.values()], args.error_range)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.env}_geometry_{trunk}_seed{args.seed}"
    trunk_desc = {
        "uref": "open-loop u_ref propagation (algorithm- and metric-independent)",
        "greedy": f"greedy {args.trunk_lookahead}-step-ahead grid control under "
                  f"{args.trunk_metric or kinds[0]} (metric-dependent; ONE trunk shared "
                  f"by all panels)",
        "cvstem_lqr": "the CV-STEM-LQR trajectory (metric-independent)",
        "lqr": "the LQR trajectory (metric-independent)",
        "sd_lqr": "the SD-LQR trajectory (metric-independent)",
    }[trunk]
    subtitle = (f"seed {args.seed} · trunk = {trunk_desc} · {args.num_chunks} "
                f"control levels per dim over the {args.u_range} box · "
                f"{args.lookahead}-step lookahead ({args.lookahead * scen.dt:.2f}s hold)")
    err_series = [(k, geo[k]["r"].numpy()) for k in kinds]

    if scen.u_dim == 1:
        fig = plt.figure(figsize=(11, 4.4 * n + 3.4), constrained_layout=True)
        gs = fig.add_gridspec(n + 1, 2, width_ratios=[1, 0.035],
                              height_ratios=[1.0] * n + [0.95])
        for i, kind in enumerate(kinds):
            ax = fig.add_subplot(gs[i, 0], projection="3d")
            draw_surface_1d(ax, t, levels[0].numpy(), geo[kind]["heat"], norm,
                            uref=scen.uref[:, 0].numpy())
            ax.set_title(kind, fontweight="bold", color=INK_2, pad=-2)
        ax_err = fig.add_subplot(gs[n, 0])
        draw_error_panel(ax_err, t, err_series)
        ax_err.set_ylabel("open-loop normalized error")
        ax_err.set_xlabel("t  [s]")
        add_colorbar(fig, norm, fig.add_subplot(gs[:, 1]))
        fig.suptitle(f"{args.env} — complete t × u error geometry per metric\n{subtitle}",
                     fontsize=12, color=INK_2)
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
            # One norm per frame, shared by all three panels: the metrics stay
            # comparable within the frame while the axis retunes to the shells on
            # screen so their change is visible.
            fnorm = error_norm(
                [geo[kd]["heat"][:, :, j] for kd in kinds for j in range(lo, f + 1)],
                args.error_range)
            add_colorbar(fig, fnorm, cax)
            for ax, kind in zip(axes3d, kinds):
                ax.clear()
                hist = [geo[kind]["heat"][:, :, j] for j in range(lo, f)]
                span = (f"  ·  shells t = {t[int(frames[lo])]:.2f}→{t[k]:.2f}s"
                        if hist else "")
                draw_frame_2d(ax, u0, u1, geo[kind]["heat"][:, :, f], fnorm,
                              uref_pt=scen.uref[k].numpy(),
                              title=f"{kind}\nstep {k}  ·  t = {t[k]:.2f} s{span}",
                              history=hist)
            ax_err.clear()
            draw_error_panel(ax_err, t, err_series)
            ax_err.axvline(t[k], color=INK_2, lw=1.2)
            ax_err.set_ylabel("open-loop normalized error")
            ax_err.set_xlabel("t  [s]")
            fig.suptitle(
                f"{args.env} — complete u₀ × u₁ error geometry per metric     "
                f"t = {t[k]:5.2f} s\n{subtitle} · × = u_ref · "
                f"faded shells = previous geometries",
                fontsize=12, color=INK_2)

        out = save_video(fig, update, len(frames),
                         os.path.join(args.output_dir, f"{stem}.mp4"), fps=args.fps)

    np.savez_compressed(
        os.path.join(args.output_dir, f"{stem}.npz"),
        t=t, u_levels=levels.numpy(), x=x.numpy(), u=u.numpy(),
        xref=scen.xref.numpy(), uref=scen.uref.numpy(),
        trunk=trunk, lookahead=args.lookahead,
        **({"frames": frames} if frames is not None else {}),
        **{f"{k}_{q}": (geo[k][q].numpy() if hasattr(geo[k][q], "numpy") else geo[k][q])
           for k in kinds for q in ("heat", "r", "r_euc")},
    )
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
