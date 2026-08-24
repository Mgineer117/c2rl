"""Feature 2 — policy controls on a shared error geometry.

The landscape here is the same geometry error_geometry.py draws: one metric,
evaluated along one trunk trajectory (``viz_common.trunk_states``, default
``cvstem_lqr``). Every policy is overlaid on that one shared landscape, which is
what makes them comparable: each is asked what it would command at the same
state, and you read off who picks the better control.

``--trunk`` moves those shared states without ever breaking that sharing, but it
is a real trade-off here and worth choosing knowingly. The default
``cvstem_lqr`` keeps the trunk in a well-tracked region, at the cost of asking
every policy to act at states CV-STEM-LQR chose — which is not neutral between
the policies, and mildly flatters CV-STEM-LQR itself. ``--trunk uref`` favours no
policy but lets the error grow unchecked, so the states drift somewhere none of
them would actually visit. The bottom rollout panel is unaffected by ``--trunk``
either way — it is each policy's own closed-loop trajectory.

Two distinct questions are answered by the two parts of the figure:

  * landscape panels — "at an identical state, which policy commands the better
    control?"  Each policy's u = pi(x_trunk(k)) is drawn on the shared
    geometry (a curve for 1-D, a floor marker for 2-D).
  * bottom panel — "which policy actually tracks better?"  Each policy's own
    closed-loop rollout normalized error, which is a property of the policy, not
    of the geometry.

Restricted to u_dim <= 2 (the full control space is plotted, nothing projected):

  * u_dim == 1  (cartpole, segway) -> static .svg: every policy's u(t) curve on
                                      one shared t x u surface.
  * u_dim == 2  (car, turtlebot)   -> .mp4: every policy's commanded control
                                      marked on one shared u0 x u1 surface per
                                      frame, with the previous --history
                                      geometries as translucent shells.

Policies:
    c3m, c2rl_ppo, c2rl_sac — rebuilt from visualization/policies/<env>/<name>.pt
        plus <name>.yaml (falls back to the task's default agent yaml).
    cvstem_lqr, lqr, sd_lqr — analytical, no checkpoint needed.

Classic envs only; standalone (nothing in the training code depends on this).

Examples:
    python visualization/policy_overlay.py --env car
    python visualization/policy_overlay.py --env car --metric ccm --policies c3m,cvstem_lqr
    python visualization/policy_overlay.py --env car --trunk cvstem_lqr
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
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
    make_policy,
    rollout,
    trunk_states,
)

ALL_POLICIES = ("c3m", "c2rl_ppo", "cvstem_lqr", "lqr", "sd_lqr")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=CLASSIC_ENVS)
    p.add_argument("--metric", choices=METRIC_KINDS, default="cvstem_pretrained",
                   help="metric conditioning the shared geometry "
                        "(default: cvstem_pretrained — batched, needs no policy checkpoint)")
    p.add_argument("--metric-ckpt", default=None,
                   help="ccm / random only: checkpoint with a 'cmg' entry")
    p.add_argument("--random-seed", type=int, default=0,
                   help="random only: seed for the untrained CMG's weight init")
    p.add_argument("--policies", default=None,
                   help=f"comma-separated subset of {ALL_POLICIES} "
                        "(default: all whose checkpoint/config is available)")
    p.add_argument("--trunk", choices=TRUNK_MODES, default="cvstem_lqr",
                   help="the trajectory the shared landscape is sampled ALONG, and the "
                        "states every policy is asked to act at (default cvstem_lqr: "
                        "keeps the trunk in a well-tracked region, at the cost of "
                        "evaluating every policy on CV-STEM-LQR's states — pass "
                        "--trunk uref for zero feedback, which favours no policy but "
                        "lets the error grow unchecked). The bottom rollout panel is "
                        "unaffected by this flag either way.")
    p.add_argument("--trunk-lookahead", type=int, default=1,
                   help="--trunk greedy only: steps the greedy control looks ahead "
                        "before committing ONE step (default 1)")
    p.add_argument("--num-chunks", type=int, default=41)
    p.add_argument("--lookahead", type=int, default=10,
                   help="steps each candidate control is HELD for before the error "
                        "is measured (default 10) — see error_geometry.py")
    p.add_argument("--u-range", choices=["physical", "uref"], default="physical")
    p.add_argument("--ccm-samples", type=int, default=None,
                   help="ccm: states for the C1/C2 fit (default: config cmg_memory_size)")
    p.add_argument("--cmg-samples", type=int, default=16384,
                   help="cvstem_pretrained: states solved for the CMG regression (cached)")
    p.add_argument("--error-range", type=float, nargs=2, metavar=("LO", "HI"),
                   default=None,
                   help="optionally PIN the surface's normalized-error axis "
                        "(colour + 3D height). Default: retuned per frame.")
    p.add_argument("--history", type=int, default=10,
                   help="u_dim==2 only: how many earlier frames' geometries to keep "
                        "on screen, drawn translucent with alpha ramping toward the "
                        "present (0 = none, 1 = just the previous one at alpha 0.5)")
    p.add_argument("--num-frames", type=int, default=150,
                   help="u_dim==2 only: video frames sampled over the episode")
    p.add_argument("--fps", type=int, default=20, help="u_dim==2 only: video frame rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-bound", type=float, default=None,
                   help="episode length in seconds; MUST divide evenly by the env's dt")
    p.add_argument("--solver", default=None, help="cvxpy SDP solver override")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    return p.parse_args()


def resolve_policies(requested: str | None, env_name: str) -> list[str]:
    if requested:
        names = [n.strip() for n in requested.split(",") if n.strip()]
        unknown = [n for n in names if n not in ALL_POLICIES + ("c2rl_sac",)]
        if unknown:
            raise SystemExit(f"unknown policies {unknown}; choose from {ALL_POLICIES}")
        return names
    names = []
    for n in ALL_POLICIES:
        if n in ("cvstem_lqr", "lqr", "sd_lqr") or find_checkpoint(env_name, n):
            names.append(n)
        else:
            print(f"[overlay] skipping {n}: no policies/{env_name}/{n}.pt")
    return names


def main():
    args = parse_args()
    t0 = time.time()

    env = make_env(args.env, args.seed, time_bound=args.time_bound)
    scen = get_scenario(env)
    metric = make_metric(args.metric, env, scen, metric_ckpt=args.metric_ckpt,
                         solver=args.solver, cmg_samples=args.cmg_samples,
                         ccm_samples=args.ccm_samples, random_seed=args.random_seed)
    names = resolve_policies(args.policies, args.env)
    print(f"[overlay] env={args.env} T={scen.T} u_dim={scen.u_dim} "
          f"metric={metric.name} policies={names}")

    # ── the shared geometry: One trunk, one landscape, every policy on it ── #
    levels = control_grid(env, scen, args.num_chunks, args.u_range)
    x_ol, u_ol = trunk_states(env, scen, args.trunk, env_name=args.env,
                              metric=metric, levels=levels,
                              horizon=args.trunk_lookahead, solver=args.solver)
    K = min(u_ol.shape[0], landscape_steps(scen, args.lookahead))
    frames = frame_indices(K, args.num_frames) if scen.u_dim == 2 else None
    heat = (compute_landscape_1d(env, scen, metric, x_ol, u_ol, levels, args.lookahead)
            if scen.u_dim == 1 else
            compute_landscape_2d(env, scen, metric, x_ol, u_ol, levels, frames, args.lookahead))

    # ── per policy: commanded control at the shared states, + own rollout ─── #
    cmd: dict[str, np.ndarray] = {}
    own: dict[str, dict] = {}
    for name in names:
        policy_fn = make_policy(name, env, scen, args.env, solver=args.solver)
        obs = torch.cat([x_ol[:K], scen.xref[:K], scen.uref[:K]], dim=-1)
        cmd[name] = policy_fn(obs).detach().numpy()      # pi(x_trunk) — comparable
        traj = rollout(env, scen, policy_fn, metric)     # own closed-loop performance
        own[name] = traj
        print(f"[overlay] {name:>11}: own-rollout final normalized error "
              f"{float(traj['r'][-1]):.4f} (euclidean {float(traj['r_euc'][-1]):.4f})")
    if getattr(metric, "infeasible", 0):
        print(f"[overlay] WARNING: CV-STEM SDP infeasible at {metric.infeasible}/"
              f"{metric.solves} states (identity metric used there)")

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
    norm = error_norm([heat], args.error_range)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.env}_{args.metric}_overlay_{args.trunk}_seed{args.seed}"
    err_series = [(n, own[n]["r"].numpy()) for n in names]
    err_euc = {n: own[n]["r_euc"].numpy() for n in names}
    trunk_desc = ("open-loop u_ref states (algorithm-independent)" if args.trunk == "uref"
                  else f"states along the {args.trunk} trunk")
    subtitle = (f"seed {args.seed} · shared geometry = {args.metric} metric over "
                f"{trunk_desc} · {args.lookahead}-step lookahead · bottom = each "
                f"policy's OWN closed-loop rollout")

    if scen.u_dim == 1:
        fig = plt.figure(figsize=(11, 11), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.035], height_ratios=[1.9, 1.0])
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        draw_surface_1d(ax3d, t, levels[0].numpy(), heat, norm,
                        curves=[(n, cmd[n][:, 0]) for n in names],
                        uref=scen.uref[:, 0].numpy())
        ax3d.legend(loc="upper right", fontsize=9)
        ax3d.set_title("controls each policy commands at the shared states",
                       color=INK_2, pad=-2)
        ax_err = fig.add_subplot(gs[1, 0])
        draw_error_panel(ax_err, t, err_series, lbd=getattr(metric, "lbd", None),
                         euclidean=err_euc)
        ax_err.set_xlabel("t  [s]")
        add_colorbar(fig, norm, fig.add_subplot(gs[:, 1]))
        fig.suptitle(f"{args.env} — policy controls on the shared t × u error geometry\n"
                     f"{subtitle}", fontsize=12, color=INK_2)
        out = os.path.join(args.output_dir, f"{stem}.{SAVE_FORMAT}")
        fig.savefig(out, bbox_inches="tight", format=SAVE_FORMAT)
    else:
        u0, u1 = levels[0].numpy(), levels[1].numpy()
        fig = plt.figure(figsize=(10, 10.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.035], height_ratios=[1.9, 1.0])
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_err = fig.add_subplot(gs[1, 0])
        cax = fig.add_subplot(gs[:, 1])

        def update(f):
            k = int(frames[f])
            lo = max(0, f - args.history)
            fnorm = error_norm([heat[:, :, j] for j in range(lo, f + 1)],
                               args.error_range)
            add_colorbar(fig, fnorm, cax)
            ax3d.clear()
            hist = [heat[:, :, j] for j in range(lo, f)]
            span = f"  ·  shells t = {t[int(frames[lo])]:.2f}→{t[k]:.2f}s" if hist else ""
            draw_frame_2d(ax3d, u0, u1, heat[:, :, f], fnorm,
                          markers=[(n, cmd[n][k]) for n in names],
                          uref_pt=scen.uref[k].numpy(), history=hist,
                          title=f"controls each policy commands at this shared state\n"
                                f"step {k}  ·  t = {t[k]:.2f} s{span}")
            ax3d.legend(loc="upper right", fontsize=9)
            ax_err.clear()
            draw_error_panel(ax_err, t, err_series, lbd=getattr(metric, "lbd", None),
                             euclidean=err_euc)
            ax_err.axvline(t[k], color=INK_2, lw=1.2)
            ax_err.set_xlabel("t  [s]")
            fig.suptitle(
                f"{args.env} — policy controls on the shared u₀ × u₁ error geometry"
                f"     t = {t[k]:5.2f} s\n{subtitle} · × = u_ref · "
                f"faded shells = previous geometries",
                fontsize=12, color=INK_2)

        out = save_video(fig, update, len(frames),
                         os.path.join(args.output_dir, f"{stem}.mp4"), fps=args.fps)

    np.savez_compressed(
        os.path.join(args.output_dir, f"{stem}.npz"),
        t=t, u_levels=levels.numpy(), heat=heat, x_trunk=x_ol.numpy(),
        xref=scen.xref.numpy(), uref=scen.uref.numpy(), trunk=args.trunk,
        **({"frames": frames} if frames is not None else {}),
        **{f"{n}_cmd": cmd[n] for n in names},
        **{f"{n}_{q}": own[n][q].numpy() for n in names for q in ("x", "u", "r", "r_euc")},
    )
    print(f"[overlay] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
