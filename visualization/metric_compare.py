"""ccm vs cvstem_pretrained CMG — where the two synthesis objectives actually
disagree, in state space rather than control space.

bound_sweep.py's control-space landscape is blind to most of M by construction
(see its docstring): an H-step-held-control sweep only explores a ~2-D
reachable slice of state space, magnitude (‖e‖) dominates the surface over
direction, and the normalized-error ratio cancels scale. Two metrics that
differ sharply in cond(M) or eigenvector field can produce near-identical
control-space landscapes for exactly that reason. This script instead compares
M(x) directly along a shared trunk:

  1. Ellipse field  — level sets of e^T M(x) e at sampled trunk states,
     projected onto (x, y) and (theta, v). Shows orientation/anisotropy
     directly: a CCM (C1/C2, no SDP) fit and a CV-STEM-regression fit can
     produce visibly different ellipse shapes even when their cond(M) traces
     coincide.
  2. Eigenvalue trace — eigenvalues of M(x) and the angle of top/bottom
     eigenvectors to the state axes, both metrics on one time axis. Shows
     where along the episode the metrics diverge, not just that they do.
  3. Empirical contraction rate — log(V_{t+1}/V_t) along the trunk (V = e^T M e,
     the Mahalanobis reward's own quantity, env_base.get_rewards), the
     certificate-relevant comparison: two metrics with similar ellipses can
     still differ here.

Classic envs only. Standalone: nothing in the training code depends on this.

Example:
    python visualization/metric_compare.py --env car
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import viz_common
from viz_common import (
    CLASSIC_ENVS,
    OUTPUT_DIR,
    SERIES_COLORS,
    get_scenario,
    load_algo_cfg,
    make_env,
    make_metric,
    trunk_states,
    wrap_diff,
)

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

# car's state layout (env.py: num_dim_x=4, angle_idx=[2]) — (x, y, theta, v).
# Falls back to generic x0..xN-1 labels for envs not listed here.
STATE_LABELS = {
    "car": ["x", "y", "theta", "v"],
    "turtlebot": ["x", "y", "theta"],
    "cartpole": ["x", "theta", "xdot", "thetadot"],
    "segway": ["x", "theta", "xdot", "thetadot"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=CLASSIC_ENVS)
    p.add_argument("--trunk", default="cvstem_lqr", choices=viz_common.TRUNK_MODES,
                   help="baseline control the shared trunk follows (default cvstem_lqr — "
                        "metric-independent, stays in the well-tracked region)")
    p.add_argument("--num-states", type=int, default=6,
                   help="how many trunk states get an ellipse panel (default 6, evenly spaced)")
    p.add_argument("--ccm-samples", type=int, default=None,
                   help="states for the ccm C1/C2 fit (default: config's cmg_memory_size)")
    p.add_argument("--cvstem-samples", type=int, default=16384,
                   help="states for the cvstem_pretrained regression fit (default 16384)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-bound", type=float, default=None)
    p.add_argument("--solver", default=None, help="cvxpy SDP solver override (SCS/CLARABEL/MOSEK)")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    return p.parse_args()


def ellipse_xy(M2: np.ndarray, center: np.ndarray, n: int = 200, scale: float = 1.0) -> np.ndarray:
    """Points on {e : e^T M2 e = scale} for a 2x2 SPD M2, in the original
    (unrotated) coordinate frame, centered at ``center``."""
    eigval, eigvec = np.linalg.eigh(M2)
    theta = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(theta), np.sin(theta)])            # (2, n)
    radii = np.sqrt(scale / np.clip(eigval, 1e-9, None))          # (2,)
    pts = eigvec @ (radii[:, None] * circle)                      # (2, n)
    return pts + center[:, None]


def main():
    args = parse_args()

    env = make_env(args.env, args.seed, time_bound=args.time_bound)
    scen = get_scenario(env)
    labels = STATE_LABELS.get(args.env, [f"x{i}" for i in range(scen.x_dim)])
    print(f"[compare] env={args.env} T={scen.T} dt={scen.dt} x_dim={scen.x_dim} state={labels}")

    base_cfg = load_algo_cfg(args.env, "c2rl_ppo")
    ccm = make_metric("ccm", env, scen, metric_cfg=base_cfg, ccm_samples=args.ccm_samples)
    ccm.name = "ccm (C1/C2)"
    cvstem = make_metric("cvstem_pretrained", env, scen, cmg_samples=args.cvstem_samples,
                          solver=args.solver)
    cvstem.name = "cvstem_pretrained (SDP-regressed)"
    metrics = {"ccm": ccm, "cvstem": cvstem}
    SERIES_COLORS.setdefault(ccm.name, SERIES_COLORS.get("ccm", "#1d3f6e"))
    SERIES_COLORS.setdefault(cvstem.name, SERIES_COLORS.get("cvstem_pretrained", "#e87ba4"))

    x, u = trunk_states(env, scen, args.trunk, env_name=args.env, solver=args.solver)
    T = x.shape[0]
    t = scen.t[:T].numpy()
    print(f"[compare] trunk={args.trunk}: {T} states")

    Ms = {}
    for key, m in metrics.items():
        M = m.M(x) if m.batched else torch.cat([m.M(x[k:k + 1]) for k in range(T)], dim=0)
        Ms[key] = M
        eig = torch.linalg.eigvalsh(M)
        cond = (eig[:, -1] / eig[:, 0].clamp_min(1e-12))
        print(f"[compare] {m.name:>32}: cond(M) median {float(cond.median()):8.3f} "
              f"(min {float(cond.min()):.3f}, max {float(cond.max()):.3f})")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.env}_metric_compare_{args.trunk}_seed{args.seed}"

    fig_ellipse = draw_ellipses(args, scen, labels, x, Ms, metrics, t)
    fig_ellipse.savefig(os.path.join(args.output_dir, f"{stem}_ellipses.svg"),
                        bbox_inches="tight")

    fig_eig = draw_eigen_trace(scen, labels, x, Ms, metrics, t)
    fig_eig.savefig(os.path.join(args.output_dir, f"{stem}_eigtrace.svg"), bbox_inches="tight")

    fig_rate = draw_contraction_rate(scen, x, Ms, metrics, t)
    fig_rate.savefig(os.path.join(args.output_dir, f"{stem}_rate.svg"), bbox_inches="tight")

    np.savez_compressed(
        os.path.join(args.output_dir, f"{stem}.npz"),
        t=t, x=x.numpy(), state_labels=np.array(labels),
        **{f"{k}_M": v.numpy() for k, v in Ms.items()},
    )
    print(f"[compare] wrote {stem}_ellipses.svg, {stem}_eigtrace.svg, "
          f"{stem}_rate.svg, {stem}.npz to {args.output_dir}")


def draw_ellipses(args, scen, labels, x, Ms, metrics, t):
    import matplotlib.pyplot as plt

    T = x.shape[0]
    idx = np.unique(np.linspace(0, T - 1, args.num_states).astype(int))
    dim_pairs = [(0, 1), (2, 3)] if scen.x_dim >= 4 else [(0, 1)]
    dim_pairs = [(a, b) for a, b in dim_pairs if b < scen.x_dim]

    fig, axes = plt.subplots(len(dim_pairs), len(idx),
                             figsize=(3.2 * len(idx), 3.2 * len(dim_pairs)),
                             squeeze=False)
    x_np = x.numpy()
    for row, (a, b) in enumerate(dim_pairs):
        for col, k in enumerate(idx):
            ax = axes[row, col]
            center = x_np[k, [a, b]]
            for key, m in metrics.items():
                M2 = Ms[key][k][np.ix_([a, b], [a, b])].numpy()
                pts = ellipse_xy(M2, center)
                ax.plot(pts[0], pts[1], color=SERIES_COLORS.get(m.name), lw=1.8, label=m.name)
            ax.scatter(*center, color="black", s=14, zorder=5)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"t={t[k]:.2f}s", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{labels[a]} — {labels[b]}\n{labels[b]}", fontsize=9)
            ax.set_xlabel(labels[a], fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{args.env} — e^T M(x) e level sets along the {args.trunk} trunk "
                 f"(ccm vs cvstem_pretrained)", fontsize=11)
    fig.tight_layout()
    return fig


def draw_eigen_trace(scen, labels, x, Ms, metrics, t):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_eig, ax_ang = axes

    for key, m in metrics.items():
        M = Ms[key]
        eigval, eigvec = torch.linalg.eigh(M)
        color = SERIES_COLORS.get(m.name)
        ax_eig.plot(t, eigval[:, -1].numpy(), color=color, lw=1.6, label=f"{m.name} λ_max")
        ax_eig.plot(t, eigval[:, 0].numpy(), color=color, lw=1.0, ls="--", label=f"{m.name} λ_min")
        # angle of the top eigenvector to the first state axis, per state —
        # a state-space orientation trace, independent of any reachable slice.
        top = eigvec[:, :, -1]
        ref = torch.zeros_like(top); ref[:, 0] = 1.0
        cosang = (top * ref).sum(-1).abs().clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cosang))
        ax_ang.plot(t, ang.numpy(), color=color, lw=1.6, label=f"{m.name}")

    ax_eig.set_yscale("log")
    ax_eig.set_ylabel("eigenvalue of M(x)  (log)")
    ax_eig.legend(fontsize=8, ncol=2)
    ax_ang.set_ylabel(f"angle(v_max, {labels[0]}-axis)  [deg]")
    ax_ang.set_xlabel("t  [s]")
    ax_ang.legend(fontsize=8)
    fig.suptitle("Eigenstructure of M(x) along the trunk — where ccm and "
                 "cvstem_pretrained actually diverge")
    fig.tight_layout()
    return fig


def draw_contraction_rate(scen, x, Ms, metrics, t):
    """log(V_{t+1}/V_t) along the trunk, V = e^T M e — the certificate-relevant
    comparison: independent of the ellipse shape's visual similarity."""
    import matplotlib.pyplot as plt

    e = wrap_diff(x - scen.xref[: x.shape[0]], scen.angle_idx)
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, m in metrics.items():
        M = Ms[key]
        V = torch.einsum("ki,kij,kj->k", e, M, e).clamp_min(1e-12)
        rate = torch.log(V[1:] / V[:-1]) / scen.dt
        ax.plot(t[:-1], rate.numpy(), color=SERIES_COLORS.get(m.name), lw=1.4, label=m.name)
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel(r"$\dot V / V$ realized  [1/s]   (< 0 = contracting)")
    ax.set_xlabel("t  [s]")
    ax.legend(fontsize=9)
    ax.set_title("Empirical contraction rate along the trunk")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
