"""Min-projected λ*(x) plots: cached data in, styled figure out.

Two jobs, deliberately separable:

  --generate   compute figures/data/minproj_<env>.npz  (the expensive half: one
               λ* bisection per grid cell, each a joint CV-STEM SDP)
  (default)    read that npz and draw figures/minproj_<env>.svg

Keeping them apart is the point. The λ* grid costs thousands of SDP solves and
never changes for a fixed (env, envelope), so restyling a figure must not re-solve
it — the cached npz is the interface, and the eight envs already cached are
replotted in seconds.

"Min-projected": the figure is a 2D slice over the two dims the dynamics actually
depend on, and every OTHER dim is projected out by taking the MINIMUM λ* over a
sample of its box. Minimum, not mean: λ* is a certificate over a set, so the
honest value at a slice point is the worst case behind it.

x0 is drawn as a DENSITY, not a scatter. Two thousand points over a 15x15 slice
is past the count where individual markers read as anything — they cover the
heatmap and hide it. A filled contour of the same samples shows where the reset
distribution actually concentrates, which is the question the panel answers.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "c2rl"))
sys.path.insert(0, str(REPO / "scripts"))

FIG_DIR = REPO / "figures"
DATA_DIR = FIG_DIR / "data"

# Every text element, one place. The defaults are matplotlib's ~10 pt, which is
# unreadable once a figure is scaled into a paper column.
FS_TITLE = 20
FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 15
FS_CBAR = 16

LINEWIDTH = 3.2          # trajectory stroke
N_TRAJ_SHOWN = 2         # rollouts drawn; more than this reads as a tangle
ARROW_FRAC = 0.026       # arrow length as a fraction of the panel
ARROW_WIDTH = 0.0075     # arrow shaft width (axes fraction)


# Title text, where capitalizing the short name is not what should be shown.
# The car pair is titled v0/v1 on BOTH sides: "Car" against "Car v1" would read
# as a base case beside a variant, when they are two settings of one plant.
DISPLAY_NAME = {"car": "Car v0", "car_v1": "Car v1"}


def env_id(env: str) -> str:
    """Short name -> gym id, via the one map the classic package owns."""
    from c2rl.tasks.direct.classic import env_id as _resolve
    return _resolve(env)


def pretty(env: str) -> str:
    """Short name -> title text. two_link_arm -> Two Link Arm; car_v1 -> Car v1."""
    if env in DISPLAY_NAME:
        return DISPLAY_NAME[env]
    return " ".join(w.capitalize() for w in env.split("_"))


# ────────────────────────────── generate ─────────────────────────────────── #

def generate(env_name: str, *, grid: int, n_other: int, n_x0: int,
             n_traj: int, seed: int) -> pathlib.Path:
    import c2rl.tasks.direct.classic  # noqa: F401
    import gymnasium as gym
    from lambda_subsets import active_dims_auto, jacobians, max_lambda  # noqa: E402

    env = gym.make(env_id(env_name), num_envs=n_traj, device="cpu").unwrapped
    x_min = env.X_MIN.detach().cpu().numpy().astype(np.float64)
    x_max = env.X_MAX.detach().cpu().numpy().astype(np.float64)

    cfg = _cm_cfg(env_name)
    kw = dict(eps=cfg["cm_eps"], dt=1.0, solver="MOSEK", r_scaler=cfg["r"],
              w_lb=cfg["w_lb"], w_ub=cfg["w_ub"])

    # The two dims to plot: the last two the dynamics actually depend on. For the
    # car that is (yaw, vel) — position drops out because f ignores x and y, so a
    # slice over position would be constant by construction.
    dims = active_dims_auto(env)[-2:]
    gx = np.linspace(x_min[dims[0]], x_max[dims[0]], grid)
    gy = np.linspace(x_min[dims[1]], x_max[dims[1]], grid)

    rng = np.random.default_rng(seed)
    # One draw of the projected-out dims, REUSED at every cell, so neighbouring
    # cells differ because of the slice coordinates and not because of the RNG.
    other = rng.uniform(x_min, x_max, size=(n_other, x_min.size))

    Z = np.empty((grid, grid), dtype=np.float64)
    print(f"[minproj] {env_name}: {grid}x{grid} cells, min over {n_other} "
          f"projected samples, dims={[env.state_names[d] for d in dims]}", flush=True)
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            pts = other.copy()
            pts[:, dims[0]] = xv
            pts[:, dims[1]] = yv
            A, B = jacobians(env, pts.astype(np.float32))
            # min over the projected dims: each sample alone is one certificate,
            # and the slice inherits the worst of them.
            lams = [max_lambda(A[k:k + 1], B[k:k + 1], kw)[0] for k in range(len(pts))]
            Z[j, i] = float(np.min(lams))
        print(f"[minproj]   column {i + 1}/{grid} done", flush=True)

    # x0 from the env's OWN reset, not a uniform box draw. reset() takes
    # x_0 = clamp(xref_0, box) + xe_0, and for car_v1 xref's velocity is drawn
    # from [0.3, 1.5] specifically so the plant sits in the weak-authority region
    # (sigma = min(1, v) = v < 1). Sampling uniformly over the box would erase the
    # very concentration this figure is supposed to show, and would also not match
    # the envs whose cached x0 was reset-drawn.
    x0 = _reset_x0(env, n_x0, seed=seed)

    # Trajectories: the env's own reference rollout, which is what the panel is
    # about — where the closed loop actually goes, not where the box allows.
    traj = _rollout(env, n_traj)

    out = DATA_DIR / f"minproj_{env_name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Z=Z, dims=np.asarray(dims), grid_x=gx, grid_y=gy,
             exact=np.asarray(True), lbd=np.asarray(cfg["lbd"]),
             r=np.asarray(cfg["r"]), w_lb=np.asarray(cfg["w_lb"]),
             w_ub=np.asarray(cfg["w_ub"]), traj=traj, x0=x0,
             state_names=np.asarray(list(env.state_names)),
             x_min=x_min, x_max=x_max)
    print(f"[minproj] wrote {out}")
    return out


def refresh_x0(env_name: str, *, n_x0: int, n_traj: int, seed: int) -> pathlib.Path:
    """Rewrite x0/traj in the cached npz from the CURRENT env; keep Z.

    lambda* depends on the plant and the envelope, so a change to how episodes
    start cannot move it. Recomputing the grid to pick up a new x0 draw would burn
    thousands of SDP solves for an identical result.
    """
    import c2rl.tasks.direct.classic  # noqa: F401
    import gymnasium as gym

    src = DATA_DIR / f"minproj_{env_name}.npz"
    if not src.exists():
        raise SystemExit(f"nothing cached at {src} — use --generate")
    old = dict(np.load(src, allow_pickle=True))

    env = gym.make(env_id(env_name), num_envs=n_traj, device="cpu").unwrapped
    lo = env.X_MIN.detach().cpu().numpy().astype(np.float64)
    hi = env.X_MAX.detach().cpu().numpy().astype(np.float64)
    x0 = _reset_x0(env, n_x0, seed=seed)
    out_frac = float(((x0 < lo - 1e-6) | (x0 > hi + 1e-6)).any(axis=1).mean())

    old["x0"] = x0
    old["traj"] = _rollout(env, n_traj)
    old["x_min"], old["x_max"] = lo, hi
    np.savez(src, **old)
    print(f"[minproj] refreshed x0/traj in {src}  "
          f"({out_frac:.2%} of x0 outside the box)")
    return src


def _reset_x0(env, n: int, *, seed: int = 0) -> np.ndarray:
    """``n`` initial states as the env actually produces them, via repeated reset.

    Batched: one reset yields ``num_envs`` states, so this loops until it has n.
    """
    out = []
    k = 0
    while sum(len(o) for o in out) < n:
        env.reset(seed=seed + k)
        out.append(env.x_t.detach().cpu().numpy().copy())
        k += 1
        if k > 4000:                     # a stuck env must not spin forever
            break
    return np.concatenate(out, axis=0)[:n].astype(np.float64)


def _cm_cfg(env_name: str) -> dict:
    """The env's shipped (lbd, r, envelope) — the figure must describe what ships."""
    import yaml
    p = (REPO / "source/c2rl/c2rl/tasks/direct/classic"
         / env_name / "agents/skrl_c2rl_ppo_cfg.yaml")
    cm = yaml.safe_load(p.read_text(encoding="utf-8"))["cm"]
    return {"lbd": float(cm["lbd"]), "r": float(cm["cvstem_r_scaler"]),
            "w_lb": float(cm["w_lb"]), "w_ub": float(cm["w_ub"]),
            "cm_eps": float(cm["cm_eps"])}


def _rollout(env, n_traj: int) -> np.ndarray:
    """Open-loop-along-reference rollout: x(t) under the env's own uref."""
    import torch
    obs, _ = env.reset(seed=0)
    T = int(env.max_episode_len)
    xs = [env.x_t.detach().cpu().numpy().copy()]
    for t in range(T):
        u = env.uref[:, min(t, env.uref.shape[1] - 1)]
        obs, *_ = env.step(u.detach().cpu().numpy()
                           if isinstance(u, torch.Tensor) else u)
        xs.append(env.x_t.detach().cpu().numpy().copy())
    return np.transpose(np.asarray(xs, dtype=np.float64), (1, 0, 2))


# ─────────────────────────────── plot ────────────────────────────────────── #

# Envs whose layout is pinned by hand rather than inferred. Only needed where
# lambda* is CONSTANT: with a flat field there is no "axis lambda* varies along"
# to detect, so marginal_axis' tie-break decides the layout arbitrarily. car is
# pinned to put vel on x, matching car_v1 -- the two are the same plant and
# read as a pair, so they must not come out transposed relative to each other.
AXIS_OVERRIDE = {"car": "y"}


def marginal_axis(Z: np.ndarray) -> str:
    """"x" or "y": the axis lambda* actually varies along.

    Z is indexed [y, x]. Comparing the peak-to-peak of the per-axis means, rather
    than the raw spread, keeps a single noisy cell from deciding the layout.
    A flat field falls through to "x", where the panel is wider and easier to read
    — see AXIS_OVERRIDE for the envs where that tie-break is not the wanted one.
    """
    var_x = float(np.ptp(Z.mean(axis=0)))
    var_y = float(np.ptp(Z.mean(axis=1)))
    return "y" if var_y > var_x else "x"


def plot(env_name: str, *, n_traj: int = N_TRAJ_SHOWN,
         axis: str | None = None, rate_only: bool = False) -> pathlib.Path:
    """The min-projected lambda* field for one env.

    ``rate_only`` drops everything that is not the state-dependent rate itself —
    no rollouts, no x0 marginal, no connectors — and writes ``rate_<env>.svg``
    instead of ``minproj_<env>.svg``, so the field can be shown on its own.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = DATA_DIR / f"minproj_{env_name}.npz"
    if not src.exists():
        raise SystemExit(f"no cached data at {src} — run with --generate first")
    z = np.load(src, allow_pickle=True)
    Z, gx, gy = z["Z"], z["grid_x"], z["grid_y"]
    dims = z["dims"]
    names = [str(s) for s in z["state_names"]]
    dx, dy = int(dims[0]), int(dims[1])
    ex, ey = _edges(gx), _edges(gy)
    traj, x0 = z["traj"], z["x0"]
    k_show = 0 if rate_only else min(int(n_traj), traj.shape[0])
    shown = traj[:k_show]

    # Always put the dim lambda* varies along on the X axis, transposing the slice
    # when it is the second one. That keeps ONE layout for every env -- field on
    # top, marginal below over x, vertical connectors -- while still giving each
    # env the marginal that carries information. car_v1 is why: its lambda* is
    # flat in yaw and swings 2.78 across vel, so vel becomes its x-axis.
    which = axis or AXIS_OVERRIDE.get(env_name) or marginal_axis(Z)
    if which == "y":
        Z = Z.T
        dx, dy = dy, dx
        gx, gy = gy, gx
        ex, ey = _edges(gx), _edges(gy)
    md = dx                                   # the marginal is always over x now

    # The colorbar gets its OWN column spanning only the top row, so both panels
    # keep an identical width. Attaching it to the top axes instead (colorbar(ax=ax))
    # steals width from that axes alone, and then a dotted vertical at the same
    # data value lands at a different PIXEL in each panel -- which defeats the
    # entire point of the shared axis.
    if rate_only:
        # Same panel width and colorbar column as the two-panel figure, so a
        # rate-only plot can sit beside a full one without rescaling.
        fig = plt.figure(figsize=(10.6, 7.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        axd = None
        cax = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=(10.6, 9.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], width_ratios=[1.0, 0.035],
                              hspace=0.06, wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        axd = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[0, 1])
        ax.tick_params(labelbottom=False)
    which = "x"                               # one code path from here down

    # A constant field gets a NEUTRAL flat colour, not a point on viridis. Mapping
    # a single value through the colormap put car and quadrotor at the very bottom
    # of the scale, so their panels rendered dark purple -- which reads as "low
    # lambda*" beside the varying-field panels where purple genuinely is low.
    if float(Z.max() - Z.min()) < 1e-6:
        mesh = ax.pcolormesh(ex, ey, np.zeros_like(Z), cmap="Greys",
                             vmin=0.0, vmax=1.0, shading="flat")
    else:
        mesh = ax.pcolormesh(ex, ey, Z, cmap="viridis", shading="flat")
    # A constant field gets NO colorbar. Normalizing a colormap around a single
    # value invents a range (car showed 4.4-5.3 for a field that is 4.884
    # everywhere) and invites reading a gradient that does not exist. The value is
    # stated on the panel instead. The colorbar COLUMN is still reserved and just
    # switched off, so all five figures keep identical panel geometry and can be
    # laid side by side.
    is_const = float(Z.max() - Z.min()) < 1e-6
    if is_const:
        cax.axis("off")
        ax.text(0.5, 0.045,
                rf"$\lambda^*(x) = {float(Z.mean()):.4g}$  everywhere",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_LABEL, color="0.10",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55",
                          alpha=0.92), zorder=8)
    else:
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(r"$\lambda^*(x)$", fontsize=FS_CBAR)
        cb.ax.tick_params(labelsize=FS_TICK)
        cb.formatter.set_useOffset(False)
        cb.update_ticks()

    for k in range(k_show):
        tx, ty = traj[k, :, dx], traj[k, :, dy]
        ax.plot(tx, ty, "-", color="#d62728", lw=LINEWIDTH, alpha=0.92,
                zorder=4, label=r"$x(t)$" if k == 0 else None)
        _arrows(ax, tx, ty)
        ax.plot(tx[0], ty[0], "o", ms=11, mfc="white", mec="#7f0f14", mew=2.4,
                zorder=7, label=r"$x_0$ of shown runs" if k == 0 else None)

    # Axes cover the union of the field and what is drawn on it: clipping to the
    # box hides every trajectory start outside it (on car, about half of them).
    if k_show:
        lo_y = min(ey[0], float(np.min(shown[:, :, dy])))
        hi_y = max(ey[-1], float(np.max(shown[:, :, dy])))
        lo_x = min(ex[0], float(np.min(shown[:, :, dx])))
        hi_x = max(ex[-1], float(np.max(shown[:, :, dx])))
    else:
        # Nothing drawn on top of the field, so the field IS the extent.
        lo_x, hi_x, lo_y, hi_y = ex[0], ex[-1], ey[0], ey[-1]
    pad_y, pad_x = 0.04 * (hi_y - lo_y), 0.02 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
    ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
    # Mark the certified box, so the blank margin is not mistaken for field.
    for yv in (ey[0], ey[-1]):
        ax.axhline(yv, color="white", lw=1.6, ls="--", alpha=0.75, zorder=5)

    ax.set_ylabel(names[dy], fontsize=FS_LABEL)
    if which == "y" or rate_only:
        ax.set_xlabel(names[dx], fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    # Env name only. The synthesis parameters (lbd, r, w_lb, w_ub) are a property
    # of the shipped config, not of what the panel shows, and they crowded the
    # one thing a reader needs to identify the figure by.
    ax.set_title(pretty(env_name), fontsize=FS_TITLE, pad=14)
    if k_show:
        ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.92)

    if rate_only:
        out = FIG_DIR / f"rate_{env_name}.svg"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[rate] wrote {out}  ({names[dx]} x {names[dy]}, lambda* spread "
              f"{float(Z.max() - Z.min()):.3g})")
        return out

    # ── the marginal, over the dim lambda* varies along ─────────────────────
    lim = ax.get_ylim() if which == "y" else ax.get_xlim()
    m = x0[:, md]
    keep = (m >= lim[0]) & (m <= lim[1])
    dens, centres = _marginal(m[keep], *lim)
    if which == "y":
        axd.fill_betweenx(centres, 0.0, dens, color="0.45", alpha=0.55, zorder=2)
        axd.plot(dens, centres, color="0.15", lw=2.0, zorder=3)
        axd.set_xlabel(rf"$x_0$ density ({names[md]})", fontsize=FS_LABEL - 2)
        axd.set_xlim(left=0.0)
        axd.set_ylim(*lim)
    else:
        axd.fill_between(centres, 0.0, dens, color="0.45", alpha=0.55, zorder=2)
        axd.plot(centres, dens, color="0.15", lw=2.0, zorder=3)
        axd.set_xlabel(names[md], fontsize=FS_LABEL)
        axd.set_ylabel(r"$x_0$ density", fontsize=FS_LABEL)
        axd.set_ylim(bottom=0.0)
        axd.set_xlim(*lim)
    axd.tick_params(labelsize=FS_TICK)

    # Connectors, oriented to match: a line from each shown rollout's start to
    # its place in the distribution.
    for k in range(k_show):
        start_val = traj[k, 0, md]
        if which == "y":
            for a in (ax, axd):
                a.axhline(start_val, ls=":", lw=1.9, color="#7f0f14", alpha=0.85,
                          zorder=6)
            axd.plot(0.0, start_val, "o", ms=9, mfc="white", mec="#7f0f14",
                     mew=2.2, zorder=7, clip_on=False)
        else:
            for a in (ax, axd):
                a.axvline(start_val, ls=":", lw=1.9, color="#7f0f14", alpha=0.85,
                          zorder=6)
            axd.plot(start_val, 0.0, "o", ms=9, mfc="white", mec="#7f0f14",
                     mew=2.2, zorder=7, clip_on=False)

    out_box = ((x0[:, dx] < ex[0]) | (x0[:, dx] > ex[-1])
               | (x0[:, dy] < ey[0]) | (x0[:, dy] > ey[-1]))
    frac = float(out_box.mean())
    if frac > 0:
        # Upper RIGHT: the marginal's left shoulder is where the interesting mass
        # sits on the weak-authority envs, and the note was landing on top of it.
        axd.text(0.985, 0.86, f"{frac:.0%} of $x_0$ outside the certified box",
                 transform=axd.transAxes, fontsize=FS_LEGEND - 3, color="#7f0f14",
                 ha="right", va="top")

    out = FIG_DIR / f"minproj_{env_name}.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[minproj] wrote {out}  ({names[dx]} x {names[dy]}, lambda* spread "
          f"{float(Z.max() - Z.min()):.3g}, marginal over {names[md]} "
          f"({which}-axis), {k_show} trajectories, {frac:.0%} of x0 outside box)")
    return out


def _marginal(u: np.ndarray, lo: float, hi: float, bins: int = 42):
    """Smoothed 1D histogram of x0 along the shared axis.

    A histogram rather than a KDE: no bandwidth to defend, and the reset draw is
    a uniform box perturbation, so the honest shape is flat-topped with real
    edges — a KDE would round exactly the corners that matter.
    """
    H, edges = np.histogram(u, bins=bins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    num = np.convolve(H, k, mode="same")
    # Same edge correction as the 2D case: convolve zero-pads, which would sag
    # the two ends and invent a taper the samples do not have.
    den = np.convolve(np.ones_like(H), k, mode="same")
    return num / np.where(den > 0, den, 1.0), centres


def _edges(c: np.ndarray) -> np.ndarray:
    """Cell centres -> edges, so pcolormesh colours the region each cell covers."""
    step = (c[-1] - c[0]) / max(len(c) - 1, 1)
    return np.concatenate([c - step / 2.0, [c[-1] + step / 2.0]])


def _smooth(a: np.ndarray) -> np.ndarray:
    """Separable [1,2,1]/4 pass — enough to stop Poisson noise reading as shape.

    Hand-rolled rather than scipy.ndimage: this module is imported by a plotting
    entry point that should not pull a new dependency for three lines of blur.
    """
    k = np.array([1.0, 2.0, 1.0])
    k = k / k.sum()

    def blur(m):
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, m)
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, out)

    # Edge correction. np.convolve(mode="same") zero-pads, so without dividing by
    # the blurred indicator the border loses mass and a UNIFORM draw renders with
    # a bright centre and dark rim -- structure the data does not have. reset()
    # samples uniformly over the box, so flat-inside is the correct picture.
    num = blur(a)
    den = blur(np.ones_like(a))
    return num / np.where(den > 0, den, 1.0)


def _arrows(ax, tx, ty, every: int = 45) -> None:
    """Direction arrows along one trajectory, at a fixed fraction of the panel.

    NOT scale_units="xy": consecutive trajectory samples are ~1e-2 apart in data
    units, so an arrow drawn at the step vector's own length collapses to a dot
    (which is what the first version did). The direction is normalized and the
    length comes from the axis range, so an arrow reads the same whatever units
    the two plotted states happen to have.
    """
    idx = np.arange(every, len(tx) - 1, every)
    if idx.size == 0:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    span_x, span_y = x1 - x0, y1 - y0
    du = np.diff(tx)[idx]
    dv = np.diff(ty)[idx]
    # Normalize in PANEL space, else a state with a wide range dominates the
    # direction and every arrow points the same way.
    nu, nv = du / span_x, dv / span_y
    mag = np.hypot(nu, nv)
    keep = mag > 1e-9
    if not keep.any():
        return
    L = ARROW_FRAC
    ax.quiver(tx[idx][keep], ty[idx][keep],
              (nu[keep] / mag[keep]) * L * span_x,
              (nv[keep] / mag[keep]) * L * span_y,
              angles="xy", scale_units="xy", scale=1.0,
              width=ARROW_WIDTH, headwidth=3.6, headlength=4.0,
              headaxislength=3.4, color="#7f0f14", zorder=6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default="car,car_v1,segway,cartpole,quadrotor",
                   help="comma-separated short env names")
    p.add_argument("--generate", action="store_true",
                   help="recompute the lambda* grid (expensive) before plotting")
    p.add_argument("--refresh-x0", "--refresh_x0", action="store_true",
                   help="redraw x0/traj from the current env, keeping the cached "
                        "lambda* grid — the cheap fix when reset() changed but the "
                        "plant and envelope did not")
    p.add_argument("--grid", type=int, default=15)
    p.add_argument("--n-other", "--n_other", type=int, default=8,
                   help="samples of the projected-out dims per cell (the min is "
                        "taken over these)")
    p.add_argument("--n-x0", "--n_x0", type=int, default=2000)
    p.add_argument("--n-traj", "--n_traj", type=int, default=10)
    p.add_argument("--show-traj", "--show_traj", type=int, default=N_TRAJ_SHOWN,
                   help="rollouts to draw (the npz may hold more)")
    p.add_argument("--marginal-axis", "--marginal_axis", choices=["x", "y"],
                   default=None,
                   help="force which axis the x0 marginal is taken over "
                        "(default: whichever one lambda* varies along)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate-only", "--rate_only", action="store_true",
                   help="also write rate_<env>.svg: the state-dependent "
                        "contraction rate alone, no rollouts or x0 marginal")
    a = p.parse_args()

    envs = [e.strip() for e in a.envs.split(",") if e.strip()]
    for e in envs:
        if a.generate or not (DATA_DIR / f"minproj_{e}.npz").exists():
            generate(e, grid=a.grid, n_other=a.n_other, n_x0=a.n_x0,
                     n_traj=a.n_traj, seed=a.seed)
        elif a.refresh_x0:
            refresh_x0(e, n_x0=a.n_x0, n_traj=a.n_traj, seed=a.seed)
        plot(e, n_traj=a.show_traj, axis=a.marginal_axis)
        if a.rate_only:
            plot(e, axis=a.marginal_axis, rate_only=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
