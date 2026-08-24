"""Matplotlib figure helpers shared by error_geometry.py and policy_overlay.py.

Styling follows the dataviz conventions: a perceptually-uniform sequential ramp
(viridis) carries the error magnitude on a log scale retuned per frame (see
error_norm),
categorical series colors come from a fixed, CVD-validated order
(viz_common.SERIES_COLORS) and are never reused for anything else, grids/axes
stay recessive, text wears ink colors rather than series colors.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  registers projection="3d"
from viz_common import SERIES_COLORS  # noqa: E402

# Every figure is saved as .svg (vector — see error_geometry.py/policy_overlay.py's
# savefig calls); pngs are not written.
SAVE_FORMAT = "svg"

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

# Sequential ramp for the error surface. viridis rather than a single-hue blue
# ramp: it is perceptually uniform and spans a much wider lightness range, so
# surface curvature stays legible where a light-blue ramp washed out — the low
# end of the landscape (exactly the basin worth reading) was the part that
# disappeared. Series overlays wear white halos to stay separable from it.
ERROR_CMAP = matplotlib.colormaps["viridis"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "savefig.dpi": 170,
})

_HALO = [pe.Stroke(linewidth=3.2, foreground=SURFACE, alpha=0.75), pe.Normal()]


def error_norm(heats: list[np.ndarray], error_range: tuple[float, float] | None = None) -> LogNorm:
    """Log scale for colour + 3D height.

    By default data-Derived from exactly the surfaces being displayed. In the
    2-D video that is the shells of the current frame only, so the axis retunes
    every frame and the difference between those shells fills the plot — which
    is the point of showing them. The cost is deliberate: shading/height are then
    comparable within a frame (all metrics share one scale) but not across
    frames, so read motion from the shells, not from absolute colour.

    ``error_range`` pins it instead, when cross-run comparability matters more.
    """
    if error_range is not None:
        return LogNorm(vmin=float(error_range[0]), vmax=float(error_range[1]))
    finite = np.concatenate([h[np.isfinite(h)].ravel() for h in heats])
    finite = finite[finite > 0]
    lo = max(finite.min(), 1e-6)
    hi = max(finite.max(), lo * 1.5)
    return LogNorm(vmin=lo, vmax=hi)


def _zlim(norm: LogNorm) -> tuple[float, float]:
    """log10 height limits from the colour scale — height and colour must always
    agree, and every panel in a frame shares one norm so the metrics stay
    comparable within that frame."""
    return float(np.log10(norm.vmin)), float(np.log10(norm.vmax))


def _style_3d(ax, norm: LogNorm):
    ax.set_zlim(*_zlim(norm))
    # Terse: the colorbar already spells the quantity out, and a long z-label
    # overruns the 3D box into the neighbouring panel's y-label.
    ax.set_zlabel("log₁₀ err", labelpad=2)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)


def draw_surface_1d(ax, t: np.ndarray, u_levels: np.ndarray, heat: np.ndarray,
                    norm: LogNorm, *, curves: list[tuple[str, np.ndarray]] = (),
                    uref: np.ndarray | None = None):
    """u_dim == 1: the complete t × u × normalized-error surface.

    heat: (C, K). Height is log₁₀ of the next-step normalized error; color is
    the same log scale as every other panel. Control curves are drawn on the
    surface (height interpolated along the u axis at each t), so a policy's
    u(t) is seen riding the landscape it actually traverses.
    """
    K = heat.shape[1]
    H = np.clip(heat, norm.vmin, norm.vmax)
    T, U = np.meshgrid(t[:K], u_levels)
    ax.plot_surface(T, U, np.log10(H), facecolors=ERROR_CMAP(norm(H)),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    def _on_surface(u_curve):
        # height of the surface at the curve's own control value, per timestep
        return np.array([np.interp(u_curve[k], u_levels, np.log10(H[:, k]))
                         for k in range(min(K, len(u_curve)))])

    if uref is not None:
        ax.plot(t[:K], uref[:K], _on_surface(uref) + 0.02, color=SERIES_COLORS["uref"],
                lw=1.4, ls="--", zorder=9)
    for name, u_curve in curves:
        z = _on_surface(u_curve)
        ax.plot(t[:len(z)], u_curve[:len(z)], z + 0.02,
                color=SERIES_COLORS.get(name, INK), lw=2.4, zorder=10, label=name)

    ax.set_xlabel("t  [s]", labelpad=8)
    ax.set_ylabel("u", labelpad=8)
    ax.set_ylim(float(u_levels[0]), float(u_levels[-1]))
    ax.set_box_aspect((1.9, 1.0, 0.8), zoom=1.25)
    ax.view_init(elev=24, azim=-58)
    _style_3d(ax, norm)


def draw_frame_2d(ax, u0: np.ndarray, u1: np.ndarray, Z: np.ndarray, norm: LogNorm,
                  *, markers: list[tuple[str, np.ndarray]] = (),
                  uref_pt: np.ndarray | None = None, title: str | None = None,
                  history: list[np.ndarray] | None = None):
    """u_dim == 2: one video frame — the complete u0 × u1 × error surface at a
    fixed timestep.

    Z: (C0, C1) normalized error. Height and colour come from ``norm``, which
    the caller retunes per frame over the shells on screen, so their spread
    fills the plot. ``markers`` places each policy's commanded control on the
    surface.

    ``history``: earlier frames' landscapes, ordered oldest → most recent,
    drawn as translucent surfaces beneath the current one with alpha ramping up
    toward the present so the newest reads strongest. Without them the
    geometry's motion is invisible in any single frame — you see where the
    surface is but not how it is moving; the stack of fading shells is that
    motion.
    """
    H = np.clip(Z, norm.vmin, norm.vmax)
    U0, U1 = np.meshgrid(u0, u1, indexing="ij")
    z_lo, z_hi = _zlim(norm)
    # Dedicated floor plane below the data range: a good controller drives the
    # trough down onto vmin, so a floor at z_lo would be flush against the
    # surface with no room to read markers against.
    z_floor = z_lo - 0.18 * (z_hi - z_lo)

    # Semi-transparent surface: mplot3d has no true occlusion (whole artists are
    # depth-sorted), so an opaque bowl hides anything beneath it — including the
    # marker in its own trough, which is exactly where a good controller sits.
    # Oldest → newest, alpha ramping up: the present reads strongest and the
    # trail shows where the geometry came from. A single previous frame gives
    # the plain alpha=0.5 ghost.
    for Zp, alpha in zip(history or [], np.linspace(0.14, 0.5, len(history or [1]))):
        Hp = np.clip(Zp, norm.vmin, norm.vmax)
        cp = ERROR_CMAP(norm(Hp))
        cp[..., 3] = alpha
        ax.plot_surface(U0, U1, np.log10(Hp), facecolors=cp,
                        rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False)
    colors = ERROR_CMAP(norm(H))
    colors[..., 3] = 0.9
    ax.plot_surface(U0, U1, np.log10(H), facecolors=colors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax.contourf(U0, U1, H, levels=14, zdir="z", offset=z_floor,
                cmap=ERROR_CMAP, norm=norm, alpha=0.9)

    def _z_at(pt):
        i = int(np.abs(u0 - pt[0]).argmin())
        j = int(np.abs(u1 - pt[1]).argmin())
        return float(np.log10(H[i, j]))

    # A hair above the floor plane: exactly coplanar with the floor contour they
    # z-fight with it and vanish on whichever panel loses.
    z_mark = z_floor + 0.02 * (z_hi - z_lo)

    def _place(pt, color, **kw):
        ax.plot([pt[0], pt[0]], [pt[1], pt[1]], [z_mark, _z_at(pt)],
                color=color, lw=1.1, alpha=0.85, zorder=11)
        ax.scatter(pt[0], pt[1], z_mark, color=color, depthshade=False, zorder=12, **kw)

    if uref_pt is not None:
        _place(uref_pt, SERIES_COLORS["uref"], s=60, marker="X", lw=1.2,
               edgecolor=SURFACE)
    for name, pt in markers:
        _place(pt, SERIES_COLORS.get(name, INK), s=95, marker="o",
               edgecolor=SURFACE, lw=1.4, label=name)

    ax.set_xlabel("u₀", labelpad=6)
    ax.set_ylabel("u₁", labelpad=6)
    ax.set_xlim(float(u0[0]), float(u0[-1]))
    ax.set_ylim(float(u1[0]), float(u1[-1]))
    ax.set_box_aspect((1.0, 1.0, 0.7), zoom=1.1)
    ax.view_init(elev=26, azim=-58)
    _style_3d(ax, norm)
    ax.set_zlim(z_floor, z_hi)  # widened to include the floor projection plane
    if title:
        ax.set_title(title, color=INK_2, pad=-2)


def _edges(centers: np.ndarray) -> np.ndarray:
    mid = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


# Series that are reference baselines rather than competing entries, mapped to
# the legend text that says so. Membership here drives gray+dashed styling in
# draw_error_panel; SERIES_COLORS must carry a matching recessive gray.
BASELINE_LABELS = {
    "random": "random (untrained baseline)",
    "euclidean (M = I)": "euclidean (M = I) — unconditioned baseline",
}


def draw_error_panel(ax, t: np.ndarray, series: list[tuple[str, np.ndarray]],
                     *, lbd: float | None = None, euclidean: dict | None = None,
                     error_range: tuple[float, float] | None = None):
    """Normalized-error-vs-time comparison (log y). series: [(name, r(t)), ...].
    Optionally a e^{-λt} contraction guide and dotted Euclidean counterparts.

    ``error_range`` optionally pins the y-axis; None (default) autoscales."""
    for name, r in series:
        # Reference baselines, not competing series: gray + dashed, so they never
        # rely on hue alone and stay visibly recessive. "random" is the untrained
        # CMG (architecture and bounds, no training); "euclidean" is no metric at
        # all. Both exist to be departed from, and the styling says so.
        label = BASELINE_LABELS.get(name)
        baseline = label is not None
        ax.plot(t[:len(r)], np.clip(r, 1e-8, None), lw=1.4 if baseline else 1.8,
                ls="--" if baseline else "-", alpha=0.9 if baseline else 1.0,
                color=SERIES_COLORS.get(name, INK), label=label or name)
    if euclidean:
        for name, r in euclidean.items():
            ax.plot(t[:len(r)], np.clip(r, 1e-8, None), lw=1.0, ls=":",
                    color=SERIES_COLORS.get(name, INK), alpha=0.7)
    if lbd is not None:
        ax.plot(t, np.exp(-lbd * t), lw=1.2, ls="-.", color=INK_2, alpha=0.8,
                label=f"e^(-{lbd:g}·t) guide")
    ax.set_yscale("log")
    if error_range is not None:
        ax.set_ylim(*error_range)
    ax.set_ylabel("normalized error  √(V/V₀)")
    ax.grid(True, which="both", alpha=0.5)
    ax.margins(x=0)
    ax.legend(loc="lower left", ncols=min(len(series) + 1, 5), fontsize=9)


def add_colorbar(fig, norm, cax, *, label="next-step normalized error  √(V/V₀)   (log scale)"):
    """Colorbar for the shared error scale, drawn into a dedicated axes.

    Built from a standalone ScalarMappable: ``plot_surface(facecolors=...)``
    produces no mappable of its own, and every panel shares one scale anyway.

    ``cax`` must be its own gridspec cell rather than ``ax=[...]``: a 3D axes'
    tick/axis labels overflow its bounding box, which constrained_layout does
    not model, so a colorbar auto-placed beside one lands on top of them.
    """
    cax.clear()  # the norm retunes per frame, so the bar is rebuilt each time
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=ERROR_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, color=INK_2)
    cb.outline.set_edgecolor(GRID)
    return cb


def save_video(fig, update_fn, num_frames: int, path: str, *, fps: int = 20):
    """Render an mp4 (ffmpeg) or gif (pillow fallback) from a per-frame callback."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers

    anim = FuncAnimation(fig, update_fn, frames=num_frames, blit=False)
    if writers.is_available("ffmpeg"):
        anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=3600))
    else:
        path = path.rsplit(".", 1)[0] + ".gif"
        print(f"[viz] ffmpeg unavailable — writing {path} instead")
        anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
