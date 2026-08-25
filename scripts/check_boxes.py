"""Two invariants that have silently broken before, checked per env.

1. x0 IN THE BOX. reset() draws x_0 = xref_0 + xe_0 from two independent boxes,
   so their sum routinely left [X_MIN, X_MAX] -- 44% of car episodes once began
   outside it. That matters because the CMG is regressed over the box only: at an
   out-of-box x, M(x) is an extrapolation certifying nothing, and every
   Stability/* metric is normalised by the error at x_0, so out-of-box starts
   move the denominator the runs are compared on.

   The fix was a clamp, which introduced its OWN defect: clamping piles
   probability onto the box faces (car went to 50.6% of episodes starting pinned
   to a face) rather than leaving a clean draw. So this reports BOTH the fraction
   outside and the fraction pinned to a face -- a clean env is 0% on both.

2. THE CERTIFIED REGION IS THE BOX. The shipped CM dataset is the certificate:
   whatever set its states cover is what W (hence the Mahalanobis reward and the
   contraction claim) is actually certified over. If those states sit inside a
   sub-box, the env is certified on less than it runs on, and nothing in the code
   would say so -- the dataset's cache key does not include the box.

    python scripts/check_boxes.py
    python scripts/check_boxes.py --envs segway --n 20000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "c2rl"))
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_ENVS = "car,car_v1,segway,cartpole,quadrotor"
FACE_TOL = 1e-6          # within this of a bound counts as pinned to the face


def _cm_cfg(env_name: str) -> dict:
    import yaml
    p = (REPO / "source/c2rl/c2rl/tasks/direct/classic"
         / env_name / "agents/skrl_c2rl_ppo_cfg.yaml")
    if not p.exists():                       # envs that share the car's configs
        p = (REPO / "source/c2rl/c2rl/tasks/direct/classic"
             / "car" / "agents/skrl_c2rl_ppo_cfg.yaml")
    cm = yaml.safe_load(p.read_text(encoding="utf-8"))["cm"]
    return {"lbd": float(cm["lbd"]), "r": float(cm["cvstem_r_scaler"]),
            "w_lb": float(cm["w_lb"]), "w_ub": float(cm["w_ub"])}


def _fmt(v: float) -> str:
    """Match build_cm_dataset's filename formatting (%g-like, no trailing .0)."""
    return f"{v:g}"


def shipped_dataset(env_name: str) -> pathlib.Path | None:
    c = _cm_cfg(env_name)
    p = (REPO / "data" / "classic" / env_name /
         f"cm_data_lbd{_fmt(c['lbd'])}_wlb{_fmt(c['w_lb'])}"
         f"_wub{_fmt(c['w_ub'])}_rs{_fmt(c['r'])}.npz")
    return p if p.exists() else None


def check_env(env_name: str, n: int, seed: int) -> dict:
    import c2rl.tasks.direct.classic as classic
    import gymnasium as gym

    env = gym.make(classic.env_id(env_name), num_envs=64, device="cpu").unwrapped
    lo = env.X_MIN.detach().cpu().numpy().astype(np.float64)
    hi = env.X_MAX.detach().cpu().numpy().astype(np.float64)
    names = [str(s) for s in env.state_names]

    # ── 1. x0 from the env's own reset ──────────────────────────────────── #
    xs = []
    k = 0
    while sum(len(a) for a in xs) < n and k < 4000:
        env.reset(seed=seed + k)
        xs.append(env.x_t.detach().cpu().numpy().copy())
        k += 1
    x0 = np.concatenate(xs, axis=0)[:n].astype(np.float64)

    below = x0 < lo - 1e-9
    above = x0 > hi + 1e-9
    out_any = (below | above).any(axis=1)
    span = np.maximum(hi - lo, 1e-300)
    on_face = ((np.abs(x0 - lo) <= FACE_TOL * span)
               | (np.abs(x0 - hi) <= FACE_TOL * span)).any(axis=1)

    worst_dim, worst_frac = "-", 0.0
    for d in range(x0.shape[1]):
        fr = float((below[:, d] | above[:, d]).mean())
        if fr > worst_frac:
            worst_frac, worst_dim = fr, names[d]

    # ── 2. the shipped CM dataset's coverage of that same box ───────────── #
    ds = shipped_dataset(env_name)
    cm = {"path": None}
    if ds is not None:
        z = np.load(ds, allow_pickle=True)
        xd = np.asarray(z["x"], dtype=np.float64)
        d_out = float(((xd < lo - 1e-9) | (xd > hi + 1e-9)).any(axis=1).mean())
        # Per-dim fraction of the box's width the data actually spans. A
        # certificate over a sub-box is the failure this catches.
        cov = (xd.max(axis=0) - xd.min(axis=0)) / span
        cm = {"path": ds.name, "n": int(xd.shape[0]), "out": d_out,
              "cov_min": float(cov.min()), "cov_dim": names[int(np.argmin(cov))],
              "cov_all": cov}
    return {"env": env_name, "n": len(x0), "dims": x0.shape[1],
            "out_frac": float(out_any.mean()), "worst_dim": worst_dim,
            "worst_frac": worst_frac, "face_frac": float(on_face.mean()),
            "cm": cm, "names": names, "lo": lo, "hi": hi}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default=DEFAULT_ENVS)
    p.add_argument("--n", type=int, default=4000, help="x0 samples per env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true",
                   help="per-dimension coverage for the CM dataset")
    a = p.parse_args()

    rows = []
    for e in [s.strip() for s in a.envs.split(",") if s.strip()]:
        try:
            rows.append(check_env(e, a.n, a.seed))
        except Exception as exc:
            print(f"[boxes] {e}: FAILED ({type(exc).__name__}: {exc})")

    hdr = (f"{'env':<10} {'n':>2}  {'x0 outside':>10} {'worst dim':>12}  "
           f"{'x0 on face':>10}  {'CM data':>8} {'in box':>8} {'min cover':>10}  dataset")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        cm = r["cm"]
        if cm["path"] is None:
            cmtxt = f"{'-':>8} {'-':>8} {'-':>10}  (none shipped)"
        else:
            cmtxt = (f"{cm['n']:>8} {1.0 - cm['out']:>7.1%} "
                     f"{cm['cov_min']:>9.1%}  {cm['path']}")
        print(f"{r['env']:<10} {r['dims']:>2}  {r['out_frac']:>10.2%} "
              f"{r['worst_dim']:>12}  {r['face_frac']:>10.2%}  {cmtxt}")

    print("\nx0 outside and x0 on face must BOTH be 0%: outside means the CMG is\n"
          "extrapolating there, on-face means the clamp is piling probability onto\n"
          "the boundary instead of drawing cleanly. 'min cover' is the smallest\n"
          "per-dimension fraction of the box the CM dataset spans -- well under\n"
          "100% means the env is certified on less than it runs on.")

    if a.verbose:
        for r in rows:
            if r["cm"]["path"] is None:
                continue
            print(f"\n{r['env']}: per-dim CM coverage of [X_MIN, X_MAX]")
            for nm, c, lo_, hi_ in zip(r["names"], r["cm"]["cov_all"],
                                       r["lo"], r["hi"]):
                print(f"  {nm:<14} {c:>7.1%}   box [{lo_:+.3f}, {hi_:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
