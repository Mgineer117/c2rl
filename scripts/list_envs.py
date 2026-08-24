"""List every registered environment with its dimensions and its dynamics.

    python scripts/list_envs.py                  # table of all envs
    python scripts/list_envs.py --keyword classic
    python scripts/list_envs.py --dynamics       # + the f(x)/B(x) source
    python scripts/list_envs.py --dynamics --keyword auv

Classic envs are pure NumPy/PyTorch and are instantiated here, so x_dim, u_dim
and the sv(B) spread are measured, not declared. Isaac envs need Isaac Sim to
construct and define their dynamics through the simulator rather than an
analytic ``get_f_and_B``; they are listed with None in those columns rather than
omitted, so the table stays a complete inventory.

The sv(B) column is the diagnostic that decides whether restricting to a state
subset can certify a faster contraction rate. B enters the CV-STEM LMI only
through ``nu*(2/r)*B B^T`` with ``nu`` shared across states, so a state with
weaker actuation cannot be rescued by any metric -- only excluded. What matters
is therefore whether B's singular values vary, not whether B depends on x at
all: a B that merely rotates (pvtol, the kinematic car) has constant singular
values and offers nothing to a subset.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSIC_DIR = ROOT / "source/c2rl/c2rl/tasks/direct/classic"
DIRECT_DIR = ROOT / "source/c2rl/c2rl/tasks/direct"
N_PROBE = 300


def _env_id(name: str) -> str:
    """Short name -> gym id, via the one map the classic package owns."""
    from c2rl.tasks.direct.classic import env_id
    return env_id(name)

def classic_names() -> list[str]:
    return sorted(p.name for p in CLASSIC_DIR.iterdir()
                  if p.is_dir() and (p / "env.py").exists())


def isaac_names() -> list[str]:
    skip = {"classic", "common", "__pycache__"}
    return sorted(p.name for p in DIRECT_DIR.iterdir()
                  if p.is_dir() and p.name not in skip and not p.name.startswith("__"))


def probe(task: str) -> dict:
    """Instantiate a classic env and measure what it actually does."""
    import gymnasium as gym
    import torch

    env = gym.make(_env_id(task), num_envs=1, device="cpu").unwrapped
    n, m = int(env.num_dim_x), int(env.num_dim_control)
    lo, hi = env.X_MIN.numpy().astype(float), env.X_MAX.numpy().astype(float)
    xs = np.random.default_rng(0).uniform(lo, hi, size=(N_PROBE, n))
    xt = torch.as_tensor(xs, dtype=torch.float32).clone().requires_grad_(True)
    f, B, _ = env.get_f_and_B(xt)
    Bn = B.detach().numpy().astype(float)

    sv = np.linalg.svd(Bn, compute_uv=False)
    spread = float((sv.max(0) / np.maximum(sv.min(0), 1e-300)).max())

    # Which dims B varies with, probed away from the box centre: a term like
    # v*sin(pitch) is invisible to a one-dim-at-a-time probe launched from the
    # centre, where pitch = 0. Segway has exactly that, and probing from the
    # centre wrongly reports vel_x_b inert.
    names = list(getattr(env, "state_names", None) or [f"x{i}" for i in range(n)])
    rng = np.random.default_rng(1)
    anchors = rng.uniform(lo, hi, size=(6, n))
    b_deps, f_deps = [], []
    for d in range(n):
        moved_b = moved_f = False
        for a in anchors:
            base = a.copy()
            pts = np.tile(base, (12, 1))
            pts[:, d] = rng.uniform(lo[d], hi[d], 12)
            t0 = torch.as_tensor(base[None, :], dtype=torch.float32)
            t1 = torch.as_tensor(pts, dtype=torch.float32)
            f0, B0, _ = env.get_f_and_B(t0)
            f1, B1, _ = env.get_f_and_B(t1)
            if float((B1 - B0).abs().max()) > 1e-8:
                moved_b = True
            if float((f1 - f0).abs().max()) > 1e-8:
                moved_f = True
        if moved_b:
            b_deps.append(names[d])
        if moved_f:
            f_deps.append(names[d])

    return dict(
        x_dim=n, u_dim=m, names=names,
        f_norm=float(f.detach().norm(dim=1).mean()),
        sv_spread=spread, b_deps=b_deps, f_deps=f_deps,
        f_src=inspect.getsource(type(env)._f_logic),
        b_src=inspect.getsource(type(env)._B_logic),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyword", default=None, help="substring filter on the env name")
    ap.add_argument("--dynamics", action="store_true",
                    help="also print the f(x) and B(x) source for each classic env")
    args = ap.parse_args()

    import c2rl.tasks.direct.classic  # noqa: F401  registers classic envs

    kw = (args.keyword or "").lower()
    classic = [t for t in classic_names() if kw in t.lower() or kw in "classic"]
    isaac = [t for t in isaac_names() if kw in t.lower() or kw in "isaac"]

    hdr = (f"{'env':22s} {'x_dim':>5s} {'u_dim':>5s} {'sv(B) spread':>12s} "
           f"{'||f||':>8s}  B(x) depends on")
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for t in classic:
        try:
            info = probe(t)
        except Exception as exc:                     # a broken env must not hide the rest
            print(f"{('classic-' + t):22s} {'ERR':>5s} {'ERR':>5s} "
                  f"{'-':>12s} {'-':>8s}  {type(exc).__name__}: {exc}")
            continue
        rows[t] = info
        dep = ", ".join(info["b_deps"]) if info["b_deps"] else "nothing (constant B)"
        print(f"{('classic-' + t):22s} {info['x_dim']:5d} {info['u_dim']:5d} "
              f"{info['sv_spread']:12.3f} {info['f_norm']:8.3f}  {dep}")

    for t in isaac:
        # Isaac envs define their dynamics through the simulator; there is no
        # analytic f/B to report, and constructing one needs Isaac Sim.
        print(f"{t:22s} {'None':>5s} {'None':>5s} {'None':>12s} {'None':>8s}  "
              f"None (Isaac Sim; no analytic get_f_and_B)")

    if rows:
        print("\nsv(B) spread = max over singular values of (max_x sv / min_x sv). "
              "1.000 means B's\nsingular values are constant, so no state subset can "
              "certify a faster lambda\n(B may still ROTATE with x -- pvtol and the "
              "kinematic car both do).")

    if args.dynamics:
        for t, info in rows.items():
            print("\n" + "=" * 78)
            print(f"classic-{t}-v0   x = ({', '.join(info['names'])})   "
                  f"u_dim = {info['u_dim']}")
            print(f"f(x) varies with: {', '.join(info['f_deps']) or 'nothing'}")
            print("-" * 78)
            print(textwrap.dedent(info["f_src"]).rstrip())
            print("-" * 78)
            print(textwrap.dedent(info["b_src"]).rstrip())
        if isaac:
            print("\n" + "=" * 78)
            print("Isaac envs: f(x) = None, B(x) = None — dynamics come from the "
                  "simulator,\nnot an analytic model, so there is no source to print.")
            for t in isaac:
                print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
