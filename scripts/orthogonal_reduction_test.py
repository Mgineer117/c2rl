"""Orthogonal Reduction Test: is an env's certified contraction rate uniform?

The proposition. If there is a map T : X -> O(n) and a FIXED pair (A0, B0) with

    T(x) A(x) T(x)^T = A0,      T(x) B(x)B(x)^T T(x)^T = B0 B0^T      for all x,

then lambda*(x) = lambda*_0 everywhere: the congruence Wbar -> T(x) Wbar T(x)^T
carries the per-state problem onto one fixed problem, and the underlying
condition is orthogonally invariant, so feasibility is identical at every x.
Contrapositive, which is what this script is for: a state-dependent rate can
only arise when NO such reduction exists.

Two halves, because they prove different things.

1. INVARIANTS (necessary). Every trace of a word in {A, A^T, M := BB^T} is
   invariant under the joint congruence, as are the spectra of A, of sym(A), and
   of M, and A's singular values. If any of them moves with x, no T can exist --
   and this is a certificate of NON-reducibility, not a heuristic: one varying
   invariant rules out every T at once.

2. CONSTRUCTION (sufficient, up to numerics). When the invariants hold still,
   that alone does not produce a T, so this actually solves

       min over T in O(n) of  ||T A(x) T^T - A0||_F^2 + ||T M(x) T^T - M0||_F^2

   at each sampled x, with (A0, M0) read off a reference state. T is
   parametrized as Q expm(S) with S skew and Q in {I, reflection}, covering both
   components of O(n), with restarts. A residual at machine-noise level across
   every sample is the reduction exhibited; a floor well above it means the
   invariants agreed by coincidence at the sampled order and the test is
   inconclusive rather than positive.

A(x) here is the drift Jacobian df/dx, which is exactly what
``figures/data/minproj_<env>.npz`` was computed from -- so the verdict and the
measured lambda* spread it is checked against describe the same object. The CM
LMI that ships uses the generalized Jacobian A(x,u) at the u-box vertices; the
reduction question for that object needs the vertex set carried through, and is
not what this script answers.

    python scripts/orthogonal_reduction_test.py                     # all five paper envs
    python scripts/orthogonal_reduction_test.py --envs segway,cartpole
    python scripts/orthogonal_reduction_test.py --self-check        # test the tester
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "c2rl"))
sys.path.insert(0, str(REPO / "scripts"))

DATA_DIR = REPO / "figures" / "data"
DEFAULT_ENVS = "car,car_v1,segway,cartpole,quadrotor"

# A varying invariant is a proof of non-reducibility, so the threshold only has
# to separate real variation from float noise. Relative spread, so it does not
# care about an env's scale.
TOL = 1e-6


# ─────────────────────────── the invariants ──────────────────────────────── #

def invariants(A, B):
    """Invariants of (A, BB^T) under the joint congruence, each with the scale it
    should be judged against.

    Returns ``{name: (value, scale)}``. The scale is the invariant's own natural
    magnitude, NOT its observed size, and that distinction is load-bearing: M is
    rank m < n, so its n - m zero eigenvalues are pure float noise, and dividing
    their spread by their own magnitude turns 1e-16 into a relative spread of
    order one. The self-check catches exactly that.

    Every trace of a word in {A, A^T, M} is invariant, because T^T T = I collapses
    between adjacent letters and the trace closes the outer pair; a degree-k word
    scales like the product of its letters' norms.
    """
    M = B @ B.T
    nA = max(float(np.linalg.norm(A, "fro")), 1e-300)
    nM = max(float(np.linalg.norm(M, "fro")), 1e-300)
    letters = {"A": A, "a": A.T, "M": M}
    norms = {"A": nA, "a": nA, "M": nM}

    out = {}
    for k in range(1, 4):
        for w in itertools.product("AaM", repeat=k):
            P = np.eye(A.shape[0])
            scale = 1.0
            for c in w:
                P = P @ letters[c]
                scale *= norms[c]
            out["tr(" + "".join(w) + ")"] = (float(np.trace(P)), scale)

    ev = np.linalg.eigvals(A)
    ev = ev[np.lexsort((ev.imag, ev.real))]
    for i, z in enumerate(ev):
        out[f"eig(A)[{i}].re"] = (float(z.real), nA)
        out[f"eig(A)[{i}].im"] = (float(abs(z.imag)), nA)   # sign is basis-dependent

    for i, w in enumerate(np.sort(np.linalg.eigvalsh(0.5 * (A + A.T)))):
        out[f"eig(sym A)[{i}]"] = (float(w), nA)
    for i, w in enumerate(np.sort(np.linalg.eigvalsh(M))):
        out[f"eig(M)[{i}]"] = (float(w), nM)
    for i, s in enumerate(np.sort(np.linalg.svd(A, compute_uv=False))):
        out[f"svd(A)[{i}]"] = (float(s), nA)
    return out


def invariant_spread(As, Bs):
    """Largest scaled spread, which invariant achieved it, and its raw range."""
    rows = [invariants(A, B) for A, B in zip(As, Bs)]
    worst, worst_key, worst_rng = 0.0, "-", (0.0, 0.0)
    for k in rows[0]:
        v = np.array([r[k][0] for r in rows])
        s = float(np.median([r[k][1] for r in rows]))
        rel = float(np.ptp(v)) / max(s, 1e-300)
        if rel > worst:
            worst, worst_key = rel, k
            worst_rng = (float(v.min()), float(v.max()))
    return worst, worst_key, worst_rng


def trace_M_range(Bs):
    """Range of tr(BB^T) over the samples.

    Quoted separately because it is the most interpretable witness available: it
    is an orthogonal invariant on its own, so if it moves with x then no T can
    exist, and for cartpole and segway it moves for a reason readable straight
    off B(x) -- both have theta in every nonzero entry.
    """
    t = np.array([float(np.trace(B @ B.T)) for B in Bs])
    return float(t.min()), float(t.max())


# ─────────────────────── constructing T when it exists ───────────────────── #

def _skew(p, n):
    S = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    S[iu] = p
    return S - S.T


def _expm(S):
    """expm of a small skew matrix via its eigendecomposition (scipy-free)."""
    w, V = np.linalg.eig(S)
    return np.real(V @ np.diag(np.exp(w)) @ np.linalg.inv(V))


def fit_T(A, M, A0, M0, *, restarts=6, seed=0):
    """min over T in O(n) of ||TAT^T - A0||^2 + ||TMT^T - M0||^2. Returns residual."""
    from scipy.optimize import minimize

    n = A.shape[0]
    k = n * (n - 1) // 2
    refl = np.eye(n)
    refl[0, 0] = -1.0                      # the det = -1 component of O(n)
    rng = np.random.default_rng(seed)

    def resid(T):
        return (np.linalg.norm(T @ A @ T.T - A0, "fro") ** 2
                + np.linalg.norm(T @ M @ T.T - M0, "fro") ** 2)

    best = np.inf
    for Q in (np.eye(n), refl):
        for r in range(restarts):
            p0 = np.zeros(k) if r == 0 else rng.normal(scale=1.0, size=k)
            out = minimize(lambda p: resid(Q @ _expm(_skew(p, n))),
                           p0, method="BFGS",
                           options={"maxiter": 600, "gtol": 1e-14})
            best = min(best, float(out.fun))
    # Report as a RELATIVE residual so envs of different scale compare.
    scale = max(np.linalg.norm(A0, "fro") ** 2 + np.linalg.norm(M0, "fro") ** 2, 1e-12)
    return float(np.sqrt(best / scale))


# ──────────────────────────────── per env ────────────────────────────────── #

def measured_spread(env_name):
    """lambda* spread from the cached minproj grid, if it is there."""
    p = DATA_DIR / f"minproj_{env_name}.npz"
    if not p.exists():
        return None
    Z = np.load(p, allow_pickle=True)["Z"]
    return float(Z.max() - Z.min()), float(Z.min()), float(Z.max())


def run_env(env_name, *, n_samples, seed, tol, fit_k):
    import c2rl.tasks.direct.classic as classic
    import gymnasium as gym
    from lambda_subsets import jacobians, sample_state_box

    env = gym.make(classic.env_id(env_name), num_envs=1, device="cpu").unwrapped
    x = sample_state_box(env.X_MIN, env.X_MAX, n=n_samples, seed=seed)
    A, B = jacobians(env, x)
    n = A.shape[1]

    spread, key, rng = invariant_spread(A, B)
    trM = trace_M_range(B)
    reducible = spread <= tol

    # Only bother constructing T when the necessary conditions hold: if an
    # invariant already moved, no T exists and the optimizer would just be
    # measuring how badly it fails.
    residual = None
    if reducible:
        M0 = B[0] @ B[0].T
        residual = max(fit_T(A[i], B[i] @ B[i].T, A[0], M0, seed=seed)
                       for i in range(1, min(fit_k, len(A))))

    if not reducible:
        verdict = "NOT reducible -> rate may be state-dependent"
    elif residual is not None and residual <= 1e-6:
        verdict = "reducible -> rate is uniform"
    else:
        verdict = "inconclusive (invariants agree, no T found)"

    return {"env": env_name, "n": n, "spread": spread, "key": key, "rng": rng,
            "trM": trM, "residual": residual, "verdict": verdict,
            "measured": measured_spread(env_name)}


# ─────────────────────────────── self-check ──────────────────────────────── #

def self_check():
    """The tester must pass a constructed reducible pair and fail a constructed
    non-reducible one -- otherwise a clean table proves nothing."""
    rng = np.random.default_rng(0)
    n = 4
    A0 = rng.normal(size=(n, n))
    B0 = rng.normal(size=(n, 2))

    # (a) reducible BY CONSTRUCTION: conjugate the same pair by random rotations.
    As, Bs = [], []
    for i in range(6):
        T = _expm(_skew(rng.normal(size=n * (n - 1) // 2), n))
        As.append(T @ A0 @ T.T)
        Bs.append(T @ B0)
    spread, key, _ = invariant_spread(As, Bs)
    assert spread < 1e-9, f"rotated copies must look invariant, got {spread:.2e} at {key}"
    res = max(fit_T(As[i], Bs[i] @ Bs[i].T, As[0], Bs[0] @ Bs[0].T) for i in range(1, 6))
    assert res < 1e-6, f"T exists by construction but was not found: residual {res:.2e}"
    print(f"  reducible pair   : spread {spread:.2e}, T recovered to {res:.2e}  OK")

    # (b) non-reducible: scale one sample, which moves the spectrum.
    As2 = [A0, 1.5 * A0]
    Bs2 = [B0, B0]
    spread2, key2, _ = invariant_spread(As2, Bs2)
    assert spread2 > 1e-3, "a rescaled A must be detected as non-reducible"
    print(f"  non-reducible    : spread {spread2:.2e} at {key2}  OK")

    # (c) same A, different actuation geometry -> M's spectrum moves.
    As3 = [A0, A0]
    Bs3 = [B0, np.column_stack([B0[:, 0], 3.0 * B0[:, 1]])]
    spread3, key3, _ = invariant_spread(As3, Bs3)
    assert spread3 > 1e-3, "a rescaled B must be detected as non-reducible"
    print(f"  B-only variation : spread {spread3:.2e} at {key3}  OK")
    print("self-check passed")


# ──────────────────────────────── driver ─────────────────────────────────── #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default=DEFAULT_ENVS, help="comma-separated short names")
    p.add_argument("--n-samples", "--n_samples", type=int, default=200,
                   help="states drawn from the certified box")
    p.add_argument("--fit-k", "--fit_k", type=int, default=12,
                   help="states to actually construct T at (the expensive half)")
    p.add_argument("--tol", type=float, default=TOL,
                   help="relative spread above which an invariant counts as varying")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-check", "--self_check", action="store_true",
                   help="verify the test on constructed reducible/non-reducible pairs")
    a = p.parse_args()

    if a.self_check:
        self_check()
        return 0

    rows = []
    for e in [s.strip() for s in a.envs.split(",") if s.strip()]:
        try:
            rows.append(run_env(e, n_samples=a.n_samples, seed=a.seed,
                                tol=a.tol, fit_k=a.fit_k))
        except Exception as exc:                      # keep going, report the gap
            print(f"[orth] {e}: FAILED ({type(exc).__name__}: {exc})")

    print(f"\nOrthogonal Reduction Test  ({a.n_samples} states per env, "
          f"T constructed at {a.fit_k})\n")
    hdr = f"{'env':<10} {'n':>2}  {'max rel spread':>15}  {'at':<10} " \
          f"{'tr(BB^T) range':>22}  {'lambda* spread':>14}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        # The fitted residual only exists when the invariants passed, i.e. when
        # constructing T was worth attempting. Carried in the verdict rather than
        # its own column, which would be empty for every non-reducible row.
        verdict = r["verdict"]
        if r["residual"] is not None:
            verdict += f"  (||T fit|| = {r['residual']:.2e})"
        if r["measured"] is None:
            meas = "not cached"
        else:
            sp, lo, hi = r["measured"]
            meas = f"{sp:.3g}" + ("" if sp > 1e-9 else " (const)")
        lo, hi = r["trM"]
        trm = f"{lo:.4g} -> {hi:.4g}" + ("" if hi - lo > 1e-12 else "  (const)")
        print(f"{r['env']:<10} {r['n']:>2}  {r['spread']:>15.3e}  {r['key']:<10} "
              f"{trm:>22}  {meas:>14}  {verdict}")

    print("""
Reading this table. The proposition is SUFFICIENT, not necessary: a reduction
forces a uniform rate, so a varying lambda* proves no reduction exists. It does
NOT say the converse. A non-reducible env is therefore PERMITTED a uniform rate,
and car is exactly that witness -- non-reducible (tr(AA^T) = 1 + v^2 moves with
v) yet lambda* = 4.884 everywhere. The only combination that would falsify
something is "reducible" beside a varying lambda*.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
