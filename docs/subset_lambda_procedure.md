# Certified contraction rate on a subset of the state space

A procedure for measuring how much of a plant's certified contraction rate is
lost to a small set of worst-case states, and for reporting the result without
overclaiming. Written to be transcribed into a paper's method section.

Implementation: `scripts/lambda_subsets.py`. Synthesis: `cvstem_joint` in
`source/c2rl/c2rl/agents/skrl/ncm_synthesis.py`.

---

## 1. Setup and notation

Plant `ẋ = f(x) + B(x)u` on a compact state box `X = [x_lo, x_hi] ⊂ R^n`, with
`A(x) = ∂f/∂x` the drift Jacobian. CV-STEM synthesis (Tsukamoto & Chung) solves
one joint SDP over a finite sample set `S = {x_1..x_N} ⊂ X` for a shared metric
scale `ν`, conditioning bound `χ`, and per-sample `W̄_k`:

```
minimize    (1/λ)·χ + ν
subject to  I ⪯ W̄_k ⪯ χI                                        for all k
            S_k ⪯ -εI                                            for all k
            ν ≤ 1/w_lb,   χ ≤ ν·w_ub                             (envelope)

  S_k = (W̄_k - I)/dt + A_k W̄_k + W̄_k A_kᵀ + 2λ W̄_k - ν(2/r)B_k B_kᵀ
```

Deployed metric `W_k = W̄_k/ν`, gain `K(x) = R⁻¹B(x)ᵀM(x)` with `M = W⁻¹`,
`R = r·I`. The rate `λ` is a *parameter*, not a variable: the program is
feasible or not at each `λ`.

Define

```
λ*(S) = sup { λ : the program above is feasible over sample set S }
```

**The monotonicity that the whole procedure rests on.** For `S' ⊆ S`, the
program over `S'` has a subset of the constraints of the program over `S` and
the same variables. Any feasible point for `S` restricts to a feasible point for
`S'`. Hence

```
S' ⊆ S   ⟹   λ*(S') ≥ λ*(S)                                          (M)
```

Every curve this procedure reports must be non-decreasing under nesting. A
decrease is not a finding — it is a diagnostic that either the sample sets were
not nested or the solver returned an inaccurate answer. Both are covered in §6.

---

## 2. Step 1 — baseline uniform rate

Draw `S_0 = {x_1..x_N}` i.i.d. uniform on `X` with a fixed seed. Compute
`λ*(S_0)` by bisection on `λ` (§4). Record `λ_0 = λ*(S_0)`, `ν_0`, `χ_0`.

This is the uniform rate the deployed controller certifies, and the denominator
of every ratio reported later.

**Report `N` and `ε` together.** They are not independent (§7).

---

## 3. Step 2 — identify the binding states

`cvstem_joint` returns `{W, ν, χ, J}` and no dual variables. Duals are not
needed: every constraint is reconstructible from the primal solution. For each
sample, undo the normalization `W̄_k = ν·W_k` and form

```
slack_lmi(k) = -ε - λ_max(S_k)          (constraint was S_k ⪯ -εI)
slack_chi(k) =  χ  - λ_max(W̄_k)         (constraint was W̄_k ⪯ χI)
slack_eye(k) =  λ_min(W̄_k) - 1          (constraint was W̄_k ⪰ I)
```

all `≥ 0` at a feasible point. Normalize `slack_lmi` by `‖S_k‖₂` so samples at
different metric magnitudes are comparable. The samples with `slack_lmi ≈ 0` are
those the rate is resting on: perturbing any of them changes `λ*`; perturbing an
interior sample does not.

**Also check the two global caps**, and check them *first*:

```
ν pinned  ⟺  ν ≥ (1/w_lb)(1-δ)
χ pinned  ⟺  χ ≥ ν·w_ub·(1-δ)              δ = 1e-3
```

`ν ≤ 1/w_lb` and `χ ≤ ν·w_ub` are *single* constraints shared by all samples, so
they never appear as per-sample tightness. Omitting this check misattributes an
envelope-limited certificate to the state box. (Observed on cartpole at `N=200`:
one sample of 200 tight on the LMI while `ν = 100.000` sat exactly on its cap of
`1/w_lb = 100`.)

**Report the count of tight samples.** It characterizes the plant:

| tight / N | interpretation |
|---|---|
| small (cartpole: 1/100) | a thin shell of worst-case states caps λ; subsets help |
| all (quadrotor: 60/60) | the constraint is uniformly active; no small set to excise |

The second case has a sharper signature worth checking for, because it means the
experiment cannot succeed and should not be run further: if λ*, ν and χ come back
*identical* across levels and across both subset rules, the optimum is fixed by a
global cap and is independent of which states are in the set. Measured on the
quadrotor — λ=0.2407, ν=100.000, χ=27.363 identical over seven solves spanning
n=150 down to n=33, both rules — with ν exactly at `1/w_lb = 100` while χ=27.4
sat far below its own cap of `ν·w_ub = 10⁴`. There `w_lb` alone throttles the
actuation term `ν(2/r)BBᵀ`. The fix is a different envelope, not a smaller region.

Confirm it directly rather than by inference — re-solve the same sample set at
looser `w_lb` and see whether λ tracks the cap. On the quadrotor:

| w_lb | ν cap | λ* | ν attained | classifier |
|---|---|---|---|---|
| 0.01 | 100 | 0.2407 | 100.000 (pinned) | envelope |
| 0.001 | 1 000 | 0.8155 | 978.5 | state box |
| 0.0001 | 10 000 | 1.6504 | 9389.3 | state box |

λ rises 3.4× then 2.0× as the cap is relaxed — diminishing, i.e. the plant
crosses over from envelope-limited to genuinely box-limited. This is also the
honest way to report the cost: `‖K‖₂ ≤ ‖B‖₂/(r·w_lb)`, so each decade of `w_lb`
buys rate by permitting a decade more gain. Quote the pair, never the rate alone.

---

## 4. Computing λ*(S)

Bisection on `λ ∈ [λ_min, λ_hi]`, each step one SDP solve, feasibility as the
oracle. Stop at relative tolerance `tol`.

Three requirements:

- **Report `tol`.** A trend below tolerance is not a trend. Use `tol ≤ 0.005`
  when the expected effect is small; `0.02` was too coarse to resolve some of
  the effects here.
- **Report ceiling hits.** If `λ_hi` itself is feasible, `λ*` is only bounded
  below. Never plot that as a value.
- **Cost.** One joint SDP over `N` samples adds an `n×n` PSD block plus two LMIs
  per sample. Measured wall-clock for a single solve: 6 s at `N=100`, 571 s at
  `N=1000`, 3444 s at `N=2500`, i.e. `T ∝ N^1.95`. With ~13-20 solves per
  bisection and one bisection per level, `N ≈ 100-300` is the practical range.

---

## 5. Step 3 — nest

Two subset rules. They answer different questions and should be reported as
different claims. **Both operate on a single draw `S_0`, filtered — never on a
fresh draw per level** (§6).

### 5a. Certificate-driven (diagnostic)

Greedily remove what binds. Two variants, and the choice is a real trade:

- **`box`** — per level, cut one axis-aligned face so the single tightest sample
  falls outside, choosing among the `2·d` candidate cuts the one that loses the
  fewest samples. The retained set stays a box `X_k ⊂ X_{k-1}`, i.e. a region
  that can be written down, stated in a paper, and checked at runtime.

- **`samples`** — per level, drop the tightest `p`-fraction outright. Faster
  rise (it removes every binding state each level, and is not restricted to
  axis-aligned cuts), but the retained set is a point cloud, not a region. This
  is an **upper bound** on what any subset of that size could certify, not a
  deployable region.

Two failure modes to avoid, both observed:

1. *Do not* drop samples and then re-derive membership from their bounding box.
   Any binding sample that was interior is back inside, and the level silently
   repeats the previous solve. (Observed on segway: three identical levels,
   `n=94, λ=6.9815`, reported as a curve.)
2. *Do not* demand a single cut past the whole tight set. When those samples are
   spread along an axis, the cheapest such cut is enormous — 100 → 14 samples in
   one step on cartpole. Exclude one sample at a time, and score cuts by samples
   lost, not by volume (volume says nothing about where samples are).

### 5b. Tube (deployment)

Sample `x = x_ref + s·x_e` with `s ~ U(0,1)` per sample, drawn once; level `k`
keeps `{s ≤ ρ_k}` for a decreasing radius schedule `ρ_k`. Nested by construction.

Do **not** instead rescale `x = x_ref + ρ_k·x_e` per level: that moves every
sample rather than dropping any, so each level is a fresh draw and (M) does not
apply. (Observed on cartpole: 0.0304, 0.0441, 0.0384, 0.0347 — dips that are
pure resampling noise.)

**State the reference distribution explicitly.** A tube about a uniformly drawn
`x_ref` is not a tube about the deployed reference trajectory, and can be
strictly harsher: on cartpole, `x_ref ~ U(X)` reaches ±60° tilt where the actual
episode reference starts upright, giving tube rates *below* the uniform-box rate
rather than above it. If the claim is about deployment, sample `x_ref` from real
reference rollouts.

---

## 6. Validity conditions — state these in the paper

1. **Nesting.** Every level's sample set is a literal subset of the previous.
   Assert it in code.
2. **Monotonicity — in the subset and in the envelope.** The reported curve is
   non-decreasing under nesting (M). The same argument applies to relaxing
   `w_lb`: a larger `1/w_lb` weakens `ν ≤ 1/w_lb`, enlarging the feasible set, so
   `λ*` can only rise. Either violation means inaccurate solves; do not interpret
   through it.

   This doubles as the **numerical-trust test**, and it is worth running before
   any loose-envelope claim. On segway, relaxing `w_lb` from 1e-6 to 1e-8 sent
   λ* from 4.8012 down to 0.0133 — a 360× *drop* where an increase was
   guaranteed. The solve at 1e-8 had also stopped tracking its own cap
   (ν = 817.8 against a bound of 1e8). Both are signatures of an interior-point
   solver failing at extreme scale, not of the plant. Any λ from that regime is
   discarded, not reported.
3. **Boundedness.** `ν` must be bounded, i.e. the envelope `w_lb/w_ub` must be
   present. Without it the program drifts toward unbounded: `ν ≈ 1.9e7` observed
   on segway, where `‖K‖ ~ ν` makes the "rate" unrealizable and the solve
   numerically untrustworthy. A large λ obtained this way is not a result.
4. **Active dimensions.** Shrinking a dimension `A(x)`/`B(x)` do not depend on
   removes volume without removing difficulty and manufactures a trend. Detect
   them by perturbation rather than by inspection: for the car this is
   `(θ, v)` of 4 states; for the quadrotor `(thrust, roll, pitch)` of 10. Report
   volume fractions over the active subspace and say so — a 3-D volume fraction
   is not comparable to a 2-D one.
5. **Sample floor.** Stop when a level falls below a stated minimum; a λ from a
   handful of samples estimates nothing.
6. **Invariance — the load-bearing caveat.** `λ*(S')` certifies contraction
   *while the trajectory remains in `S'`*. It composes into a statement about
   the closed loop only if `S'` is forward invariant, or if excursions are
   separately bounded. Uniform λ over the whole box exists precisely to avoid
   owing this argument. **Any subset claim owes it.** Without an invariance
   argument the correct wording is "the certificate is limited by a thin set of
   states", not "the controller contracts at the faster rate".

---

## 7. On ε and N

ε converts a *finite-sample* certificate into a *box-wide* one. If the LMI
residual is `⪯ -εI` at every sample and the map `x ↦ S(x)` is `L`-Lipschitz on
`X`, then every state within `ε/L` of a sample satisfies the constraint. So ε
buys a covering radius `ε/L`, and the two knobs trade directly:

- few samples → large gaps to cover → **larger ε** required
- many samples → small gaps → **smaller ε** sufficient

Using a coarse-and-conservative pair (`N` small, ε large) for a *search* and a
fine pair (`N` large, ε small) for *synthesis* is therefore legitimate, under
one condition: **the search proposes, synthesis certifies.** The reported λ must
be the one feasible at the deployed `(N, ε)`, verified there. Feasibility is a
hard pass/fail so there is no circularity. What is not legitimate is reporting a
search-time λ never re-checked at deployment settings, or lowering ε until
something turns feasible.

Two riders: `L` must actually be bounded (from `‖∂A/∂x‖`, `‖W̄‖`, `‖∂W̄/∂x‖`) for
this to be a theorem rather than a heuristic with the right sign; and smaller ε
puts the labels closer to the PSD boundary, which is measurable in the metric
network's regression error and worth checking rather than assuming.

**Keep ε fixed across every number in a comparison.** Envs in this repo do not
all ship the same ε (segway 0.01 vs 0.1 elsewhere), and ε changes what
certifies.

---

## 8. Reporting checklist

Per experiment: `N`, ε, `dt`, `r`, envelope `[w_lb, w_ub]`, solver, seed,
bisection `tol`, active dims, subset rule and its parameter, samples retained
per level.

Per level: `λ*`, `ν`, `χ`, tight-sample counts (LMI and χ separately), whether
ν or χ was pinned, and the retained region (box bounds, or "point cloud" for
`--drop-mode samples`).

Headline number as a ratio with its cost: *"λ rose 2.23× for 11% of the active
volume removed"* — never the ratio alone.

State plainly whether the mechanism was identified. The strongest form of this
result is when the cuts the SDP chooses match an independent analytical
prediction: on cartpole `B(x) ∝ cos θ` degrades input authority as the pole
tilts, and the procedure — with no such knowledge — cut `pitch` at every level
and never once cut `pitch_rate`.

---

## 8b. All four plants on one footing (`w_lb=1e-3`, `w_ub=1e3`, ε=0.1)

Certificate rule, `--drop-mode box`, `p=0.10`, 6 levels. The one envelope every
plant is feasible under, so these four numbers are directly comparable.

At the **deployed** `r` for each plant (the one `find_uniform_lambda`'s actuator
check returns), `p=0.25` per level:

| plant | r | λ* full box → subset | gain | vol kept | tight/N at L0 |
|---|---|---|---|---|---|
| **cartpole** | 12.8 | 0.0890 → **0.5990** | **6.73×** | 27% | 1/100 |
| **segway** | 1.0 | 0.9621 → **1.5297** | **1.59×** | 25% | 1/100 |
| car | 1.6 | 5.0146 → 5.0146 | 1.00× | 42% | 100/100 |
| quadrotor | 12.8 | 0.8156 → 0.8156 | 1.00× | 43% | 60/60 |

**`r` is a third axis, alongside λ and the envelope, and it must match the
config.** Cartpole measured 2.18× at `r=1.6` and 6.73× at its deployed `r=12.8`
— same plant, same draw, same rule. A larger `r` shrinks the actuation term
`ν(2/r)BBᵀ`, so authority is scarcer, so the variation in `B(x)` costs more and
excising the weak-authority states pays more. Same shape as the envelope result
in the next section: **the tighter the constraint, the more a subset is worth.**

The r=1.6 cartpole numbers were never deployable — that is exactly why the
actuator check raised `r`. Quote the sweep at the shipped `r` or the gain is
measured on a controller that would leave the actuator box.

### These λ are LMI-only — the actuator check is not applied

`lambda_subsets.py` bisects λ against **LMI feasibility alone**.
`find_uniform_lambda` additionally requires ≤`--viol-frac` of the drawn
`u = uref − K(x)·e` to stay inside the applied box, and the two disagree:

| plant | sweep λ (LMI only) | certified λ (LMI + actuator) |
|---|---|---|
| cartpole | 0.0890 → 0.5990 | 0.0771 |
| segway | 0.9621 → 1.5297 | **none — INFEASIBLE at every envelope tried** |
| car | 5.0146 | 0.3902 |
| quadrotor | 0.8156 | 0.2601 |

So a sweep gain is a comparison of **LMI-certifiable rates over nested sets**. It
is not a claim that a controller achieving those rates fits the actuator box —
at every level above L0 the rate is un-checked, and even L0 exceeds the
actuator-checked λ.

Segway is the extreme case: it has no certified (λ, r) at all, because its `r`
branch is inert (the LMI absorbs `r` into `ν`, so the gain never shrinks and the
violation rate never improves — measured to r=819.2 before the LMI died). The
`r=1.0` its sweep ran at is a stale config value, not a certified one. Report
segway's 1.59× as an LMI result or not at all.

Wording that survives review: *"the certificate is limited by a thin set of
states — removing them admits a λ this many times faster"*, not *"the controller
contracts this many times faster."* The second claim needs the actuator check and
the invariance argument of §6.6.

Trustworthy depth: quote **L3** (n=40), not L5 (n=20-22). λ* is certified over
Sampled states, so a thin set is optimistic — the same covering argument as §7,
applied to the subset. At L3: cartpole 0.5089 (**5.72×**, 47% volume), segway
1.4443 (**1.50×**, 42% volume).

**The tight-sample count at L0 predicts the outcome exactly**: the two plants with
a thin binding shell (1 of 100) gain 1.5–1.7×, the two with a uniformly active
constraint (all of N) gain nothing. That is one solve, not a six-level sweep, and
it is the cheapest thing in this document to run.

Both plants that gain are gravity-driven pendulums whose input authority or drift
varies strongly with tilt, and on both the gain comes almost entirely from
narrowing `pitch` while the rate dimension barely moves. The two that do not gain
fail for structurally different reasons: the car's θ enters `A(x)` only through a
rotation, so its samples are interchangeable (λ, ν, χ identical to 6 digits while
the tight set churns between 25 and 82 members); the quadrotor's LMI is active at
every sample simultaneously.

## 8c. Why a plant does or does not respond — measure it in one pass

The joint program shares one ν and one χ across every sample, so

```
λ*(S) ≤ min_{x ∈ S} λ*(x)                                             (W)
```

where `λ*(x)` is the rate a single state admits (a one-sample SDP). The worst
state is a ceiling on the whole set, so a subset can raise λ* only by removing
states that are worse than the rest. Whether such states exist is a property of
the plant, measurable per state without any nesting machinery.

Measured at `w_lb=1e-3`, ε=0.1, 40 uniform states per plant:

| plant | λ*(x) min | med | max | **max/min** | frac within 10% of worst | sweep gain |
|---|---|---|---|---|---|---|
| cartpole | 0.733 | 1.383 | 1.667 | **2.27×** | 10% | 1.70× |
| segway | 1.063 | 1.466 | 1.759 | **1.65×** | 5% | 1.47× |
| car | 4.981 | 4.981 | 4.981 | **1.00×** | 100% | 1.00× |
| quadrotor | 0.816 | 0.816 | 0.816 | **1.00×** | 100% | 1.00× |

Car and quadrotor return the same `λ*(x)` at all 40 states, to solver precision.
Every state is exactly as hard as every other, so there is nothing a subset can
remove — 1.00× is forced, not incidental. Cartpole and segway have a 2.27× and
1.65× spread with only 10% and 5% of states near the worst: a thin hard shell,
which is exactly what the sweep exploits.

**`max/min` upper-bounds the achievable gain** (shrinking to the single best
state is the limit), and both responsive plants realise 75–89% of it: cartpole
1.70 of 2.27, segway 1.47 of 1.65.

(W) is also tight where it should be: cartpole's joint λ*(S) = 0.7332 equals its
worst pointwise 0.733 exactly. On segway the joint 0.9621 falls below the worst
pointwise 1.063 — the gap is the cost of one shared ν/χ across heterogeneous
states, which is a second, independent reason a subset helps there.

**This is a cheap diagnostic**: 40 one-sample SDPs, no nesting, no bisection over
sets. If `max/min ≈ 1`, the sweep cannot produce an effect and should not be run.

### Cheaper still, and it is the actual cause: does `B(x)` vary?

`B` enters the LMI only through `ν(2/r)BBᵀ`, the single term supplying control
authority, and `ν` is shared across the set. `W̄_k` is a free per-state variable,
so it absorbs whatever `A(x)` does; what it cannot absorb is a state with less
actuation than its neighbours, because the authority budget is global. So
per-state difficulty is uniform whenever `B` is constant, no matter how `A` moves.

Measured over 200 uniform states per plant — one SVD per sample, no SDP:

| plant | **sv(B) spread** | sv(A) spread | λ*(x) spread | sweep gain |
|---|---|---|---|---|
| cartpole | **2.208×** | 6.066× | 2.25× | 1.70× |
| segway | **1.139×** | 1.357× | 1.81× | 1.47× |
| car | **1.000×** | 1.988× | 1.00× | 1.00× |
| quadrotor | **1.000×** | 3.806× | 1.00× | 1.00× |

`sv(B)` separates the four exactly. `sv(A)` does not: the quadrotor has the
second-largest `A` variation and gains nothing, so a varying drift Jacobian is
neither necessary nor sufficient. In this repo the split is structural —
cartpole's `B ∝ cos θ/(m_c+m_p sin²θ)` and segway's `B ∝ (a cos θ+b)/(cos θ+c)`
against car's and quadrotor's constant 0/1 matrices.

**Criterion: a subset can raise λ* only if `B(x)`'s singular values vary over the
box.** Check this first; it costs one SVD per sample.

## 8d. The shipped pipeline

Three stages, each at its own `(N, ε)`, and the split is deliberate — see §7.

```
1. SEARCH      find_uniform_lambda   N=100    eps=0.1   --cm-dt 1.0
               --w-lb 0.001 --w-ub 1000        -> proposes (lbd, r)
2. CERTIFY +   cvstem_metric_dataset N=10000  eps=0.01
   GENERATE    same lbd/r/envelope             -> caches the metric, or reports
                                                  INFEASIBLE (= lbd too high)
3. TRAIN       every algorithm config carries that (lbd, r, w_lb, w_ub, cm_eps)
```

`ε` falls 0.1 → 0.01 between stages 1 and 2 while `N` rises 100 → 10000. The two
move in opposite directions on feasibility (smaller ε relaxes `S ⪯ -εI`, more
samples adds constraints), so neither dominates and stage 2 is a genuine test,
not a formality. That is why the generation solve doubles as the certification:
a λ that only held on a thin draw fails there.

A full search at N=10000 is not an option: one joint solve is ~15 h at that size
(`T ∝ N^1.95`), and a search is 10–20 solves, i.e. 150–300 h.

### Values in the configs

Measured at the stage-1 point above (`N=100`, ε=0.1, `--cm-dt 1.0`,
`w=[1e-3,1e3]`), each at the `r` its own config ships. Every row passes the 5%
actuator check, so this table is now what the configs carry rather than a
snapshot that drifted from them:

| env | λ | r | ν | χ | ‖K‖₂ max | out of box | budget |
|---|---|---|---|---|---|---|---|
| car | 0.3902 | 1.6 | 4.394 | 4.084 | 2.137 | 4.34% | 1.5 |
| car_v1 | 0.0771 | 3.2 | 13.56 | 8.417 | 2.043 | 4.82% | 1.5 |
| cartpole | 0.3902 | 3.2 | 960.8 | 155.4 | 30.03 | 3.00% | 6 |
| segway | 0.0152 | 6.4 | 1000 | 1012 | 22.68 | 0.65% | 6 |
| quadrotor | 1.3169 | 0.1 | 33.00 | 198.1 | — | 0.35% | 30 |

Stage 2 is the BINDING gate, and stage 1 alone can over-reach. Re-verified at
`N=1000`, ε=0.01, `--cm-dt 1.0`, same `r` and envelope:

| env | λ | ν | χ | ‖K‖₂ max | out of box | stage 2 |
|---|---|---|---|---|---|---|
| car | 0.3902 | 4.233 | 3.853 | 2.066 | 3.55% | pass |
| car_v1 | 0.0771 | 9.354 | 6.138 | 1.926 | 1.83% | pass |
| cartpole | 0.3902 | 1000 | 151.4 | 34.67 | 4.77% | pass |
| segway | 0.0152 | 1000 | 1319 | 23.43 | 0.68% | pass |
| quadrotor | 1.3169 | 32.09 | 192.7 | 26.94 | 0.47% | pass |

car's ν=4.233 / χ=3.853 here reproduce `8a64182`'s recorded numbers exactly,
which is the check that this harness measures the same thing the configs claim.

**segway is why both stages exist.** Stage 1 returns λ=0.0228 for it (0.59% out
of box at N=100/ε=0.1) — one rung faster than the shipped 0.0152. That rung is
wrong: at `N=1000`/ε=0.01 the LMI is **infeasible** at 0.0228, so it could never
have survived the N=10000 build either, since more samples is strictly harder. A
stage-1 result is a proposal, not a certificate; promoting one without stage 2
buys a rate that evaporates at deploy density. 0.0152 is the fastest rate that
clears both.

Note the two stages move ε and N in opposite directions on difficulty, so
neither dominates: ε 0.1 → 0.01 relaxes `S ⪯ -εI` while N 100 → 1000 adds
constraints. Running stage 2 at ε=0.1 instead (N=1000, ε=0.1) is therefore
stricter than anything that ships, and it reports λ=0.0514 for car_v1 and
0.0045 for segway — real numbers, but for a program the pipeline never solves.

**segway is not infeasible** — the row below claiming so predates `8a64182`,
which certified it at λ=0.0152 with `r=6.4`. Its ν does pin at exactly
`1/w_lb = 1000`, so it sits on the envelope boundary and nothing looser is
available to it, which is what the §"segway does not fit this envelope"
discussion below is really describing. That section's *mechanism* still holds;
its verdict does not.

`car_v1` is the class-III half of the car pair
(`v ∈ [0.2, 2]` makes the Hautus margin `σ = min(1, v) = v < 1`), and it
certifies 5× slower than the otherwise-identical car at 2× the control-effort
weight. Verified at stage 2's `N=1000`/ε=0.01: ν=9.354, χ=6.138, ‖K‖₂max=1.926,
1.83% out of box.

The values this table used to list — cartpole 0.0771/12.8, quadrotor
0.2601/12.8, segway infeasible — were superseded by `8a64182` and no longer
match any config. They are kept only in the historical note that follows.

Loosening `w_lb` 0.01 → 1e-3 raised λ on the two plants that were envelope-limited
(cartpole 2.2×, quadrotor 1.5×) and left the car exactly where it was — the car's
certificate is set by something the envelope does not touch. Both plants that
moved also needed `r=12.8` to keep the generated controls inside the actuator
box, so the deployed gain `‖K‖₂ ≤ ‖B‖₂/(r·w_lb)` is softer than the raw λ
suggests. Verify against a rollout AUC before quoting these as improvements.

Applied to `skrl_cvstem_lqr_cfg.yaml`, `skrl_c2rl_ppo_cfg.yaml`,
`skrl_c2rl_sac_cfg.yaml` and `skrl_c3m_cfg.yaml` per env. `c3m` takes λ and the
envelope only — it is a CCM method with no Riccati term, so it has no `r` analog.

Cartpole moved a long way under the new envelope: λ 0.0343 → 0.0771 (2.2×) and
r 1.6 → 12.8 (8×). A looser `w_lb` lets ν rise, which admits a higher λ, but the
actuator check then forces `r` up to keep the generated controls inside the box.
The deployed gain is correspondingly softer (`‖K‖₂ ≤ ‖B‖₂/(r·w_lb)`), so the
rollout AUC is worth re-checking rather than trusting the certificate alone.

### segway does not fit this envelope

`find_uniform_lambda` returns INFEASIBLE at `w=[1e-3,1e3]` down to λ=0.01. Not a
solver failure and not an env bug — the LMI **alone** certifies segway to λ=1.41
there (§8b measured exactly that, since the sweep applies no actuator check).
The conflict is between the two:

* the 5% actuator check drives `r` up to 12.8,
* the LMI sees `ν` and `r` only as `ν/r`, so that needs ν ~12.8× larger,
* `ν ≤ 1/w_lb = 1000` forbids it.

So segway's plant wants gains this envelope does not permit — consistent with it
also being infeasible at the older `w=[0.01,100]`, and with ν pinning at exactly
1e4 when given `w_lb=1e-4`. Resolving it needs either a looser `w_lb` for segway
alone or a larger actuator-violation budget; it is not fixable by lowering λ.

## 9. Worked example (cartpole)

`N=100`, ε=0.1, `dt=1.0`, `r=1.6`, envelope `[0.01, 100]`, MOSEK, seed 0,
`tol=0.005`, active dims `(pitch, pitch_rate)`, rule = certificate/`box`.

| level | n | vol | pitch box | λ* | ν | χ |
|---|---|---|---|---|---|---|
| 0 | 100 | 1.000 | ±60.0° | 0.0441 | 100.0 | 412.5 |
| 1 | 97 | 0.982 | −57.8°, +60.0° | 0.0528 | 100.0 | 368.1 |
| 2 | 94 | 0.955 | −57.8°, +56.7° | 0.0597 | 100.0 | 341.5 |
| 3 | 92 | 0.939 | −56.0°, +56.7° | 0.0736 | 100.0 | 247.0 |
| 4 | 90 | 0.925 | −56.0°, +55.0° | 0.0736 | 100.0 | 244.1 |
| 5 | 88 | 0.892 | −52.1°, +55.0° | 0.0983 | 100.0 | 233.8 |

λ rose **2.23×** for **11%** of the active volume, monotone throughout.

With `--drop-mode box` and the same `p = 0.10` (several face cuts per level until
10% of samples are excluded), the retained set stays a box and the result is
reportable as a region:

| L | n | vol | λ* | gain | pitch | pitch_rate |
|---|---|---|---|---|---|---|
| 0 | 100 | 1.000 | 0.0441 | 1.00× | ±60.0° | ±1.000 |
| 1 | 90 | 0.931 | 0.0597 | 1.35× | −57°, +55° | ±1.000 |
| 2 | 80 | 0.840 | 0.1276 | 2.89× | −49°, +52° | ±1.000 |
| 3 | 72 | 0.783 | 0.1487 | 3.37× | −45°, +49° | ±1.000 |
| 4 | 62 | 0.691 | 0.2553 | 5.79× | −45°, +39° | −1.000, +0.977 |
| 5 | 56 | 0.611 | **0.3167** | **7.18×** | **−37°, +39°** | **−0.967, +0.977** |

i.e. `X = {|pitch| ≤ 60°, |pitch_rate| ≤ 1}` certifies λ = 0.0441, while
`X₅ = {pitch ∈ [−37°, +39°], pitch_rate ∈ [−0.967, +0.977]}` certifies
λ = 0.3167. `pitch_rate` is shaved by 3%; essentially the entire gain comes from
narrowing `pitch`, which is what `B(x) ∝ cos θ` predicts independently.

Same settings, rule = certificate/`samples` with `p = 0.10` (drop the tightest
10% each level):

| level | n | vol | λ* | χ | λ·χ | gain |
|---|---|---|---|---|---|---|
| 0 | 100 | 1.000 | 0.0441 | 412.5 | 18.19 | 1.00× |
| 1 | 90 | 0.907 | 0.0528 | 368.1 | 19.43 | 1.20× |
| 2 | 81 | 0.808 | 0.1418 | 135.4 | 19.20 | 3.22× |
| 3 | 73 | 0.767 | 0.1455 | 133.7 | 19.45 | 3.30× |
| 4 | 66 | 0.653 | 0.2553 | 72.8 | 18.58 | 5.79× |
| 5 | 59 | 0.600 | 0.3102 | 50.5 | 15.65 | **7.04×** |

**λ·χ is constant to ±4% over five levels while λ rises sevenfold.** With ν
pinned at its cap, χ is the only free scalar in the program, and the certified
rate moves inversely with the conditioning the metric must tolerate:

```
λ*(S) · χ(S) ≈ const                                                  (†)
```

(†) is the mechanism behind every positive result here. A subset raises λ
exactly insofar as it lets `W` become rounder. It is also why the envelope-
limited plants are flat: when a global cap fixes the solution, χ does not move
either. Reported as an observation on one plant, not a theorem — but it is
directly checkable on any run, since χ is already in the solver output.

**(†) is not general — it is a cartpole observation and segway refutes it.**
Three measurements, weakest constraint last:

| sweep | λ·χ across levels | spread |
|---|---|---|
| cartpole, shipped `w_lb=0.01` (ν pinned) | 18.19 … 19.45 | ±4% |
| cartpole, loose `w_lb=1e-4` (ν has slack) | 2292 … 3870 | ±25% |
| **segway, `w_lb=1e-3` (ν pinned)** | **22 981 … 101 249** | **4.4×** |

On segway χ *rose* 23 886 → 71 627 while λ rose 0.9621 → 1.4137 — the opposite
sign to (†), under the same ν-pinned condition that was supposed to make it hold.
So "λ rises because χ falls" is not the mechanism in general; it is what happened
on cartpole. The defensible statement is only that λ* rises when the subset drops
states the certificate was resting on, and that how the metric changes to permit
it is plant-dependent. Relaxing the envelope also breaks (†) across plants:
quadrotor gives λ·χ = 6.6, 57.4, 597.5 at `w_lb` = 0.01, 0.001, 0.0001.

Report λ·χ if it is informative on your plant; do not present it as a law.

## The envelope is ours, not Tsukamoto's — and it cannot simply be dropped

The reference `cvstem0` declares `nu` and `chi` as `cp.Variable(nonneg=True)`
and never bounds them above; its only constraints are `I ⪯ W̄ ⪯ χI` and the
contraction LMI. `ν ≤ 1/w_lb` and `χ ≤ ν·w_ub` are a project addition, added
because `‖M‖₂ ≤ ν` makes ν the direct cap on the deployed gain
`‖K‖₂ ≤ ‖B‖₂/(r·w_lb)`. `cvstem_joint` with `w_lb=w_ub=None` is his program
exactly, and that is its default.

Removing the envelope to recover the reference program does **not** yield a
faithful-but-valid comparison on these plants. It yields no measurement:

| plant | λ* range over the sweep | ν | monotonicity violations |
|---|---|---|---|
| cartpole | 4.60 – 8.38 | 1.4e7 – 1.2e8 | 2 |
| car | 271 – 357 | 1.5e8 – 2.9e8 | 2 |
| quadrotor | 18.8 – 23.4 | 5.4e8 – **1.6e9** | 2 |
| segway | 4.80, then 0.0133 | 9.9e5 – 1.9e7 | 1 (360× drop) |

Every plant produced a λ that *decreased* on a nested subset — impossible for
the true quantity by (M), hence proof of solver failure rather than evidence
about the plant. Quadrotor's final "0.89×" and car's "0.95×" are artifacts.

This is not a defect in the reference method. Tsukamoto penalizes ν through the
objective `J = d1·b̄·χ/α + d2·ν` and reads `Jcv` as the steady-state error bound,
so a runaway ν surfaces as a bad objective value to be rejected on inspection. A
bisection that asks only "feasible at this λ?" has no such signal: an unbounded ν
is free, and the solve drifts into a regime where the interior-point method stops
being reliable.

**Recommendation: `w_lb=1e-3, w_ub=1e3`**, described as a *numerical
regularizer* rather than a design choice. Measured feasibility at L0, ε=0.1:

| envelope | cartpole | car | quadrotor | segway | ν | reliable? |
|---|---|---|---|---|---|---|
| `[0.01, 100]` | 0.0441 | 2.1183 | 0.2407 | **infeasible** | 10² | yes |
| **`[1e-3, 1e3]`** | **0.7348** | **5.0148** | **0.8187** | **0.9637** | 10³ | yes |
| `[1e-4, 1e4]` | 1.3649 | 11.24 | 1.6701 | 2.3414 | 10⁴ | yes |
| none | 4.60 | 287 | 21.2 | 6.78 | 10⁷–10⁹ | **no** |

`1e-3/1e3` is the tightest setting that is feasible on all four: it bounds the
deployed gain 10× harder than `1e-4` at ~2× lower λ, keeps ν three orders below
where the solver degrades, and — since a tighter envelope makes subsets worth
more — shows a larger subset effect than `1e-4` would.

Feasibility is **not guaranteed** by any envelope: it is a property of the pair
(λ, envelope) for a given plant, ε and N. What does hold is monotonicity — a
looser envelope only weakens constraints, so anything feasible at `[0.01,100]`
remains feasible at `[1e-3,1e3]` and at `[1e-4,1e4]`. Feasibility transfers
toward looser settings, never toward tighter.

State that the unconstrained program is the reference one but admits no reliable
solve at these scales, and cite the monotonicity violations as the evidence —
that is a far stronger argument than asserting ill-conditioning.

Empirically λ ∝ (1/w_lb)^≈0.3 across four plants and four decades of cap
(car: 2.12 → 5.01 → 11.24 → 53.23 at `w_lb` = 1e-2 … 1e-6), so the rate a paper
reports is meaningless without its envelope printed beside it.

## A tighter envelope makes subsets worth more

Same plant, same draw, same rule, nearly the same volume removed — only the
envelope differs:

| envelope | λ* full box | λ* at ~60% volume | gain | χ over the sweep |
|---|---|---|---|---|
| shipped `w_lb=0.01` | 0.0441 | 0.3167 | **7.18×** | 412 → 50 (8.3× fall) |
| loose `w_lb=1e-4` | 1.3649 | 2.0546 | **1.51×** | 1950 → 1741 (1.1×) |

This inverts the natural expectation that relaxing the envelope should *expose*
the subset effect. It suppresses it, and (†) says why: a tight envelope caps how
anisotropic `W` may be, so the metric cannot stretch to accommodate hard states
and deleting them is the only remaining lever. Loosen the cap and the metric
absorbs those states itself — χ barely falls, so λ barely rises.

The practical consequence is favourable: **the subset effect is largest exactly
where it is useful**, at a realistic gain bound, and a loose-envelope experiment
understates it. Diagnose at the deployed envelope, not a relaxed one.

Note the two rules differ by 3× (2.23× vs 7.04×) on the same draw. Report which
one produced the number: `box` is a region you can deploy on, `samples` is an
upper bound on what any subset of that size could certify.

The mechanism is in χ, not ν: ν is pinned at `1/w_lb = 100` at every level and
never moves, while χ falls 412.5 → 233.8. With the metric scale capped, the only
thing the program can improve is the conditioning it must tolerate; extreme-tilt
states force an ill-conditioned metric, and removing them lets `W` become rounder,
which certifies a faster rate. `pitch_rate` was never cut. Cuts alternated
`lo, hi, lo, hi, lo, hi` — consistent with `cos θ` degrading symmetrically, so
the worst state moves to the opposite end after each cut.

Note this is an envelope-active case (`ν` pinned throughout) that nonetheless
gains 2.23×. An active envelope does **not** imply shrinking is futile; the cap
and the LMI bind together.
