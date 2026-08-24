# Why a CV-STEM program is feasible, and what a state subset can buy

Two structural facts govern every result in this repo. Both are read directly off
the LMI, and both are confirmed by measurement on the plants in `tasks/direct/classic`.

The program (`ncm_synthesis.cvstem_joint`, Tsukamoto's `cvstem0`) is, over states
`k = 1..N` sharing scalars `ν, χ`:

```
(W̄ₖ − I)/dt + A(xₖ)W̄ₖ + W̄ₖA(xₖ)ᵀ + 2λW̄ₖ − ν(2/r)·B(xₖ)B(xₖ)ᵀ  ⪯  −εI      (LMI)
I ⪯ W̄ₖ ⪯ χI,     ν ≤ 1/w_lb,     χ ≤ ν·w_ub
```

Deployed: `W = W̄/ν`, `M = W⁻¹`, `K = R⁻¹BᵀM` with `R = rI`.

Note what is per-state and what is shared. **`W̄ₖ` is free at every state; `ν` and
`χ` are one number for the whole program.** Everything below follows from that
asymmetry.

---

## 1. Controllability rank: an exact feasibility barrier

### 1.1 The uncontrollable block sees a control-free LMI

Fix a state and drop the index. Let `q = rank[B, AB, …, Aⁿ⁻¹B]`. If `q < n`, put
the pair in Kalman controllability form by an orthogonal `T` (columns 1..q span
the controllable subspace `R`):

```
TᵀAT = [ A₁₁  A₁₂ ]        TᵀB = [ B₁ ]
       [  0   A₂₂ ]               [  0 ]
```

The bottom block is exact: `R` is `A`-invariant and contains `range(B)`, so the
`(2,1)` block of `A` and the second block of `B` both vanish.

Congruence preserves the LMI, so apply `T` and partition `W̄ = [[W₁₁, W₁₂],[W₁₂ᵀ, W₂₂]]`.
The `(2,2)` block of `AW̄` is `[0  A₂₂]·[W₁₂; W₂₂] = A₂₂W₂₂`, and the `(2,2)` block
of `BBᵀ` is `0`. So the LMI **implies**, as a necessary condition:

```
(W₂₂ − I)/dt + A₂₂W₂₂ + W₂₂A₂₂ᵀ + 2λW₂₂  ⪯  −εI                        (LMI₂₂)
```

with `W₂₂ ⪰ I`. **The control term is gone.** `(LMI₂₂)` is a pure Lyapunov
inequality on the uncontrollable dynamics — `ν`, `r`, `B` and the envelope cannot
touch it.

### 1.2 The exact condition

`(LMI₂₂)` is a Lyapunov inequality for `A₂₂ + λI` (the `(W₂₂−I)/dt` term is `⪰ 0`
since `W₂₂ ⪰ I`, so it only tightens). Feasibility therefore **requires**

```
Re λᵢ(A₂₂) < −λ   for every uncontrollable mode i
```

i.e. the plant must be **λ-stabilizable**, not merely stabilizable. In PBH form:

```
rank[ A(x) − sI ,  B(x) ]  =  n     for every s with Re(s) ≥ −λ
```

Two consequences worth stating separately:

* Uncontrollable modes are not fatal *per se* — they are fatal unless they decay
  faster than the rate being certified. A plant with a well-damped uncontrollable
  mode certifies fine up to that mode's decay rate.
* **Marginally stable uncontrollable modes are fatal at every `λ > 0`.** If
  `Re λᵢ(A₂₂) = 0`, no positive rate certifies. And with `W₂₂ ⪰ I` and `ε > 0`,
  even `λ = 0` fails: for `A₂₂` nilpotent the left side of `(LMI₂₂)` is `⪰ 0`,
  which can never be `⪯ −εI`.

### 1.3 Measured

`scripts/list_envs.py` reports the pointwise rank; every classic plant except
PVTOL is pointwise controllable.

```
env              x_dim   ctrb rank
pvtol                6           4     UNCONTROLLABLE, 2 directions
auv                  3           3
ball_and_beam        4           4
two_link_arm         4           4
cartpole             4           4
segway               4           4
car                  4           4
```

PVTOL's uncontrollable block, extracted numerically:

```
A₂₂ = [[ 0.369075,  0.837318],
       [-0.162682, -0.369075]]        eig(A₂₂) = {0, 0}       ‖B₂‖ = 2e-16
```

`A₂₂` is nilpotent, so §1.2 predicts infeasibility at every `λ > 0`. Measured:
infeasible at every `λ` down to 0.01, at every `cm_dt ∈ {0.02, 1, 10, 100, 1e4}`
and every `w_lb ∈ {1e-3, 1e-5, 1e-8}`, and 0 of 40 states certify even
individually. The prediction is exact.

**Nilpotency alone is not the problem** — a double integrator (`A` nilpotent,
`B` rank 1 of 2, fully controllable) certifies immediately (`ν = 3.1, χ = 1.2` at
`λ = 0.01`). Rank deficiency is the problem.

### 1.4 Why PVTOL is physically controllable but fails here

PVTOL steers laterally by *tilting*: thrust enters `v̇ₓ` through `−sin φ`, so
lateral motion is produced by first moving `φ`. That is a genuinely nonlinear
mechanism — it lives in the Lie bracket `[f, g]`, not in `B`. CV-STEM freezes
`A(x)` and `B(x)` at each sampled state, and **at frozen attitude the tilting
mechanism does not exist**. The system is nonlinearly controllable (indeed
differentially flat) while its pointwise linearization is not.

This is a limitation of pointwise linearization, not of the plant, and it is the
same failure as the driftless turtlebot (`f ≡ 0 ⟹ A = 0`, uncontrollable block
`A₂₂ = 0`). Both are the `Re λ(A₂₂) = 0` case.

---

## 2. Singular values of `B`: why they, and not `A`, decide subset gains

### 2.1 The metric absorbs `A` but cannot touch `B`

Start from the Riccati/contraction condition on `M`:

```
Ṁ + AᵀM + MA − 2MBR⁻¹BᵀM + 2λM  ⪯  0
```

and apply the congruence `W = M⁻¹` (pre- and post-multiply by `W`, which is what
convexifies the problem):

```
Ẇ + AW + WAᵀ − 2BR⁻¹Bᵀ + 2λW  ⪯  0
```

Look at what happened to each term:

| term | after congruence | scales with the metric? |
|---|---|---|
| drift | `AW + WAᵀ` | **yes** — linear in `W` |
| control | `−2BR⁻¹Bᵀ` | **no** — `W` has cancelled out |

**This is the whole story.** In `W`-coordinates the control authority is a fixed
matrix. A better metric can reshape how the drift acts on the error; it cannot
manufacture control authority where `B` has none.

With the normalization `W = W̄/ν` (multiply by `ν`, impose `W̄ ⪰ I`) the control
term becomes `−2ν R⁻¹BBᵀ`, which is the `(LMI)` at the top. So `ν` is precisely
*"how much control authority the program is allowed relative to the metric
scale"*, and the envelope caps it at `ν ≤ 1/w_lb`.

### 2.2 The worst direction, and the per-state price

Test `(LMI)` against a unit direction `v`:

```
vᵀ[(W̄−I)/dt + AW̄ + W̄Aᵀ + 2λW̄]v  ≤  (2ν/r)‖Bᵀv‖² − ε
```

The available authority in direction `v` is `(2ν/r)‖Bᵀv‖²`, and

```
σ_min(B)² ≤ ‖Bᵀv‖² ≤ σ_max(B)²
```

So the binding direction at a state is `B`'s smallest right-singular vector, where
authority is `(2ν/r)σ_min(B)²`. Define the **price of a state**

```
ν_k(λ) = min { ν : ∃ W̄ ⪰ I satisfying (LMI) at state k }        ν_k ∝ 1/σ_min(B_k)²
```

`W̄ₖ` is free per state, so whatever `A(xₖ)` does is absorbed into the choice of
`W̄ₖ` — it changes *which* metric that state wants, not the shared budget it
consumes. `B(xₖ)` is different: its effect is a coefficient on the shared `ν`.

### 2.3 The subset lemma falls out

The joint program is feasible over a set `S` iff one shared `ν` pays for every
state in it:

```
ν_required(λ, S) = max_{k ∈ S} ν_k(λ)        feasible ⟺ ν_required(λ, S) ≤ 1/w_lb
λ*(S) = max { λ : max_{k ∈ S} ν_k(λ) ≤ 1/w_lb }
```

`ν_k` is increasing in `λ`, and a max over a smaller set cannot be larger, so for
`S' ⊆ S`:

```
λ*(S') ≥ λ*(S)                                              (monotonicity)
```

with **strict** improvement exactly when shrinking removes the argmax — the
worst-actuated state. Hence:

* `σ(B)` **constant over the box** ⟹ every `ν_k` is essentially equal ⟹ removing
  states does not lower the max ⟹ **no λ gain, at any subset**.
* `σ(B)` **varying** ⟹ the small-`σ_min` states dominate the max ⟹ excluding them
  lowers `ν_required` ⟹ **λ rises**.

`σ(A)` has no analogous role: `W̄ₖ` is free, so `A`-variation is gauged away.
Measured confirmation — `σ(A)` spread has no predictive power whatsoever:

```
env          sv(B) spread   sv(A) spread   subset λ gain
cartpole            2.213          5.979         2.52x
segway              1.139          1.837         5.63x
car                 1.000          1.998         none
quadrotor           1.000          7.077         none      <- largest sv(A), zero gain
turtlebot           1.000          0.000         none
pvtol               1.000          0.000         none
```

**`σ(B)` spread predicts the sign of the effect; it does not predict its size.**
Segway (1.139) out-gains cartpole (2.213) because the gain depends on how much of
the `ν`-budget the excluded states were consuming relative to the `1/w_lb` cap,
not on the spread alone.

### 2.4 The same mechanism sets the gain — and can make a plant unusable

The deployed gain inherits `ν` directly:

```
M = νW̄⁻¹,   ‖W̄⁻¹‖ ≤ 1,   so   ‖M‖ ≤ ν ≤ 1/w_lb
K = R⁻¹BᵀM     ⟹     ‖K‖ ≤ ‖B‖ / (r · w_lb)
```

So **the state that forces a large `ν` also inflates the gain at every other
state**, because `ν` is shared. The AUV is the clean example:

```
σ_min(B) = 0.0117  (fin authority ∝ V², nearly zero at V = 0.3 m/s)
⟹ ν = 250 needed at λ = 0.01
⟹ ‖K‖ mean 527, max 905,  against a ±2 control box and errors ~0.1–0.3
⟹ 99.99% of generated controls out of box
```

All 40 sampled states certify *individually* (pointwise `λ*` from 0.27 to 10.0),
so this is not a structural failure like PVTOL's — it is entirely the shared `ν`.

**And `r` cannot fix it.** The LMI sees `ν` and `r` only through `ν/r`. If the
program is tight enough that `χ` is pinned (measured: `χ = 2.87 → 2.94`, frozen),
the solved `ν` scales proportionally with `r`, so

```
K = (1/r)·Bᵀ·νW̄⁻¹  ∝  ν/r = const
```

Raising `r` buys nothing and eventually drives `ν` into the `1/w_lb` cap, where
the LMI dies. That is exactly the observed ladder: feasible at `r = 0.1` with
`ν = 568`, infeasible at `r = 0.2`.

**The irony:** AUV's `σ(B)` spread of 92.7 — the property that made it an
attractive benchmark for subset gains — is precisely what makes it infeasible
under a shared `ν`. The "worst state governs" mechanism of §2.3 surfaces here as
an unusable gain rather than a slow `λ`.

### 2.5 What would fix it

Both fixes break the shared-scalar coupling rather than tuning around it:

* **Diagonal `R`** instead of `R = rI`: lets each input channel be weighted
  separately, so a fin whose authority collapses at low speed does not force the
  thruster channel's gain up with it.
* **State-dependent envelope** `w_lb(x)`: caps `‖M(x)‖` per state instead of
  globally, so the weak corner stops setting everyone's gain.

Neither is implemented today; `cvstem_joint` takes scalar `r` and scalar `w_lb`.

---

## 3. Summary

| question | answer | binding object |
|---|---|---|
| when is CV-STEM feasible at rate `λ`? | iff `Re λᵢ(A₂₂) < −λ` for every uncontrollable mode | `rank[B, AB, …]` |
| when can a state subset raise `λ`? | iff `σ(B)` varies over the box | `σ_min(B(x))` |
| why not `σ(A)`? | `W̄ₖ` is free per state and absorbs `A`; after the `W = M⁻¹` congruence the control term is metric-independent | the congruence |
| why does one bad state hurt everywhere? | `ν` is shared and capped at `1/w_lb`; `‖K‖ ≤ ‖B‖/(r·w_lb)` | shared `ν` |
