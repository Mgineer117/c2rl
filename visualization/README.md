# visualization/ — standalone contraction-metric visualizations (classic envs)

Self-contained scripts for inspecting the error geometry a contraction metric
induces over the control space, and where policies sit in it. Nothing here is
imported by the training code, and nothing here modifies it — the scripts only
*read* the classic envs, the agents' network modules, and stored checkpoints.

## Scope: u_dim ≤ 2 only

| env | u_dim | output |
|---|---|---|
| `cartpole`, `segway` | 1 | static **t × u × error** surface (`.svg`) |
| `car`, `turtlebot` | 2 | **u₀ × u₁ × error** surface per timestep, animated over t (`.mp4`) |

`quadrotor` (u_dim = 4) is deliberately excluded. The restriction is the point:
for one or two inputs the **full control space is plotted directly**, so the
landscape is complete and nothing is projected away.

This replaced an earlier design that swept a scalar axis. Any reduction to 1-D
for m ≥ 2 is provably lossy in the way that matters for a plot: no *continuous*
injection ℝᵐ → ℝ¹ exists (a continuous bijection from a compact space to a
Hausdorff space is a homeomorphism, but removing a point disconnects an interval
and not a square), so bijections like digit-interleaving are necessarily
discontinuous and destroy the neighborhood structure a surface's shape depends
on. Cardinality is preserved; topology — the thing a plot needs — is not. For
m > 2 the principled reduction is a *linear* projection onto
`span(Bᵀ(2Me + w))`, the single direction the contraction rate depends on to
first order; for m ≤ 2 none is needed, so none is used.

## The trunk (`--trunk`, default `uref,cvstem_lqr,greedy`)

Two separate control streams build these figures, and the distinction is the
whole design:

* **the trunk** — the trajectory the landscape is sampled *along*. One state per
  timestep.
* **the branches** — at each trunk state, every control on the sweep grid is held
  for `--lookahead` steps and its resulting error becomes the surface height. The
  branches are throwaway what-ifs: **they never feed back into the trunk**. So
  frame *k+1* is not rooted at frame *k*'s argmin — the geometry is a fan of
  hypotheticals off a fixed spine, not a rollout.

`--trunk` takes a **comma-separated list** and writes **one file per trunk** —
the default `uref,cvstem_lqr,greedy` gives the three reference points: no
feedback, a real controller, and the best available on the grid. Metrics are
built **once** and reused across trunks, so the extra trunks are nearly free
(car: 126s for the first, 5.8s for all three once the CMG is cached).

Within each file the trunk is **always one trajectory shared by every panel**, so
the panels differ only by their conditioning metric. What varies is which region
of state space you end up looking at:

| `--trunk` | trajectory | metric-independent? | note |
|---|---|---|---|
| `cvstem_lqr` *(default set)* | CV-STEM-LQR | yes (fixed control law) | well-tracked region; one SDP solve per step |
| `lqr` | LQR, linearized at `xref` | yes (fixed control law) | well-tracked region |
| `sd_lqr` | state-dependent LQR, linearized at `x` | yes (fixed control law) | gain stays valid far from `xref` |
| `uref` *(default set)* | zero feedback, `u = u_ref` | **yes** — no policy, no metric | the only fully neutral choice; error grows unchecked into a region no working controller visits |
| `greedy` *(default set)* | best grid control under `--trunk-metric`, `--trunk-lookahead` ahead, committing one step (receding horizon) | **no** | the best any controller confined to the grid could do |

The three defaults are complementary rather than interchangeable: `uref` is the
only fully neutral one but its error diverges (car: 0.819 → **2.741**), so it
shows a region no working controller visits; `cvstem_lqr` shows the well-tracked
region but conditions the landscape on where CV-STEM-LQR went; `greedy` bounds
what any grid-confined controller could achieve. Rendering all three is why the
default is a list.

**`greedy` is the one that needs care.** It needs an `M` to know what "decrement"
means, so it is metric-*dependent*. A single designated `--trunk-metric`
(default: the first of `--metrics`) therefore drives one trunk for **all** panels
— if each panel followed its own metric they would sit on different
trajectories and their differences would no longer be attributable to the metric
alone.

Measured on car (euclidean `|e|` 0.819 → final, 3 s — the one number no metric
influences): `uref` **2.741** (diverges, as it must), `lqr` 0.128, `sd_lqr` 0.129,
`cvstem_lqr` 0.065, `greedy` **0.027**.

`greedy` depends on `--trunk-metric` (default: the first of `--metrics`, i.e.
`ccm`), so its quality tracks that metric's quality directly — it is worth
watching as a diagnostic. With a properly-fit C1/C2 `ccm` it reaches 0.027, the
best of the five and better than CV-STEM-LQR itself, which is what you expect
from "the best control available on the grid". When the metric is bad the greedy
trunk degrades with it: driven by a starved fit, or by `c3m.pt`'s
controller-co-adapted CMG, the same trunk only reached 0.227 at
`--trunk-lookahead 1`, saturating on the box edge (the one-step effect
[the lookahead section](#the-lookahead---lookahead-default-10) describes:
`‖u*‖ ~ ‖e‖/dt` sits far outside the actuator box). If `greedy` looks bang-bang,
suspect the metric before the trunk.

## Metric conditioning — the geometries (`--metrics`)

**Each metric is defined by the objective its CMG minimizes** — trained here from
the config, *not* loaded from whichever checkpoint happens to be on disk, so each
panel means exactly what its name says.

| kind | the CMG minimizes | batched |
|---|---|---|
| `ccm` | the **C1/C2 contraction losses** (`train_cmg_ccm` — no SDP, no regression); weights cached under `visualization/cache/` | yes |
| `cvstem_pretrained` | **MSE regression loss onto CV-STEM SDP solutions** (`build_cm_dataset` + `regress_cmg`; Tsukamoto NCM = `cvstem_lqr`'s `metric_source="pretrained"`); `{x, W*}` dataset cached | yes |
| `cvstem_online` | *(no CMG)* — the CV-STEM SDP re-solved at every visited state | no (one solve/timestep) |
| `random` | *(nothing — untrained)* `ccm`'s architecture, config and `w_lb`/`w_ub` with no training at all | yes |

So `ccm` vs `cvstem_pretrained` is a clean comparison of the two **synthesis
formulations** (C1/C2 gradient descent vs SDP + regression) on identical
networks; `cvstem_online` vs `cvstem_pretrained` **is** the regression error of
that fit; and both vs `random` is what their objective bought.

`--metric-ckpt` overrides `ccm` with a stored CMG. Use it knowingly:

| checkpoint | how its CMG was trained | a pure C1/C2 metric? |
|---|---|---|
| `c2rl_ppo.pt` | `cmg_method: ccm` → `train_cmg_ccm`, offline then frozen | yes (only if `cmg_method: ccm`) |
| `c3m.pt` | `pd_loss + c1_loss + c2_loss (+ os_loss)`, one **joint** step with the controller | **no** — co-adapted to that controller |

> ### ⚠ `--ccm-samples` sizes the C1/C2 fit, and starving it is silent
>
> The C1/C2 path has no SDP, so samples are cheap; it defaults to the config's own
> `cmg_memory_size` (car: 131072). **Do not shrink it to `--cmg-samples`' size** —
> that flag budgets the CV-STEM *SDP* dataset, where each sample costs a solve.
> The gradient-step budget is `samples/batch_size × epochs`, so at 2048 samples
> and `cmg_regress_epochs: 10` you get **20 steps** and the fit does nothing
> (`c1_loss` 5.13 → 4.22, and the "trained" metric lands *behind* `random`). At
> the config's budget it converges properly: `c1_loss` **1.125 → 0**, `c2_loss`
> 0.233 → 0.0036.

### The `random` baseline

Same architecture, same config, same `w_lb`/`w_ub` as `ccm` — the *only*
difference is that `train_cmg_ccm` never runs. Needs no checkpoint (`ccm` builds
from config too), so it works on every env. Drawn gray and dashed because it is a
baseline, not a fourth competing metric.

**A random CMG is not "no metric".** It is still a bounded SPD field, so its
landscape is not featureless: whatever structure it shows is structure the
architecture and bounds impose *for free*, before any learning. Only the excess
over it is creditable to the objective. Across `--random-seed 0/1/2` it gives a
tight spread, so it reflects the architecture rather than one lucky draw.

> ### ⚠ Do not read the error panel as a ranking across metrics
>
> Normalized error is `r = √(eᵀM(x)e / e₀ᵀM(x₀)e₀)` — **each metric normalizes by
> its own `M`**, so each panel's curve is measured with a different ruler. On car
> the C1/C2 `ccm` reads 9.41 where `random` reads 3.68 on the same `uref` trunk;
> that is not "`ccm` tracks worse", it is two rulers disagreeing about what the
> error even is. `c1_loss → 0` certifies that *some* control contracts under that
> metric — it claims nothing about the zero-feedback `uref` trajectory.
>
> Compare **shapes** (where the basin is, how it moves) across panels, and
> absolute `r` only *within* one panel or against `random` under the same metric.
> The one genuinely cross-metric number is the trunk's own euclidean `|e|`, which
> no metric influences.

> **Caveat:** the CV-STEM LMI is **structurally infeasible on segway and
> turtlebot** (0/N feasible). `cvstem_pretrained` is skipped there with a
> message, and `cvstem_online` degrades to the identity metric (counted and
> reported) — so its landscape is *not* a real contraction certificate. This is
> consistent with the known driftless-system LMI issue in project memory. Use
> `--metrics ccm,random` on those envs — both are CMG-based, need no SDP, and no
> longer need a checkpoint either. `car` works with all four.

## The lookahead (`--lookahead`, default 10)

Each candidate control is **held for H steps** before its error is measured.
This is not a tuning knob to skip past — at `H=1` every landscape is a monotone
wall with no interior optimum, because the error-minimizing control satisfies
`‖u*‖ ~ ‖e‖/(H·dt)` ≈ 33‖e‖ at dt=0.03, far outside the actuator box: the
control has no time to act. Raising H brings `‖u*‖` inside the box and the basin
appears — with no dependence on any controller. Measured on car: `H=1` gives a
16–35 % spread over the control box and *zero* interior optima; `H=12` gives
117–151 % and a real basin in 65–100 % of frames.

## Scripts

### 1. `error_geometry.py` — one geometry per metric

```bash
python visualization/error_geometry.py --env car   # all metrics x 3 trunks -> 3 files
python visualization/error_geometry.py --env car --metrics ccm,cvstem_online
python visualization/error_geometry.py --env segway           # 1-D → svg
```

1-D gives one `t × u` surface per metric; 2-D gives an mp4 with one
`u₀ × u₁` surface per metric per frame. Height and colour share one log scale
within each frame, so the metrics are directly comparable there. Each 2-D
frame also keeps the previous `--history` geometries as translucent shells, and
every panel is titled with its timestep and the shells' span — see
[The geometry history](#the-geometry-history---history-default-10) below.

### 2. `policy_overlay.py` — policies on that geometry

Overlays every policy on **one shared** landscape, which is what makes them
comparable. Two distinct questions get two parts of the figure:

* **landscape** — "at an identical state, which policy commands the better
  control?" Each policy's `u = π(x_trunk(k))` is drawn on the shared geometry.
* **bottom panel** — "which policy actually tracks better?" Each policy's own
  closed-loop rollout error, a property of the policy rather than the geometry
  and **unaffected by `--trunk`**.

`--trunk` moves the shared states without breaking the sharing, but pick it
knowingly **here especially**: the default `cvstem_lqr` keeps the trunk
well-tracked at the cost of asking every policy to act at states CV-STEM-LQR
chose — not neutral between policies, and mildly flattering to CV-STEM-LQR
itself. `--trunk uref` favours no policy but lets the error grow into states none
of them would visit. If you are ranking policies against CV-STEM-LQR, prefer
`--trunk uref` or read the bottom panel, which no trunk affects.

```bash
python visualization/policy_overlay.py --env car
python visualization/policy_overlay.py --env car --metric ccm --policies c3m,cvstem_lqr
python visualization/policy_overlay.py --env car --trunk uref   # policy-neutral states
```

## `policies/<env>/` — checkpoints + configs

| file | needed by | notes |
|---|---|---|
| `c3m.pt` | `c3m` policy, `ccm` metric | any skrl C3M checkpoint (`policy` + `cmg` entries) |
| `c3m.yaml` | actor/CMG dims | optional — falls back to the task's `skrl_c3m_cfg.yaml` |
| `c2rl_ppo.pt` | `c2rl_ppo` policy, `ccm` metric | skrl C2RL-PPO checkpoint (`policy` + `cmg`) |
| `c2rl_ppo.yaml` | actor/CMG dims | optional — falls back to `skrl_c2rl_ppo_cfg.yaml` |

`cvstem_lqr`, `lqr` and `sd_lqr` are analytical — **no model files needed**;
their gains are recomputed from the env dynamics and their task yaml
(`skrl_lqr_cfg.yaml` / `skrl_sdlqr_cfg.yaml` / `skrl_cvstem_lqr_cfg.yaml`).

Store the yaml **that the run actually used** next to the checkpoint whenever it
differs from the task default — the config determines the actor architecture
(backbone + hidden dims) used to rebuild the network.

## The geometry history (`--history`, default 10)

Each 2-D frame keeps the previous **N** geometries on screen as translucent
shells, alpha ramping up toward the present (0.14 → 0.5, current at 0.9), so the
geometry's motion is readable inside a single frame rather than only in
playback. `--history 1` gives a plain alpha-0.5 previous-frame ghost; `0`
disables it. Each panel's title carries the timestep and the shells' time span
(`step 64 · t = 1.92 s · shells t = 0.54→1.92s`).

**The normalized-error axis retunes every frame**, over exactly the shells
currently on screen, so their spread fills the plot and the per-timestep change
stays visible. All panels in a frame share one norm, so the metrics remain
comparable *within* a frame — but colour/height are **not** comparable across
frames, so read motion from the shells rather than from absolute colour. Pass
`--error-range LO HI` to pin the axis instead when cross-run comparability
matters more.

## Commands

```bash
# --- the geometries, one panel per metric (feature 1) ---
python visualization/error_geometry.py --env car                 # 2-D → mp4, all 3 metrics
python visualization/error_geometry.py --env segway              # 1-D → svg
python visualization/error_geometry.py --env car --metrics ccm,cvstem_online

# --- what did training actually buy? trained CCM vs its untrained init ---
python visualization/error_geometry.py --env car --metrics ccm,random
python visualization/error_geometry.py --env car --metrics random --random-seed 1

# --- the trunk: which trajectory the landscape is sampled along ---
# default renders THREE trunks (uref, cvstem_lqr, greedy) -> one file each.
python visualization/error_geometry.py --env car --trunk uref        # just one
python visualization/error_geometry.py --env car --trunk lqr,sd_lqr  # any subset

# --- policies on the shared geometry (feature 2) ---
python visualization/policy_overlay.py --env car
python visualization/policy_overlay.py --env car --metric ccm --policies c3m,cvstem_lqr
python visualization/policy_overlay.py --env car --trunk uref --policies c3m,lqr,sd_lqr

# --- fast iteration (short episode, coarse grid, few frames) ---
python visualization/error_geometry.py --env car --time-bound 3.0 \
    --num-chunks 25 --num-frames 16 --fps 6 --cmg-samples 4096 --solver SCS

# --- publication quality (slow: 41² controls × 150 frames × 3 metrics) ---
python visualization/error_geometry.py --env car --num-chunks 41 --num-frames 150 --fps 20
```

Run from the repo root under the `base` conda env (classic envs never import
Isaac). `--solver SCS` needs no licence; drop it to use the config's MOSEK.

## Useful flags

`--trunk` the trajectory the landscape is sampled along
(`uref`|`greedy`|`cvstem_lqr`|`lqr`|`sd_lqr`, see above) · `--trunk-metric` the
one metric driving a `greedy` trunk, shared by all panels · `--trunk-lookahead`
steps `greedy` looks ahead before committing one step (default 1) ·
`--num-chunks` control levels per dimension (2-D cost is the square of this) ·
`--lookahead` steps each control is held (see above) · `--history` earlier
geometries kept on screen per 2-D frame (default 10; 0 = none, 1 = just the
previous at alpha 0.5) · `--error-range LO HI` pin the surface axis instead of
retuning it per frame · `--random-seed` weight-init
seed for the `random` baseline (vary it to separate architecture from draw) ·
`--cmg-samples`
states solved for the `cvstem_pretrained` regression (cached) · `--num-frames` /
`--fps` video length and rate · `--u-range physical|uref` which control box to
sweep · `--seed` · `--time-bound` episode length in seconds
(**must divide evenly by the env's `dt`** — `env_base` builds its time vector
with `torch.arange`, so e.g. `2.0/0.03` yields 67 steps against a
`max_episode_len` of 66 and raises a shape error) · `--solver` cvxpy SDP solver.

Outputs (svg/mp4 + raw `.npz` arrays) land in `visualization/output/`; the
`cvstem_pretrained` SDP dataset is cached in `visualization/cache/`.
