# C2RL

Contraction-based reinforcement learning research repo built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
All algorithms run through a unified **skrl** backend — Isaac Sim environments and lightweight
classic (pure-NumPy, no-Isaac) environments alike.

## Table of Contents

1. [Installation](#installation)
2. [Listing Environments](#listing-environments)
3. [Environments](#environments)
   - [Velocity-tracking (Isaac)](#velocity-tracking-isaac)
   - [Path-tracking (Isaac)](#path-tracking-isaac)
   - [Classic analytical environments](#classic-analytical-environments)
   - [Cartpole prototype (Isaac)](#cartpole-prototype-isaac)
4. [Action convention](#action-convention)
5. [Training](#training)
6. [Reference Trajectory Generation](#reference-trajectory-generation)
7. [Evaluation / Play](#evaluation--play)
8. [W&B Logging](#wb-logging)
9. [Algorithm Reference](#algorithm-reference)
10. [Config Files](#config-files)
11. [Project Structure](#project-structure)
12. [Testing](#testing)
13. [Reproducing the Paper Results](#reproducing-the-paper-results)
14. [Citing](#citing)
15. [License](#license)

---

## Installation

### 1. Install Isaac Lab

Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Conda install is recommended:

```bash
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
# … follow Isaac Lab conda steps …
```

Isaac Lab/Sim is only needed for the Isaac environments (humanoid/quadruped/manipulator
vel/path-tracking, the `Cartpole-v0` prototype). The `classic-*` environments are pure
NumPy/gymnasium and need none of this — see [Classic analytical environments](#classic-analytical-environments).

### 2. Clone this repo

```bash
git clone <repo-url> C2RL
cd C2RL
```

### 3. Install the C2RL extension

```bash
python -m pip install -e source/c2rl
```

*This also installs `skrl`, `wandb`, and `torch`.*

### 4. Verify dependencies

```bash
python -c "import torch, scipy, gymnasium, skrl; print('OK')"
```

### 5. (Optional) MOSEK — a higher-accuracy SDP solver for C2RL

C2RL's contraction-metric SDP (`cm_solver` in the yaml's `cm:` block — see
[ncm_synthesis.py](source/c2rl/c2rl/agents/skrl/ncm_synthesis.py))
defaults to **SCS**, an open-source solver that needs no license and is already
installed via the `cvxpy` dependency above. **MOSEK** is a commercial
interior-point solver — the one Tsukamoto's original Neural Contraction Metric
code uses — that gives materially tighter/more accurate SDP solutions if you
have a license. `pip install -e source/c2rl` already installs the
`mosek` Python package, but it won't actually solve anything until you add a
license file:

1. **Get a license** — free for academic email addresses:
   https://www.mosek.com/products/academic-licenses/ (or a trial/commercial
   license from the same site). MOSEK emails you a `mosek.lic` file.
2. **Place the license file**:
   ```bash
   mkdir -p ~/mosek
   mv /path/to/downloaded/mosek.lic ~/mosek/mosek.lic
   ```
   (or set `MOSEKLM_LICENSE_FILE=/path/to/mosek.lic` instead of using the
   default location).
3. **Verify cvxpy sees it**:
   ```bash
   python -c "import cvxpy as cp; print(cp.installed_solvers())"  # should list MOSEK
   ```
4. **Use it** — set `cm_solver: MOSEK` in the yaml's `cm:` block for whichever
   C2RL config you're running (e.g.
   [skrl_c2rl_ppo_cfg.yaml](source/c2rl/c2rl/tasks/direct/classic/car/agents/skrl_c2rl_ppo_cfg.yaml)).

Without a license, leave `cm_solver: SCS` (the shipped default) — it needs no
extra setup.

---

## Listing Environments

Scan task `__init__.py` files without loading Isaac Sim:

```bash
python scripts/list_envs.py

# Filter by keyword
python scripts/list_envs.py --keyword vel_tracking
python scripts/list_envs.py --keyword path_tracking
python scripts/list_envs.py --keyword classic
```

| Task ID | Type | Isaac Sim? |
|---|---|---|
| `Humanoid-VelTracking-v0` | velocity-tracking | yes |
| `Humanoid-PathTracking-v0` | path-tracking | yes |
| `Quadruped-VelTracking-v0` | velocity-tracking | yes |
| `Quadruped-PathTracking-v0` | path-tracking | yes |
| `Manipulator-VelTracking-v0` | velocity-tracking | yes |
| `Manipulator-PathTracking-v0` | path-tracking | yes |
| `Cartpole-v0` | prototype (obs=4, act=1) | yes |
| `classic-car-v0` | path-tracking, analytical | no |
| `classic-cartpole-v0` | path-tracking, analytical | no |
| `classic-segway-v0` | path-tracking, analytical | no |
| `classic-turtlebot-v0` | path-tracking, analytical | no |

---

## Environments

Every path-tracking environment (Isaac and classic) shares the same contraction-compatible
observation layout `obs = [x, x_ref, u_ref]` and quadratic tracking reward `r = -‖x - x_ref‖²`,
so all seven algorithms (PPO, SAC, C3M, LQR, SD-LQR, C2RL-PPO, C2RL-SAC) can train/evaluate in any of them.

### Velocity-tracking (Isaac)

**Tasks:** `Quadruped-VelTracking-v0`, `Humanoid-VelTracking-v0`, `Manipulator-VelTracking-v0`

**Purpose:** pre-train a locomotion/manipulation policy. Its rollouts are recorded and become
the reference trajectories `[x_ref, u_ref]` that the path-tracking envs track (see
[Reference Trajectory Generation](#reference-trajectory-generation)) — these envs are not
contraction-tracking envs themselves.

#### Commands sampled each episode

```
vx  ~ Uniform(vx_range)            [m/s] forward velocity, constant per episode
vy  ~ Uniform(vy_range)            [m/s] lateral velocity, constant per episode
yaw_rate(t) = A·sin(ω·t + φ)             sinusoidal yaw, makes the robot curve
   A ~ Uniform(yaw_A_range)  [rad/s]
   ω ~ Uniform(yaw_omega_range) — up to one full cycle per episode (2π / episode_length_s)
   φ ~ Uniform(0, 2π)
```
(manipulator tracks an end-effector Cartesian velocity command `[vx, vy, vz, yaw_rate]` instead.)

#### Observation, action, reward, termination

| | Quadruped | Humanoid | Manipulator |
|---|---|---|---|
| **Physical state** | `lin_vel_b(3)+ang_vel_b(3)+proj_gravity_b(3)+joint_pos_rel(12)+joint_vel(12)` = **33** | `lin_vel_b(3)+ang_vel_b(3)+proj_gravity_b(3)+joint_pos_rel(19)+joint_vel(19)` = **47** | `joint_pos(7)+joint_vel(7)+ee_pos_local(3)+ee_lin_vel(3)+ee_yaw_vel(1)` = **21** |
| **Obs** = state + cmds(4) + prev_action | **49** | **70** | **32** |
| **Action** | 12 joint targets: `default_pos + 0.25·action` | 19 joint targets: `default_pos + 0.25·action` | 7 joint targets: `mid ± half_range·action` (soft-limit midpoint form) |
| **Reward terms** | `rew_alive(0.5) + rew_lin(exp(-lin_err/0.25)·2.0) + rew_yaw(exp(-yaw_err/0.25)·0.5) + rew_flat(-0.5·Σg_xy²) + rew_z(-0.5·vz²) + rew_rp(-0.05·Σω_xy²) + rew_torque(-1e-5·Στ²) + rew_action_rate(-0.01·Σ(a-a')²)` | `rew_alive(1.0) + rew_lin(exp(-lin_err/0.1)·2.0) + rew_yaw(exp(-yaw_err/0.1)·0.5) + rew_flat(-1.0·Σg_xy²) + rew_z(-0.5·vz²) + rew_rp(-0.05·Σω_xy²) + rew_torque(-1e-5·Στ²) + rew_action_rate(-0.01·Σ(a-a')²)` | `rew_vel(-1.0·‖ee_vel-cmd‖²) + rew_yaw(-0.5·(ee_yaw_vel-cmd)²) + rew_action_rate(-0.01·Σ(a-a')²) + rew_joint_limits(-0.1·soft-limit penalty)` — no exp-shaping, no alive bonus |
| **Termination** | fall: `base_height < 0.20 m` or tilt `proj_gravity_b[z] > -0.71` | fall: `base_height < 0.50 m` or tilt `proj_gravity_b[z] > -0.71` | time-out only, no failure condition |
| **Decimation / control freq** | 4 @ sim dt=1/200s → **50 Hz** (step dt=0.02s) | 4 @ 1/200s → **50 Hz** | 2 @ sim dt=1/120s → **60 Hz** (step dt≈0.0167s) |
| **Episode length** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=15.0` → **900 steps** |

Episode lengths were sized so each episode covers roughly 4 contraction time-constants
(`T ≥ 4/λ`) at the target rate — `λ≈0.1` for the high-DoF locomotion envs (quadruped/humanoid,
hence 40 s) and `λ≈0.3` elsewhere (manipulator/classic, hence 15 s); see
[W&B Logging](#wb-logging) for how the achieved rate is actually measured.

**Reference-trajectory quality gate** (see [Reference Trajectory Generation](#reference-trajectory-generation)):
half of the theoretical best-case total episode reward, i.e. `0.5 · (rew_alive+rew_lin_vel+rew_yaw_rate) · T`
— **3000** for quadruped, **3500** for humanoid, **0** for manipulator (its per-step reward has no
positive term, so its best case is exactly 0 and the gate is a no-op unless you pass `--min_ref_quality`).

---

### Path-tracking (Isaac)

**Tasks:** `Quadruped-PathTracking-v0`, `Humanoid-PathTracking-v0`, `Manipulator-PathTracking-v0`

**Purpose:** follow a pre-recorded reference trajectory `[x_ref, u_ref]` step-by-step — this is
where the contraction algorithms (C3M/LQR/SD-LQR/C2RL) are actually trained/evaluated, alongside
PPO/SAC baselines.

```
Episode reset:
  1. Sample a reference trajectory xref[0..T], uref[0..T] from the .npz buffer
  2. Initialise the robot to xref[0] (+ small noise so e(0) != 0)

Each step t:
  obs    = [x_current, x_ref[t], u_ref[t]]
  error  = wrap_diff(x_current - x_ref[t], angle_idx)   ← shortest-angle on yaw, raw elsewhere
  reward = -‖error‖²
  u_applied = action              ← the env applies the policy's action DIRECTLY
```

See [Action convention](#action-convention) — the env never adds `u_ref` back in; algorithms
that need to track `u_ref` (CLActor for C3M/C2RL, LQR/SD-LQR) fold it into their own output.

Quadruped/humanoid use **"Option A"**: the state is the world SE(2) pose `(xy, yaw)` — the raw
angle, cyclic w.r.t. the dynamics, so it only appears in the state to be *tracked*, never in
f(x)/B(x) — plus body-frame twist and joint state. `yaw` is embedded as `(cos, sin)` at every
network's input (`agents/skrl/angle_utils.py`); the env/error/dynamics stay in raw radians.

| | Quadruped | Humanoid | Manipulator |
|---|---|---|---|
| **state_dim** (exported physical state) | `xy_rel(2)+yaw(1)+proj_gravity_b(3)+joint_pos_rel(12)+base_lin_vel_b(3)+base_ang_vel_b(3)+joint_vel(12)` = **36** | same layout, 19 joints = **50** | `joint_pos(7)+joint_vel(7)+ee_pos_local(3)+ee_lin_vel(3)+ee_yaw_vel(1)` = **21** |
| **angle_idx** (raw angle → (cos,sin) embedded at every network's input) | `[2]` (yaw) | `[2]` (yaw) | `[]` (none) |
| **Obs** = x + x_ref + u_ref | 36+36+12 = **84** | 50+50+19 = **119** | 21+21+7 = **49** |
| **Action** | 12: `default_pos + 0.25·action` | 19: `default_pos + 0.25·action` | 7: soft-limit midpoint form |
| **Termination** | non-terminating by default (`terminate_on_fall: False`) — episodes always run full length; the fall check (`base_height < 0.20/0.50 m`) still resets on divergence (NaN/exploded state) regardless | non-terminating by default, same divergence-reset | time-out only |
| **Decimation / control freq** | 4 @ 1/200s → **50 Hz** | 4 @ 1/200s → **50 Hz** | 2 @ 1/120s → **60 Hz** |
| **Episode length** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=40.0` → **2000 steps** | `episode_length_s=15.0` → **900 steps** |
| **Reference file** | `data/quadruped/dynamics_data.npz` | `data/humanoid/dynamics_data.npz` | `data/manipulator/dynamics_data.npz` |

#### Supported algorithms

All seven algorithms work in every path-tracking env:

| Algorithm | Entry point key |
|-----------|----------------|
| PPO | `skrl_cfg_entry_point` |
| SAC | `skrl_sac_cfg_entry_point` |
| C3M | `skrl_c3m_cfg_entry_point` |
| LQR | `skrl_lqr_cfg_entry_point` |
| SD-LQR | `skrl_sdlqr_cfg_entry_point` |
| C2RL-PPO | `skrl_c2rl_ppo_cfg_entry_point` |
| C2RL-SAC | `skrl_c2rl_sac_cfg_entry_point` |

---

### Classic analytical environments

**Tasks:** `classic-car-v0`, `classic-cartpole-v0`, `classic-segway-v0`, `classic-turtlebot-v0`

Pure-Python/NumPy gymnasium envs, no Isaac Sim dependency (`tasks/direct/classic/`). All share
`common/env_base.py`'s `BaseEnv` and the same `obs = [x, x_ref, u_ref]` / `r = -0.5·‖x-x_ref‖²`
contraction-compatible interface. Unlike the Isaac envs, the reference trajectory is **not** a
recorded file — each env analytically synthesizes a fresh `xref/uref` every `reset()` by sampling
a random Fourier-sum reference control and rolling it forward through the same dynamics used at
train time (`system_reset()` in each env.py).

Dynamics are control-affine, `ẋ = f(x) + B(x)u`:

| Env | State `x` | Control `u` | dt | `time_bound` | steps | Dynamics |
|---|---|---|---|---|---|---|
| car | `[p_x,p_y,θ,v]` (4) | `[ω,a]` (2) | 0.03s | 15.0s | ≤500 | Dubins-like car with velocity state |
| cartpole | `[p,θ,v,ω]` (4) | `[F]` (1) | 0.03s | 15.0s | ≤500 | standard nonlinear cartpole ODE |
| segway | `[x,θ,v,ω]` (4) | `[τ]` (1) | 0.03s | 15.0s | ≤500 | segway balance ODE (fixed numeric coefficients) |
| turtlebot | `[x,y,θ]` (3) | `[v,ω]` (2) | 0.05s | 30.0s | ≤600 | pure kinematic unicycle (`f(x)=0`) |

Reward config for all four: `q=1.0, r=0.0`, i.e. effective reward is `-0.5·‖x-x_ref‖²` with no
control-cost term active (`control_effort` is computed but its weight `r` is 0).

**Termination** is always `False` — episodes only truncate at their sampled length (`time_steps
== episode_len - 1`), and `episode_len` can be *shorter* than `time_bound/dt` if the analytically
generated reference exits its position bounds early during `system_reset`.

Episode lengths above (15s for car/cartpole/segway, 30s for turtlebot) target a contraction rate
λ≈0.3, same reasoning as the manipulator env — see the note under
[Velocity-tracking](#velocity-tracking-isaac).

Uses `--classic` flag (no Isaac Sim needed). All six algorithms are supported.

---

### Cartpole prototype (Isaac)

**Task:** `Cartpole-v0`

Stock IsaacLab cartpole tutorial environment — a minimal scaffold for rapid PPO/SAC
prototyping, **not** part of the path/velocity-tracking contraction pipeline (no `u_ref`,
no reference trajectory).

- **Obs (4)**: `[pole_pos, pole_vel, cart_pos, cart_vel]`
- **Action (1)**: direct joint-effort target, `action·100.0 N`
- **Reward**: `rew_alive(1.0·(1-terminated)) + rew_termination(-2.0·terminated) + rew_pole_pos(-1.0·pole_pos²) + rew_cart_vel(-0.01·|cart_vel|) + rew_pole_vel(-0.005·|pole_vel|)`
- **Termination**: `|cart_pos| > 3.0 m` or `|pole_pos| > π/2`
- **Decimation/control freq**: 2 @ sim dt=1/120s → **60 Hz**
- **Episode length**: `episode_length_s=5.0` → **300 steps**

Supports PPO and SAC only.

---

## Action convention

Every environment (Isaac and classic) applies the policy's action **directly** —
`u_applied = action`, no environment ever adds `u_ref` back in:

- Isaac locomotion envs: `default_pos + action_scale · action` (or the manipulator's soft-limit
  midpoint form) — a joint-space delta, unrelated to `u_ref`.
- Classic envs: `self.current_u = u.copy()` — applied as-is.

Instead, it's the **agent** that folds `u_ref` into its own output where relevant:

- `CLActor` (used by C3M/C2RL): `mu = uref + w2 · l1(x, xref)` — feedback added on top of `uref`,
  sliced out of the observation.
- `LQR` / `SD-LQR`: `u = uref - K·(x - xref)`.
- PPO/SAC on **all** path-tracking envs (Isaac and classic) default to `models.policy.backbone:
  control` in their yaml — this swaps the stock skrl Gaussian MLP for `CLActorModel`, so their
  policy mean is *also* `uref + feedback`, not a from-scratch `u`. Set `backbone: mlp` explicitly
  to opt out and have the policy learn the full control (this is the only option for
  vel-tracking envs, which have no `u_ref` to fold in).

**Bound note:** `uref` ranges over the same declared `action_space` as the feedback term (both
`[UREF_MIN, UREF_MAX]` on classic envs), so `uref + feedback` can reach up to 2x that range.
PPO/C3M backbones handle this via `clip_actions: True` (skrl's stock post-sample clamp — log_prob
is computed on the pre-clamp sample, same as any unbounded-Gaussian clip_actions setup). Squashed
SAC backbones (`control-squashed`/`mlp-squashed`) clamp `uref + rescale(tanh(u))` to the action
space directly inside `_TanhSquashMixin.act()` *after* computing log_prob/entropy — tanh-squashing
alone only bounds the feedback term, not the sum once a residual is added.

---

## Training

### Velocity-tracking (locomotion pre-training)

```bash
# PPO — quadruped (recommended starting config)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --num_envs 4096 --headless

# PPO — humanoid
python scripts/skrl/train.py \
    --task Humanoid-VelTracking-v0 --algorithm ppo \
    --num_envs 4096 --headless

# PPO — manipulator
python scripts/skrl/train.py \
    --task Manipulator-VelTracking-v0 --algorithm ppo \
    --num_envs 2048 --headless

# SAC (lower env count — replay buffer makes it sample-efficient)
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm sac \
    --num_envs 64 --headless

# Resume from checkpoint
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --headless
```

Reference trajectories for path-tracking are **auto-generated** at the end of a vel-tracking
run if the policy clears the quality gate — see
[Reference Trajectory Generation](#reference-trajectory-generation).

**HP overrides:**

```bash
# PPO
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo \
    --ppo_lr 3e-4 --ppo_rollouts 64 --ppo_learning_epochs 5 --ppo_mini_batches 4 \
    --headless

# SAC
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm sac \
    --sac_lr 1e-4 --sac_batch_size 512 --sac_gradient_steps 4 --sac_memory_size 100000 \
    --headless
```

### Path-tracking (contraction control)

Requires a reference file at `data/{robot}/dynamics_data.npz` (auto-generated after
vel-tracking training, or manually via `scripts/generate_ref_traj.py`).

```bash
# C3M — quadruped (trains NeuralDynamics online from the reference buffer)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m --headless

# SD-LQR / LQR — quadruped (analytical, no gradient training)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm sdlqr --headless
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm lqr --headless

# C2RL — quadruped (two-policy online RL, on top of PPO or SAC)
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c2rl-ppo --headless
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c2rl-sac --headless

# PPO baseline
python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm ppo --headless

# Same for humanoid / manipulator
python scripts/skrl/train.py --task Humanoid-PathTracking-v0 --algorithm c3m --headless
python scripts/skrl/train.py --task Manipulator-PathTracking-v0 --algorithm c2rl-ppo --headless
```

### Classic environments

No Isaac Sim needed. Pass `--classic` flag. `--num_envs` here is a *process-level* parallel-env
count (plain Python instances via `gymnasium.vector.SyncVectorEnv`), not GPU-batched.

```bash
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c3m --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm lqr --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm sdlqr --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-sac --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm ppo --num_envs 4
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm sac --num_envs 4

# Same for classic-cartpole-v0 / classic-segway-v0 / classic-turtlebot-v0
```

### Hyperparameter sweeps (W&B)

All sweeps go through one entry point, `commands/search.sh`. It prompts for the
algorithm, env, agent count and GPU, previews the exact space it will search, and
launches detached (`nohup`) so the sweep survives closing the terminal:

```bash
./commands/search.sh                                 # fully interactive
./commands/search.sh --algorithm c3m --env all --gpu 0 -y
./commands/search.sh --algorithm cvstem-lqr --env car --num-agents 3 --gpu 0 -y
```

Every sweep lands in the single W&B project **`c2rl-Search`**, named
`<env>-<algorithm>` (e.g. `classic-car-v0-cvstem-lqr`) — so relaunching the same
env+algorithm accumulates in one place instead of scattering across per-launch projects.

**The searched space is not in the script.** It lives in `search/configs/`, one yaml
per algorithm, each applying to *every* env — so `commands/search.sh` never needs editing to
change a range or add an algorithm (see `search/configs/README.md`):

| config | metric optimized |
|---|---|
| `ppo`, `sac`, `c2rl-{ppo,sac}-ccm` | `Reward / Total reward (mean)` |
| `c3m` | `Stability/contraction_score_mean` (`= contraction_rate / overshoot`) |
| `lqr`, `sdlqr`, `cvstem-lqr`, `c2rl-{ppo,sac}-cvstem` | `Stability/auc_mean` (minimized) |

C3M optimizes the contraction score rather than raw reward because maximizing reward
alone doesn't guarantee the certified contraction property C3M is meant to produce.
The LQR-family and CV-STEM configs optimize the normalized-error AUC, which measures
the certified contraction quantity directly rather than a training proxy.

**Infeasible SDPs are recorded, not skipped.** The CV-STEM configs sweep the knobs
that decide whether the metric SDP is solvable at all, and much of that space is
structurally infeasible. Those trials run through `search/sweep_runner.py`, which kills
the trial on the first infeasibility and writes a poison AUC (`1e3` for `cvstem-lqr`,
`1e4` for `c2rl-*-cvstem`). Without it an infeasible trial finishes with *no metric* —
and a metric-less run is silently ignored by bayes bookkeeping, so the same dead corner
gets resampled forever.

`cvstem-lqr` is the original CV-STEM pipeline (`AstroHiro/ncm`): One joint SDP over
uniform state-box samples with `nu`/`chi` shared, then a `chol(M)` network — solved once
at construction, no per-step solve and no per-state lambda backoff. An infeasible joint
program aborts immediately, before any rollout — there is no partial feasibility to
record. Note `agent.r_scaler` is inert for this agent: with `nu` free the solved
`nu` scales with `r` and the gain is unchanged, so only `cm.lbd`/`cm.cm_eps` move the
certificate (measured — see `scripts/find_uniform_lambda.py`'s docstring).

**PPO/SAC — Isaac locomotion pre-training** uses the same entry point, via
`search/configs/{ppo,sac}.yaml`. Those two optimize plain `Reward / Total reward (mean)`
rather than a `Stability/*` metric — reward is the right target here, since there's no
contraction certificate to check:

```bash
bash commands/search.sh --algorithm ppo --env Quadruped-VelTracking-v0
bash commands/search.sh --algorithm sac --env Quadruped-VelTracking-v0
```

---

## Reference Trajectory Generation

Isaac path-tracking envs need a `.npz` trajectory buffer at `data/{robot}/dynamics_data.npz`
(keys `x`, `u`, `x_dot`, `lengths`, consumed by `TrajectoryBuffer`). Classic envs need no such
file — see [Classic analytical environments](#classic-analytical-environments).

**Automatic** — `_generate_ref_trajs()` in `scripts/skrl/train.py` runs at the end of every
Isaac vel-tracking training run:
1. Loads `best_agent.pt` and checks the quality gate (mean episode reward ≥ half of the
   theoretical max — see the per-robot numbers under
   [Velocity-tracking](#velocity-tracking-isaac); 0 for manipulator).
2. Rolls out a candidate pool of `--ref_oversample_factor × --ref_num_trajs` episodes across
   **every** parallel env (not just the first `num_trajs` — this maximizes the pool the
   selection gets to pick from, since Isaac can't shrink the batch anyway), keeping only
   trajectories that survive `--min_ref_traj_length_frac` (default 50%) of the episode.
3. Keeps the **longest** `--ref_num_trajs` of that pool — early termination is exactly what a
   poor rollout looks like, so ranking by survival length favors complete, high-quality
   reference data.

```bash
# Tune the collection (defaults shown)
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo --headless \
    --ref_num_trajs 1000 --ref_oversample_factor 2.0 --min_ref_traj_length_frac 0.5

# Skip the quality gate entirely
python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo --headless \
    --min_ref_quality 0
```

**Manual** — if auto-generation was skipped, or you want to regenerate from an older checkpoint:

```bash
python scripts/generate_ref_traj.py \
    --task Quadruped-VelTracking-v0 \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --robot quadruped \
    --num_envs 128 \
    --num_trajs 2000 \
    --headless
```

Output defaults to `data/{robot}/dynamics_data.npz` (matching each path-tracking env's default
`traj_path`) — override `--out_dir` only if you've also overridden `traj_path` in the env config.

---

## Evaluation / Play

Isaac only (`scripts/skrl/play.py` has no `--classic` route):

```bash
# GUI — watch the trained policy in real-time
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint logs/skrl/quadruped_vel_tracking/<RUN>/checkpoints/best_agent.pt \
    --num_envs 4

# Velocity arrow overlay (blue=command, green=actual)
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint <PATH> --num_envs 4 --debug_vis

# Record one episode as MP4, headless
python scripts/skrl/play.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --checkpoint <PATH> --num_envs 1 \
    --video --video_length 600 --headless

# Path-tracking evaluation
python scripts/skrl/play.py \
    --task Quadruped-PathTracking-v0 --algorithm c3m \
    --checkpoint <PATH> --num_envs 4
```

---

## W&B Logging

```bash
wandb login   # once — or set WANDB_API_KEY env var

# W&B is on by default (project=C2RL). Pass --no_wandb to disable.
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --wandb_run_name quad-ppo-v1 \
    --video --video_length 200 --video_interval 2000 \
    --headless

# Different project
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm sac \
    --wandb_project my-project --headless

# Disable
python scripts/skrl/train.py \
    --task Quadruped-VelTracking-v0 --algorithm ppo \
    --no_wandb --headless
```

### Tabs

Metric keys are grouped into W&B "tabs" by whatever precedes the first `/` in the key. skrl's
own built-in tracker uses space-padded keys (`"Episode / Total timesteps (mean)"`), so custom
metrics use the same `"Tab / key"` spacing to land in the *same* section rather than creating a
visually-identical-but-distinct duplicate tab.

| Tab | Contents |
|---|---|
| `Reward` | skrl's own `Reward / Total reward (max/min/mean)`, plus (vel-tracking envs) `discounted_return`, `avg_reward_per_step`, `total_reward_ci95` (95% CI of total reward — `undiscounted_return` itself is intentionally **not** logged again, since it's the same quantity as skrl's own total reward). Classic C3M's periodic eval also reports `Reward / reward_mean` here. |
| `Stability` | *(path-tracking envs only)* per-episode contraction diagnostics + their 95% CIs: `auc` (Σ‖error‖ over the episode), `contraction_rate` (empirical λ, fit as `e(T)=e(0)·exp(-λT)`), `overshoot` (peak error / initial error — 1.0 = no overshoot), and **`contraction_score = contraction_rate / overshoot`** (higher is better: fast contraction with little to no overshoot). Classic C3M's periodic eval reports the same four keys here too. |
| `Episode` | skrl's own `Episode / Total timesteps (...)`, plus `contraction_flag` (fraction of steps where error strictly decreased) and `performance_score` (-mean error) for path-tracking envs, and `auc` (velocity-tracking error AUC) for vel-tracking envs. |
| `Loss` | C3M's `Loss / C3M/loss/*`, `Loss / C3M/dynamics/mse`, `Loss / Pretrain/dynamics_mse`, PPO/SAC's own loss terms. |
| `Eval` | Isaac path-tracking evaluator output (`reward_mean`, `auc`, ...) from `train.py`'s post-training evaluation. |

`Stability/*` and vel-tracking's `Reward/*` are populated from `env.extras["log"]`, forwarded by
skrl's trainer only when the value is a `torch.Tensor` (not a Python float) — keep that in mind
if you add new per-episode metrics to an env's `_reset_idx`.

For classic C3M runs specifically, `Stability/*` (and `Reward/reward_mean`) come from
`C3MSkrlTrainer.eval()`, called every `eval_interval` training steps (default 100 in each
`skrl_c3m_cfg.yaml`) — this is also what the `c3m` sweep in `search/configs/` optimizes
(`Stability / contraction_score`, `goal: maximize`).

---

## Algorithm Reference

### PPO / SAC

Standard skrl implementations (real `skrl.agents.torch.ppo.PPO` / `skrl.agents.torch.sac.SAC`,
built through `CLActorRunner`, see [Action convention](#action-convention) for the `backbone:`
choices). See [skrl docs](https://skrl.readthedocs.io) for the full training-loop/parameter
reference. Configs: `agents/skrl_ppo_cfg.yaml`, `agents/skrl_sac_cfg.yaml`.

**Normalization**: every shipped config now sets **`use_state_norm: false`** and (PPO only)
**`use_value_norm: true`** — observations are left raw, value targets are normalized. When
`use_state_norm` is enabled, `train.py` wires a `RunningStandardScaler` observation preprocessor
into the agent config before construction (`observation_preprocessor` for Isaac envs, remapped
from the legacy `state_preprocessor` yaml key for classic envs — see
[skrl's `Runner._process_cfg`](https://skrl.readthedocs.io)); it's applied (not updated) inside
`act()` and updated (`train=True`) inside `update()`'s gradient step, over the full observation
vector `[x, xref, uref]` (path-tracking envs) with no special-casing of any sub-range. Value
normalization (`use_value_norm`, PPO only — SAC has no state-value network) adds a separate
`RunningStandardScaler(size=1)` on the value output.

Why state norm is off by default: on the residual backbones (`control`/`mlp`, which compute
`u = uref + feedback` by slicing `uref` out of the observation), normalizing the observation also
normalizes the sliced `uref`, so the control law silently becomes `uref_norm + feedback` — the
reference-tracking residual is distorted. Keeping observations raw avoids this and (for C2RL) keeps
the contraction certificate consistent with the deployed controller (see the C2RL section).

---

### C3M (Control Contraction Metric)

Jointly trains a state-dependent Riemannian metric `M(x) = W(x)⁻¹` and a tracking controller
`δu = CLActor(x, x_ref, u_ref)` such that all trajectories contract toward each other at rate `λ`.

Three matrix-valued conditions must hold everywhere in state space:

- **Cu ≺ 0** — closed-loop: `Ṁ + 2·sym(M(A+BK)) + 2λM ≺ 0`
- **C1 ≺ 0** — drift: `Bₗᵀ(−Ẇ_f + 2·sym(Df/Dx·W) + 2λW)Bₗ ≺ 0`
- **C2 = 0** — compatibility: `Bₗᵀ(Ẇ_b − 2·sym(∂B/∂x·W))Bₗ = 0`

On Isaac envs, `NeuralDynamics` (`ẋ = f_net(x) + B_net(x)·u`) is trained online from trajectory data.
On classic envs with `use_analytical_dynamics: true`, the env's exact `get_f_and_B(x)` is used.

**Per-update workflow** (`C3MAgent.update()`, once per training `timestep`):

1. **Dynamics** — one MSE gradient step on `NeuralDynamics` from a fresh `(x, u, x_dot)` batch
   (skipped entirely when `use_analytical_dynamics: true`).
2. Anneal the controller's exploration `log_std` by training progress.
3. **One full pass over the training buffer** (`(x, xref, uref)` triples from `get_rollout`), in
   `batch_size` chunks; each chunk does `cmg_updates_per_policy_update` CMG-only gradient steps
   (controller frozen) followed by one controller-only gradient step (CMG frozen) — both directions
   optimise the same `pd_loss + c1_loss + c2_loss (+ os_loss)`, alternating which side actually
   receives the gradient is what keeps the joint (metric, controller) optimization stable.

`cfg.lbd` is the *certified/target* contraction rate baked into the training loss; the
`Stability/contraction_rate` W&B metric is the *empirically measured* rate on rollouts — they
are related but not the same number, and `Stability/contraction_score` (`=contraction_rate/overshoot`)
is what the `c3m` hyperparameter sweep in `search/configs/` actually optimizes.

**Normalization**: none. The policy/CMG networks are bare `nn.Module`s, never wrapped in a
PPO/SAC base agent, so there is no observation/value preprocessor anywhere — every network call
sees the same raw physical-unit `(x, xref, uref)` at both training and eval time.

| Param | Default | Notes |
|-------|---------|-------|
| `lbd` | 0.01 | Target contraction rate λ |
| `w_ub` / `w_lb` | 10.0 / 0.1 | Metric bounds |
| `W_lr` / `u_lr` | 3e-4 | CMG / CLActor learning rates |
| `buffer_size` | 4096 | Training data buffer |
| `eval_interval` | 100 | Steps between `eval()` calls that populate `Stability/*` |

---

### SD-LQR (State-Dependent LQR)

Linearises `ẋ = f(x) + B(x)u` at the **current state** `x`, solves care, applies `u = uref − K(x)·e`.
No training — analytical per-step computation (`update()` is a no-op). Every `act()` call:
autodiff `get_f_and_B` (analytical for classic envs, or a `NeuralDynamics` checkpoint pretrained
by C3M for Isaac envs) at the linearization point to get `f, B` and their Jacobians, form
`A = ∂f/∂x + Σⱼ uref_j·∂Bⱼ/∂x`, solve the care per-environment (`scipy`, CPU-only — falls back to
zero feedback if `(A, B)` isn't stabilizable at that point rather than aborting the batch), then
apply the gain.

**Normalization**: none — there are no learned weights whose input distribution could drift, only
a per-step closed-form solve on the raw physical state.

| Param | Default | Notes |
|-------|---------|-------|
| `Q_scaler` | 1.0 | `Q = Q_scaler · I` |
| `R_scaler` | 0.01 | `R = R_scaler · I` — must be > 0 |

---

### LQR

Same as SD-LQR but linearises at the **reference state** `x_ref` instead of the current state.
Applies `u = uref − K(x_ref)·e`.

---

### C2RL (Contraction-Certified RL)

Two-policy architecture, built on top of a chosen base algorithm (`c2rl-ppo` uses two official
skrl `PPO` sub-agents, `c2rl-sac` uses two official skrl `SAC` sub-agents) that share one CMG:

- **con_policy** ("contracting", `gamma_con` ≈ 0) and **opt_policy** ("optimal", `gamma_opt`, e.g.
  0.99) optimise the **same** Mahalanobis tracking reward
  `−tracking_scaler·‖e‖²_M/std − control_scaler·‖u−uref‖²/std` (`M(x)` = the CMG's current metric)
  — they differ only in discount factor, not in reward. opt_policy is what's actually deployed at
  inference unless `con_only: true`.
- con_policy's mean control and its state-Jacobian `K = du/dx` are what the CMG (CCM generator,
  same architecture as C3M) is trained against — opt_policy plays no role in shaping the metric.
  The CMG loss is `pd_loss + c1_loss + c2_loss`: `pd_loss` certifies the closed-loop system under
  con_policy (`Cu ≺ 0`), while `c1_loss`/`c2_loss` are C3M's own `C1`/`C2` conditions on the CMG
  alone (no policy involvement), shaping `W(x)` to admit some contracting controller.

**Per-epoch workflow** (`C2RLSkrlTrainer.train()`, one iteration of the outer loop):

1. **Con rollout** — collect `rollouts` env steps with con_policy acting (its own replay/rollout
   buffer — PPO: reset every epoch; SAC: a persistent replay buffer, sized independently via
   `memory_size`).
2. **update_con** — recompute the Mahalanobis reward from the raw just-collected observations and
   overwrite the environment reward with it, then take one PPO/SAC gradient step for con_policy
   (PPO updates explicitly here; SAC recomputes the reward per-minibatch inside a patched
   `memory.sample()` and defers the actual gradient step to `con_agent.post_interaction()`, so
   `learning_starts` is still respected).
3. **Opt rollout** — same as step 1, with opt_policy acting (skipped if `con_only`).
4. **update_opt** — same as step 2, for opt_policy.
5. **update_cmg** — refresh the CMG training buffer (fresh `(x, xref, uref)` from `get_rollout`)
   and take `cmg_updates_per_iter` contraction pd-loss gradient steps against con_policy's mean
   control/Jacobian.

**Normalization**:

- **Observations** — `use_state_norm: false` in every shipped config (see the PPO/SAC
  normalization note above for why: with `control`/`mlp` residual backbones, normalizing the obs
  also normalizes the sliced `uref`, distorting the `u = uref + feedback` law). When enabled,
  con_agent and opt_agent each get their own independent `RunningStandardScaler` over
  `[x, xref, uref]` — con_agent's fit to the states con_policy visits, opt_agent's to opt_policy's,
  and the two can diverge over training. PPO additionally normalizes its value function per
  sub-agent when `use_value_norm: true`.
- **Reward** — a *separate* mechanism from the above: the Mahalanobis term (and the optional
  control-effort term) are each divided by the sqrt of their own EMA'd batch variance
  (`reward_norm_beta`), tracked on the outer C2RL agent (not per sub-agent), so their scale stays
  roughly stationary as the CMG's metric shape changes during training.
- **CMG metric `M(x)` + Mahalanobis reward always use raw observations** — both read straight from
  the rollout/`get_rollout` tensors, never through a preprocessor. This is deliberate: `M(x)` and
  `e = x − xref` are physical quantities, and per-dimension normalization would scale `x` and
  `xref` independently (different observation indices, independent running stats), distorting `e`.
- **The certified policy Jacobian stays consistent with the deployed policy** — the one part of the
  CMG loss that touches the policy network (`K = du/dx`) now routes its input through con_agent's
  *own* observation preprocessor before calling con_policy. With state norm off (the default) that's
  the identity, so `K` is the raw Jacobian as before; with state norm on, con_policy sees the same
  normalized obs it was trained/deployed on, and because the normalization stays in the autograd
  graph while `jacobian(u, x)` still differentiates w.r.t. raw `x`, `K` correctly absorbs the
  `1/std` factor — the certificate always matches the network's actual input distribution. (This
  resolves the earlier raw-vs-normalized caveat; C3M never had it, since its policy is never wrapped
  in an observation-normalizing base agent.)

| Param | Default | Notes |
|-------|---------|-------|
| `gamma_con` | 0.0 | Discount for the contracting policy |
| `gamma_opt` | 0.99 | Discount for the optimal policy (the one deployed, unless `con_only`) |
| `pd_loss_num_samples` | 128 | Random directions for the `c1_loss`/`c2_loss` PD-violation hinge loss (mirrors C3M) |
| `tracking_scaler` | 1.0 | Mahalanobis reward scale (Q, shared by both policies) |
| `control_scaler` | 0.0 | Control-effort penalty weight (R, shared by both policies; 0 = disabled) |
| `reward_norm_beta` | 0.99 | EMA momentum for the reward-variance normalisation |
| `cmg_updates_per_iter` | 3 | CMG gradient steps per epoch |

---

## Config Files

Each environment owns its own agent configs, stored alongside the environment:

```
tasks/direct/
  quadruped_vel_tracking/agents/      skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  quadruped_path_tracking/agents/     skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  humanoid_vel_tracking/agents/        skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  humanoid_path_tracking/agents/       skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  manipulator_vel_tracking/agents/     skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  manipulator_path_tracking/agents/    skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  cartpole/agents/                    skrl_ppo_cfg.yaml   skrl_sac_cfg.yaml
  classic/car/agents/                 skrl_ppo_cfg.yaml       skrl_sac_cfg.yaml
                                      skrl_c3m_cfg.yaml       skrl_lqr_cfg.yaml
                                      skrl_sdlqr_cfg.yaml     skrl_c2rl_ppo_cfg.yaml
                                      skrl_c2rl_sac_cfg.yaml
  classic/cartpole/agents/            (same 7 as car)
  classic/segway/agents/              (same 7 as car)
  classic/turtlebot/agents/           (same 7 as car)
```

---

## Project Structure

```
C2RL/
├── commands/                         # every shell entry point lives here
│   ├── search.sh                     # the sweep entry point (interactive; nohup-detached)
│   └── run_seeds.sh                  # multi-seed training + aggregated Stability CSV
├── search/                           # the searched SPACE and its logs — data, not commands
│   ├── build_sweep.py                # configs/<algo>.yaml + env -> a W&B sweep yaml
│   ├── sweep_runner.py               # Trial wrapper: records a poison metric on an
│   │                                 #   infeasible SDP instead of a metric-less run
│   ├── configs/                      # THE searched space — one yaml per algorithm,
│   │                                 #   applying to every env (see its README.md)
│   └── logs/                         # Sweep launch logs + the generated sweep.yaml
│                                     #   (gitignored; one dir per launch)
├── data/                             # GENERATED DATA, gitignored — inputs to later runs
│   ├── {robot}/dynamics_data.npz     #   reference trajectories path-tracking envs track
│   └── classic/{env}/cm_data*.npz    #   synthesized contraction-metric caches
├── logs/                             # RUN OUTPUT — checkpoints, tensorboard, eval json.
│                                     #   Disposable: deleting it never forces a re-synthesis.
├── tests/                            # Runs WITHOUT Isaac Sim (see "Testing")
│   ├── test_configs.py               #   every yaml key is actually applied
│   ├── test_contraction_metrics.py   #   the four Stability/* metrics' math
│   ├── test_env_contract.py          #   the getattr-discovered env interface
│   └── test_isaac_parity.py          #   classic <-> Isaac capability drift
├── scripts/
│   ├── list_envs.py                  # List all envs without Isaac Sim
│   ├── generate_ref_traj.py          # Manually generate reference trajectories
│   └── skrl/
│       ├── train.py                  # skrl training entry point (Isaac + --classic route)
│       ├── train_utils.py            #   shared W&B/patch/eval helpers for both routes
│       └── play.py                   # Multi-model Stability evaluator (both routes)
│
└── source/c2rl/c2rl/
    ├── agents/skrl/
    │   ├── math_utils.py             # Jacobians, PD losses (pure PyTorch)
    │   ├── nn_modules.py             # MLP, CCM_Generator, CLActor, NeuralDynamics
    │   ├── models.py                 # skrl model wrappers (CMGModel, CLActorModel)
    │   ├── c3m.py                    # C3MAgent + C3MSkrlTrainer (train/eval, Stability metrics)
    │   ├── sdlqr.py                  # SDLQRAgent + LQRAgent
    │   ├── c2rl.py                    # C2RLAgent + C2RLSkrlTrainer
    │   └── eval_metrics.py           # fit_exponential_envelope, mean_confidence_interval
    │
    └── tasks/direct/
        ├── classic/                  # car / cartpole / segway / turtlebot (analytical, no Isaac)
        │   └── common/env_base.py    # BaseEnv shared by all 4 classic envs
        ├── cartpole/                 # Cartpole-v0  (Isaac prototype, obs=4, act=1)
        ├── common/
        │   ├── vel_commands.py       # Sinusoidal yaw + constant vxy commands
        │   ├── path_tracking_base.py # [x, x_ref, u_ref] layout + Stability/Episode W&B logging
        │   ├── eval_metrics.py       # episode_metrics/batch_episode_metrics, mean_confidence_interval
        │   └── traj_buffer.py        # .npz trajectory loader
        ├── quadruped_vel_tracking/   # Quadruped-VelTracking-v0   (obs 49, act 12)
        ├── quadruped_path_tracking/  # Quadruped-PathTracking-v0  (obs 66, act 12)
        ├── humanoid_vel_tracking/    # Humanoid-VelTracking-v0    (obs 70, act 19)
        ├── humanoid_path_tracking/   # Humanoid-PathTracking-v0   (obs 101, act 19)
        ├── manipulator_vel_tracking/ # Manipulator-VelTracking-v0 (obs 32, act  7)
        └── manipulator_path_tracking/# Manipulator-PathTracking-v0 (obs 49, act  7)
```

---

## Testing

The test suite runs **without Isaac Sim** — the classic environments are pure
NumPy/torch, and the Isaac-side contracts are verified statically (AST + config
inspection) rather than by launching a simulator.

```bash
pip install pytest
python -m pytest tests -q          # 157 tests, ~8 s
```

What each module guards:

| File | Guards against |
|---|---|
| `tests/test_configs.py` | A yaml key that is **silently dropped** because it is not a field of the algorithm's `Cfg` dataclass — the run then trains a different algorithm than its config describes, without erroring. Also checks SAC↔squashed / PPO↔unbounded backbone pairing. |
| `tests/test_contraction_metrics.py` | The streaming `auc` / `contraction_rate` / `overshoot` / `contraction_score` math, pinned against closed-form cases (a known exponential must report back its own λ). |
| `tests/test_env_contract.py` | The duck-typed interface `ContractionRunner` discovers by `getattr`: control-affine `get_f_and_B` shapes, `Bᵀ·B_null ≈ 0`, `x_dot` (not `x_next`) from dynamics rollouts, the divergence guard under saturated actions, and that `set_ccm` actually switches the reward to the Mahalanobis form. |
| `tests/test_isaac_parity.py` | Capability drift between the two env families — a member wired up on classic but missing on Isaac, mismatched `set_ccm` signatures, divergent reward forms, or an analytic controller left with no dynamics source on Isaac. |

CI (`.github/workflows/ci.yml`) runs these on Python 3.10/3.11, plus one short
end-to-end training run per algorithm on `classic-car-v0` — because a config key
that stops being applied still imports and still passes unit tests.

## Reproducing the Paper Results

```bash
# 1. Single run: one algorithm, one environment
python scripts/skrl/train.py --classic --task classic-car-v0 --algorithm c2rl-ppo

# 2. Multi-seed: every algorithm x every classic env, aggregated with 95% CIs
./commands/run_seeds.sh             # see the script header for seed/env selection
python scripts/aggregate_seeds.py   # -> mean +/- CI table across seeds

# 3. Compare trained checkpoints head-to-head on the Stability metrics
python scripts/skrl/play.py --classic --task classic-car-v0
```

Isaac Sim environments follow the same commands without `--classic`, but need
reference trajectories first — see
[Reference Trajectory Generation](#reference-trajectory-generation). The
analytic controllers (`lqr`, `sdlqr`, `cvstem-lqr`) additionally need
`--dynamics_checkpoint <dynamics.pt>` there, since Isaac exposes no analytical
`get_f_and_B` and those algorithms train no dynamics model of their own.

All algorithms report the same four contraction-stability metrics under
`Stability/*` (normalized-error AUC, contraction rate λ, overshoot, contraction
score), computed by the same code path — so numbers are directly comparable
across algorithms and across both environment families. See
[Algorithm Reference](#algorithm-reference).

## Citing

```bibtex
@software{cho_c2rl,
  author  = {Cho, Minjae},
  title   = {{C2RL}: contraction-metric reinforcement learning on a unified skrl backend},
  year    = {2026},
  url     = {https://github.com/Mgineer117/C2RL},
  license = {Apache-2.0}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Portions derive from
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause), whose notices
are retained in the affected files.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), which
documents the invariants (silent-failure classes) this codebase is built around.
