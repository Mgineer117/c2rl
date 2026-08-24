# Contributing

## Setup

The classic (analytical) environments and the full test suite need **no Isaac Sim**:

```bash
pip install -e source/c2rl
pip install pytest ruff
python -m pytest tests -q
```

Isaac Sim is only required to run the `*-VelTracking-v0` / `*-PathTracking-v0`
environments. See [README.md](README.md#installation) for that setup.

## Before opening a PR

```bash
python -m pytest tests -q     # must be green without Isaac Sim
ruff check source scripts     # lint
pre-commit run --all-files    # formatting/codespell (see .pre-commit-config.yaml)
```

If you touched anything an algorithm reads, also smoke-run the matrix — a config
key that stops being applied does **not** raise:

```bash
for e in car cartpole turtlebot segway quadrotor; do
  for a in ppo sac c3m lqr sdlqr cvstem-lqr c2rl-ppo c2rl-sac; do
    python scripts/skrl/train.py --classic --task classic-$e-v0 --algorithm $a \
        --num_timesteps 300 --num_envs 16 --no_wandb --skip_final_eval || echo "FAIL $e/$a"
  done
done
```

## Invariants

Each of these is enforced by a test; breaking one produces a run that looks
healthy while training something other than what its config describes.

**Never let a config key be silently ignored.** `rl_glue.filter_cfg_fields` drops
any key that is not a declared field of the algorithm's `Cfg` dataclass. A run
with an ignored key trains a *different* algorithm than its config describes and
looks completely healthy. When you add a yaml knob, add the dataclass field in
the same commit; `tests/test_configs.py` enforces this.

**Never let a missing capability degrade quietly.** Prefer raising over falling
back. `C2RLSkrlTrainer._inject_ccm` raises when no env accepted `set_ccm`
precisely because the fallback — training on the plain baseline reward — is
invisible in every metric.

**Keep the two environment families interchangeable.** Anything a contraction
agent discovers by `getattr` (`get_f_and_B`, `get_rollout`, `set_ccm`, `x_dim`,
`u_dim`) must exist with the same signature on *both* `classic/common/env_base.py`
and `common/path_tracking_base.py`. `tests/test_isaac_parity.py` checks this
statically, so it runs without Isaac Sim.

**Do not modify vendored `skrl` code.** Behaviour changes go in
`agents/skrl/agent_patches.py` (post-construction patches) or in project-side
subclasses, so upgrading skrl stays a dependency bump.

**SDP infeasibility is a signal, not a nuisance.** If CV-STEM synthesis reports
0% feasible, do not lower `min_feasibility_rate` to make it pass. Read the
envelope first (`w_lb`, `cvstem_r_scaler`) — and note that CV-STEM imposes its
LMI at the *drift* Jacobian, so a driftless plant (e.g. the unicycle turtlebot,
`f ≡ 0`) is infeasible at every λ by construction and must use
`cmg_method: ccm` instead.

**Actions must not be clipped for contraction controllers.** `torch.clamp` has
exactly zero gradient at saturation, which collapses the feedback Jacobian and
silently reduces the certified condition to the open-loop drift. Use the
tanh-squashed backbones (`control-squashed` / `mlp-squashed`) when you need
bounded actions.

## Adding an environment

1. Subclass `classic/common/env_base.py` (analytical) or
   `common/path_tracking_base.py` (Isaac) and implement its abstract members.
2. Register it in the package `__init__.py` with one
   `skrl_<algorithm>_cfg_entry_point` per algorithm you support.
3. Add the env's short name to `CLASSIC_ENVS` in `tests/conftest.py` — the
   contract and config suites will then cover it automatically.
