# `search/configs/` — one search space per algorithm

Each `<algorithm>.yaml` here declares the hyperparameters and ranges swept for
that algorithm, and applies to **every** env. `search.sh` prompts for the
algorithm + env, and `build_sweep.py` merges the chosen config with the env to
emit a W&B sweep yaml.

## File schema

```yaml
label:        c2rl (rl=ppo, cm=cvstem)   # shown in search.sh's preview
algorithm:    c2rl-ppo                   # train.py --algorithm value
num_envs:     1024                       # parallel envs per trial

metric:            # verbatim W&B sweep `metric:` block
  name: "..."
  goal: maximize | minimize

parameters:        # verbatim W&B sweep `parameters:` block
  agent.discount_factor:
    values: [...]

# ── optional ──────────────────────────────────────────────────────────────
runner:
  wrapper: true    # run trials through sweep_runner.py, which detects an
                   # infeasible SDP, kills the trial, and records bad_value on
                   # the metric instead of leaving a metric-less (ignored) run
  bad_value: 1e4   # sign follows metric.goal: large for minimize, very
                   # negative for maximize
  one_episode: true  # cap the trial at exactly one episode (episode length is
                     # read from the env's ENV_CONFIG: time_bound / dt)
```

## Which algorithms exist

| config | train.py `--algorithm` | metric | notes |
|---|---|---|---|
| `ppo.yaml` | `ppo` | reward | |
| `sac.yaml` | `sac` | reward | |
| `c3m.yaml` | `c3m` | contraction score | |
| `lqr.yaml` | `lqr` | AUC | analytical |
| `sdlqr.yaml` | `sdlqr` | AUC | analytical |
| `cvstem-lqr.yaml` | `cvstem-lqr` | AUC | analytical, online SDP, 1-episode eval |
| `c2rl-<ppo\|sac>-<ccm\|cvstem>.yaml` | `c2rl-ppo` / `c2rl-sac` | reward (ccm) / AUC (cvstem) | 4 combinations |

The `c2rl-*-cvstem` configs sweep the RL hyperparameters *and* the CV-STEM LMI
knobs that decide feasibility, so they use the wrapper. `c2rl-*-ccm` sweeps
`w_lb`/`w_ub` as a joint categorical because the pair must satisfy
`w_lb < w_ub` — see the comment in those files.

## Adding an algorithm

Drop a new `<name>.yaml` here following the schema. `search.sh` discovers
configs by globbing this directory — nothing else needs editing.
