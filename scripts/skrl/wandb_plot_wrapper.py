import sys

import numpy as np


class WandbPlotWrapper:
    """
    Gym wrapper that periodically pushes train/normalized_error and
    train/path_tracking trajectory curves to wandb.

    Plot data is read directly from the inner StatManagerEnvWrapper's
    already-validated buffer (`trajectories()` / `_compute_count`) instead of
    being independently re-derived here — StatManagerEnvWrapper anchors e(0)
    at the true reset, sqrt's classic BaseEnv's squared "tracking_error", and
    produces exactly `max_episode_length`-long curves (no extra post-episode
    transition point), so reusing it sidesteps that whole class of bugs
    rather than re-solving it a second time in this wrapper.
    """
    def __init__(self, env, total_timesteps=None, num_plots=10):
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        self._total_steps = 0
        # Plot cadence is measured in `.step()` calls (== the trainer's "timesteps"/
        # "global_step" unit, i.e. Not multiplied by num_envs), so it scales with an
        # env's actual episode length instead of a fixed episode count. The previous
        # "every 50 completed episodes" default silently never fired for any run
        # whose (timesteps / episode_len) never reached a multiple of 50 — e.g. a
        # 30000-timestep run of a 500-step-episode classic env completes only ~60
        # episodes total, so it produced at most a single plot (or none, for longer-
        # episode Isaac path-tracking envs). Spacing plots evenly across the whole
        # run guarantees ~num_plots of them regardless of episode length.
        self._step_calls = 0
        self._plot_freq_steps = max(1, int(total_timesteps) // num_plots) if total_timesteps else 1000
        self._last_plot_call = -self._plot_freq_steps
        # StatManagerEnvWrapper's buffer-completion counter last seen — a plot
        # is only pushed once this has actually advanced, so a slow env (whose
        # eval buffer hasn't completed a fresh round yet) never gets a stale
        # re-push of the previous round's curves.
        self._last_compute_count = getattr(self.env, "_compute_count", 0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += self.num_envs
        self._step_calls += 1

        compute_count = getattr(self.env, "_compute_count", None)
        if compute_count is not None and compute_count != self._last_compute_count:
            if (self._step_calls - self._last_plot_call) >= self._plot_freq_steps \
                    and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
                self._plot_and_push()
                self._last_plot_call = self._step_calls
            self._last_compute_count = compute_count

        # Lift "log" dicts so SKRL's trainer can log them natively.
        # SyncVectorEnv structures info differently depending on autoreset mode:
        #   NEXT_STEP (default): no "final_info"; per-env "log" dicts are flattened
        #       by _add_info into info["log"] = {metric_key: np.array(num_envs)}
        #       with a boolean mask info["_log"] indicating which envs have data.
        #   SAME_STEP: info["final_info"] is a tuple of per-env dicts (or None).
        device = getattr(self.env, "device", "cpu")
        lifted = False

        # --- Format 1: NEXT_STEP autoreset (default SyncVectorEnv) ---
        # _add_info recursively flattens env_info["log"] into info["log"] as
        # a dict of numpy arrays, with info["_log"] as a boolean mask.
        if "log" in info and "_log" in info and not lifted:
            mask = info["_log"]  # bool array (num_envs,)
            if isinstance(mask, np.ndarray) and mask.any():
                import torch
                from c2rl.tasks.direct.common.eval_metrics import mean_confidence_interval
                raw_log = info["log"]
                # raw_log is a dict whose leaf values are numpy arrays and whose
                # sub-keys starting with "_" are masks — skip those.
                new_log = {}
                for k, v in raw_log.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, np.ndarray):
                        valid = v[mask]
                        if valid.size > 0:
                            v_m, v_ci = mean_confidence_interval(valid)
                            new_log[f"{k}_mean"] = torch.tensor(v_m, dtype=torch.float32, device=device)
                            new_log[f"{k}_ci95"] = torch.tensor(v_ci, dtype=torch.float32, device=device)
                    elif isinstance(v, (int, float)):
                        new_log[f"{k}_mean"] = torch.tensor(float(v), dtype=torch.float32, device=device)
                        new_log[f"{k}_ci95"] = torch.tensor(0.0, dtype=torch.float32, device=device)
                if new_log:
                    info["log"] = new_log
                    lifted = True

        # --- Format 2: SAME_STEP autoreset (final_info is a tuple/list/ndarray) ---
        # gymnasium's SyncVectorEnv._add_info funnels "final_info" through the
        # Same generic per-key aggregation as every other info key: since each
        # env's `info["final_info"]` value is a `dict`, `_init_info_arrays`
        # allocates an object-DTYPE NUMPY array (`np.zeros(num_envs, dtype=object)`),
        # not a tuple/list — checking only (tuple, list) silently skipped this
        # branch on gymnasium>=0.29's SyncVectorEnv, so no Stability/* from
        # classic envs' terminal info dict ever reached wandb.
        if not lifted and "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, (tuple, list, np.ndarray)):
                logs_list = {}
                for fin_info in final_info:
                    if fin_info is not None and isinstance(fin_info, dict) and "log" in fin_info:
                        for k, v in fin_info["log"].items():
                            logs_list.setdefault(k, []).append(float(v))

                if logs_list:
                    import torch
                    from c2rl.tasks.direct.common.eval_metrics import mean_confidence_interval
                    info["log"] = {}
                    for k, v_list in logs_list.items():
                        v_m, v_ci = mean_confidence_interval(np.array(v_list))
                        info["log"][f"{k}_mean"] = torch.tensor(v_m, dtype=torch.float32, device=device)
                        info["log"][f"{k}_ci95"] = torch.tensor(v_ci, dtype=torch.float32, device=device)

        return obs, reward, terminated, truncated, info

    def close(self, **kwargs):
        # Safety net for envs whose episodes never complete within the run
        # (episode length > total timesteps) — the buffer-completion trigger
        # above then never fires. Push whatever the StatManager buffer holds
        # so far rather than emitting nothing for the whole run.
        if self._last_compute_count == 0 and getattr(self.env, "_compute_count", 0) > 0 \
                and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
            self._plot_and_push()
        return self.env.close(**kwargs)

    def _plot_and_push(self):
        # Delegate to the shared plotter so the standalone PPO/SAC/LQR/SD-LQR
        # path emits the same "train/normalized_error" and "train/path_tracking"
        # figures (same style/keys) that C3M/C2RL emit from their eval() — a
        # single source of truth for all tracking plots, backed by
        # StatManagerEnvWrapper's buffer (already normalized by e(0)).
        from c2rl.agents.skrl.contraction_metrics import (
            log_metric_distributions,
            log_tracking_plots,
        )

        # Histograms of the per-env AUC / lambda / running lambda / peak
        # overshoot behind the Stability scalars. Pushed before the early
        # returns below: those guard on trajectory recording (traj_x), which is
        # a separate buffer — the metric distributions are computed from
        # _eval_buffer and are available whenever the StatManager has reduced,
        # so gating them on trajectories would silently drop them.
        log_metric_distributions(
            self.env.distributions(), prefix="train",
            step=self._total_steps, title="Train",
        )

        traj_x, traj_xref, traj_err = self.env.trajectories()
        if not traj_err:
            return
        # trajectories() is keyed by buffer slot (up to num_envs_for_eval,
        # e.g. 64). Log all slots that actually recorded an episode — the plotter
        # lays them out as ceil(n/5) subplots, 5 curves each. Unused slots still
        # carry a full-length all-zero error row in traj_err but an empty traj_x
        # list, so they are filtered here (gating on non-empty traj_x) — that
        # both drops flat placeholder lines and keeps every figure
        # (normalized_error / normalized_maha_error / path_tracking) on the same
        # env set and ordering.
        # len(), not truthiness — a numpy array raises on `if arr`.
        plot_keys = sorted(k for k in traj_err
                           if traj_x.get(k) is not None and len(traj_x[k]) > 0)
        if not plot_keys:
            return
        traj_x = {k: traj_x.get(k, []) for k in plot_keys}
        traj_xref = {k: traj_xref.get(k, []) for k in plot_keys}
        traj_err = {k: traj_err[k] for k in plot_keys}

        # C2RL only: the metric-weighted analog of normalized_error. Empty (→ no
        # extra figure) for envs that emit no Mahalanobis error (see
        # StatManagerEnvWrapper.maha_trajectories).
        maha_fn = getattr(self.env, "maha_trajectories", None)
        traj_maha = maha_fn() if callable(maha_fn) else {}
        traj_maha = {k: traj_maha[k] for k in plot_keys if k in traj_maha}

        # For classic envs, self.env.unwrapped is the SyncVectorEnv itself —
        # `dt` lives on its per-env `.envs[i]`, not on the vector env — so the
        # generic getattr chain below silently fell back to 1.0 (wrong AUC/
        # legend scale). Isaac envs expose `step_dt` directly on `.unwrapped`.
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "envs"):
            dt = getattr(unwrapped.envs[0], "dt", 1.0)
        else:
            dt = getattr(unwrapped, "step_dt", None) or getattr(unwrapped, "dt", 1.0)
        log_tracking_plots(
            traj_x, traj_xref, traj_err,
            dt=float(dt), prefix="train", step=self._total_steps, title="Train",
            traj_maha_error=traj_maha,
        )
