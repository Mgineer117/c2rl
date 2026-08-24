"""Shared NeuralDynamics pretraining loop for contraction agents (C3M, C2RL).

Both C3M and C2RL learn ``ẋ = f(x) + B(x)·u`` with a NeuralDynamics network that
is pretrained *before* the main training loop and then refined online each
update/epoch. This module holds the single pretraining implementation so the two
agents cannot drift apart.

The agent passed in must expose the (identical across C3M/C2RL) interface:
    ._neural_dynamics, ._train_dynamics(data), ._get_rollout(n, "dynamics"),
    ._cfg.dynamics_batch_size, ._dynamics_optimizer, ._dynamics_lr_scheduler,
    .track_data(), .writer, .write_tracking_data()  (skrl Agent methods)
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as _tqdm

from .math_utils import EarlyStopper, train_val_split


def load_offline_dynamics_data(data_path: str, tag: str = "") -> dict:
    """Load + flatten an offline ``dynamics_data.npz`` (keys ``x``, ``u``,
    ``x_dot``, optional ``lengths``) into flat ``(n, dim)`` arrays.

    Filters out whole episodes containing any NaN, and — when a ``lengths``
    array is present — masks out padding steps (the last valid state repeated
    past each episode's true length) so downstream fits/samples aren't biased
    toward artificial ``x_dot ~ 0`` padding. Shared by ``pretrain_dynamics``
    (NeuralDynamics fit) and C2RL's ``synthesize_cmg`` (CMG SDP-dataset states),
    so both read the same offline states the same way.
    """
    print(f"{tag} Loading dynamics pretrain data from {data_path}")
    npz = np.load(data_path)

    nan_mask = (
        np.isnan(npz["x"]).any(axis=(1, 2))
        | np.isnan(npz["u"]).any(axis=(1, 2))
        | np.isnan(npz["x_dot"]).any(axis=(1, 2))
    )
    lengths_arr = npz["lengths"] if "lengths" in npz else None
    if nan_mask.any():
        print(f"{tag} WARNING: Found NaNs in {nan_mask.sum()} offline episodes! Filtering them out...")
        valid_mask = ~nan_mask
        x_arr = npz["x"][valid_mask]
        u_arr = npz["u"][valid_mask]
        x_dot_arr = npz["x_dot"][valid_mask]
        if lengths_arr is not None:
            lengths_arr = lengths_arr[valid_mask]
    else:
        x_arr, u_arr, x_dot_arr = npz["x"], npz["u"], npz["x_dot"]

    if lengths_arr is not None:
        T_len = x_arr.shape[1]
        step_mask = np.arange(T_len)[None, :] < lengths_arr[:, None]  # (N, T)
        x_arr = x_arr[step_mask]
        u_arr = u_arr[step_mask]
        x_dot_arr = x_dot_arr[step_mask]
        print(f"{tag} Unpacked {step_mask.sum()} valid samples from {step_mask.shape[0]} trajectories (padding masked)")

    x_arr = x_arr.reshape(-1, x_arr.shape[-1])
    u_arr = u_arr.reshape(-1, u_arr.shape[-1])
    x_dot_arr = x_dot_arr.reshape(-1, x_dot_arr.shape[-1])
    return {
        "x": x_arr.astype(np.float32),
        "u": u_arr.astype(np.float32),
        "x_dot": x_dot_arr.astype(np.float32),
    }


def load_offline_trajectories(data_path: str, tag: str = "") -> dict:
    """Load an offline ``dynamics_data.npz`` without flattening — returns
    trajectory-structured ``x`` ``(N, T, x_dim)`` and each trajectory's valid
    length ``lengths`` ``(N,)``, preserving both the within- and
    across-trajectory order/boundaries that ``load_offline_dynamics_data``
    intentionally discards (that loader exists for i.i.d. NeuralDynamics
    fitting, not for a temporal material-derivative term).

    Used by ``ncm_synthesis.build_cm_dataset``'s ``cm_wdot_trajectory=True``
    path (see ``C2RLAgent.synthesize_cmg`` / ``c2rl.py``'s ``cm_wdot_trajectory``
    docstring) to thread a real ``Ẇ ≈ (W̄_t − W̄_{t−1})/dt`` through the CV-STEM
    SDP using the actual previous state of the same reference trajectory,
    instead of dropping ``Ẇ`` or using Tsukamoto's static ``(W̄-I)/dt`` proxy.

    Raises if the file has no ``lengths`` array — that means it isn't a
    trajectory-structured ``dynamics_data.npz`` (see
    ``scripts/skrl/train.py``'s ``_generate_ref_trajs``), so there is no way
    to recover per-trajectory boundaries from it. Filters out whole episodes
    containing any NaN, same as ``load_offline_dynamics_data``.
    """
    print(f"{tag} Loading offline reference trajectories from {data_path}")
    npz = np.load(data_path)
    if "lengths" not in npz:
        raise ValueError(
            f"{tag} {data_path} has no 'lengths' array — it isn't a trajectory-structured "
            f"dynamics_data.npz (see scripts/skrl/train.py's _generate_ref_trajs), so "
            f"cm_wdot_trajectory can't recover per-trajectory boundaries from it."
        )
    x = npz["x"]
    lengths = npz["lengths"]
    nan_mask = np.isnan(x).any(axis=(1, 2))
    if nan_mask.any():
        print(f"{tag} WARNING: Found NaNs in {nan_mask.sum()} offline episodes! Filtering them out...")
        x = x[~nan_mask]
        lengths = lengths[~nan_mask]
    return {"x": x.astype(np.float32), "lengths": lengths.astype(np.int64)}


def pretrain_dynamics(agent, *, epochs: int, data_path: str | None,
                      timesteps: int, memory_size: int | None = None,
                      num_controls_per_state: int | None = None,
                      log_interval: int | None = None,
                      tag: str = "",
                      val_frac: float = 0.1,
                      early_stop_patience: int = 10) -> None:
    """Pretrain ``agent._neural_dynamics`` for ``epochs`` epochs over a fixed
    ``memory_size``-sized buffer of ``(x, u, x_dot)`` samples, drawn once before
    the epoch loop — the same "sample a fixed dataset, then multi-epoch it"
    structure ``ncm_synthesis.build_cm_dataset``/``regress_cmg`` use for CMG
    synthesis (see ``C2RLAgent._sample_cmg_x``/``synthesize_cmg``), so both
    pretraining phases behave the same way.

    The buffer comes from ``load_offline_dynamics_data(data_path)`` when
    ``data_path`` points at an offline ``dynamics_data.npz`` — uniformly
    subsampled (without replacement) to ``memory_size``, with a warning and a
    cap to what's on disk if ``memory_size`` asks for more (Isaac envs
    typically only have this fixed on-disk supply). Otherwise it's drawn fresh
    via ``agent._get_rollout(memory_size, "dynamics", num_control_per_state=...)``
    — classic envs can feasibly sample any ``memory_size`` since it's synthetic
    analytic sampling, no cap needed; Isaac envs without a data_path instead
    draw from their in-memory reference-trajectory buffer (also no fixed cap).

    No-op when the agent has no NeuralDynamics (analytical dynamics) or
    epochs<=0. Pretraining metrics (dynamics MSE + LR) are flushed to the
    agent's writer at negative timesteps (so they precede training on the
    ``global_step`` x-axis). The flush cadence is derived from ``epochs`` — not
    the agent's training ``write_interval``, which is typically
    ``timesteps//100`` and would never fire within a short pretraining loop —
    and the final epoch always flushes. ``log_interval`` optionally caps that
    cadence; ``tag`` labels console prints.

    ``val_frac`` holds out that fraction of the (fixed, once-sampled) buffer as
    a validation split never trained on; ``early_stop_patience`` stops the loop
    once the held-out MSE hasn't improved for that many consecutive epochs,
    restoring the best-val-epoch weights (see ``math_utils.EarlyStopper``).
    ``val_frac<=0`` or a validation split with 0 samples disables both and
    falls back to always running the full ``epochs`` budget.
    """
    if agent._neural_dynamics is None or epochs <= 0:
        return

    # ~100 wandb points regardless of the (huge) training write_interval.
    log_every = max(1, epochs // 100)
    if log_interval and log_interval > 0:
        log_every = min(log_every, log_interval)
    has_writer = getattr(agent, "writer", None) is not None

    dev = agent._neural_dynamics.device

    if data_path:
        offline = load_offline_dynamics_data(data_path, tag=tag)
        x_all, u_all, xdot_all = offline["x"], offline["u"], offline["x_dot"]
        n_avail = x_all.shape[0]
        n_samples = memory_size if memory_size is not None else n_avail
        if n_samples > n_avail:
            print(f"{tag} WARNING: emp_dynamics_memory_size={n_samples} exceeds the "
                  f"{n_avail} available offline dynamics samples — using {n_avail} instead.")
            n_samples = n_avail
        idx = np.random.choice(n_avail, size=n_samples, replace=False)
        x_np, u_np, xdot_np = x_all[idx], u_all[idx], xdot_all[idx]
    else:
        n_samples = memory_size if memory_size is not None else agent._cfg.dynamics_batch_size
        data = agent._get_rollout(n_samples, "dynamics", num_control_per_state=num_controls_per_state)
        x_np, u_np, xdot_np = data["x"], data["u"], data["x_dot"]

    x = torch.as_tensor(x_np).to(torch.float32).to(dev)
    u = torch.as_tensor(u_np).to(torch.float32).to(dev)
    x_dot = torch.as_tensor(xdot_np).to(torch.float32).to(dev)
    n = x.shape[0]

    train_idx, val_idx = train_val_split(n, val_frac, device=dev)
    n_train = train_idx.shape[0]
    n_val = val_idx.shape[0]
    x_val, u_val, xdot_val = x[val_idx], u[val_idx], x_dot[val_idx]
    stopper = EarlyStopper(patience=early_stop_patience if n_val > 0 else 0)
    batches_per_epoch = max(1, n_train // agent._cfg.dynamics_batch_size)

    dyn_pbar = _tqdm.tqdm(range(epochs), desc="Pretraining dynamics", file=sys.stdout)
    for epoch in dyn_pbar:
        for _ in range(batches_per_epoch):
            dbz = min(agent._cfg.dynamics_batch_size, n_train)
            idx = train_idx[torch.randint(0, n_train, (dbz,), device=dev)]
            loss_val = agent._train_dynamics({"x": x[idx], "u": u[idx], "x_dot": x_dot[idx]})

        postfix = {"loss": f"{loss_val:.3g}"}

        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent._dynamics_lr_scheduler.step()

        agent.track_data("Loss / Pretrain/dynamics_mse", loss_val)
        if getattr(agent, "_dynamics_lr_scheduler", None) is not None:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_lr_scheduler.get_last_lr()[0])
        else:
            agent.track_data("Pretrain/dynamics_lr", agent._dynamics_optimizer.param_groups[0]["lr"])

        stop = False
        if n_val > 0:
            with torch.no_grad():
                val_loss = F.mse_loss(
                    agent._neural_dynamics.predict_x_dot(x_val, u_val), xdot_val
                ).item()
            agent.track_data("Loss / Pretrain/dynamics_val_mse", val_loss)
            postfix["val"] = f"{val_loss:.3g}"
            stop = stopper.step(val_loss, agent._neural_dynamics, epoch)

        dyn_pbar.set_postfix(**postfix)

        # Flush at negative timesteps so pretraining precedes training on the
        # global_step x-axis; always flush the final/stopping epoch so short
        # runs log too.
        if has_writer and ((epoch + 1) % log_every == 0 or epoch == epochs - 1 or stop):
            agent.write_tracking_data(timestep=epoch - epochs, timesteps=timesteps)

        if stop:
            print(f"{tag} Dynamics pretraining early-stopped at epoch {epoch + 1}/{epochs} "
                  f"(best val MSE {stopper.best:.4g} @ epoch {stopper.best_epoch + 1}).")
            dyn_pbar.close()
            break

    if n_val > 0:
        # Restore the best-val-epoch weights whether the loop was early-stopped
        # or simply ran out its full epoch budget (the final epoch is not
        # necessarily the best one either way) — this print always fires so
        # the actually-used epoch/MSE is visible even when the loop ran to
        # completion without early stopping ever triggering.
        stopper.restore_best(agent._neural_dynamics)
        print(f"{tag} Dynamics pretraining: using best-val epoch "
              f"{stopper.best_epoch + 1}/{epochs} (val MSE {stopper.best:.4g}).")
