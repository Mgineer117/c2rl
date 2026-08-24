#!/usr/bin/env python3
"""W&B sweep trial wrapper that fails an INFEASIBLE trial fast, and loudly.

Why this exists
---------------
Several configs in ``search/configs/`` sweep the knobs that decide whether a
contraction SDP is solvable at all — the ``{w_lb, w_ub}`` envelope, ``λ``, ``ε``
and the Riccati ``R`` scaler, i.e. every term of

    A·W̄ + W̄·Aᵀ - 2ν·B R⁻¹Bᵀ + 2λ·W̄ ⪯ -ε·I,   ν ≤ 1/w_lb,  χ ≤ ν·w_ub

Large parts of that space are structurally infeasible (``ncm_synthesis.py``'s
module docstring measures 0% feasible on segway at ``w_lb=0.1``, 100% at
``w_lb=0.001``).

Left alone, an infeasible trial is the worst possible sweep outcome — not
because it fails, but because it fails without a metric:

  * offline (C2RL cvstem): it either grinds through all 131072 per-state SDP
    solves before ``build_cm_dataset`` raises, or trains a full run on a CMG
    regressed onto a handful of surviving states.
  * online (CV-STEM-LQR): every infeasible state silently falls back to
    ``u = u_ref``, so the trial finishes with a plausible-looking AUC produced
    by a controller that was partly open-Loop.

Either way the sweep controller sees a run with no metric — and a metric-less
run is simply ignored by bayes bookkeeping. Nothing is learned from it, so the
same dead region gets sampled again, forever.

So this wrapper runs the trial as a child, watches its output, and the first
time infeasibility is reported it kills the child, writes ``--bad-value`` to the
sweep metric, and exits 0. The trial becomes a real, very bad datapoint instead
of a hole — which is also what makes bayes usable over a partly-infeasible
space: the wall is an observation the surrogate can learn to avoid.

Nothing in the solver path is modified; detection is pure output inspection.
Crucially, the child's own config decides what counts as a terminal miss — this
wrapper only watches for the signals that mean the child actually dropped a
state or aborted. Strictness is a per-algorithm choice made in the config:

                                     #   the state drops and the ratio/rate fires
    cm.min_feasibility_rate:  1.0    # offline: child raises if any state is dropped

a few genuine drops. A rescued state is feasible, so it never trips a detection
signal — which is why the ``infeasible at λ=`` rescue WARNING is not a kill
signal (see ``_HARD_MARKERS``). Only a state the child could not rescue (dropped)
reaches the wrapper, via the ratio/rate paths.

Detection signals:
  * ``CVSTEM-LQR INFEASIBLE``                            — cvstem-lqr's joint SDP
    has no solution at this (λ, ε) (``cvstem_lqr.INFEASIBLE_MARKER``; the two
    must stay in sync). Raised at construction, before any rollout: one program
    covers every sample, so there is nothing per-state to rescue.
  * tqdm postfix ``feasible=<k>/<i>`` with ``k < i``     — earliest offline signal,
    fires within the first 128 states (a dropped, not rescued, state)
  * ``NCM synthesis: <k>/<n> states feasible`` with k<n  — end-of-solve summary
  * ``produced 0 feasible metrics`` / ``below min_feasibility_rate`` — the raises

Run-id handling: the parent picks (or inherits) ``WANDB_RUN_ID`` and exports it
to the child, so after killing the child it can resume that exact run and write
the metric into its summary — which is the value the sweep controller reads.

Sign convention: ``--bad-value`` must match ``metric.goal``. For the AUC metrics
these configs use (minimized) it is a large positive number, not a negative one.

Usage (normally invoked by the sweep yaml search/build_sweep.py generates):
    python search/sweep_runner.py --task classic-car-v0 --algorithm cvstem-lqr \
        --num_envs 64 --metric-name "Stability/auc_mean" --bad-value=1e3 [...]
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_PY = _ROOT / "scripts" / "skrl" / "train.py"

# ── Infeasibility signals ─────────────────────────────────────────────────── #
# Substrings that unambiguously mean "the SDP failed" regardless of counts.
_HARD_MARKERS = (
    # cvstem_lqr.INFEASIBLE_MARKER — the online abort, raised both by CV-STEM-LQR's
    # control law. (C2RL's per-step "online" metric used to raise it too; that
    # path was removed -- a per-state SDP certifies nothing about contraction.)
    "CVSTEM-LQR INFEASIBLE",
    "CVSTEMInfeasibleError",       # same, as it appears in the traceback
    "produced 0 feasible metrics",
    "below min_feasibility_rate",
    # build_cm_dataset now solves ONE joint SDP over all samples, which raises this
    # instead of reporting a per-state feasibility rate. Without it here an
    # infeasible offline synthesis is not recognised as a hard failure, so the
    # trial finishes metric-less -- and a metric-less trial is silently dropped
    # from the bayes bookkeeping, which is the exact failure this module exists to
    # prevent. The two markers above can no longer fire from the offline path
    # (that code is unreachable); they are kept for the online metric.
    "joint CV-STEM SDP INFEASIBLE",
)
# NOTE: the per-state ratio patterns below are legacy. Synthesis now solves ONE
# joint SDP over every sample, so there is no per-state feasibility rate to fall
# short of and no per-state λ-backoff to rescue a miss: the program either
# certifies λ over the whole draw or raises the marker above. The k/n patterns are
# kept because they still match older logs, not because a current run emits them.
# tqdm postfix (`pbar.set_postfix(feasible=f"{len(xs)}/{i+1}")`) and the final
# summary line — both carry a k/n pair that is only healthy when k == n.
_RATIO_PATTERNS = (
    re.compile(r"feasible=(\d+)/(\d+)"),
    re.compile(r"NCM synthesis:\s*(\d+)/(\d+)\s*states feasible"),
)


def _infeasibility_reason(text: str) -> str | None:
    """Return a human-readable reason if ``text`` reports infeasibility."""
    for marker in _HARD_MARKERS:
        if marker in text:
            return f"synthesis reported: {marker!r}"
    for pattern in _RATIO_PATTERNS:
        for feasible, seen in pattern.findall(text):
            if int(feasible) < int(seen):
                return f"only {feasible}/{seen} states feasible"
    return None


def _iter_output(proc: subprocess.Popen):
    """Yield chunks of the child's merged stdout, split on \\n and \\r.

    tqdm redraws its bar with a bare carriage return, so the postfix — our
    earliest infeasibility signal — never arrives on a newline-terminated line,
    and readline() would sit on it until the whole progress bar finished.

    Reads at the file descriptor level (os.read returns as soon as any bytes are
    available) rather than proc.stdout.read(n), which blocks until it has
    collected exactly n chars or hits eof. That difference is the whole point of
    this function: the child prints the infeasibility marker and then can sit
    there — mid-SDP-solve, or blocked — for a long time without emitting enough
    further output to fill a fixed-size read. A buffered read would not surface
    the marker until that padding arrived, which is precisely when we most need
    to have already killed the child.
    """
    fd = proc.stdout.fileno()
    buf = ""
    pending = b""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        # Decode incrementally: a 4096-byte boundary can land mid-Utf-8 (the
        # markers contain λ), so hold any partial trailing sequence over.
        pending += chunk
        try:
            text = pending.decode("utf-8")
            pending = b""
        except UnicodeDecodeError:
            # Keep the last few bytes and decode what is safely complete.
            for cut in range(1, min(4, len(pending)) + 1):
                try:
                    text = pending[:-cut].decode("utf-8")
                    pending = pending[-cut:]
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue
        buf += text
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for part in parts:
            yield part, True
        # A marker can arrive with no trailing newline (the child may stall
        # immediately after printing it), so the in-progress tail must be
        # scannable too. Flagged partial: the caller scans it but does not print
        # it, and it stays buffered so the eventual complete line is still
        # yielded — and printed — exactly once.
        if buf:
            yield buf, False
    if pending:
        buf += pending.decode("utf-8", errors="replace")
    if buf:
        yield buf, True


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the child's whole process group, SIGKILL if it lingers."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _log_bad_metric(metric_name: str, bad_value: float, reason: str, run_id: str | None) -> None:
    """Resume the child's run and write ``bad_value`` into the sweep metric.

    The sweep controller reads the run summary, so a single log call is enough.
    Best-effort: a failure here must not turn a cleanly-detected infeasibility
    into a nonzero exit (which would make the agent look like it crashed).
    """
    try:
        import wandb
    except ImportError:
        print("[sweep-runner] wandb unavailable — cannot record the bad metric.", flush=True)
        return
    try:
        if wandb.run is None:
            wandb.init(
                id=run_id,
                resume="allow",
                project=os.environ.get("WANDB_PROJECT"),
                settings=wandb.Settings(console="off"),
            )
        wandb.log({metric_name: bad_value, "sdp_infeasible": 1, "global_step": 0})
        if wandb.run is not None:  # log() already sets the summary; pin it explicitly
            wandb.run.summary[metric_name] = bad_value
            wandb.run.summary["sdp_infeasible"] = 1
            wandb.run.summary["sdp_infeasible_reason"] = reason
        wandb.finish(exit_code=0)
        print(f"[sweep-runner] recorded {metric_name}={bad_value} for this trial.", flush=True)
    except Exception as exc:  # noqa: BLE001 — never mask the real failure
        print(f"[sweep-runner] failed to record the bad metric: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Env id, e.g. classic-car-v0.")
    parser.add_argument("--algorithm", required=True, help="train.py --algorithm value.")
    parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, required=True)
    parser.add_argument("--num_timesteps", "--num-timesteps", dest="num_timesteps",
                        type=int, default=None,
                        help="Forwarded to train.py; set by one_episode configs.")
    parser.add_argument("--classic", action="store_true",
                        help="Forwarded to train.py for classic envs.")
    parser.add_argument("--metric-name", default="Stability/auc_mean",
                        help="Sweep metric to poison. Must match the yaml's metric.name.")
    parser.add_argument("--bad-value", type=float, default=1e4,
                        help="Value written to --metric-name on infeasibility. Sign follows "
                             "the metric's goal. Pass as --bad-value=<v>, one token.")
    args, passthrough = parser.parse_known_args()

    # Own the run id so the child's run can be resumed after we kill it. Under a
    # real `wandb agent` the id is already set — never override it, that is the
    # id the sweep controller is tracking.
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        try:
            import wandb
            run_id = wandb.util.generate_id()
        except Exception:  # noqa: BLE001
            run_id = None

    cmd = [sys.executable, str(_TRAIN_PY)]
    if args.classic or args.task.startswith("classic-"):
        cmd.append("--classic")
    cmd += [
        "--task", args.task,
        "--algorithm", args.algorithm,
        "--num_envs", str(args.num_envs),
    ]
    if args.num_timesteps is not None:
        cmd += ["--num_timesteps", str(args.num_timesteps)]
    cmd += passthrough

    env = dict(os.environ)
    if run_id:
        env["WANDB_RUN_ID"] = run_id
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[sweep-runner] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        # Binary + unbuffered: _iter_output reads the fd directly and decodes
        # incrementally, so a TextIOWrapper in between would only add a buffer
        # that can hide the marker we are racing to see.
        cmd, cwd=str(_ROOT), env=env, bufsize=0,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, so _terminate takes the tree
    )

    reason: str | None = None
    tail: list[str] = []
    for line, complete in _iter_output(proc):
        if complete:
            print(line, flush=True)
            tail.append(line)
            del tail[:-200]
        if reason is None:
            reason = _infeasibility_reason(line)
            if reason is not None:
                if not complete:  # never printed above — show what triggered it
                    print(line, flush=True)
                print(f"\n[sweep-runner] INFEASIBLE — {reason}. Terminating this trial.", flush=True)
                _terminate(proc)
                break

    proc.wait()

    # The raise can also surface only in the final traceback (e.g. the
    # min_feasibility_rate check, whose message lands after the solve loop).
    if reason is None and proc.returncode != 0:
        reason = _infeasibility_reason("\n".join(tail))

    if reason is not None:
        _log_bad_metric(args.metric_name, args.bad_value, reason, run_id)
        return 0  # a recorded bad trial, not an agent crash

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
