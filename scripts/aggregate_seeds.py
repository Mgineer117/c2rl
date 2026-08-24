#!/usr/bin/env python3
"""Aggregate multi-seed best-policy evaluations into a single CSV.

Each training run writes ``eval_results.json`` next to its checkpoints — the
best_agent.pt rollout produced by train_utils._evaluate_best_model /
_evaluate_classic_path_tracking. That file already holds the Stability numbers
for one seed (auc, overshoot C, contraction rate lambda, contraction score,
total reward), plus the run identity written by train_utils._run_metadata.

This script collects those per-seed files and reduces them across seeds,
emitting two CSVs:

  * ``<out>_runs.csv``    one row per seed — the raw per-run numbers.
  * ``<out>.csv``         one row per (task, algorithm) — mean, std and 95% CI
                          across seeds, plus n_seeds.

The two CI columns mean different things and are deliberately both kept:
``*_ci95`` inside a run is the within-run CI over that run's eval episodes,
while ``*_ci95`` in the summary is the across-SEED CI — the one that answers
"is this algorithm better", since seeds are the independent samples. The
per-seed value fed into the summary is each run's ``*_mean``.

Usage:
    python scripts/aggregate_seeds.py --run-tag seeds_20260720_120000
    python scripts/aggregate_seeds.py --logs logs --out results/my_table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

# Metrics reduced across seeds. Each is the "_mean" key of an eval_results.json;
# any that a given run does not have (e.g. contraction_score is only fitted when
# enough envs clear the e(0) threshold) is simply absent from that run's row and
# skipped in the summary rather than treated as zero.
_METRICS = [
    "auc_mean",
    "contraction_rate_mean",
    "overshoot_mean",
    "contraction_score_mean",
    "total_reward_mean",
]


def _t95(n: int) -> float:
    """Two-sided 95% t multiplier for n samples (n-1 dof).

    Seed counts here are small (3-10), where the normal 1.96 is meaningfully
    too tight, so a short table is used rather than pulling in scipy — this
    script is meant to run without the Isaac env active.
    """
    # Keyed by sample count n (not dof): table[n] is t(0.975, n-1).
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
             14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
             20: 2.093, 21: 2.086, 26: 2.060, 31: 2.042}
    if n < 2:
        return 0.0
    if n in table:
        return table[n]
    # Above the table t is nearly flat and monotonically approaching 1.96;
    # 31 (dof=30) is close enough for any seed count this is used with.
    return 1.96 if n > 31 else table[min(table, key=lambda k: abs(k - n))]


def collect(logs_root: str, run_tag: str | None) -> list[dict]:
    """Load every eval_results.json under logs_root, newest last."""
    rows = []
    pattern = os.path.join(logs_root, "**", "eval_results.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[aggregate] skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        # Runs from before _run_metadata existed have no run_tag/seed. They are
        # kept only when no --run-tag filter is in force, so an unfiltered call
        # still sees historical runs instead of silently returning nothing.
        if run_tag is not None and data.get("run_tag") != run_tag:
            continue
        data["run_dir"] = os.path.dirname(path)
        rows.append(data)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Group per-seed rows by (task, algorithm) and reduce across seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("task") or "?", r.get("algorithm") or "?")
        groups.setdefault(key, []).append(r)

    out = []
    for (task, algo), grp in sorted(groups.items()):
        # Distinct seeds, not row count: a re-run of the same seed would
        # otherwise inflate n and shrink the CI on duplicated evidence.
        seeds = sorted({r.get("seed") for r in grp if r.get("seed") is not None})
        summary = {
            "task": task,
            "algorithm": algo,
            "n_runs": len(grp),
            "n_seeds": len(seeds) or len(grp),
            "seeds": " ".join(str(s) for s in seeds),
        }
        for metric in _METRICS:
            vals = [float(r[metric]) for r in grp
                    if isinstance(r.get(metric), (int, float))
                    and math.isfinite(float(r[metric]))]
            base = metric[:-5] if metric.endswith("_mean") else metric
            if not vals:
                summary[f"{base}_mean"] = ""
                summary[f"{base}_std"] = ""
                summary[f"{base}_ci95"] = ""
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            ci = _t95(len(vals)) * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            summary[f"{base}_mean"] = round(mean, 6)
            summary[f"{base}_std"] = round(std, 6)
            summary[f"{base}_ci95"] = round(ci, 6)
        out.append(summary)
    return out


def _write_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default="logs",
                   help="Root to search for eval_results.json (default: logs).")
    p.add_argument("--run-tag", default=None,
                   help="Only aggregate runs carrying this CRL_RUN_TAG "
                        "(set by run_seeds.sh). Default: every run found.")
    p.add_argument("--out", default="results/seed_summary",
                   help="Output path WITHOUT extension; writes <out>.csv and "
                        "<out>_runs.csv (default: results/seed_summary).")
    args = p.parse_args()

    rows = collect(args.logs, args.run_tag)
    if not rows:
        where = f" with run_tag={args.run_tag}" if args.run_tag else ""
        print(f"[aggregate] No eval_results.json found under {args.logs}{where}.",
              file=sys.stderr)
        print("[aggregate] Runs finish their eval only WITHOUT --skip_final_eval.",
              file=sys.stderr)
        return 1

    summary = summarize(rows)

    run_cols = (["task", "algorithm", "seed", "run_tag", "run_dir", "checkpoint",
                 "num_episodes"]
                + [c for m in _METRICS for c in (m, m.replace("_mean", "_ci95"))])
    summary_cols = (["task", "algorithm", "n_seeds", "n_runs", "seeds"]
                    + [c for m in _METRICS
                       for c in (m, m.replace("_mean", "_std"), m.replace("_mean", "_ci95"))])

    runs_csv = f"{args.out}_runs.csv"
    summary_csv = f"{args.out}.csv"
    _write_csv(runs_csv, rows, run_cols)
    _write_csv(summary_csv, summary, summary_cols)

    print(f"[aggregate] {len(rows)} run(s) → {runs_csv}")
    print(f"[aggregate] {len(summary)} (task, algorithm) group(s) → {summary_csv}")
    for s in summary:
        auc = s.get("auc_mean", "")
        ci = s.get("auc_ci95", "")
        print(f"  {s['task']:<28} {s['algorithm']:<14} n={s['n_seeds']:<3} "
              f"auc={auc} ± {ci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
