"""Self-check for the X_TERMINATION_* early-termination box.

    python scripts/check_x_termination.py

Covers the things that can silently break:
  1. On by default — leaving the box ends the episode, on `truncated` (not
     `terminated`), so skrl's GAE keeps bootstrapping and there is no suicide
     bonus.
  2. Turning it off restores the old never-terminating behaviour exactly, which
     is what every pre-flip number was measured under.
  3. A termination box wider than [X_MIN, X_MAX] raises instead of no-opping
     (step() clamps first, so a wider bound could never fire).
  4. StatManagerEnvWrapper drops early-ended episodes from AUC/lambda instead of
     padding them into a full-length curve, and reports the dropped fraction.
"""
import pathlib
import sys

import c2rl.tasks  # noqa: F401  (registers classic-*-v0)
import gymnasium as gym
import torch
from c2rl.agents.skrl.contraction_metrics import StatManagerEnvWrapper
from c2rl.tasks.direct.classic.segway.env import SegwayEnv

NUM_ENVS = 8
# pitch 1.0 is inside the box (pi/3 = 1.047), but the rate carries it out in one
# dt: 1.0 + 3.0*0.03 = 1.09. Setting pitch alone would not move at all — pitch
# only changes through pitch_rate.
OUT_OF_BOX = torch.tensor([0.0, 1.0, 0.0, 3.0])


def _env(**kw):
    return SegwayEnv(num_envs=NUM_ENVS, device="cpu", **kw)


# ── 1. on by default, and reported as truncation ─────────────────────────── #
env = _env()
assert env.terminate_out_of_box is True, "the box must be armed by default"
assert env.X_TERMINATION_MIN is not None
env.reset()
env.x_t[:] = OUT_OF_BOX
_, _, term, trunc, info = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert info["episode_ended_early"].all(), "pitch ran past pi/3 but nothing fired"
assert trunc.all(), "excursion must be reported as truncation"
assert not term.any(), "terminated zeroes the GAE bootstrap = suicide bonus"
assert (env.time_steps == 0).all(), "env must have auto-reset"
print("1. on by default -> truncated         ok")

# ── 2. turning it off restores the old behaviour ─────────────────────────── #
env = _env()
env.set_terminate_out_of_box(False)
assert env.terminate_out_of_box is False
assert env.X_TERMINATION_MIN is not None, "constants must exist even when off"
env.reset()
# Slam the state far outside the box; with the feature off nothing ends early.
env.x_t[:] = torch.tensor([100.0, 3.0, 50.0, 50.0])
_, _, term, trunc, info = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert not term.any() and not trunc.any(), "off must never terminate"
assert "episode_ended_early" not in info, "flag leaks when the feature is off"
print("2. off restores old behaviour         ok")

# opt-in terminal flag still available for anyone who wants it
env = _env()
env.set_terminate_out_of_box(True, terminal=True)
env.reset()
env.x_t[:] = OUT_OF_BOX
_, _, term, _, _ = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert term.all(), "terminal=True must report on `terminated`"
print("2b. terminal=True honoured            ok")

# ── 3. a box wider than [X_MIN, X_MAX] raises ────────────────────────────── #
env = _env()
env.X_TERMINATION_MAX = env.X_MAX + 1.0
try:
    env.set_terminate_out_of_box(True)
except ValueError as e:
    assert "inside [X_MIN, X_MAX]" in str(e)
    print("3. wider-than-state-box raises        ok")
else:
    raise AssertionError("a bound outside the clamp is a silent no-op — must raise")

# ── 4. StatManager drops early-ended episodes ────────────────────────────── #
# Through the real wrapper stack (train.py:737): StatManagerEnvWrapper only ever
# sees the flat observation the skrl wrapper produces, never the env's Dict.
# Half the envs are held on the reference (they run the full horizon), half are
# kicked out of the box, so those slots must be excluded rather than padded.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "skrl"))
from train_utils import BatchedGymnasiumWrapper  # noqa: E402

raw = gym.make("classic-segway-v0", num_envs=NUM_ENVS, device="cpu")
env = raw.unwrapped
env.set_terminate_out_of_box(True)
wrapped = StatManagerEnvWrapper(BatchedGymnasiumWrapper(raw), num_envs_for_eval=NUM_ENVS)
wrapped.reset()
zero_u = torch.zeros(NUM_ENVS, env.num_dim_control)
kick = torch.zeros(NUM_ENVS, dtype=torch.bool)
kick[: NUM_ENVS // 2] = True
for t in range(env.max_episode_len + 2):
    # Hold the survivors on the reference (error ~ 0) and shove the rest out.
    env.x_t[~kick] = env.xref[~kick, min(t, env.max_episode_len - 1)]
    env.x_t[kick] = OUT_OF_BOX
    wrapped.step(zero_u)
    if wrapped._compute_count > 0:
        break

summary = wrapped.stability_summary()
assert wrapped._compute_count > 0, "buffer never reduced"
frac = summary.get("early_end_frac")
assert frac is not None, "early_end_frac must always be reported"
assert frac > 0.0, f"kicked envs were not excluded (early_end_frac={frac})"
if wrapped._recent_valid_n:
    assert "auc_mean" in summary, "survivors must still produce metrics"
    for k in ("auc_mean", "contraction_rate_mean", "overshoot_mean"):
        assert summary[k] == summary[k], f"{k} is NaN — invalid rows leaked in"
else:
    assert set(summary) == {"early_end_frac"}, \
        "with no survivors the metrics must be ABSENT, not stale"
print(f"4. early-ended slots excluded         ok "
      f"(early_end_frac={frac:.2f}, valid={wrapped._recent_valid_n})")

# ── 5. feature off is the old behaviour, metric for metric ───────────────── #
# The regression that matters most: every pre-flip number was produced with no
# termination box, so with it off nothing may be dropped and every metric must
# still be reported — that is what --no_terminate_out_of_box has to reproduce.
raw = gym.make("classic-segway-v0", num_envs=NUM_ENVS, device="cpu")
env = raw.unwrapped
env.set_terminate_out_of_box(False)
wrapped = StatManagerEnvWrapper(BatchedGymnasiumWrapper(raw), num_envs_for_eval=NUM_ENVS)
wrapped.reset()
for _ in range(env.max_episode_len + 2):
    wrapped.step(torch.zeros(NUM_ENVS, env.num_dim_control))
    if wrapped._compute_count > 0:
        break
summary = wrapped.stability_summary()
assert wrapped._compute_count > 0, "buffer never reduced with the feature off"
assert summary["early_end_frac"] == 0.0, "nothing may be dropped when the box is off"
assert wrapped._recent_valid_n == NUM_ENVS, "every slot must count when the box is off"
for k in ("auc_mean", "contraction_rate_mean", "overshoot_mean", "contraction_score_mean"):
    assert k in summary and summary[k] == summary[k], f"{k} missing/NaN with the feature off"
print(f"5. off == previous behaviour          ok (auc={summary['auc_mean']:.3f})")

import ast  # noqa: E402

SHARED = {"set_terminate_out_of_box", "_left_termination_box", "_init_termination_box"}
SRC = pathlib.Path(__file__).resolve().parent.parent / "source/c2rl/c2rl/tasks/direct"
MIXIN = SRC / "common/termination_box.py"
HOSTS = {
    "classic": (SRC / "classic/common/env_base.py", "BaseEnv"),
    "isaac": (SRC / "common/path_tracking_base.py", "PathTrackingBase"),
}


def _defs(path):
    return {n.name for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef)}


def _bases(path, cls):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [b.id for b in node.bases if isinstance(b, ast.Name)]
    raise AssertionError(f"class {cls} not found in {path}")


# Parity is now structural: One implementation, inherited by both hosts. That is
# strictly stronger than comparing two copies' signatures — which is what this
# check used to do, and which caught them drifting on an argument name.
assert _defs(MIXIN) >= SHARED, f"mixin is missing {SHARED - _defs(MIXIN)}"
for side, (path, cls) in HOSTS.items():
    assert "TerminationBoxMixin" in _bases(path, cls), \
        f"{cls} must inherit TerminationBoxMixin, else the two can drift again"
    redefined = _defs(path) & SHARED
    assert not redefined, f"{side} re-defines {redefined} — that is the drift this prevents"
    assert "episode_ended_early" in path.read_text(), f"{side} never publishes episode_ended_early"
print(f"6. one shared impl, both hosts        ok ({len(SHARED)} methods)")

# ── 7. evaluation always measures the full horizon ───────────────────────── #
# The nastiest failure this feature can cause: AUC = integral of ||e||/||e0||
# over a fixed horizon, so truncating an eval episode does not merely shorten
# the integral, it inverts the metric — a policy that falls at step 20 stops
# accumulating error and outscores one that tracks imperfectly for 500 steps.
from train_utils import _disarm_termination_for_eval  # noqa: E402

raw = gym.make("classic-segway-v0", num_envs=NUM_ENVS, device="cpu")
env = raw.unwrapped
assert env.terminate_out_of_box is True, "precondition: armed by default"
_disarm_termination_for_eval(raw, "[test]")
assert env.terminate_out_of_box is False, \
    "eval env must never terminate early — AUC would reward failing sooner"
env.reset()
env.x_t[:] = OUT_OF_BOX
_, _, term, trunc, _ = env.step(torch.zeros(NUM_ENVS, env.num_dim_control))
assert not term.any() and not trunc.any(), "eval env still ended an episode early"
print("7. eval measures full horizon         ok")

print("\nall checks passed")
