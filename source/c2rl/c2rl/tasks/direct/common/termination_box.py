"""The early-termination state box, shared by both env families.

What the box is for
-------------------
An episode ends the first step the state leaves [X_TERMINATION_MIN,
X_TERMINATION_MAX]. Without it a diverged env keeps producing off-distribution
transitions for the rest of the horizon — on segway, 500 steps of already-fallen
data per failure — and those transitions dominate the rollout batch, which is a
large part of why that env's outcome swings with the seed.

Which done flag
---------------
The excursion is reported as truncation, not termination. skrl's GAE is
``not_done = ~terminated`` (agents/skrl/agent_patches.compute_gae), so
``terminated`` replaces the true continuation value V(x) with zero. Every reward
in this repo is a cost (V < 0), so that would make ending the episode strictly
better than continuing — a suicide bonus, which some seeds find and others do
not, i.e. exactly the variance the box exists to remove. On the truncation
channel with ``time_limit_bootstrap: true`` (set in every classic ppo/c2rl-ppo
yaml) skrl adds gamma*V(x_final) back and the agent is indifferent to the cut,
with no terminal penalty to tune. ``terminal=True`` opts into the other
behaviour and says loudly what it costs.
"""

from __future__ import annotations

import torch

from c2rl.agents.skrl.angle_utils import wrap_diff


class TerminationBoxMixin:
    """``set_terminate_out_of_box`` / ``_left_termination_box`` for both families.

    A host class calls :meth:`_init_termination_box` once during ``__init__``
    and then folds :meth:`_left_termination_box` into its own done logic.
    ``self.angle_idx`` must exist by then (both families set it well before).
    """

    # Set by _init_termination_box; declared here so the attributes exist even
    # if a host forgets the call (they then read as "disarmed" rather than
    # raising AttributeError from deep inside a step).
    X_TERMINATION_MIN: torch.Tensor | None = None
    X_TERMINATION_MAX: torch.Tensor | None = None
    terminate_out_of_box: bool = False
    x_termination_terminal: bool = False

    def _init_termination_box(self, lo, hi, *, clamp_box=None, armed: bool = True,
                              terminal: bool = False, tag: str = "Env") -> None:
        """Install the box. ``lo``/``hi`` may be None (host declares no box).

        ``clamp_box`` is the host's own (X_MIN, X_MAX) when its step clamps the
        state into that box before anything else sees it — the termination
        bounds must then lie inside it or they could never be crossed. Pass None
        when the host does no clamping (the Isaac family).
        """
        self._term_tag = tag
        self._term_clamp_box = clamp_box
        self.X_TERMINATION_MIN = None if lo is None else torch.as_tensor(
            lo, device=self.device, dtype=torch.float32).flatten()
        self.X_TERMINATION_MAX = None if hi is None else torch.as_tensor(
            hi, device=self.device, dtype=torch.float32).flatten()
        if (self.X_TERMINATION_MIN is None) != (self.X_TERMINATION_MAX is None):
            raise ValueError(
                f"[{tag}] x_termination_min and x_termination_max must be set "
                "together — one without the other silently never terminates.")
        self.terminate_out_of_box = False
        self.x_termination_terminal = False
        self.set_terminate_out_of_box(armed, terminal=terminal, quiet=True)

    def set_terminate_out_of_box(self, flag: bool, *, terminal: bool = False,
                                 quiet: bool = False) -> None:
        """Arm or disarm the box. See the module docstring for why ``terminal``
        defaults to False and what turning it on costs."""
        flag = bool(flag)
        tag = getattr(self, "_term_tag", "Env")
        if flag:
            if self.X_TERMINATION_MIN is None:
                raise ValueError(
                    f"[{tag}] terminate_out_of_box=True but this env declares no "
                    "x_termination_min/max — it would silently never terminate.")
            clamp = getattr(self, "_term_clamp_box", None)
            if clamp is not None:
                x_min, x_max = clamp
                # The host clamps the state into [X_MIN, X_MAX] before anything
                # else sees it, so a wider termination bound can never be
                # crossed and the whole feature would be a silent no-op.
                if (x_min > self.X_TERMINATION_MIN).any() or (x_max < self.X_TERMINATION_MAX).any():
                    raise ValueError(
                        f"[{tag}] the termination box must lie inside [X_MIN, X_MAX] "
                        "(step() clamps to it, so a wider bound never fires):\n"
                        f"  X_MIN             {x_min.tolist()}\n"
                        f"  X_TERMINATION_MIN {self.X_TERMINATION_MIN.tolist()}\n"
                        f"  X_TERMINATION_MAX {self.X_TERMINATION_MAX.tolist()}\n"
                        f"  X_MAX             {x_max.tolist()}")
        self.terminate_out_of_box = flag
        # Only meaningful while armed. Latching it on a disarmed box would print
        # the warning below for a setting that cannot fire — e.g. the flag pair
        # `--no_terminate_out_of_box --terminate_as_terminal`.
        self.x_termination_terminal = bool(terminal) and flag
        if flag and not quiet:
            # An all-infinite box (most Isaac envs, whose _state_bounds is ±inf)
            # can never be crossed. Say so rather than let it look armed.
            finite = bool(torch.isfinite(self.X_TERMINATION_MIN).any()
                          or torch.isfinite(self.X_TERMINATION_MAX).any())
            if not finite:
                print(f"[{tag}] terminate_out_of_box=True but no finite state bounds "
                      "are declared, so the box is INERT. Override _state_bounds or "
                      "set x_termination_min/max to arm it.")
            else:
                print(f"[{tag}] terminate_out_of_box=True on "
                      f"{'terminated' if self.x_termination_terminal else 'truncated'}: "
                      f"episodes end on leaving {self.X_TERMINATION_MIN.tolist()} .. "
                      f"{self.X_TERMINATION_MAX.tolist()}")
        if self.x_termination_terminal:
            print(f"[{tag}] WARNING: x_termination_terminal=True zeroes the GAE "
                  "bootstrap at the cut. On a cost reward that is a suicide bonus — "
                  "use truncation unless you have added a matching terminal penalty. "
                  "See termination_box.py's module docstring.")

    def _left_termination_box(self, x: torch.Tensor):
        """Per-env bool: did the state just leave the box? ``None`` when
        disarmed, so callers can tell "no excursion" from "not checking".

        Angle dims are wrapped first, so a bound inside (-pi, pi] is comparable
        against the same representation the state is stored in. Non-finite counts
        as out-of-box: a NaN/Inf state is precisely the divergence the guards
        keep alive, and it is the event this ends.
        """
        if not self.terminate_out_of_box:
            return None
        xw = wrap_diff(x, getattr(self, "angle_idx", ()))
        out = (xw < self.X_TERMINATION_MIN) | (xw > self.X_TERMINATION_MAX)
        return out.any(dim=-1) | (~torch.isfinite(x)).any(dim=-1)
