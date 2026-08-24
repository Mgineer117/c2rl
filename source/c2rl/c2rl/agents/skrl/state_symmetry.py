"""Named state dimensions -> the symmetry group of the tracking problem.

Why names instead of index lists
--------------------------------
An env used to declare its structure as two integer facts: ``pos_dimension``
(how many leading dims are position) and ``angle_idx`` (which dims wrap). That
was enough for the translation quotient but not for rotation, and it was easy to
get wrong silently — nothing tied index 2 to "this is a yaw", so a reader (or a
new env) had no way to know which dims must co-rotate.

Instead each env now declares ``state_names``: one name per state dim, e.g.

    car        = ("pos_x", "pos_y", "yaw", "vel")
    turtlebot  = ("pos_x", "pos_y", "yaw")
    segway     = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
    quadrotor  = ("pos_x", "pos_y", "pos_z", "vel_x_w", "vel_y_w", "vel_z_w",
                  "thrust", "roll", "pitch", "yaw")

``pos_dimension`` and ``angle_idx`` are then derived from the names (and kept as
properties, so every existing consumer keeps working).

The frame suffix on velocities is load-bearing
----------------------------------------------
``vel_x_w`` is a world-frame component: rotating the scene rotates it, so it
must be co-rotated with the position pair. ``vel_x_b`` is a body-frame
component: it is already yaw-invariant and must be left alone. Getting this
backwards silently breaks the rotation quotient, so an unrecognised name
disables the quotient rather than being assumed invariant (see
``from_names``) — "I don't know what this dim is" must never be treated as
"this dim is invariant".

The two symmetries
------------------
``translation``  x and xref shifted together along the ``pos_*`` dims.
                 Valid whenever f and B don't depend on absolute position,
                 which holds for every env in this repo (verified numerically).

``se2``          translation + rotation about z. Additionally requires a
                 ``yaw`` dim and a ``pos_x``/``pos_y`` pair, and that every
                 other dim is yaw-invariant. Canonicalizing means rotating the
                 whole scene so the reference yaw is zero, which turns the
                 position error into (longitudinal, lateral) error in the
                 reference frame and drops the absolute heading entirely.

For the car this takes the gain-net input from 10 dims (absolute) to 8
(translation) to 5 (SE(2)): ``(e_long, e_lat, dyaw, v, v_ref)`` — the complete
invariant of the 8-dim (x, xref) pair under a 3-dim symmetry group.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .angle_utils import embed_angles, embedded_dim, wrap_diff

# ── Name vocabulary ──────────────────────────────────────────────────────── #
# World-frame planar position: translates and rotates.
_POS_XY = ("pos_x", "pos_y")
# Translates but does not rotate (the rotation axis).
_POS_Z = ("pos_z",)
# The planar heading. Rotating the scene by phi adds phi to it.
_YAW = "yaw"
# World-frame planar velocity: rotates with position (suffix _w = world frame).
_VEL_XY_W = ("vel_x_w", "vel_y_w")
# Dims that hold a raw wrapping angle and so need (cos, sin) embedding.
_ANGULAR = frozenset({"roll", "pitch", "yaw"})
# Everything below is invariant to both translation and yaw rotation. Body-frame
# quantities (suffix _b), joint states, and projected gravity are yaw-invariant
# by construction; scalar speeds and forces have no planar direction at all.
_INVARIANT_EXACT = frozenset({
    "vel", "vel_z_w", "vel_x_b", "vel_y_b", "vel_z_b",
    "roll", "pitch", "roll_rate", "pitch_rate", "yaw_rate",
    "ang_vel_x_b", "ang_vel_y_b", "ang_vel_z_b",
    "thrust", "force", "height",
})
# Prefixes for repeated/indexed dims (joint_pos_0, proj_grav_z, ...).
_INVARIANT_PREFIXES = ("joint_pos", "joint_vel", "joint_torque", "proj_grav",
                       "gravity_b", "ee_vel_b", "contact", "phase")


def _is_invariant(name: str) -> bool:
    return name in _INVARIANT_EXACT or name.startswith(_INVARIANT_PREFIXES)


@dataclass(frozen=True)
class StateSymmetry:
    """The symmetry the networks may quotient out, plus the feature maps.

    ``mode`` is the only thing callers branch on: "none" reproduces the historic
    absolute-observation behaviour exactly, so an env whose names are not fully
    understood, or whose dynamics fail the numerical equivariance check, keeps
    the previous semantics rather than being mis-modelled.
    """

    x_dim: int
    names: tuple[str, ...]
    mode: str                                   # "none" | "translation" | "se2"
    pos_idx: tuple[int, ...] = ()               # translation directions
    yaw_idx: int | None = None
    rot_pairs: tuple[tuple[int, int], ...] = () # (x, y) index pairs that co-rotate
    angle_idx: tuple[int, ...] = ()

    # ── construction ────────────────────────────────────────────────────── #
    @classmethod
    def from_names(cls, names: Sequence[str], *, allow_se2: bool = True) -> StateSymmetry:
        names = tuple(str(n) for n in names)
        x_dim = len(names)
        angle_idx = tuple(i for i, n in enumerate(names) if n in _ANGULAR)
        pos_idx = tuple(i for i, n in enumerate(names) if n in _POS_XY + _POS_Z)
        if not pos_idx:
            return cls(x_dim, names, "none", angle_idx=angle_idx)

        # Rotation additionally needs a yaw, a full pos_x/pos_y pair, and every
        # remaining dim to be a recognised yaw-invariant name. An unknown name
        # fails closed to translation.
        idx = {n: i for i, n in enumerate(names)}
        has_planar = _YAW in idx and all(n in idx for n in _POS_XY)
        rot_pairs: tuple[tuple[int, int], ...] = ()
        if has_planar:
            rot_pairs = ((idx["pos_x"], idx["pos_y"]),)
            if all(n in idx for n in _VEL_XY_W):
                rot_pairs += ((idx["vel_x_w"], idx["vel_y_w"]),)
        rotating = {i for pair in rot_pairs for i in pair} | ({idx[_YAW]} if has_planar else set())
        unexplained = [names[i] for i in range(x_dim)
                       if i not in rotating and not _is_invariant(names[i])
                       and names[i] not in _POS_Z]
        se2_ok = allow_se2 and has_planar and not unexplained
        return cls(
            x_dim=x_dim, names=names,
            mode="se2" if se2_ok else "translation",
            pos_idx=pos_idx,
            yaw_idx=idx[_YAW] if se2_ok else None,
            rot_pairs=rot_pairs if se2_ok else (),
            angle_idx=angle_idx,
        )

    def demote_to_translation(self) -> StateSymmetry:
        """Drop back to the translation quotient (e.g. the numerical rotation
        equivariance check failed for this env's dynamics)."""
        if self.mode != "se2":
            return self
        return StateSymmetry(self.x_dim, self.names, "translation",
                             pos_idx=self.pos_idx, angle_idx=self.angle_idx)

    def disabled(self) -> StateSymmetry:
        return StateSymmetry(self.x_dim, self.names, "none", angle_idx=self.angle_idx)

    # ── derived legacy views ────────────────────────────────────────────── #
    @property
    def pos_dimension(self) -> int:
        """Back-compat: the historic leading-block position count. Only equal to
        ``len(pos_idx)`` when the position dims really are the leading block —
        which they are in every env here, but the feature maps below use
        ``pos_idx`` directly so they do not depend on it."""
        return len(self.pos_idx)

    # ── index bookkeeping ───────────────────────────────────────────────── #
    def _kept_idx(self) -> tuple[int, ...]:
        """State dims a network sees per block: everything except the position
        dims (pure symmetry directions) and, under SE(2), the absolute yaw
        (which the canonicalization fixes to zero)."""
        drop = set(self.pos_idx)
        if self.mode == "se2":
            drop.add(self.yaw_idx)
        return tuple(i for i in range(self.x_dim) if i not in drop)

    def _kept_angle_idx(self) -> list[int]:
        kept = self._kept_idx()
        return [kept.index(i) for i in self.angle_idx if i in kept]

    # ── single-block features: W(x), NeuralDynamics f/B ─────────────────── #
    def single_dim(self) -> int:
        if self.mode == "none":
            return embedded_dim(self.x_dim, self.angle_idx)
        return embedded_dim(len(self._kept_idx()), self._kept_angle_idx())

    def single_features(self, x: torch.Tensor) -> torch.Tensor:
        """One state block, symmetry directions removed.

        Dropping (not zeroing) the columns is what makes the invariance exact:
        d/d(dropped dim) is identically 0 by construction rather than a small
        number the optimiser has to push down.

        NOTE for SE(2): this drops the absolute yaw without rotating anything,
        so it is a valid map only for quantities that are themselves yaw
        invariant. W(x) is not — a metric transforms as a tensor, W(g.x) =
        dg W(x) dg^T — so the CMG deliberately stays on the translation quotient
        (see contraction_runner._resolve_symmetry).
        """
        if self.mode == "none":
            return embed_angles(x, self.angle_idx)
        kept = list(self._kept_idx())
        return embed_angles(x[..., kept], self._kept_angle_idx())

    # ── pair features: the actor's gain nets and the critics ────────────── #
    def pair_dim(self) -> int:
        if self.mode == "none":
            return 2 * embedded_dim(self.x_dim, self.angle_idx)
        n_rel = len(self.pos_idx) + (1 if self.mode == "se2" else 0)
        return n_rel + 2 * self.single_dim()

    def pair_features(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """Complete invariant of the ``(x, xref)`` pair under the symmetry.

        translation: ``[x_pos - xref_pos, feats(x), feats(xref)]``
        se2:         ``[R(-yaw_ref) (x_pos - xref_pos), wrap(yaw - yaw_ref),
                        feats(R(-yaw_ref) x), feats(R(-yaw_ref) xref)]``

        Lossless in both cases: the dynamics and the tracking reward are
        themselves invariant, so this is a bijection onto the reduced state
        space, not an approximation.
        """
        if self.mode == "none":
            return torch.cat([embed_angles(x, self.angle_idx),
                              embed_angles(xref, self.angle_idx)], dim=-1)
        if self.mode == "translation":
            pos = list(self.pos_idx)
            return torch.cat([x[..., pos] - xref[..., pos],
                              self.single_features(x),
                              self.single_features(xref)], dim=-1)

        # ── SE(2): rotate the whole scene so the reference yaw is zero ──── #
        yaw_ref = xref[..., self.yaw_idx]
        cos, sin = torch.cos(-yaw_ref), torch.sin(-yaw_ref)
        x_r, xref_r = self._rotate(x, cos, sin), self._rotate(xref, cos, sin)
        pos = list(self.pos_idx)
        d_pos = x_r[..., pos] - xref_r[..., pos]
        d_yaw = torch.remainder(
            x[..., self.yaw_idx] - yaw_ref + math.pi, 2 * math.pi) - math.pi
        return torch.cat([d_pos, d_yaw.unsqueeze(-1),
                          self.single_features(x_r),
                          self.single_features(xref_r)], dim=-1)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the planar rotation to every co-rotating (x, y) pair and to yaw.

        ``cos``/``sin`` are per-sample (shape ``x.shape[:-1]``), so each sample is
        rotated into its own reference frame.
        """
        out = x.clone()
        for ix, iy in self.rot_pairs:
            vx, vy = x[..., ix], x[..., iy]
            out[..., ix] = cos * vx - sin * vy
            out[..., iy] = sin * vx + cos * vy
        # yaw is only read through single_features (which drops it) and d_yaw
        # above, but keep it consistent so _rotate is a true group action.
        ang = torch.atan2(sin, cos)
        out[..., self.yaw_idx] = torch.remainder(
            x[..., self.yaw_idx] + ang + math.pi, 2 * math.pi) - math.pi
        return out

    # ── the feedback error vector ───────────────────────────────────────── #
    def error_features(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """The tracking error CLActor's bilinear feedback multiplies, expressed in
        the canonical frame. Full ``x_dim`` width — every state dim still gets
        feedback; only the frame changes.

        Canonicalizing the gain-net inputs is not enough on its own: the control
        law is ``u = W2(feats) tanh(W1(feats) e)``, so if ``e`` stays in world
        coordinates its position components rotate with the scene and ``u`` moves
        even when the features do not. Under SE(2) the position pairs are rotated
        by ``R(-yaw_ref)``, which makes the feedback act on the longitudinal /
        lateral error in the reference frame — the physically meaningful
        parameterisation for a planar vehicle — and the heading term on the
        shortest-angle yaw error.

        ``e == 0`` still implies ``u == uref`` (a rotation is invertible and fixes
        the origin), which is what the contraction certificate needs.
        """
        e = wrap_diff(x - xref, self.angle_idx)
        if self.mode != "se2":
            return e
        yaw_ref = xref[..., self.yaw_idx]
        cos, sin = torch.cos(-yaw_ref), torch.sin(-yaw_ref)
        out = e.clone()
        for ix, iy in self.rot_pairs:
            ex, ey = e[..., ix], e[..., iy]
            out[..., ix] = cos * ex - sin * ey
            out[..., iy] = sin * ex + cos * ey
        return out

    # ── numerical verification against the env's own dynamics ───────────── #
    def verify_translation(self, get_f_and_B, x_min, x_max, *, samples=256,
                           offset=10.0, atol=1e-5) -> bool:
        """f and B must not depend on the position dims at all."""
        try:
            x = self._sample(x_min, x_max, samples)
            shifted = x.clone()
            for i in self.pos_idx:
                shifted[:, i] += (torch.rand(samples, device=x.device) * 2 - 1) * offset
            f1, B1, _ = get_f_and_B(x)
            f2, B2, _ = get_f_and_B(shifted)
            return bool(torch.allclose(f1, f2, atol=atol) and torch.allclose(B1, B2, atol=atol))
        except Exception:
            return False

    def verify_rotation(self, get_f_and_B, x_min, x_max, *, samples=256,
                        atol=1e-4) -> bool:
        """f and B must be equivariant, not invariant: rotating the state must
        rotate f and B's rows the same way.

            f(R.x) = R f(x)      B(R.x) = R B(x)

        This is what fails for the quadrotor, whose roll/pitch are world
        referenced so a yaw rotation mixes them — caught here rather than
        assumed away.
        """
        if self.mode != "se2":
            return False
        try:
            x = self._sample(x_min, x_max, samples)
            phi = (torch.rand(samples, device=x.device) * 2 - 1) * math.pi
            cos, sin = torch.cos(phi), torch.sin(phi)
            f1, B1, _ = get_f_and_B(x)
            f2, B2, _ = get_f_and_B(self._rotate(x, cos, sin))
            # Rotate f (a state-space vector) and B's rows (state-space columns)
            f1r = self._rotate_tangent(f1, cos, sin)
            B1r = torch.stack([self._rotate_tangent(B1[..., j], cos, sin)
                               for j in range(B1.shape[-1])], dim=-1)
            return bool(torch.allclose(f1r, f2, atol=atol) and torch.allclose(B1r, B2, atol=atol))
        except Exception:
            return False

    def _rotate_tangent(self, v: torch.Tensor, cos, sin) -> torch.Tensor:
        """Rotate a tangent vector (f, or a column of B). Same planar pairs as
        the state, but yaw's tangent component is a rate, which is rotation
        invariant — so unlike ``_rotate`` there is no additive yaw shift."""
        out = v.clone()
        for ix, iy in self.rot_pairs:
            vx, vy = v[..., ix], v[..., iy]
            out[..., ix] = cos * vx - sin * vy
            out[..., iy] = sin * vx + cos * vy
        return out

    def _sample(self, x_min, x_max, n) -> torch.Tensor:
        lo = x_min.flatten().float()
        hi = x_max.flatten().float()
        return lo + torch.rand(n, lo.numel(), device=lo.device) * (hi - lo)

    def describe(self) -> str:
        return (f"mode={self.mode} x_dim={self.x_dim} pos={list(self.pos_idx)} "
                f"yaw={self.yaw_idx} rot_pairs={list(self.rot_pairs)} "
                f"angles={list(self.angle_idx)} -> pair_dim={self.pair_dim()} "
                f"single_dim={self.single_dim()}")
