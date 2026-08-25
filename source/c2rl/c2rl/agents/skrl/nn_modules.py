"""Self-contained neural network modules for c2rl.

Provides all network building blocks needed by C3M, SD-LQR, LQR, and C2RL with
no mjrl dependency.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .state_symmetry import StateSymmetry


def _sym_from_names(names):
    """Rebuild the symmetry a checkpoint was trained with, so a reload cannot
    silently change the network input layout."""
    if not names:
        return None
    return StateSymmetry.from_names(names)


from .angle_utils import embed_angles, embedded_dim
from .math_utils import rescale_residual
from .ref_window import Feats, RefWindow

_MIN_LOG_STD = math.log(0.001)  # ≈ -6.908; annealing floor


# ─────────────────────────────────────────────────────────────────────────── #
# MLP
# ─────────────────────────────────────────────────────────────────────────── #

class MLP(nn.Module):
    """Standard feedforward MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        try:
            gain = nn.init.calculate_gain("tanh")
        except Exception:
            gain = 1.0
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(in_d, out_d)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.1)
            layers += [lin, activation]
        self.output_dim = dims[-1]
        if output_dim is not None:
            lin = nn.Linear(dims[-1], output_dim)
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            lin.bias.data.fill_(0.0)
            layers.append(lin)
            self.output_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────── #
# CCM_Generator
# ─────────────────────────────────────────────────────────────────────────── #

class CCM_Generator(nn.Module):
    """Contraction-metric generator W(x) = VᵀV (symmetric PSD by construction).

    In stochastic mode, samples from a diagonal Gaussian over the matrix entries
    and reports entropy for optional regularisation (used in C2RL).
    """
    bounded = False

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list[int],
        activation: str | nn.Module = "tanh",
        mode: str = "stochastic",
        device: str = "cpu",
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.mode = mode
        self.angle_idx = list(angle_idx)
        # Translation quotient: W depends only on the non-position dims (f and B
        # do, so the metric should too). pos_dim=0 keeps the absolute behaviour.
        self.sym = sym

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()]

        # Network sees the continuous (cos, sin) embedding of any angle dims, with
        # the translation directions dropped; W(x) itself is still indexed/shaped
        # by the raw x_dim (see forward).
        self.backbone = MLP((sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx)),
                            list(hidden_dim), activation=activation)
        h = self.backbone.output_dim
        self.mu_head = nn.Linear(h, x_dim * x_dim)
        self.logstd_head = nn.Linear(h, x_dim * x_dim)
        self.to(device)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        n = x.shape[0]
        h = self.backbone(self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        mu = self.mu_head(h)

        if self.mode == "deterministic" or deterministic:
            W_flat = mu
            info = {
                "entropy": torch.zeros(n, 1, device=x.device, dtype=x.dtype),
                "logprobs": torch.zeros(n, 1, device=x.device, dtype=x.dtype),
            }
        else:
            logstd = torch.clamp(self.logstd_head(h), -5, 2)
            dist = Normal(mu, logstd.exp())
            W_flat = dist.rsample()
            info = {
                "entropy": dist.entropy().sum(-1, keepdim=True),
                "logprobs": dist.log_prob(W_flat).sum(-1, keepdim=True),
            }

        W = W_flat.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)  # symmetric PSD: VᵀV
        return W, info


# ─────────────────────────────────────────────────────────────────────────── #
# CholMetric — the original NCM network: x -> chol(M), M = RᵀR
# ─────────────────────────────────────────────────────────────────────────── #

class CholMetric(nn.Module):
    """Tsukamoto's NCM network (``classncm.train`` + ``ncm``/``cholM2M``).

    A plain MLP ``x ↦ vec(R)`` with ``R`` upper-triangular and ``M = RᵀR``, so
    the deployed metric is SPD by construction at every state — including ones
    the SDP never sampled — with no eigenvalue clamp anywhere. That is the whole
    reason the reference regresses ``chol(M)`` rather than ``W``: ``ν`` is free in
    its SDP, so no ``[w_lb, w_ub]`` envelope exists to clamp to, and clamping a
    regressed ``W`` would silently deploy a metric the SDP never certified.

    Deliberately not ``CCM_Generator``: that one outputs ``W`` and relies on
    ``bound_W`` for definiteness, which is the repo's envelope-bearing variant
    (still what C2RL uses). This one is only for CV-STEM-LQR.

    Raw ``x`` in, no feature map and no angle embedding — the reference feeds the
    state vector directly. Angles therefore enter unwrapped; a state box that
    spans ±π has a discontinuity the network must fit through, which is the
    reference's behaviour, not an oversight here.

    Output layer is plain linear, so ``R``'s diagonal may be zero or negative and
    ``M`` is PSD rather than strictly PD. Nothing downstream inverts ``M`` (the
    gain is ``K = R⁻¹BᵀM``), so a singular ``M`` degrades feedback smoothly
    instead of raising.
    """

    def __init__(self, x_dim: int, hidden_dims: Sequence[int] = (128, 128),
                 activation: nn.Module | None = None) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.n_out = self.x_dim * (self.x_dim + 1) // 2
        self.net = MLP(self.x_dim, list(hidden_dims), self.n_out,
                       activation=activation if activation is not None else nn.ReLU())
        iu = torch.triu_indices(self.x_dim, self.x_dim)
        # Same row-major upper-triangular order as ncm_synthesis.M_to_cholvec —
        # the labels and the reconstruction must agree on it.
        self.register_buffer("_iu", iu, persistent=False)

    def chol(self, x: torch.Tensor) -> torch.Tensor:
        """Upper-triangular ``R`` with ``M = RᵀR``, ``(b, x_dim, x_dim)``."""
        vec = self.net(x)
        R = x.new_zeros(x.shape[0], self.x_dim, self.x_dim)
        R[:, self._iu[0], self._iu[1]] = vec
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.chol(x)
        return torch.bmm(R.transpose(1, 2), R)


# ─────────────────────────────────────────────────────────────────────────── #



# ─────────────────────────────────────────────────────────────────────────── #
# Structured preview residuals — π(s) variants for residual RL on the
# CV-STEM-LQR base. Each explicitly separates the roles of the current state x
# (physics / stabilizing gains), the future preview P (path planning), and the
# invariant tracking error e (what the feedback acts on) — versus the original
# CLActor, which blends x, xref and P into one context feeding both W1 and W2.
#
# Contract (all variants): forward(state_feats, e, preview) -> residual (b, u_dim),
# added on top of the analytic base u_base = uref - K·e. state_feats (b, S) are
# the invariant current-config features (CLActor._state_feats), e (b, x_dim) the
# canonical-frame error (CLActor._error_vec), preview (b, P) the future-reference
# tail. All are zero-initialized (W2's output layer) so the actor starts exactly
# at the CV-STEM-LQR base and PPO grows the residual from there. c = 3·x_dim is
# the bilinear latent width (matches CLActor).
# ─────────────────────────────────────────────────────────────────────────── #

def _zero_last_linear(mlp: MLP) -> None:
    """Zero an MLP's final Linear so its output starts at exactly 0."""
    last = None
    for m in mlp.net:
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


# ─────────────────────────────────────────────────────────────────────── #
# Superseded by the single actor architecture  u = urefs[0] + W2(xrefs)·tanh(W1(x)·e)
# (see CLActor below). These were residual heads for warm-starting on an analytic
# CV-STEM-LQR base; kept commented rather than deleted.
# ─────────────────────────────────────────────────────────────────────── #
# class DecoupledResidual(nn.Module):
#     """Option 1 — Decoupled matrices (physical / geometric split):
#
#         π = W2(P) · tanh( W1(x) · e )
#
#     ``W1`` (error → latent) is generated from the current state only — it
#     transforms the error according to the robot's present physical configuration.
#     ``W2`` (latent → control) is generated from the preview only — it allocates
#     control effort for the upcoming path geometry. Physics and path-planning are
#     thus produced by disjoint sub-networks."""
#
#     def __init__(self, x_dim, u_dim, state_dim, preview_dim, hidden, activation):
#         super().__init__()
#         self.x_dim, self.u_dim, self.c = x_dim, u_dim, 3 * x_dim
#         self.w1 = MLP(state_dim, hidden, self.c * x_dim, activation=activation)      # W1(x)
#         self.w2 = MLP(preview_dim, hidden, self.u_dim * self.c, activation=activation)  # W2(P)
#
#     def zero_output(self):
#         """Zero W2's output so π starts at 0 — used only when warm-starting on an
#         analytic base (u = base + π). For the default u = uref + π (learn feedback
#         from scratch) the standard init is kept so the policy starts with feedback."""
#         _zero_last_linear(self.w2)
#
#     def forward(self, state_feats, e, preview, xref_feats=None, uref=None):
#         n = e.shape[0]
#         w1 = self.w1(state_feats).reshape(n, self.c, self.x_dim)
#         w2 = self.w2(preview).reshape(n, self.u_dim, self.c)
#         l1 = torch.tanh(torch.matmul(w1, e.unsqueeze(-1)))
#         return torch.matmul(w2, l1).squeeze(-1)
#
#
# class LatentPreviewResidual(nn.Module):
#     """Option 2 (recommended for generalization) — Latent preview bias:
#
#         π = W2(x) · tanh( W1(x) · e + f_prev(P) )
#
#     ``W1``/``W2`` are strict state-dependent gain schedulers (physics only). The
#     preview enters as an additive bias ``f_prev(P)`` in the latent pre-tanh space,
#     so it preemptively nudges the error signal while remaining bounded by the tanh
#     saturation the state-dependent physical gains set. The preview can never alter
#     the stabilizing gains themselves — only bias what they act on — which is why
#     this generalizes best."""
#
#     def __init__(self, x_dim, u_dim, state_dim, preview_dim, hidden, activation):
#         super().__init__()
#         self.x_dim, self.u_dim, self.c = x_dim, u_dim, 3 * x_dim
#         self.w1 = MLP(state_dim, hidden, self.c * x_dim, activation=activation)   # W1(x)
#         self.w2 = MLP(state_dim, hidden, self.u_dim * self.c, activation=activation)  # W2(x)
#         self.f_prev = MLP(preview_dim, hidden, self.c, activation=activation)     # latent bias
#
#     def zero_output(self):
#         """Zero W2's output (see DecoupledResidual.zero_output)."""
#         _zero_last_linear(self.w2)
#
#     def forward(self, state_feats, e, preview, xref_feats=None, uref=None):
#         n = e.shape[0]
#         w1 = self.w1(state_feats).reshape(n, self.c, self.x_dim)
#         w2 = self.w2(state_feats).reshape(n, self.u_dim, self.c)
#         bias = self.f_prev(preview).unsqueeze(-1)                    # (n, c, 1)
#         l1 = torch.tanh(torch.matmul(w1, e.unsqueeze(-1)) + bias)
#         return torch.matmul(w2, l1).squeeze(-1)
#
#
class PreviewSequenceEncoder(nn.Module):
    """Encodes the preview tail as a sequence of ``num_points`` future
    reference rows (each ``point_dim`` wide), instead of flattening it into
    one vector for a plain MLP. Drop-in replacement for
    ``MLP(preview_dim, hidden, out_dim)`` — same ``(n, preview_dim) -> (n,
    out_dim)`` signature — so it can be swapped in for any FiLMResidual γ-gate
    that reads "preview" without touching FiLMResidual.forward at all.

    "gru": fed farthest point first (the tail is stored nearest-first, so this
    reverses it). The final hidden state is then dominated by the near-term
    point, the far future having already been squashed by the forget gates —
    which is what makes "an RNN's forgetting resembles discounting" literally
    true here. Fed nearest-first the final state would be dominated by the
    farthest point, the opposite of what discounting wants. The encoder is never
    told gamma, unlike env_base's explicit geometric ladder, so any horizon-like
    behavior is purely learned.

    "attn": one learned query attends over all points via softmax
    (order-independent) — lets training discover which offsets matter, with no
    recency prior, as a deliberate contrast to "gru".

    "mlp": flattens the (possibly strided) points into one vector through a
    plain MLP — no order/recency structure at all, a deliberate contrast to
    "gru"/"attn".

    ``stride`` applies uniformly to all three modes (keep every stride-th
    point, nearest-first — offset 0 is never skipped): "mlp" sees fewer,
    wider-spaced points to flatten (bounds its input width, which otherwise
    scales with ``num_points`` and gets wasteful once the tail is a full
    episode trajectory — see ``env_base.BaseEnv._full_trajectory_offsets``);
    "gru"/"attn" see a shorter, cheaper sequence. ``stride=1`` (default)
    keeps every point — the original dense behavior for any mode.
    """

    MODES = ("mlp", "gru", "attn")

    def __init__(self, preview_dim, point_dim, hidden, out_dim, mode="gru", stride: int = 1):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"PreviewSequenceEncoder: mode must be one of {self.MODES}, got {mode!r}")
        if preview_dim % point_dim != 0:
            raise ValueError(f"PreviewSequenceEncoder: preview_dim={preview_dim} is not a "
                             f"multiple of point_dim={point_dim}")
        self.num_points = preview_dim // point_dim
        self.point_dim = point_dim
        self.mode = mode
        self.stride = max(1, int(stride))
        num_kept = len(range(0, self.num_points, self.stride))
        if mode == "gru":
            self.rnn = nn.GRU(input_size=point_dim, hidden_size=hidden, batch_first=True)
            self.proj = nn.Linear(hidden, out_dim)
        elif mode == "attn":
            self.key = nn.Linear(point_dim, hidden)
            self.value = nn.Linear(point_dim, hidden)
            self.query = nn.Parameter(torch.randn(hidden) * (hidden ** -0.5))
            self.proj = nn.Linear(hidden, out_dim)
        else:  # mlp
            self.mlp = MLP(num_kept * point_dim, [hidden], out_dim, activation=nn.Tanh())


    def train(self, mode: bool = True) -> PreviewSequenceEncoder:
        """Keep the GRU submodule permanently in train mode, regardless of
        what the surrounding policy is set to. cuDNN's RNN kernel refuses to
        run backward at all when its own training flag is False (either a
        flat "called in eval mode" error, or — if a caller upstream needs
        create_graph=True — a hard "double backwards not supported"
        NotImplementedError) — and skrl's own act()/update() cycle toggles
        eval()/train() on the whole policy in ways this module has no control
        over (including inside vendored skrl/ code, off-limits to edit). This
        GRU has no dropout, so train vs eval never changes its output — only
        whether cuDNN allows differentiating through it — so forcing it to
        stay in train mode is free."""
        super().train(mode)
        if self.mode == "gru":
            self.rnn.train(True)
        return self

    def forward(self, preview_flat: torch.Tensor) -> torch.Tensor:
        n = preview_flat.shape[0]
        points = preview_flat.reshape(n, self.num_points, self.point_dim)
        if self.stride > 1:
            points = points[:, ::self.stride, :]  # nearest-first, every stride-th point kept
        if self.mode == "mlp":
            return self.mlp(points.reshape(n, -1))
        if self.mode == "gru":
            seq = points.flip(dims=[1])          # farthest-first -> nearest-last
            _, h = self.rnn(seq)
            return self.proj(h.squeeze(0))
        k = self.key(points)                                     # (n, P, hidden)
        v = self.value(points)                                   # (n, P, hidden)
        scores = torch.einsum("h,nph->np", self.query, k) / (k.shape[-1] ** 0.5)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)          # (n, P, 1)
        return self.proj((w * v).sum(dim=1))


# class FiLMResidual(nn.Module):
#     """Option 3 — Preview-modulated gain scheduling (FiLM):
#
#         π = ( W2(x) ⊙ γ2(G2) ) · tanh( ( W1(x) ⊙ γ1(G1) ) · e )
#
#     ``W1(x)``/``W2(x)`` are the baseline stabilizing matrices; ``γ1(G1)``/``γ2(G2)``
#     are positive per-latent scales (softplus) that stiffen or relax the gains.
#     Scaling is element-wise over the latent dimension ``c``. Crucially, ``e``
#     enters multiplicatively inside the tanh (``(W1⊙γ1)·e``), so at e=0 the tanh
#     argument is exactly 0 regardless of the gate inputs — u(e=0)=uref always holds,
#     unlike LatentPreviewResidual's additive pre-tanh bias (see project memory:
#     that design breaks u(e=0)=uref once preview is nonzero → diverges).
#
#     ``gate1_source``/``gate2_source`` independently pick ``G1``/``G2``:
#     "preview" (default, future-Uref window), "xref_preview" (future-Xref-relative
#     window, needs ``xref_preview_dim>0``), "xref" (invariant features of the
#     Current reference point — not raw xref, which would break translation/SE(2)
#     invariance), or "uref" (already frame-safe raw).
#
#     The two preview sources are not separate tensors: ``preview`` is the combined
#     tail construct_state produces ([future-uref] ++ [future-xref-relative]),
#     sliced internally at ``xref_preview_dim`` from the end. 0 (default) means no
#     xref-relative block and "preview" reads the whole tensor.
#
#     ``gate_encoder`` picks how a preview-window gate turns its P-point block
#     into the γ-network's input: "mlp"/"gru"/"attn" (PreviewSequenceEncoder —
#     see that class; ``gate_stride`` applies uniformly to all three there).
#     Ignored for "xref"/"uref" gates, which are single points, not sequences."""
#
#     _GATE_SOURCES = ("preview", "xref_preview", "xref", "uref")
#     _SEQUENCE_SOURCES = ("preview", "xref_preview")
#     _GATE_ENCODERS = PreviewSequenceEncoder.MODES
#
#     def __init__(self, x_dim, u_dim, state_dim, preview_dim, hidden, activation,
#                  xref_feat_dim=None, gate1_source="preview", gate2_source="preview",
#                  gate_encoder="mlp", xref_preview_dim=0, gate_stride=1):
#         super().__init__()
#         self.x_dim, self.u_dim, self.c = x_dim, u_dim, 3 * x_dim
#         for name in (gate1_source, gate2_source):
#             if name not in self._GATE_SOURCES:
#                 raise ValueError(f"FiLMResidual: gate source must be one of "
#                                  f"{self._GATE_SOURCES}, got {name!r}")
#         if gate_encoder not in self._GATE_ENCODERS:
#             raise ValueError(f"FiLMResidual: gate_encoder must be one of "
#                              f"{self._GATE_ENCODERS}, got {gate_encoder!r}")
#         self.gate1_source, self.gate2_source = gate1_source, gate2_source
#         self.gate_encoder = gate_encoder
#         # "preview" reads the first (preview_dim - xref_preview_dim) columns of
#         # the combined tail (the uref block); "xref_preview" reads the last
#         # xref_preview_dim columns (see construct_state's tail ordering).
#         self.xref_preview_dim = int(xref_preview_dim)
#         self.uref_preview_dim = preview_dim - self.xref_preview_dim
#         gate_dims = {"preview": self.uref_preview_dim, "xref_preview": self.xref_preview_dim,
#                      "xref": xref_feat_dim, "uref": u_dim}
#         point_dims = {"preview": u_dim, "xref_preview": x_dim}
#         gate_hidden = hidden[-1] if isinstance(hidden, list) else hidden
#
#         def _make_gate(source):
#             if source in self._SEQUENCE_SOURCES:
#                 return PreviewSequenceEncoder(gate_dims[source], point_dim=point_dims[source],
#                                               hidden=gate_hidden, out_dim=self.c,
#                                               mode=gate_encoder, stride=gate_stride)
#             # "xref"/"uref": a single point, not a sequence — stride is a no-op.
#             return MLP(gate_dims[source], hidden, self.c, activation=activation)
#
#         self.w1 = MLP(state_dim, hidden, self.c * x_dim, activation=activation)   # W1(x)
#         self.w2 = MLP(state_dim, hidden, self.u_dim * self.c, activation=activation)  # W2(x)
#         self.g1 = _make_gate(gate1_source)  # γ1(G1)
#         self.g2 = _make_gate(gate2_source)  # γ2(G2)
#
#     def zero_output(self):
#         """Zero W2's output (see DecoupledResidual.zero_output)."""
#         _zero_last_linear(self.w2)
#
#     def _gate_input(self, source, preview, xref_feats, uref):
#         if source == "preview":
#             return preview[:, : self.uref_preview_dim]
#         if source == "xref_preview":
#             return preview[:, self.uref_preview_dim:]
#         return {"xref": xref_feats, "uref": uref}[source]
#
#     def forward(self, state_feats, e, preview, xref_feats=None, uref=None):
#         n = e.shape[0]
#         w1 = self.w1(state_feats).reshape(n, self.c, self.x_dim)
#         w2 = self.w2(state_feats).reshape(n, self.u_dim, self.c)
#         g1_in = self._gate_input(self.gate1_source, preview, xref_feats, uref)
#         g2_in = self._gate_input(self.gate2_source, preview, xref_feats, uref)
#         g1 = F.softplus(self.g1(g1_in)).unsqueeze(-1)   # (n, c, 1) — scale W1 rows (latent)
#         g2 = F.softplus(self.g2(g2_in)).unsqueeze(-2)   # (n, 1, c) — scale W2 cols (latent)
#         l1 = torch.tanh(torch.matmul(w1 * g1, e.unsqueeze(-1)))
#         return torch.matmul(w2 * g2, l1).squeeze(-1)
#
#
# # Registry the models layer uses to build a variant by name (see
# # models.CLActorModel). "bilinear" (original) and "feedforward" (preview-only,
# # ignores e) are handled directly by CLActor/CLActorModel, not here.
# PREVIEW_RESIDUAL_VARIANTS = {
#     "decoupled": DecoupledResidual,
#     "latent_bias": LatentPreviewResidual,
#     "film": FiLMResidual,
# }


# ─────────────────────────────────────────────────────────────────────────── #
# CLActor
# ─────────────────────────────────────────────────────────────────────────── #

class CLActor(nn.Module):
    """The contracting controller, over a ``{x, xrefs, urefs}`` observation:

        u = urefs[0] + W2(xrefs) @ tanh( W1(x) @ e ),   e = error(x, xrefs[0])

    Roles are disjoint by construction, which is the whole point of the split:

    ``W1(x)``      reads the current configuration only (``Feats.single`` — the
                   symmetry directions dropped). It shapes how the tracking error
                   is weighted according to the present physics.
    ``W2(xrefs)``  reads the reference PATH only, as a sequence of ``length``
                   points each expressed relative to the current ``x``
                   (``Feats.sequence`` — relative position, wrapped angles). It
                   allocates control effort for the geometry that is coming. The
                   sequence goes through ``PreviewSequenceEncoder``, so mlp/gru/
                   attn is a searchable architectural choice.
    ``e``          the canonical-frame tracking error, the only thing the
                   feedback multiplies — so ``e == 0 => u == urefs[0]`` exactly,
                   which is what the contraction certificate requires.

    Differentiable in x, so ``K = du/dx`` feeds the contraction condition
    without a separate Jacobian network.
    """

    def __init__(
        self,
        window: RefWindow,
        feats: Feats,
        anneal_stddev: bool = False,
        hidden_dim: list[int] | None = None,
        activation: nn.Module | str = nn.Tanh(),
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
    ):
        super().__init__()
        self.window = window
        self.feats = feats
        self.x_dim = window.x_dim
        self.u_dim = window.u_dim

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()]
        hidden = list(hidden_dim) if hidden_dim else [128, 128]
        self.c = 3 * self.x_dim          # bilinear latent width

        self.w1 = MLP(feats.single_dim, hidden, self.c * self.x_dim, activation=activation)
        # W2 consumes the window as a sequence of `length` points, each
        # `feats.pair_dim` wide. point_dim is what makes gru/attn see points
        # rather than one flat vector.
        self.w2 = PreviewSequenceEncoder(
            preview_dim=window.length * feats.pair_dim,
            point_dim=feats.pair_dim,
            hidden=encoder_hidden,
            out_dim=self.u_dim * self.c,
            mode=encoder,
            stride=encoder_stride,
        )

        self.anneal = anneal_stddev
        self.logstd = nn.Parameter(torch.zeros(1, self.u_dim), requires_grad=not anneal_stddev)
        self._init_logstd = 0.0

    def anneal_stddev(self, progress: float, mode: str = "exponential") -> None:
        """Anneal log_std from 0 to log(0.001) (≈-6.9) — prevents KL collapse."""
        if not self.anneal:
            return
        progress = float(max(0.0, min(1.0, progress)))
        ratio = progress ** 5.0 if mode == "exponential" else progress
        new_logstd = self._init_logstd * (1.0 - ratio) + _MIN_LOG_STD * ratio
        with torch.no_grad():
            self.logstd.data.fill_(float(max(_MIN_LOG_STD, min(2.0, new_logstd))))

    def trim_state(self, state: torch.Tensor):
        """``(x, xrefs[0], urefs[0])`` — the current triple, for callers that
        only need the present reference (the analytic controllers, the squashed
        wrappers). The full window stays available via ``window.split``."""
        x, xrefs, urefs = self.window.split(state)
        return x, xrefs[:, 0], urefs[:, 0]

    def _feedback(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw (unbounded) feedback ``W2(xrefs) @ tanh(W1(x) @ e)`` and ``urefs[0]``."""
        x, xrefs, urefs = self.window.split(state)
        n = x.shape[0]
        e = self.feats.error(x, xrefs[:, 0]).unsqueeze(-1)          # (n, x_dim, 1)
        seq = self.feats.sequence(x, xrefs).reshape(n, -1)          # (n, L*pair_dim)
        w1 = self.w1(self.feats.single(x)).reshape(n, self.c, self.x_dim)
        w2 = self.w2(seq).reshape(n, self.u_dim, self.c)
        feedback = torch.matmul(w2, torch.tanh(torch.matmul(w1, e))).squeeze(-1)
        return feedback, urefs[:, 0]

    def mean_control(self, state: torch.Tensor) -> torch.Tensor:
        feedback, uref = self._feedback(state)
        return uref + feedback

    def mean_control_squashed(self, state: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Deterministic control law bounded into ``(low, high)`` by construction.

        ``u = urefs[0] + rescale_residual(tanh(feedback), urefs[0], low, high)`` —
        the feedback (not ``uref``) is squashed and ``uref`` added after, so
        ``feedback == 0`` still gives exactly ``u == uref`` (see
        ``math_utils.rescale_residual``). Unlike ``torch.clamp``, ``tanh``'s
        gradient never hits exact zero, so ``K = jacobian(u, x)`` stays
        trainable even where the raw feedback would have saturated — see
        project memory (project_c3m_clip_actions_divergence.md).
        """
        feedback, uref = self._feedback(state)
        action, _ = rescale_residual(torch.tanh(feedback), uref, low, high)
        return action


# ─────────────────────────────────────────────────────────────────────────── #
# NeuralDynamics
# ─────────────────────────────────────────────────────────────────────────── #

_ACT_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class NeuralDynamics(nn.Module):
    """Control-affine neural dynamics  ẋ = f(x) + B(x)·u.

    Trained online by C3MAgent from trajectory buffer transition data and then
    loaded by SDLQRAgent / LQRAgent / C2RLAgent via ``NeuralDynamics.load()``.
    """

    _dtype = torch.float32

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: Sequence[int] = (256, 256),
        activation: str = "relu",
        device: str | None = None,
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.null_dim = x_dim - u_dim
        # f and B provably do not depend on the position dims (that is the very
        # property the quotient rests on), so the nets should not see them.
        self.sym = sym
        self._hidden_dim = list(hidden_dim)
        self._activation_str = activation
        self.device = torch.device(device or "cpu")
        self.angle_idx = list(angle_idx)

        act = _ACT_MAP.get(activation, nn.ReLU)()
        # Nets see the continuous embedding of x's angle dims; f/B are still
        # shaped/indexed by the raw x_dim (outputs are raw-coordinate ẋ / rows).
        emb_dim = (sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx))
        self.f_net = MLP(emb_dim, list(hidden_dim), x_dim, activation=act)
        self.B_net = MLP(emb_dim, list(hidden_dim), x_dim * u_dim, activation=act)
        self.to(self.device)

    def get_f_and_B(self, x: torch.Tensor):
        """Return (f, B, B_null) — autodiff-compatible for C3M Jacobians."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self._dtype).to(self.device)
        x_emb = (self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        f = self.f_net(x_emb)
        B = self.B_net(x_emb).reshape(-1, self.x_dim, self.u_dim)
        B_null = self._compute_B_null(B)
        return f, B, B_null

    def _compute_B_null(self, B: torch.Tensor) -> torch.Tensor:
        n = B.shape[0]
        if self.null_dim <= 0:
            return torch.zeros(n, self.x_dim, 1, device=B.device, dtype=B.dtype)
        with torch.no_grad():
            U, _, _ = torch.linalg.svd(B.detach(), full_matrices=True)
        return U[:, :, self.u_dim:].contiguous()

    def predict_x_dot(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        f, B, _ = self.get_f_and_B(x)
        return f + (B @ u.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x.astype(np.float32)).to(self.device)
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0)
        return self.get_f_and_B(x)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "x_dim": self.x_dim,
                "u_dim": self.u_dim,
                "hidden_dim": self._hidden_dim,
                "activation": self._activation_str,
                "angle_idx": self.angle_idx,
                "state_names": list(getattr(self.sym, "names", ()) or ()),
                "state_dict": self.state_dict(),
            },
            path,
        )
        print(f"[NeuralDynamics] saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> NeuralDynamics:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            x_dim=ckpt["x_dim"],
            u_dim=ckpt["u_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
            device=device,
            angle_idx=ckpt.get("angle_idx", []),
            sym=_sym_from_names(ckpt.get("state_names")),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[NeuralDynamics] loaded ← {path}  (x_dim={ckpt['x_dim']}, u_dim={ckpt['u_dim']})")
        return model

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            result.device = torch.device(args[0])
        return result

class BoundedCCM_Generator(nn.Module):
    """CMG with hard eigenvalue bounds baked into the forward pass.

    ``outputs_metric`` selects which of the two mutually-inverse matrices the
    forward pass produces, and it is set from ``cmg_method``:

      * ``cvstem`` -> ``outputs_metric=True``: the head emits ``M`` directly,
        with eigenvalues squashed into ``[1/w_ub, 1/w_lb]``. The reward needs
        ``e^T M e``, so emitting ``M`` removes a batched SPD inverse from every
        env step (env_base.get_rewards used to call ``spd_inverse(W)`` per step
        per env, on top of the eigh this forward already does). The regression
        target is inverted once offline instead.
      * ``ccm`` -> ``outputs_metric=False``: the head emits ``W``, because the
        C1/C2 contraction losses are written in ``W`` and there is no SDP
        dataset to pre-invert. The reward inverts per step, as before.

    Fitting ``M`` directly is also better aligned with its use: inverting a
    fitted ``W`` amplifies relative error exactly where ``W`` is smallest, which
    is the stiff, weakly-actuated region the certificate cares most about.
    """
    bounded = True

    def __init__(
        self,
        x_dim: int,
        hidden_dim: list,
        activation: str | nn.Module = "tanh",
        mode: str = "deterministic",
        w_lb: float = 0.1,
        w_ub: float = 10.0,
        device: str = "cpu",
        angle_idx: Sequence[int] = (),
        sym: StateSymmetry | None = None,
        outputs_metric: bool = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.mode = mode
        self.device = device
        self.w_lb = w_lb
        self.w_ub = w_ub
        self.outputs_metric = bool(outputs_metric)
        # The eigenvalue box the forward pass squashes into. For M the envelope
        # simply inverts: w_lb <= eig(W) <= w_ub  <=>  1/w_ub <= eig(M) <= 1/w_lb.
        self._eig_lo = (1.0 / w_ub) if self.outputs_metric else w_lb
        self._eig_hi = (1.0 / w_lb) if self.outputs_metric else w_ub
        self.angle_idx = list(angle_idx)
        self.sym = sym

        if isinstance(activation, str):
            activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}.get(
                activation.lower(), nn.Tanh()
            )
        # Network sees the continuous embedding minus the translation directions;
        # W(x) itself stays x_dim x x_dim.
        self.model = MLP(
            input_dim=(sym.single_dim() if sym is not None else embedded_dim(x_dim, self.angle_idx)),
            hidden_dims=hidden_dim, activation=activation,
        )

        self.model.to(device)

        out_dim = x_dim * x_dim
        self.mu = nn.Linear(self.model.output_dim, out_dim).to(device)
        self.logstd = nn.Linear(self.model.output_dim, out_dim).to(device)

    def _to_bounded_spd(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape → symmetrise → sigmoid-on-eigenvalues → bounded SPD."""
        n = flat.shape[0]
        S_raw = flat.view(n, self.x_dim, self.x_dim)
        S = 0.5 * (S_raw + S_raw.mT)           # symmetric; eigenvalues span ℝ
        lam, V = self._robust_eigh(S)
        lam = self._eig_lo + (self._eig_hi - self._eig_lo) * torch.sigmoid(lam)
        return V @ torch.diag_embed(lam) @ V.mT  # SPD, λ ∈ (_eig_lo, _eig_hi)

    @staticmethod
    def _robust_eigh(S: torch.Tensor):
        """``torch.linalg.eigh`` with a jitter-retry fallback. eigh can raise
        LinAlgError (code 3) on a symmetric matrix with (near-)degenerate
        eigenvalues — sporadic during a long CMG regression as the raw network
        output momentarily produces repeated eigenvalues. Adding a tiny diagonal
        jitter breaks the degeneracy without meaningfully moving the bounded
        eigenvalues (they pass through a sigmoid), turning a hard crash into a
        negligible numerical nudge. Falls back to CPU (mps lacks eigh).

        Also sanitizes actual non-finite entries (NaN/Inf) in ``S`` before
        decomposing: a raw ``mu`` head can overflow float32 for one unlucky
        batch during a long regression (the training loop's grad-norm clip
        gates the optimizer step, not the forward pass, so one bad batch's
        activations can still overflow even though no bad weight update ever
        lands). Both eigh and its SVD last-resort raise LinAlgError on
        non-finite input ("failed to converge") rather than just misordering
        eigenvalues, so without this the fallback chain above is defeated by
        the exact failure mode it exists to survive."""
        on_cpu = S.device.type == "mps"
        work = S.cpu() if on_cpu else S
        if not torch.isfinite(work).all():
            work = torch.nan_to_num(work, nan=0.0, posinf=1e4, neginf=-1e4)
        eye = torch.eye(work.shape[-1], device=work.device, dtype=work.dtype)
        for jitter in (0.0, 1e-6, 1e-4, 1e-2):
            try:
                lam, V = torch.linalg.eigh(work + jitter * eye)
                if on_cpu:
                    lam, V = lam.to(S.device), V.to(S.device)
                return lam, V
            except torch._C._LinAlgError:
                continue
        # Last resort: symmetric eigendecomposition via SVD (no convergence
        # failure mode), sign-corrected — always succeeds on a real symmetric S.
        U, s, _ = torch.linalg.svd(work + 1e-2 * eye)
        lam = s
        if on_cpu:
            lam, U = lam.to(S.device), U.to(S.device)
        return lam, U

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        logits = self.model(self.sym.single_features(x) if self.sym is not None else embed_angles(x, self.angle_idx))
        mu = self.mu(logits)

        # Return-dict keys mirror CCM_Generator so the two are drop-in compatible.
        if self.mode == "deterministic" or deterministic:
            W = self._to_bounded_spd(mu)
            logprobs = torch.zeros(x.shape[0], 1, device=x.device)
            return W, {
                "dist": None,
                "probs": torch.ones_like(logprobs),
                "logprobs": logprobs,
                "entropy": torch.zeros_like(logprobs),
            }

        logstd = self.logstd(logits).clamp(-5, 2)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        sample = dist.rsample()
        W = self._to_bounded_spd(sample)
        logprobs = dist.log_prob(sample).sum(-1, keepdim=True)
        return W, {
            "dist": dist,
            "probs": torch.exp(logprobs),
            "logprobs": logprobs,
            "entropy": dist.entropy().sum(-1, keepdim=True),
        }


