"""Pure-PyTorch math utilities for contraction RL algorithms.

These standalone functions provide the Jacobian computations, positive-definiteness
losses, and metric utilities needed by C3M, SD-LQR, LQR, and C2RL — with no mjrl
dependency.
"""
from __future__ import annotations

import torch
from torch.autograd import grad


def jacobian(f: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Batched autograd Jacobian ∂f/∂x → (n, f_dim, x_dim)."""
    n, f_dim, x_dim = x.shape[0], f.shape[-1], x.shape[-1]
    if not f.requires_grad:
        return torch.zeros((n, f_dim, x_dim), device=x.device, dtype=x.dtype)
    J = torch.zeros(n, f_dim, x_dim, device=x.device, dtype=x.dtype)
    for i in range(f_dim):
        g = grad(f[:, i].sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            J[:, i, :] = g
    return J


def b_jacobian(B: torch.Tensor, x: torch.Tensor, u_dim: int, create_graph: bool = True) -> torch.Tensor:
    """Jacobian of each column of B: ∂B_j/∂x → (n, x_dim, x_dim, u_dim)."""
    n, x_dim, _ = B.shape
    B_flat = B.reshape(n, x_dim * u_dim)
    J_flat = jacobian(B_flat, x, create_graph=create_graph)
    J = J_flat.reshape(n, x_dim, u_dim, x_dim)
    return J.transpose(2, 3)


def Jacobian_Matrix(M: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n, device=x.device, dtype=x.dtype)
    for i in range(m):
        for j in range(m):
            g = grad(M[:, i, j].sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                J[:, i, j, :] = g
    return J


def weighted_gradients(W: torch.Tensor, v: torch.Tensor, x: torch.Tensor,
                       detach: bool = False, create_graph: bool = True) -> torch.Tensor:
    """Material derivative Ẇ = Σᵢ (∂W/∂xᵢ) vᵢ → (n, d, d)."""
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x, create_graph).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x, create_graph) * v.view(bs, 1, 1, -1)).sum(dim=3)


_RESCALE_EPS = 1e-6  # keeps half-scales/atanh args away from 0 / +-1 (log(0)/atanh divergence)


def rescale_residual(
    tanh_u: torch.Tensor, residual: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(-1, 1) -> (low, high) via an asymmetric, residual-centered rescale.

    ``tanh_u >= 0`` maps into ``[0, high - residual]``, ``tanh_u < 0`` into
    ``[-(residual - low), 0]``, using a different per-sample half-width for
    each side (``residual`` varies per state, so these half-widths do too).
    Guarantees, for every ``residual`` in ``[low, high]``: ``tanh_u == 0 =>
    action == residual`` exactly (feedback=0 preserves the reference law),
    and ``action`` lands in ``(low, high)`` for every ``tanh_u``, with no
    post-hoc clamping ever required (unlike a fixed-scale rescale-then-add,
    which can push the sum outside the bounds whenever ``residual`` is
    off-center). Shared by ``_TanhSquashMixin`` (SAC's stochastic squash +
    log_prob correction) and ``CLActor.mean_control_squashed`` (C3M's
    deterministic certified control law) — this formula is easy to get
    subtly wrong, so it lives in exactly one place.

    Returns ``(action, scale)`` — ``scale`` is the per-sample, per-side
    half-width used (needed by callers computing a log-det correction).
    """
    s_hi = torch.clamp(high - residual, min=_RESCALE_EPS)
    s_lo = torch.clamp(residual - low, min=_RESCALE_EPS)
    scale = torch.where(tanh_u >= 0, s_hi, s_lo)
    return residual + tanh_u * scale, scale


def loss_pos_matrix_random_sampling(A: torch.Tensor, num_samples: int = 1024) -> torch.Tensor:
    """PD-violation hinge loss via random directional sampling (no eigendecomposition).

    Projects ``A`` onto ``num_samples`` random unit vectors and penalizes only the
    directions where ``zᵀAz < 0``. Unlike an exact eigenvalue loss, this never
    differentiates through ``eigh``/``eigvalsh``, so it stays numerically stable
    even when double-differentiated (as happens via ``weighted_gradients``).
    """
    device, dtype = A.device, A.dtype
    d = A.size(-1)
    z = torch.randn(num_samples, d, device=device, dtype=dtype)
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1, num_samples, -1)).sum(dim=2).reshape(-1)
    negative = zTAz < 0
    if negative.any():
        return -zTAz[negative].mean()
    return torch.zeros((), device=device, dtype=dtype, requires_grad=True)


def loss_pos_matrix_eigen(A: torch.Tensor, reg: bool = True):
    """PD loss via eigenvalues: relu(-λ).sum() → (loss, reg_loss)."""
    device = A.device
    if device.type == "mps":
        lambdas = torch.linalg.eigvalsh(A.cpu()).to(device)
    else:
        lambdas = torch.linalg.eigvalsh(A)
    loss_eigen = torch.relu(-lambdas).sum(dim=-1).mean()
    loss_reg = torch.zeros(1, device=device, dtype=A.dtype)
    if reg:
        loss_reg = loss_reg + torch.relu(lambdas[:, -1] - 500).mean()
    return loss_eigen, loss_reg


def build_lr_scheduler(optimizer: torch.optim.Optimizer, name: str, kwargs: dict | None = None):
    """Instantiate a ``torch.optim.lr_scheduler`` by class name, or ``None`` if ``name`` is falsy.

    Shared by every pretraining loop that anneals an Adam LR by epoch (C2RL's
    NeuralDynamics fit in ``dynamics_pretrain.py`` and CMG regression in
    ``ncm_synthesis.regress_cmg``) so they build schedulers the same way — each
    still configures its own ``name``/``kwargs`` independently, this just avoids
    duplicating the ``getattr(torch.optim.lr_scheduler, ...)`` lookup.
    """
    if not name:
        return None
    sched_cls = getattr(torch.optim.lr_scheduler, name)
    return sched_cls(optimizer, **(kwargs or {}))


def bound_W(raw_W: torch.Tensor, w_lb: float, x_dim: int, bounded: bool = False) -> torch.Tensor:
    """Add w_lb * I to raw CCM output to ensure strict positive definiteness."""
    if bounded:
        return raw_W
    I = torch.eye(x_dim, device=raw_W.device, dtype=raw_W.dtype)
    return raw_W + w_lb * I


def train_val_split(n: int, val_frac: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ``(train_idx, val_idx)`` index split of ``n`` samples.

    ``val_frac<=0`` returns an empty ``val_idx`` and every index in
    ``train_idx`` (validation/early-stopping disabled). Shared by
    ``dynamics_pretrain.pretrain_dynamics`` and ``ncm_synthesis.regress_cmg`` so
    both hold out their fixed pretraining buffer the same way.
    """
    perm = torch.randperm(n, device=device)
    n_val = int(n * val_frac) if val_frac > 0 else 0
    return perm[n_val:], perm[:n_val]


class EarlyStopper:
    """Tracks a held-out validation loss across epochs and signals when to stop.

    Shared by ``dynamics_pretrain.pretrain_dynamics`` (NeuralDynamics fit) and
    ``ncm_synthesis.regress_cmg`` (CMG regression) — both fit a fixed,
    once-sampled buffer for a configured epoch budget, but with no held-out
    signal that budget is just a guess: too many epochs overfits the buffer,
    too few undershoots it. Holding out a validation split and stopping once
    its loss stops improving lets the epoch budget be an upper bound instead
    of the actual stopping point.

    ``patience`` is the number of consecutive non-improving epochs tolerated
    before stopping (``<=0`` disables early stopping — ``step`` always returns
    ``False`` and the caller runs the full configured epoch count).
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.num_bad_epochs = 0

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Record ``val_loss`` for ``epoch``; returns True if training should stop now."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.patience > 0 and self.num_bad_epochs >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        """Load back the state_dict from the best (lowest val_loss) epoch seen."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def spd_inverse(W: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of a batch of SPD matrices via Cholesky.

    ``W`` is SPD by construction (``VᵀV + w_lb·I`` or eigenvalue-bounded), so a
    Cholesky solve is both faster and more stable than the general LU-based
    ``torch.linalg.solve`` — and it won't silently return garbage if ``W`` drifts
    ill-conditioned.
    """
    n, d, _ = W.shape
    I = torch.eye(d, device=W.device, dtype=W.dtype).expand(n, d, d).contiguous()
    L = torch.linalg.cholesky(W)
    return torch.cholesky_solve(I, L)
