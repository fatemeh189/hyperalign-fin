"""
Baseline: FactorCL-inspired shared/unique multi-view fusion
================================================================
A simplified, faithfully-motivated adaptation of the core idea in
Liang et al. (2023), "Factorized Contrastive Learning: Going Beyond
Multi-view Redundancy" (NeurIPS): rather than assuming all
task-relevant information lives in what is SHARED between two views
(as standard contrastive multi-view learning does), explicitly
factorize each view's representation into a shared component (aligned
across views) and a unique component (decorrelated from the shared
component), so that both redundant and view-unique information are
preserved and available to downstream tasks.

Documented simplification (not hidden): the original FactorCL uses a
task-relevant information-theoretic factorization based on labeled
downstream data. Since our setting is self-supervised (no downstream
label at representation-learning time, matching HyperAlign-Fin's own
protocol for a fair comparison), we adapt the core structural idea --
explicit shared vs. unique factorization -- using a self-supervised
implementation: shared components are aligned via InfoNCE (as in
standard multi-view contrastive learning), and unique components are
trained to be decorrelated from the shared components via a
Barlow-Twins-style cross-covariance penalty (Zbontar et al., 2021),
a standard, well-established redundancy-reduction technique. This is
a FactorCL-INSPIRED baseline, not a literal reimplementation of the
original paper's exact estimators, and is labeled as such throughout.

This baseline reuses the SAME visual (GAF-CNN) and graph (HGNN)
encoders as HyperAlign-Fin (via hyperalign_model.py), so that the
comparison isolates the fusion/alignment MECHANISM, not the input
representations -- a stricter, more informative test than comparing
against TS2Vec (which lacks any graph branch at all).

Usage: see run_factorcl_comparison.py
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedUniqueHeads(nn.Module):
    """Splits a latent_dim embedding into shared and unique sub-spaces."""

    def __init__(self, latent_dim: int, shared_dim: int = 64):
        super().__init__()
        self.shared_dim = shared_dim
        self.unique_dim = latent_dim - shared_dim
        assert self.unique_dim > 0, "shared_dim must be < latent_dim"
        self.shared_proj = nn.Linear(latent_dim, shared_dim)
        self.unique_proj = nn.Linear(latent_dim, self.unique_dim)

    def forward(self, x: torch.Tensor):
        return self.shared_proj(x), self.unique_proj(x)


def info_nce_shared(shared_v: torch.Tensor, shared_g: torch.Tensor,
                     temperature: float = 0.1) -> torch.Tensor:
    """
    Standard InfoNCE between the SHARED components of the two views
    (asset-to-asset matching within a window, same structure as
    HyperAlign-Fin's alignment loss, for a like-for-like comparison
    of the fusion mechanism alone).
    shared_v, shared_g: (B, N, shared_dim)
    """
    B, N, d = shared_v.shape
    zv = F.normalize(shared_v, dim=-1)
    zg = F.normalize(shared_g, dim=-1)
    logits = torch.einsum("bnd,bmd->bnm", zv, zg) / temperature
    targets = torch.arange(N, device=logits.device).unsqueeze(0).expand(B, N)
    loss_v2g = F.cross_entropy(logits.reshape(B * N, N), targets.reshape(-1))
    loss_g2v = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), targets.reshape(-1))
    return 0.5 * (loss_v2g + loss_g2v)


def barlow_twins_decorrelation(unique: torch.Tensor, shared: torch.Tensor,
                                 lambda_offdiag: float = 5e-3) -> torch.Tensor:
    """
    Barlow-Twins-style redundancy-reduction penalty (Zbontar et al.,
    2021), repurposed here to decorrelate a UNIQUE representation from
    the corresponding SHARED representation, rather than decorrelating
    two augmented views of the same input as in the original paper.
    Encourages the cross-covariance matrix between unique and shared
    features toward the identity being penalized (i.e., toward zero
    cross-correlation), so unique captures information the shared
    component does not.

    unique: (M, d_u), shared: (M, d_s), M = flattened batch*assets
    """
    unique = (unique - unique.mean(dim=0)) / (unique.std(dim=0) + 1e-8)
    shared = (shared - shared.mean(dim=0)) / (shared.std(dim=0) + 1e-8)
    M = unique.shape[0]
    cross_cov = (unique.T @ shared) / M  # (d_u, d_s)
    # penalize ALL entries toward zero (unique should share nothing with shared)
    return lambda_offdiag * (cross_cov ** 2).sum()


class FactorCLFusion(nn.Module):
    """
    Replaces HyperAlign-Fin's RegimeAwareAlignment + GatedFusion with a
    FactorCL-inspired shared/unique factorization. Produces a fused
    representation Z = concat(shared_avg, unique_V, unique_G) followed
    by a linear projection back to latent_dim, for direct comparability
    with HyperAlign-Fin's Z in downstream evaluation.
    """

    def __init__(self, latent_dim: int, shared_dim: int = 64):
        super().__init__()
        self.heads_v = SharedUniqueHeads(latent_dim, shared_dim)
        self.heads_g = SharedUniqueHeads(latent_dim, shared_dim)
        unique_dim = latent_dim - shared_dim
        self.fuse = nn.Linear(shared_dim + 2 * unique_dim, latent_dim)

    def forward(self, V: torch.Tensor, G: torch.Tensor):
        """V, G: (B, N, latent_dim). Returns Z, and the loss components
        needed by the training loop (shared_v, shared_g, unique_v, unique_g)."""
        shared_v, unique_v = self.heads_v(V)
        shared_g, unique_g = self.heads_g(G)
        shared_avg = 0.5 * (shared_v + shared_g)
        Z = self.fuse(torch.cat([shared_avg, unique_v, unique_g], dim=-1))
        return Z, shared_v, shared_g, unique_v, unique_g


def factorcl_loss(shared_v: torch.Tensor, shared_g: torch.Tensor,
                   unique_v: torch.Tensor, unique_g: torch.Tensor,
                   temperature: float = 0.1, lambda_decorr: float = 5e-3) -> torch.Tensor:
    """
    Total FactorCL-inspired loss:
        alignment loss on shared components (InfoNCE)
      + decorrelation penalty: unique_v vs shared_v, unique_g vs shared_g
    """
    B, N, d = shared_v.shape
    align_loss = info_nce_shared(shared_v, shared_g, temperature=temperature)

    sv_flat = shared_v.reshape(B * N, -1)
    sg_flat = shared_g.reshape(B * N, -1)
    uv_flat = unique_v.reshape(B * N, -1)
    ug_flat = unique_g.reshape(B * N, -1)

    decorr_loss = (
        barlow_twins_decorrelation(uv_flat, sv_flat, lambda_decorr)
        + barlow_twins_decorrelation(ug_flat, sg_flat, lambda_decorr)
    )
    return align_loss + decorr_loss


# ================================================================
# Smoke test
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, d = 4, 32, 128

    V = torch.randn(B, N, d)
    G = torch.randn(B, N, d)

    fusion = FactorCLFusion(latent_dim=d, shared_dim=64)
    Z, shared_v, shared_g, unique_v, unique_g = fusion(V, G)
    loss = factorcl_loss(shared_v, shared_g, unique_v, unique_g)

    print(f"Z: {tuple(Z.shape)}")
    print(f"shared_v: {tuple(shared_v.shape)}  unique_v: {tuple(unique_v.shape)}")
    print(f"loss: {loss.item():.4f}")

    loss.backward()
    print("backward() OK -- gradients flow correctly")
