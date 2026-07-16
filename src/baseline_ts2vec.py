"""
Baseline: TS2Vec (Yue et al., AAAI 2022) -- simplified, faithful implementation
==================================================================================
A general-purpose self-supervised time-series representation learning
baseline, applied to the SAME per-asset price windows used by
HyperAlign-Fin (no GAF, no hypergraph -- just the raw series), for a
fair "does our theory-driven architecture beat an established
general-purpose method" comparison, which every Q1 reviewer will ask
for.

Documented simplifications from the original paper (not hidden):
  - Timestamp MASKING only as the augmentation (not random cropping),
    since T=20 is short and crop-alignment adds complexity without
    much benefit at this length. Masking is one of the two
    augmentations used in the original paper itself.
  - Single-scale contrastive loss (temporal + instance-wise), not the
    full hierarchical multi-scale pooling pyramid -- T=20 is too short
    for multiple pooling levels to be meaningful.

Reference: Yue, Z. et al. "TS2Vec: Towards Universal Representation of
Time Series." AAAI 2022.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 1. Encoder: dilated causal-ish CNN (TS2Vec-style)
# ================================================================

class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class TS2VecEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, repr_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.input_fc = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, dilation=2 ** i) for i in range(n_layers)
        ])
        self.output_fc = nn.Linear(hidden_dim, repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) raw series -> (B, T, repr_dim) per-timestep representation
        h = self.input_fc(x.unsqueeze(-1))     # (B, T, hidden)
        h = h.transpose(1, 2)                   # (B, hidden, T)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)                   # (B, T, hidden)
        return self.output_fc(h)                # (B, T, repr_dim)


# ================================================================
# 2. Augmentation + hierarchical contrastive loss
# ================================================================

def mask_timestamps(x: torch.Tensor, mask_prob: float = 0.3) -> torch.Tensor:
    """x: (B, T). Returns a masked copy (Bernoulli, zeroed) -- one of
    TS2Vec's two original augmentations."""
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


def hierarchical_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                                   temperature: float = 0.1) -> torch.Tensor:
    """
    z1, z2: (B, T, D) representations of two augmented views of the
    same batch of series. Combines TS2Vec's two contrastive terms:
        - instance-wise: at a fixed timestep, same series (across
          views) is positive, other series in the batch are negative.
        - temporal: within a series, same timestep (across views) is
          positive, other timesteps are negative.
    """
    B, T, D = z1.shape
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)

    # instance-wise (per timestep, contrast across batch)
    z1t = z1n.transpose(0, 1)   # (T, B, D)
    z2t = z2n.transpose(0, 1)
    logits_inst = torch.bmm(z1t, z2t.transpose(1, 2)) / temperature   # (T, B, B)
    targets_inst = torch.arange(B, device=z1.device).unsqueeze(0).expand(T, B)
    inst_loss = F.cross_entropy(logits_inst.reshape(T * B, B), targets_inst.reshape(-1))

    # temporal (per series, contrast across time)
    logits_temp = torch.bmm(z1n, z2n.transpose(1, 2)) / temperature    # (B, T, T)
    targets_temp = torch.arange(T, device=z1.device).unsqueeze(0).expand(B, T)
    temp_loss = F.cross_entropy(logits_temp.reshape(B * T, T), targets_temp.reshape(-1))

    return 0.5 * (inst_loss + temp_loss)


# ================================================================
# 3. Training + embedding extraction
# ================================================================

def _prep_series(price_window: torch.Tensor) -> torch.Tensor:
    """
    price_window: (B, T, N) raw prices.
    Returns (B*N, T): per-asset, per-window normalized log-price series
    (z-scored within the window, matching Definition 1's normalization
    style so the baseline sees comparably-scaled input, not raw prices
    at wildly different levels).
    """
    B, T, N = price_window.shape
    logp = torch.log(price_window + 1e-8)
    logp = (logp - logp.mean(dim=1, keepdim=True)) / (logp.std(dim=1, keepdim=True) + 1e-8)
    return logp.permute(0, 2, 1).reshape(B * N, T)


def pretrain_ts2vec(loader, seed: int, epochs: int = 30, lr: float = 1e-3,
                     hidden_dim: int = 64, repr_dim: int = 128,
                     mask_prob: float = 0.3, device: str = "cpu",
                     verbose: bool = True) -> TS2VecEncoder:
    """Trains one TS2Vec encoder, pooling ALL assets as independent series."""
    torch.manual_seed(seed)
    encoder = TS2VecEncoder(hidden_dim=hidden_dim, repr_dim=repr_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            series = _prep_series(batch["price_window"].to(device))
            view1 = mask_timestamps(series, mask_prob)
            view2 = mask_timestamps(series, mask_prob)
            z1 = encoder(view1)
            z2 = encoder(view2)
            loss = hierarchical_contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and (epoch == 0 or epoch == epochs - 1 or epoch % max(epochs // 5, 1) == 0):
            print(f"    [ts2vec] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    encoder.eval()
    if verbose:
        drop = loss_history[0] - loss_history[-1]
        print(f"    [ts2vec] loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}  "
              f"(drop={drop:.4f}){'  <-- WARNING: barely moved, check hyperparameters' if drop < 0.1 else ''}")
    encoder.loss_history = loss_history  # attached for external diagnostics
    return encoder


def check_embedding_health(encoder: TS2VecEncoder, loader, device: str) -> dict:
    """
    Diagnostic: is the TS2Vec representation actually varying across
    samples, or has it collapsed to a near-constant vector (which would
    make ANY downstream probe fail regardless of the label, and should
    NOT be reported as "TS2Vec loses" without this check)?
    """
    import math
    all_emb = []
    for batch in loader:
        emb = extract_ts2vec_embedding(encoder, batch["price_window"], device)
        B, N, d = emb.shape
        all_emb.append(emb.reshape(B * N, d).cpu())
        if len(all_emb) >= 20:  # enough for a quick diagnostic, no need for the full split
            break
    Z = torch.cat(all_emb, dim=0)
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z_centered)
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 1e-12]
    eff_rank = float(math.exp(-(p * p.log()).sum().item()))
    per_dim_var = Z.var(dim=0)
    dead_dims = int((per_dim_var < 1e-6).sum().item())
    result = {
        "effective_rank": eff_rank, "max_rank": min(Z.shape),
        "dead_dims": dead_dims, "total_dims": Z.shape[1],
        "mean_pairwise_std": float(Z.std(dim=0).mean()),
    }
    collapsed = eff_rank < 0.05 * min(Z.shape) or dead_dims > 0.5 * Z.shape[1]
    print(f"    [ts2vec health] effective_rank={eff_rank:.1f}/{min(Z.shape)}  "
          f"dead_dims={dead_dims}/{Z.shape[1]}  "
          f"{'*** COLLAPSED -- results below are not trustworthy ***' if collapsed else '(looks healthy)'}")
    result["collapsed"] = collapsed
    return result


@torch.no_grad()
def extract_ts2vec_embedding(encoder: TS2VecEncoder, price_window: torch.Tensor,
                              device: str) -> torch.Tensor:
    """
    price_window: (B, T, N) -> per-asset pooled embedding (B, N, repr_dim),
    via max-pooling over time (the standard TS2Vec instance-level
    representation for downstream tasks).
    """
    B, T, N = price_window.shape
    series = _prep_series(price_window.to(device))
    z = encoder(series)                 # (B*N, T, repr_dim)
    pooled = z.max(dim=1).values         # (B*N, repr_dim)
    return pooled.reshape(B, N, -1)


# ================================================================
# 4. Smoke test
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, N = 4, 20, 15
    price_window = torch.cumprod(1 + 0.01 * torch.randn(B, T, N), dim=1) * 100

    encoder = TS2VecEncoder(hidden_dim=32, repr_dim=64, n_layers=3)
    series = _prep_series(price_window)
    print(f"series: {tuple(series.shape)}")

    view1 = mask_timestamps(series, 0.3)
    view2 = mask_timestamps(series, 0.3)
    z1 = encoder(view1)
    z2 = encoder(view2)
    print(f"z1: {tuple(z1.shape)}  z2: {tuple(z2.shape)}")

    loss = hierarchical_contrastive_loss(z1, z2)
    print(f"loss: {loss.item():.4f}")

    emb = extract_ts2vec_embedding(encoder, price_window, device="cpu")
    print(f"pooled embedding: {tuple(emb.shape)}")
