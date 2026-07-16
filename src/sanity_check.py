"""
Post-training sanity check
=============================
A decreasing loss curve alone does NOT prove the model learned useful
structure -- the most common silent failure mode in contrastive /
self-supervised training is EMBEDDING COLLAPSE (all embeddings converge
to a near-constant vector, which can still produce a low InfoNCE loss
if the positive/negative structure happens to align trivially with a
degenerate solution).

This script checks, on the BEST checkpoint from train_hyperalign.py:
    1. Final val_loss vs. the random-init baseline (ln(N))
    2. Effective rank of Z (via singular values) -- low effective rank
       means the embedding space has collapsed to a low-dimensional
       (or single-point) subspace, regardless of what the loss says.
    3. Per-dimension variance of Z -- near-zero variance in most
       dimensions is another collapse signature.

Usage:
    python sanity_check.py --checkpoint ./run1/best_checkpoint.pt \
        --data_dir ./cache --n_assets 119 --seq_len 20
"""

from __future__ import annotations
import argparse
import math

import numpy as np
import torch

from dataset import make_loader
from hyperalign_model import (
    HyperAlignFin, hyperalign_loss, build_node_features, build_same_group_mask,
)


def effective_rank(Z: torch.Tensor) -> float:
    """
    Effective rank via entropy of normalized singular values (Roy & Vetterli, 2007).
    Z: (M, d) flattened embeddings across a batch of assets/windows.
    Returns a value in [1, min(M,d)]; close to 1 => collapsed, close to
    min(M,d) => full-rank / healthy spread.
    """
    Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum().item()
    return float(math.exp(entropy))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="dir with val.pt")
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rho_star = ckpt["rho_star"]

    model = HyperAlignFin(
        seq_len=args.seq_len, n_assets=args.n_assets, rho_star=rho_star,
        latent_dim=args.latent_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    import os
    val_loader = make_loader(os.path.join(args.data_dir, "val.pt"),
                              batch_size=args.batch_size, shuffle=False)

    all_Z, all_loss, all_baseline = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            gaf = batch["gaf"].to(device)
            incidence = batch["incidence"].to(device)
            hyperedge_w = batch["hyperedge_w"].to(device)
            rho_mean = batch["rho_mean"].to(device)
            node_feat = build_node_features(batch["price_window"]).to(device)

            out = model(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)
            loss = hyperalign_loss(out, incidence=incidence, mask_same_hyperedge_negatives=True)
            all_loss.append(loss.item())

            # IMPORTANT: with same-hyperedge negatives masked, the random
            # -chance baseline is NOT ln(N) anymore -- it's ln(k_i) where
            # k_i = 1 (self) + number of UNMASKED negatives for asset i,
            # which varies per row depending on hyperedge structure. Using
            # flat ln(N) here would understate the model's real baseline
            # and make the model look worse than it is.
            mask = build_same_group_mask(incidence)          # (B,N,N) True=masked-out
            n_valid = (~mask).sum(dim=-1).float()             # (B,N) includes the diagonal (never masked)
            baseline_per_row = torch.log(n_valid.clamp_min(1.0))
            all_baseline.append(baseline_per_row.mean().item())

            B, N, d = out.Z.shape
            all_Z.append(out.Z.reshape(B * N, d).cpu())

    Z = torch.cat(all_Z, dim=0)
    mean_loss = float(np.mean(all_loss))
    N = args.n_assets
    baseline_loss = float(np.mean(all_baseline))  # masked-aware baseline, NOT flat ln(N)

    eff_rank = effective_rank(Z)
    per_dim_var = Z.var(dim=0)
    dead_dims = int((per_dim_var < 1e-4).sum().item())

    print("=" * 60)
    print("  Post-training Sanity Check")
    print("=" * 60)
    print(f"  Checkpoint epoch: {ckpt['epoch']}")
    print(f"  rho_star used:    {rho_star:.4f}")
    print()
    print(f"  Val loss (this run):      {mean_loss:.4f}")
    print(f"  Random-chance baseline (masked-aware, avg ln(k_i)): {baseline_loss:.4f}  "
          f"(N={N}, same-hyperedge negatives excluded)")
    if mean_loss < 0.5 * baseline_loss:
        print("  -> Loss is well below baseline: model IS learning "
              "asset-to-asset correspondence, not random.")
    elif mean_loss < 0.9 * baseline_loss:
        print("  -> Loss is somewhat below baseline: some learning, "
              "but check if more epochs / better hyperparameters help.")
    else:
        print("  -> WARNING: loss is close to the random-init baseline. "
              "The model may not be learning meaningful alignment.")
    print()
    print(f"  Effective rank of Z:  {eff_rank:.1f}  (out of max {min(Z.shape):d})")
    print(f"  Dead dimensions (var<1e-4): {dead_dims} / {Z.shape[1]}")
    if eff_rank < 0.05 * min(Z.shape) or dead_dims > 0.5 * Z.shape[1]:
        print("  -> WARNING: possible embedding collapse. Z occupies a "
              "very low-dimensional subspace. Consider: lower learning "
              "rate, add a variance/covariance regularizer (e.g. "
              "VICReg-style), or check the alignment temperature.")
    else:
        print("  -> Embedding spread looks healthy (no collapse signature).")
    print("=" * 60)


if __name__ == "__main__":
    main()
