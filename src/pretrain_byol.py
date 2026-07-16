"""
HyperAlign-Fin — BYOL Self-Supervised Pretraining for the Visual Encoder
==========================================================================
Implements Definition 2's "trained self-supervised (BYOL)" requirement
for f_theta (GAFEncoder), operating directly on GAF matrices.

Why not standard image augmentations (color jitter, horizontal flip,
etc.): GAF matrices are not natural images — they encode angular
relationships between time steps (Definition 2, Eq. for [G_i]_{t,s}).
Flipping or color-jittering them destroys the temporal structure and is
not a meaningful invariance to learn. Instead we use two augmentations
that preserve the GAF's semantic content while creating distinct views:

    1. Additive Gaussian noise (models measurement/estimation noise)
    2. Random block masking (models missing/occluded time segments,
       forcing the encoder to learn robust, non-local structure)

This is a design choice, not a theorem — report and ablate it as such.

Usage:
    python pretrain_byol.py --data ./cache/train.pt --epochs 50 \
        --out ./checkpoints/gaf_encoder_byol.pt
"""

from __future__ import annotations
import argparse
import copy
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hyperalign_model import GAFEncoder
from dataset import HyperAlignDataset


# ================================================================
# Augmentations for GAF matrices
# ================================================================

def augment_gaf(gaf: torch.Tensor, noise_std: float = 0.05,
                 mask_frac: float = 0.15) -> torch.Tensor:
    """
    gaf: (M, 1, T, T)  batch of GAF matrices (M = flattened batch*assets)
    Returns an augmented copy: additive noise + one random square block masked to 0.
    """
    out = gaf.clone()
    out = out + noise_std * torch.randn_like(out)

    M, C, T, _ = out.shape
    block = max(1, int(mask_frac * T))
    for m in range(M):
        r0 = torch.randint(0, T - block + 1, (1,)).item()
        c0 = torch.randint(0, T - block + 1, (1,)).item()
        out[m, :, r0:r0 + block, c0:c0 + block] = 0.0
    return out


# ================================================================
# BYOL components
# ================================================================

class MLPHead(nn.Module):
    """Projector / predictor head, standard BYOL 2-layer MLP."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    Standard BYOL wrapper around GAFEncoder.

    online network  = encoder -> projector -> predictor
    target network  = EMA(encoder) -> EMA(projector)   [no predictor, no grad]
    """

    def __init__(self, encoder: GAFEncoder, latent_dim: int,
                 proj_dim: int = 128, ema_decay: float = 0.996):
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = MLPHead(latent_dim, out_dim=proj_dim)
        self.predictor = MLPHead(proj_dim, out_dim=proj_dim)

        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.online_encoder.parameters(),
                                       self.target_encoder.parameters()):
            target_p.data = self.ema_decay * target_p.data + (1 - self.ema_decay) * online_p.data
        for online_p, target_p in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            target_p.data = self.ema_decay * target_p.data + (1 - self.ema_decay) * online_p.data

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        view1, view2: (M, N, 1, T, T)  two augmented views of the same
        batch (M windows, N assets each). Encoder expects (B,N,1,T,T);
        here B=M for this pretraining stage.
        """
        def online_pass(v):
            z = self.online_encoder(v)              # (M, N, latent_dim)
            M, N, d = z.shape
            z = z.reshape(M * N, d)
            p = self.online_projector(z)
            return self.predictor(p)                 # (M*N, proj_dim)

        @torch.no_grad()
        def target_pass(v):
            z = self.target_encoder(v)
            M, N, d = z.shape
            z = z.reshape(M * N, d)
            return self.target_projector(z)           # (M*N, proj_dim)

        pred1, pred2 = online_pass(view1), online_pass(view2)
        targ1, targ2 = target_pass(view1), target_pass(view2)

        loss = byol_loss(pred1, targ2.detach()) + byol_loss(pred2, targ1.detach())
        return loss.mean()


def byol_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity, as in the original BYOL paper."""
    pred = F.normalize(pred, dim=-1)
    targ = F.normalize(targ, dim=-1)
    return 2 - 2 * (pred * targ).sum(dim=-1)


# ================================================================
# Training loop
# ================================================================

def pretrain(data_path: str, out_path: str, epochs: int = 50,
             batch_size: int = 16, lr: float = 3e-4, latent_dim: int = 128,
             seq_len: int = 20, device: str = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[byol] device = {device}")

    ds = HyperAlignDataset(data_path)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: torch.stack([b["gaf"] for b in batch], dim=0),
        drop_last=True,
    )

    encoder = GAFEncoder(seq_len=seq_len, latent_dim=latent_dim)
    model = BYOL(encoder, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters()),
        lr=lr,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for gaf in loader:                       # gaf: (B, N, T, T)
            gaf = gaf.unsqueeze(2).to(device)     # (B, N, 1, T, T)
            B, N, C, T, _ = gaf.shape
            flat = gaf.reshape(B * N, C, T, T)

            view1 = augment_gaf(flat).reshape(B, N, C, T, T)
            view2 = augment_gaf(flat).reshape(B, N, C, T, T)

            loss = model(view1, view2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "seconds": dt})
        print(f"[byol] epoch {epoch:03d}/{epochs}  loss={avg_loss:.4f}  ({dt:.1f}s)")

    torch.save({
        "encoder_state_dict": model.online_encoder.state_dict(),
        "latent_dim": latent_dim,
        "seq_len": seq_len,
        "history": history,
    }, out_path)
    print(f"[byol] saved pretrained encoder -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to train.pt from data_pipeline.py")
    p.add_argument("--out", default="./checkpoints/gaf_encoder_byol.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=20)
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pretrain(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, latent_dim=args.latent_dim, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
