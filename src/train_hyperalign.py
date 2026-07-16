"""
HyperAlign-Fin — Main Training Loop
======================================
Ties together: data_pipeline.py -> dataset.py -> (optional) pretrain_byol.py
-> hyperalign_model.py

Pipeline:
    1. Load train/val splits produced by data_pipeline.py
    2. Calibrate rho* from TRAIN correlations ONLY (Corollary 3)
    3. Build HyperAlignFin with the calibrated threshold
    4. Optionally load a BYOL-pretrained visual encoder (pretrain_byol.py)
    5. Train with the contrastive alignment loss (Corollary 1), logging
       every epoch to a CSV file and checkpointing best + last models
    6. Save a loss-curve plot at the end

Usage:
    python train_hyperalign.py --data_dir ./cache --out_dir ./run1 \
        --epochs 100 --batch_size 8 --n_assets 119 --seq_len 20

Windows note: num_workers=0 is used throughout to avoid multiprocessing
issues; run this script directly (not from an interactive shell without
the __main__ guard) if you increase num_workers later.
"""

from __future__ import annotations
import argparse
import csv
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import HyperAlignDataset, make_loader
from hyperalign_model import (
    HyperAlignFin, hyperalign_loss, calibrate_rho_star,
    build_node_features, build_same_group_mask,
)


# ================================================================
# Calibration (Corollary 3) — TRAIN split only
# ================================================================

def calibrate_from_train(train_path: str, percentile: float = 75.0) -> float:
    ds = HyperAlignDataset(train_path)
    corr_stack = np.stack([s["corr"].numpy() for s in ds.samples], axis=0)  # (n_w, N, N)
    rho_star, diag = calibrate_rho_star(corr_stack, percentile=percentile)
    print(f"[calibration] rho_star = {rho_star:.4f}  "
          f"(crisis={diag['crisis_pct']:.1f}% of TRAIN windows, "
          f"percentile={percentile})")
    return rho_star


# ================================================================
# Train / eval epochs
# ================================================================

def run_epoch(model, loader, optimizer, device, train: bool):
    """
    Returns (mean_loss, mean_baseline) where mean_baseline is the
    masked-aware random-chance loss (avg ln(k_i) over unmasked negatives
    + self) for THIS split. Comparing loss to ITS OWN split's baseline
    is essential: train and val periods can have very different
    hyperedge density (e.g. a high-correlation crisis-heavy val period
    masks out more negatives than a calmer train period), which shifts
    the baseline itself -- comparing raw train_loss to raw val_loss
    directly is misleading without this.
    """
    model.train(mode=train)
    total_loss, total_baseline, n_batches = 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            gaf = batch["gaf"].to(device)
            incidence = batch["incidence"].to(device)
            hyperedge_w = batch["hyperedge_w"].to(device)
            rho_mean = batch["rho_mean"].to(device)
            node_feat = build_node_features(batch["price_window"]).to(device)

            out = model(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)
            loss = hyperalign_loss(out, incidence=incidence, mask_same_hyperedge_negatives=True)

            with torch.no_grad():
                mask = build_same_group_mask(incidence)
                n_valid = (~mask).sum(dim=-1).float()
                baseline = torch.log(n_valid.clamp_min(1.0)).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_baseline += baseline.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_baseline / max(n_batches, 1)


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dir with train.pt / val.pt from data_pipeline.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5,
                    help="L2 regularization; increase (e.g. 1e-4) if overfitting "
                         "(val loss worsening while train loss keeps improving)")
    p.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    p.add_argument("--calibration_percentile", type=float, default=75.0)
    p.add_argument("--pretrained_encoder", default=None,
                    help="optional path to a checkpoint from pretrain_byol.py")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    train_path = os.path.join(args.data_dir, "train.pt")
    val_path = os.path.join(args.data_dir, "val.pt")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(
            f"Expected train.pt and val.pt in {args.data_dir}. "
            f"Run data_pipeline.py first."
        )

    # --- Step 1: calibrate rho* on TRAIN only (Corollary 3) ---
    rho_star = calibrate_from_train(train_path, percentile=args.calibration_percentile)

    # --- Step 2: dataloaders ---
    train_loader = make_loader(train_path, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_path, batch_size=args.batch_size, shuffle=False)

    # --- Step 3: model ---
    model = HyperAlignFin(
        seq_len=args.seq_len, n_assets=args.n_assets, rho_star=rho_star,
        latent_dim=args.latent_dim,
    ).to(device)

    # --- Step 4: optional BYOL-pretrained visual encoder ---
    if args.pretrained_encoder is not None:
        ckpt = torch.load(args.pretrained_encoder, map_location=device, weights_only=False)
        model.visual_encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"[train] loaded BYOL-pretrained visual encoder from "
              f"{args.pretrained_encoder}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Step 5: training loop with early stopping + checkpointing ---
    log_path = os.path.join(args.out_dir, "training_log.csv")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_baseline", "train_gap",
                          "val_loss", "val_baseline", "val_gap", "seconds"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_baseline = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_baseline = run_epoch(model, val_loader, optimizer, device, train=False)
        dt = time.time() - t0
        train_gap = train_baseline - train_loss   # positive = better than chance, for THIS split
        val_gap = val_baseline - val_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "train_gap": train_gap, "val_gap": val_gap, "seconds": dt})
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_baseline, train_gap,
                                     val_loss, val_baseline, val_gap, dt])

        print(f"[train] epoch {epoch:03d}/{args.epochs}  "
              f"train: loss={train_loss:.4f} (baseline={train_baseline:.4f}, "
              f"gap={train_gap:+.4f})  "
              f"val: loss={val_loss:.4f} (baseline={val_baseline:.4f}, "
              f"gap={val_gap:+.4f})  ({dt:.1f}s)")

        # checkpoint: last
        torch.save({
            "model_state_dict": model.state_dict(),
            "rho_star": rho_star,
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        }, os.path.join(args.out_dir, "last_checkpoint.pt"))

        # checkpoint: best
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "rho_star": rho_star,
                "epoch": epoch,
                "val_loss": val_loss,
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_checkpoint.pt"))
            print(f"  -> new best (val_loss={val_loss:.4f}), checkpoint saved")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[train] early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # --- Step 6: loss curves (raw + gap-vs-own-baseline) ---
    epochs_arr = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_gaps = [h["train_gap"] for h in history]
    val_gaps = [h["val_gap"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs_arr, train_losses, label="Train loss")
    axes[0].plot(epochs_arr, val_losses, label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("InfoNCE loss")
    axes[0].set_title("Raw loss (NOT directly comparable across splits --\n"
                       "train/val periods have different hyperedge density)")
    axes[0].legend()

    axes[1].plot(epochs_arr, train_gaps, label="Train: baseline - loss")
    axes[1].plot(epochs_arr, val_gaps, label="Val: baseline - loss")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1, label="Chance level")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Gap vs. own-split baseline (nats)")
    axes[1].set_title("Better metric: improvement over EACH split's own\n"
                       "random-chance baseline (comparable across splits)")
    axes[1].legend()

    plt.suptitle(f"HyperAlign-Fin training (rho*={rho_star:.3f})", fontweight="bold")
    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    print(f"[train] saved loss curve -> {curve_path}")
    print(f"[train] best val_loss = {best_val_loss:.4f}")
    print(f"[train] all outputs in {args.out_dir}: "
          f"training_log.csv, best_checkpoint.pt, last_checkpoint.pt, loss_curve.png")


if __name__ == "__main__":
    main()
