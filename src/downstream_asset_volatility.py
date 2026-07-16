"""
Downstream Task 2 — Per-Asset Volatility Quartile Classification
=====================================================================
Task 1 (regime classification) favored G almost by construction (the
regime label is derived from the same correlation matrix G is built
from) -- it wasn't a fair test of whether V contributes anything.

This task is the complement: predict whether an asset's OWN realized
volatility (within its window) is in the top quartile, from V_i, G_i,
Z_i (per-asset, not pooled). GAF (Definition 2) is explicitly built to
encode a single asset's price-shape dynamics, so V should carry real
signal here; G (sector + correlation-cluster identity) should carry
much less, since cluster membership is a weak proxy for a specific
asset's own volatility level.

If Corollary 2 holds generally (not just for G-favoring tasks), Z
should now be competitive with or better than V here, and clearly
better than G alone -- the mirror image of Task 1's result.

Usage:
    python downstream_asset_volatility.py \
        --checkpoint ./run3/best_checkpoint.pt --data_dir ./cache \
        --n_assets 119 --seq_len 20
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from dataset import make_loader
from hyperalign_model import HyperAlignFin, build_node_features


# ================================================================
# 1. Extract PER-ASSET embeddings + realized-volatility labels
# ================================================================

@torch.no_grad()
def extract_per_asset(model, loader, device):
    all_V, all_G, all_Z, all_vol = [], [], [], []

    for batch in loader:
        gaf = batch["gaf"].to(device)
        incidence = batch["incidence"].to(device)
        hyperedge_w = batch["hyperedge_w"].to(device)
        rho_mean = batch["rho_mean"].to(device)
        price_window = batch["price_window"].to(device)          # (B, T, N)
        node_feat = build_node_features(price_window).to(device)  # (B, N, 8): idx 1 = std_r

        out = model(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)

        B, N, d = out.V.shape
        all_V.append(out.V.reshape(B * N, d).cpu())
        all_G.append(out.G.reshape(B * N, d).cpu())
        all_Z.append(out.Z.reshape(B * N, d).cpu())

        realized_vol = node_feat[:, :, 1]   # std_r, per asset per window (B, N)
        all_vol.append(realized_vol.reshape(B * N).cpu())

    return (torch.cat(all_V), torch.cat(all_G), torch.cat(all_Z), torch.cat(all_vol))


def binarize_top_quartile(vol_train: torch.Tensor, *other_vols: torch.Tensor):
    """Threshold fit on TRAIN only, applied to all splits (no leakage)."""
    q75 = torch.quantile(vol_train, 0.75)
    labels = [(vol_train >= q75).long()]
    for v in other_vols:
        labels.append((v >= q75).long())
    return labels, q75.item()


# ================================================================
# 2. Linear probe (same as Task 1)
# ================================================================

class LinearProbeHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(X_train, y_train, X_val, y_val, in_dim, epochs=200, lr=1e-2, device="cpu"):
    probe = LinearProbeHead(in_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    best_val_acc, best_state = -1.0, None
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = loss_fn(probe(X_train), y_train)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_acc = (probe(X_val).argmax(dim=-1) == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe


def evaluate_probe(probe, X, y, device="cpu"):
    probe.eval()
    with torch.no_grad():
        pred = probe(X.to(device)).argmax(dim=-1).cpu()
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}


# ================================================================
# 3. Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="./downstream_volatility")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[downstream-2] device = {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rho_star = ckpt["rho_star"]
    model = HyperAlignFin(seq_len=args.seq_len, n_assets=args.n_assets,
                           rho_star=rho_star, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"[downstream-2] loaded frozen encoder (epoch {ckpt['epoch']})")

    train_loader = make_loader(os.path.join(args.data_dir, "train.pt"), args.batch_size, shuffle=False)
    val_loader = make_loader(os.path.join(args.data_dir, "val.pt"), args.batch_size, shuffle=False)
    test_path = os.path.join(args.data_dir, "test.pt")
    has_test = os.path.exists(test_path)
    test_loader = make_loader(test_path, args.batch_size, shuffle=False) if has_test else None

    print("[downstream-2] extracting per-asset embeddings + realized volatility ...")
    V_tr, G_tr, Z_tr, vol_tr = extract_per_asset(model, train_loader, device)
    V_va, G_va, Z_va, vol_va = extract_per_asset(model, val_loader, device)
    if has_test:
        V_te, G_te, Z_te, vol_te = extract_per_asset(model, test_loader, device)
        (y_tr, y_va, y_te), q75 = binarize_top_quartile(vol_tr, vol_va, vol_te)
    else:
        (y_tr, y_va), q75 = binarize_top_quartile(vol_tr, vol_va)

    print(f"[downstream-2] top-quartile threshold (train-fit): std_r >= {q75:.5f}")
    print(f"[downstream-2] train positive rate: {y_tr.float().mean():.3f}  "
          f"val positive rate: {y_va.float().mean():.3f}"
          + (f"  test positive rate: {y_te.float().mean():.3f}" if has_test else ""))

    views = {"V (visual only)": (V_tr, V_va), "G (graph only)": (G_tr, G_va), "Z (fused)": (Z_tr, Z_va)}
    views_test = {"V (visual only)": V_te, "G (graph only)": G_te, "Z (fused)": Z_te} if has_test else None

    results = {}
    print("\n[downstream-2] training linear probes (predict OWN top-quartile volatility) ...")
    for name, (X_tr, X_va) in views.items():
        probe = train_probe(X_tr, y_tr, X_va, y_va, in_dim=X_tr.shape[1], device=device)
        val_metrics = evaluate_probe(probe, X_va, y_va, device)
        entry = {"val": val_metrics}
        if has_test:
            entry["test"] = evaluate_probe(probe, views_test[name], y_te, device)
        results[name] = entry
        line = f"  {name:<20} val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        if has_test:
            line += f"  test_acc={entry['test']['accuracy']:.4f}  test_f1={entry['test']['f1']:.4f}"
        print(line)

    split = "test" if has_test else "val"
    acc_v = results["V (visual only)"][split]["accuracy"]
    acc_g = results["G (graph only)"][split]["accuracy"]
    acc_z = results["Z (fused)"][split]["accuracy"]
    f1_v = results["V (visual only)"][split]["f1"]
    f1_g = results["G (graph only)"][split]["f1"]
    f1_z = results["Z (fused)"][split]["f1"]
    print(f"\n[Corollary 2 check -- TASK 2, complementary to regime task] ({split} split)")
    print(f"  Accuracy -- V: {acc_v:.4f}  G: {acc_g:.4f}  Z: {acc_z:.4f}  "
          f"(reference only, can be skewed by class imbalance)")
    print(f"  F1       -- V: {f1_v:.4f}  G: {f1_g:.4f}  Z: {f1_z:.4f}  (deciding metric)")
    if f1_v > f1_g + 0.05:
        print("  -> As expected, V >> G here (volatility is asset-specific, "
              "not a sector/correlation property). Confirms V and G carry "
              "genuinely different information (Theorem 3).")
    if f1_z > max(f1_v, f1_g):
        print(f"  -> PASS: fused Z (F1={f1_z:.4f}) beats both individual views "
              f"(max F1={max(f1_v, f1_g):.4f}). Corollary 2 supported.")
    elif f1_z >= max(f1_v, f1_g) - 0.03:
        print(f"  -> PARTIAL: Z (F1={f1_z:.4f}) is within noise of the best "
              f"individual view (F1={max(f1_v, f1_g):.4f}).")
    else:
        print(f"  -> FAIL: Z (F1={f1_z:.4f}) underperforms the best individual "
              f"view (F1={max(f1_v, f1_g):.4f}). Fusion is NOT helping on the "
              f"metric that matters here -- worth investigating the fusion "
              f"layer itself (Section 3.4).")

    names = list(results.keys())
    val_accs = [results[n]["val"]["accuracy"] for n in names]
    test_accs = [results[n]["test"]["accuracy"] for n in names] if has_test else None

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, val_accs, width, label="Val", color="steelblue")
    if has_test:
        ax.bar(x + width/2, test_accs, width, label="Test", color="coral")
    ax.axhline(0.75, color="gray", linestyle="--", label="Majority-class baseline (0.75)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_title("Downstream Task 2: Per-Asset Volatility Quartile\n"
                 "(complementary check: should favor V, not G)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "volatility_classification_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n[downstream-2] saved figure -> {fig_path}")

    torch.save(results, os.path.join(args.out_dir, "results.pt"))
    print(f"[downstream-2] saved raw results -> {args.out_dir}/results.pt")


if __name__ == "__main__":
    main()
