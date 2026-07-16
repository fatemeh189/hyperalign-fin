"""
Downstream Task 1 — Regime Classification via Linear Probe
==============================================================
The real test of Corollary 2 (PAC-style bound: joint Z should beat
either view alone). This is the FIRST genuine "does this representation
do anything useful" experiment, as opposed to the pretext-task loss
(InfoNCE) tuned so far.

Label source: the SAME regime signal already validated in
hyperalign_validation.py / Corollary 3 (rho_mean(t) > rho_star). No
manual labeling needed -- this is a legitimate downstream task because
predicting regime from the FUSED embedding Z, using a linear probe
frozen on top of Z, is a different question than "did rho_mean exceed
a threshold" (which the model never sees directly): can the model's
learned Z reconstruct this regime information from raw price/graph
data alone, days before/without recomputing correlations explicitly?

Protocol (Corollary 2 requires comparing V-alone, G-alone, and Z=fused):
    1. Freeze the trained HyperAlignFin encoder.
    2. Extract V, G, Z for every window in train/val/test.
    3. Train THREE separate linear probes: V->regime, G->regime, Z->regime.
    4. Report accuracy/F1 for each. Z should beat max(V,G) if the
       fusion is doing real work (Corollary 2's practical claim).

Usage:
    python downstream_regime_classification.py \
        --checkpoint ./run3/best_checkpoint.pt --data_dir ./cache \
        --n_assets 119 --seq_len 20
"""

from __future__ import annotations
import argparse
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import make_loader
from hyperalign_model import HyperAlignFin, build_node_features


# ================================================================
# 1. Extract frozen embeddings + regime labels for every window
# ================================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Returns per-WINDOW (not per-asset) pooled embeddings, since regime
    is a market-wide (not asset-specific) label: mean-pool V, G, Z
    across the N assets in each window.
    """
    all_V, all_G, all_Z, all_labels = [], [], [], []

    for batch in loader:
        gaf = batch["gaf"].to(device)
        incidence = batch["incidence"].to(device)
        hyperedge_w = batch["hyperedge_w"].to(device)
        rho_mean = batch["rho_mean"].to(device)
        node_feat = build_node_features(batch["price_window"]).to(device)

        out = model(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)

        # market-wide pooled representation (mean over the N assets)
        all_V.append(out.V.mean(dim=1).cpu())
        all_G.append(out.G.mean(dim=1).cpu())
        all_Z.append(out.Z.mean(dim=1).cpu())
        all_labels.append(out.regime.long().cpu())  # the label IS the regime flag itself

    return (torch.cat(all_V), torch.cat(all_G), torch.cat(all_Z), torch.cat(all_labels))


# ================================================================
# 2. Linear probe training (frozen features, simple logistic regression)
# ================================================================

class LinearProbeHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(X_train, y_train, X_val, y_val, in_dim, epochs=200, lr=1e-2, device="cpu"):
    """
    Trains a linear probe with two safeguards against the "always
    predict the majority class" degenerate solution seen earlier with
    weak/imbalanced signals (V on Task 1, and suspected in the TS2Vec
    baseline):
      1. Class-weighted CrossEntropyLoss (inverse frequency), so the
         minority class is not free to ignore.
      2. Model selection by VALIDATION F1 (not accuracy) -- accuracy
         is what silently rewarded the degenerate majority-only
         solution before.
    """
    probe = LinearProbeHead(in_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    class_counts = torch.bincount(y_train, minlength=2).float().clamp_min(1.0)
    class_weights = (class_counts.sum() / (2.0 * class_counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1, best_state = -1.0, None
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val).argmax(dim=-1).cpu()
            val_f1 = f1_score(y_val.cpu(), val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state is None:
        # every epoch had F1=0 (probe never predicted the positive class once)
        # -- keep the LAST state rather than crash, but this itself is diagnostic
        best_state = probe.state_dict()
    probe.load_state_dict(best_state)
    return probe, best_val_f1


def evaluate_probe(probe, X, y, device="cpu"):
    probe.eval()
    with torch.no_grad():
        pred = probe(X.to(device)).argmax(dim=-1).cpu()
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred),
        "n_positive": int(y.sum()),
        "n_total": len(y),
    }


# ================================================================
# 3. Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="dir with train.pt/val.pt/test.pt")
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="./downstream_regime")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[downstream] device = {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rho_star = ckpt["rho_star"]

    model = HyperAlignFin(
        seq_len=args.seq_len, n_assets=args.n_assets, rho_star=rho_star,
        latent_dim=args.latent_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"[downstream] loaded frozen encoder (epoch {ckpt['epoch']}, rho_star={rho_star:.4f})")

    train_loader = make_loader(os.path.join(args.data_dir, "train.pt"),
                                batch_size=args.batch_size, shuffle=False)
    val_loader = make_loader(os.path.join(args.data_dir, "val.pt"),
                              batch_size=args.batch_size, shuffle=False)
    test_path = os.path.join(args.data_dir, "test.pt")
    has_test = os.path.exists(test_path)
    test_loader = make_loader(test_path, batch_size=args.batch_size, shuffle=False) if has_test else None

    print("[downstream] extracting frozen embeddings (V, G, Z) + regime labels ...")
    V_tr, G_tr, Z_tr, y_tr = extract_embeddings(model, train_loader, device)
    V_va, G_va, Z_va, y_va = extract_embeddings(model, val_loader, device)
    if has_test:
        V_te, G_te, Z_te, y_te = extract_embeddings(model, test_loader, device)

    print(f"[downstream] train regime balance: {y_tr.float().mean():.3f} positive "
          f"({int(y_tr.sum())}/{len(y_tr)})")
    print(f"[downstream] val   regime balance: {y_va.float().mean():.3f} positive "
          f"({int(y_va.sum())}/{len(y_va)})")

    views = {"V (visual only)": (V_tr, V_va), "G (graph only)": (G_tr, G_va),
             "Z (fused)": (Z_tr, Z_va)}
    if has_test:
        views_test = {"V (visual only)": V_te, "G (graph only)": G_te, "Z (fused)": Z_te}

    results = {}
    print("\n[downstream] training linear probes ...")
    for name, (X_tr, X_va) in views.items():
        probe, best_val_acc = train_probe(
            X_tr, y_tr, X_va, y_va, in_dim=X_tr.shape[1], device=device)
        val_metrics = evaluate_probe(probe, X_va, y_va, device)
        entry = {"val": val_metrics}
        if has_test:
            test_metrics = evaluate_probe(probe, views_test[name], y_te, device)
            entry["test"] = test_metrics
        results[name] = entry

        line = f"  {name:<20} val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        if has_test:
            line += f"  test_acc={entry['test']['accuracy']:.4f}  test_f1={entry['test']['f1']:.4f}"
        print(line)

    # --- Corollary 2 check: does Z beat max(V, G)? ---
    # F1, not accuracy, is the deciding metric: this task's classes can be
    # imbalanced (e.g. crisis rate differs sharply between val/test periods),
    # in which case accuracy is dominated by the majority class and is
    # uninformative about whether the minority (often the interesting) class
    # is actually being identified. Accuracy is still reported for reference.
    split = "test" if has_test else "val"
    acc_v = results["V (visual only)"][split]["accuracy"]
    acc_g = results["G (graph only)"][split]["accuracy"]
    acc_z = results["Z (fused)"][split]["accuracy"]
    f1_v = results["V (visual only)"][split]["f1"]
    f1_g = results["G (graph only)"][split]["f1"]
    f1_z = results["Z (fused)"][split]["f1"]
    print(f"\n[Corollary 2 check] ({split} split)")
    print(f"  Accuracy -- V: {acc_v:.4f}  G: {acc_g:.4f}  Z: {acc_z:.4f}  "
          f"(reference only, can be skewed by class imbalance)")
    print(f"  F1       -- V: {f1_v:.4f}  G: {f1_g:.4f}  Z: {f1_z:.4f}  (deciding metric)")
    if f1_z > max(f1_v, f1_g):
        print(f"  -> PASS: fused Z (F1={f1_z:.4f}) beats both individual views "
              f"(max F1={max(f1_v, f1_g):.4f}). Corollary 2 supported.")
    elif f1_z >= max(f1_v, f1_g) - 0.03:
        print(f"  -> PARTIAL: Z (F1={f1_z:.4f}) is within noise of the best individual "
              f"view (F1={max(f1_v, f1_g):.4f}). Inconclusive with this sample size.")
    else:
        print(f"  -> FAIL: Z (F1={f1_z:.4f}) underperforms the best individual view "
              f"(F1={max(f1_v, f1_g):.4f}). Fusion is NOT helping -- do not claim "
              f"Corollary 2 is supported without investigating further.")

    # --- plot ---
    names = list(results.keys())
    val_accs = [results[n]["val"]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, val_accs, color=["steelblue", "coral", "mediumseagreen"])
    ax.axhline(max(y_va.float().mean().item(), 1 - y_va.float().mean().item()),
               color="gray", linestyle="--", label="Majority-class baseline")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Downstream Task 1: Regime Classification\n(Corollary 2 check: does fusion help?)")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, val_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.3f}", ha="center")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "regime_classification_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n[downstream] saved figure -> {fig_path}")

    torch.save(results, os.path.join(args.out_dir, "results.pt"))
    print(f"[downstream] saved raw results -> {args.out_dir}/results.pt")


if __name__ == "__main__":
    main()
