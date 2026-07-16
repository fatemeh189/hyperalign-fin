"""
FactorCL-Inspired Baseline: Training and Comparison
=======================================================
Trains a model using the SAME visual (GAF-CNN) and graph (HGNN)
encoders as HyperAlign-Fin, but replaces the regime-aware alignment +
gated fusion with the FactorCL-inspired shared/unique factorization of
baseline_factorcl.py (see that file for the documented simplifications
relative to Liang et al., 2023). This isolates whether HyperAlign-Fin's
specific, theory-derived fusion mechanism outperforms a general
multi-view redundancy-vs-uniqueness baseline given IDENTICAL inputs --
a stricter comparison than TS2Vec, which lacks a graph branch entirely.

Runs n seeds, evaluates on both downstream tasks with the same
class-weighted, F1-selected linear probe protocol used throughout this
project, and reports a paired significance test against the final
HyperAlign-Fin (gated) results already saved in raw_results.json.

Usage:
    python run_factorcl_comparison.py --data_dir ./cache \
        --pretrained_encoder ./checkpoints/gaf_byol.pt \
        --hyperalign_raw_results ./ablation_results/raw_results.json \
        --n_assets 119 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_dir ./factorcl_results
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from dataset import make_loader
from hyperalign_model import GAFEncoder, HGNNEncoder, build_node_features
from baseline_factorcl import FactorCLFusion, factorcl_loss
from train_hyperalign import calibrate_from_train
from downstream_regime_classification import train_probe, evaluate_probe
from downstream_asset_volatility import binarize_top_quartile


# ================================================================
# 1. Combined model: reuse HyperAlign-Fin's own encoders + FactorCL fusion
# ================================================================

class FactorCLModel(torch.nn.Module):
    def __init__(self, seq_len: int, n_assets: int, latent_dim: int = 128,
                 shared_dim: int = 64, hgnn_in_feat_dim: int = 8):
        super().__init__()
        self.visual_encoder = GAFEncoder(seq_len=seq_len, latent_dim=latent_dim)
        self.graph_encoder = HGNNEncoder(n_assets=n_assets, latent_dim=latent_dim,
                                          in_feat_dim=hgnn_in_feat_dim)
        self.fusion = FactorCLFusion(latent_dim=latent_dim, shared_dim=shared_dim)

    def forward(self, gaf, incidence, hyperedge_w, node_feat=None):
        V = self.visual_encoder(gaf)
        G = self.graph_encoder(incidence, hyperedge_w, node_feat)
        Z, shared_v, shared_g, unique_v, unique_g = self.fusion(V, G)
        return V, G, Z, shared_v, shared_g, unique_v, unique_g


def run_epoch_factorcl(model, loader, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            gaf = batch["gaf"].to(device)
            incidence = batch["incidence"].to(device)
            hyperedge_w = batch["hyperedge_w"].to(device)
            node_feat = build_node_features(batch["price_window"]).to(device)

            _, _, _, shared_v, shared_g, unique_v, unique_g = model(
                gaf, incidence, hyperedge_w, node_feat=node_feat)
            loss = factorcl_loss(shared_v, shared_g, unique_v, unique_g)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_one_run(seed, args, train_path, val_path, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FactorCLModel(seq_len=args.seq_len, n_assets=args.n_assets,
                           latent_dim=args.latent_dim).to(device)
    if args.pretrained_encoder:
        ckpt = torch.load(args.pretrained_encoder, map_location=device, weights_only=False)
        model.visual_encoder.load_state_dict(ckpt["encoder_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_path, args.batch_size, shuffle=True)
    val_loader = make_loader(val_path, args.batch_size, shuffle=False)

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        run_epoch_factorcl(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch_factorcl(model, val_loader, optimizer, device, train=False)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


# ================================================================
# 2. Downstream evaluation (Z from FactorCL fusion, same protocol)
# ================================================================

@torch.no_grad()
def extract_regime(model, loader, device):
    all_Z, all_rho = [], []
    for batch in loader:
        gaf = batch["gaf"].to(device)
        incidence = batch["incidence"].to(device)
        hyperedge_w = batch["hyperedge_w"].to(device)
        rho_mean = batch["rho_mean"]
        node_feat = build_node_features(batch["price_window"]).to(device)
        _, _, Z, *_ = model(gaf, incidence, hyperedge_w, node_feat=node_feat)
        all_Z.append(Z.mean(dim=1).cpu())
        all_rho.append(rho_mean)
    return torch.cat(all_Z), torch.cat(all_rho)


@torch.no_grad()
def extract_per_asset(model, loader, device):
    all_Z, all_vol = [], []
    for batch in loader:
        gaf = batch["gaf"].to(device)
        incidence = batch["incidence"].to(device)
        hyperedge_w = batch["hyperedge_w"].to(device)
        price_window = batch["price_window"].to(device)
        node_feat = build_node_features(price_window).to(device)
        _, _, Z, *_ = model(gaf, incidence, hyperedge_w, node_feat=node_feat)
        B, N, d = Z.shape
        all_Z.append(Z.reshape(B * N, d).cpu())
        all_vol.append(node_feat[:, :, 1].reshape(B * N).cpu())
    return torch.cat(all_Z), torch.cat(all_vol)


def evaluate_downstream(model, args, train_path, val_path, test_path, rho_star, device):
    train_loader = make_loader(train_path, args.batch_size, shuffle=False)
    val_loader = make_loader(val_path, args.batch_size, shuffle=False)
    test_loader = make_loader(test_path, args.batch_size, shuffle=False)

    Z_tr, rho_tr = extract_regime(model, train_loader, device)
    Z_va, rho_va = extract_regime(model, val_loader, device)
    Z_te, rho_te = extract_regime(model, test_loader, device)
    y_tr = (rho_tr > rho_star).long()
    y_va = (rho_va > rho_star).long()
    y_te = (rho_te > rho_star).long()

    probe1, _ = train_probe(Z_tr, y_tr, Z_va, y_va, in_dim=Z_tr.shape[1], device=device)
    m1 = evaluate_probe(probe1, Z_te, y_te, device)

    Zp_tr, vol_tr = extract_per_asset(model, train_loader, device)
    Zp_va, vol_va = extract_per_asset(model, val_loader, device)
    Zp_te, vol_te = extract_per_asset(model, test_loader, device)
    (yv_tr, yv_va, yv_te), _ = binarize_top_quartile(vol_tr, vol_va, vol_te)

    probe2, _ = train_probe(Zp_tr, yv_tr, Zp_va, yv_va, in_dim=Zp_tr.shape[1], device=device)
    m2 = evaluate_probe(probe2, Zp_te, yv_te, device)

    return {"task1_f1": m1["f1"], "task1_acc": m1["accuracy"],
            "task2_f1": m2["f1"], "task2_acc": m2["accuracy"]}


# ================================================================
# 3. Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pretrained_encoder", default=None)
    p.add_argument("--hyperalign_raw_results", required=True)
    p.add_argument("--hyperalign_fusion_type", default="gated")
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--out_dir", default="./factorcl_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[factorcl] device={device}  seeds={args.seeds}")

    train_path = os.path.join(args.data_dir, "train.pt")
    val_path = os.path.join(args.data_dir, "val.pt")
    test_path = os.path.join(args.data_dir, "test.pt")
    rho_star = calibrate_from_train(train_path)

    partial_path = os.path.join(args.out_dir, "factorcl_raw_partial.json")
    raw = {}
    completed = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            saved = json.load(f)
        raw, completed = saved["raw"], set(saved["completed"])
        print(f"[factorcl] RESUMING: {len(completed)} seeds already done.")

    def save_partial():
        with open(partial_path, "w") as f:
            json.dump({"raw": raw, "completed": list(completed)}, f, indent=2)

    for seed in args.seeds:
        if seed in completed:
            print(f"[factorcl] seed={seed} -- SKIPPED (already done)")
            continue
        print(f"\n[factorcl] seed={seed} -- training ...")
        model, best_val = train_one_run(seed, args, train_path, val_path, device)
        print(f"  best_val_loss={best_val:.4f}")
        metrics = evaluate_downstream(model, args, train_path, val_path, test_path, rho_star, device)
        print(f"  task1_f1={metrics['task1_f1']:.4f}  task2_f1={metrics['task2_f1']:.4f}")
        for k, v in metrics.items():
            raw.setdefault(k, []).append(v)
        completed.add(seed)
        save_partial()
        print(f"  [saved progress: {len(completed)}/{len(args.seeds)} seeds done]")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(args.hyperalign_raw_results) as f:
        hyperalign_raw = json.load(f)[args.hyperalign_fusion_type]

    print("\n" + "=" * 70)
    print(f"  HyperAlign-Fin ({args.hyperalign_fusion_type}) vs. FactorCL-inspired baseline")
    print("=" * 70)

    summary = {}
    for task_key, task_name in [("task1", "Task 1: Regime Classification"),
                                 ("task2", "Task 2: Volatility Classification")]:
        hf = np.array(hyperalign_raw[f"{task_key}_Z_f1"])
        fc = np.array(raw[f"{task_key}_f1"])
        n = min(len(hf), len(fc))
        hf, fc = hf[:n], fc[:n]
        print(f"\n{task_name} (n={n} matched seeds)")
        print(f"  HyperAlign-Fin: {hf.mean():.4f} +/- {hf.std(ddof=1) if n>1 else 0:.4f}")
        print(f"  FactorCL-insp.: {fc.mean():.4f} +/- {fc.std(ddof=1) if n>1 else 0:.4f}")
        if n >= 2:
            t_stat, p_val = stats.ttest_rel(hf, fc)
            try:
                _, w_p = stats.wilcoxon(hf, fc)
            except ValueError:
                w_p = float("nan")
            diff = hf.mean() - fc.mean()
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f"  diff: {diff:+.4f}  paired t: t={t_stat:.3f} p={p_val:.4f} ({sig})")
            print(f"  Wilcoxon: p={w_p:.4f}")
            summary[task_name] = {"hyperalign_mean": float(hf.mean()), "factorcl_mean": float(fc.mean()),
                                   "diff": float(diff), "p_ttest": float(p_val), "p_wilcoxon": float(w_p)}

    with open(os.path.join(args.out_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task_key, task_name) in zip(axes, [("task1", "Task 1: Regime"),
                                                  ("task2", "Task 2: Volatility")]):
        hf = np.array(hyperalign_raw[f"{task_key}_Z_f1"])
        fc = np.array(raw[f"{task_key}_f1"])
        n = min(len(hf), len(fc))
        means = [hf[:n].mean(), fc[:n].mean()]
        stds = [hf[:n].std(ddof=1) if n > 1 else 0, fc[:n].std(ddof=1) if n > 1 else 0]
        ax.bar(["HyperAlign-Fin", "FactorCL-insp."], means, yerr=stds, capsize=8,
               color=["mediumseagreen", "steelblue"], alpha=0.85)
        ax.set_ylabel("F1 (test)")
        ax.set_title(task_name)
        ax.set_ylim(0, 1.0)
    plt.suptitle(f"HyperAlign-Fin vs. FactorCL-inspired baseline (n={len(args.seeds)} seeds)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "factorcl_comparison.png"), dpi=150)
    print(f"\n[factorcl] saved figure -> {args.out_dir}/factorcl_comparison.png")


if __name__ == "__main__":
    main()
