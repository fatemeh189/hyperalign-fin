"""
Multi-Seed Ablation: concat vs. gated fusion
================================================
Runs N_SEEDS independent training runs for EACH fusion_type
("concat" and "gated"), evaluates both downstream tasks on each
trained model, and reports mean +/- std across seeds -- plus a paired
significance test (matched by seed, which controls for run-to-run
variance shared by both fusion types at a given seed).

This is what makes the concat-vs-gated comparison citable: a single
run's difference could be initialization noise; this script tells you
whether it's a real, statistically supported effect.

The BYOL-pretrained visual encoder is loaded ONCE and reused across all
seeds (it is a fixed feature-extraction preprocessing step, not part of
the ablation) -- only the alignment/fusion layers and HGNN encoder vary
by seed, which is what the ablation is actually about.

Usage:
    python run_multiseed_ablation.py --data_dir ./cache \
        --pretrained_encoder ./checkpoints/gaf_byol.pt \
        --n_assets 119 --seq_len 20 --seeds 0 1 2 3 4 \
        --epochs 60 --patience 10 --out_dir ./ablation_results
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
from hyperalign_model import HyperAlignFin
from train_hyperalign import calibrate_from_train, run_epoch
from downstream_regime_classification import extract_embeddings as extract_regime
from downstream_asset_volatility import extract_per_asset, binarize_top_quartile
from downstream_regime_classification import train_probe, evaluate_probe


# ================================================================
# 1. Train one model for one (seed, fusion_type) combination
# ================================================================

def train_one_run(seed: int, fusion_type: str, args, rho_star: float,
                   train_path: str, val_path: str, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HyperAlignFin(
        seq_len=args.seq_len, n_assets=args.n_assets, rho_star=rho_star,
        latent_dim=args.latent_dim, fusion_type=fusion_type,
    ).to(device)

    if args.pretrained_encoder:
        ckpt = torch.load(args.pretrained_encoder, map_location=device, weights_only=False)
        model.visual_encoder.load_state_dict(ckpt["encoder_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_path, args.batch_size, shuffle=True)
    val_loader = make_loader(val_path, args.batch_size, shuffle=False)

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, _ = run_epoch(model, val_loader, optimizer, device, train=False)
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
# 2. Downstream evaluation of one trained model (both tasks)
# ================================================================

def evaluate_downstream(model, args, train_path, val_path, test_path, device):
    train_loader = make_loader(train_path, args.batch_size, shuffle=False)
    val_loader = make_loader(val_path, args.batch_size, shuffle=False)
    has_test = os.path.exists(test_path)
    test_loader = make_loader(test_path, args.batch_size, shuffle=False) if has_test else None

    out = {}

    # --- Task 1: regime classification (market-wide, pooled) ---
    V_tr, G_tr, Z_tr, y_tr = extract_regime(model, train_loader, device)
    V_va, G_va, Z_va, y_va = extract_regime(model, val_loader, device)
    train_feats = {"V": V_tr, "G": G_tr, "Z": Z_tr}
    val_feats = {"V": V_va, "G": G_va, "Z": Z_va}
    if has_test:
        V_te, G_te, Z_te, y_te = extract_regime(model, test_loader, device)
        test_feats = {"V": V_te, "G": G_te, "Z": Z_te}
    for name in ("V", "G", "Z"):
        probe, _ = train_probe(train_feats[name], y_tr, val_feats[name], y_va,
                                in_dim=train_feats[name].shape[1], device=device)
        split_X, split_y = (test_feats[name], y_te) if has_test else (val_feats[name], y_va)
        metrics = evaluate_probe(probe, split_X, split_y, device)
        out[f"task1_{name}_f1"] = metrics["f1"]
        out[f"task1_{name}_acc"] = metrics["accuracy"]

    # --- Task 2: per-asset volatility quartile ---
    Vp_tr, Gp_tr, Zp_tr, vol_tr = extract_per_asset(model, train_loader, device)
    Vp_va, Gp_va, Zp_va, vol_va = extract_per_asset(model, val_loader, device)
    train_feats2 = {"V": Vp_tr, "G": Gp_tr, "Z": Zp_tr}
    val_feats2 = {"V": Vp_va, "G": Gp_va, "Z": Zp_va}
    if has_test:
        Vp_te, Gp_te, Zp_te, vol_te = extract_per_asset(model, test_loader, device)
        test_feats2 = {"V": Vp_te, "G": Gp_te, "Z": Zp_te}
        (yv_tr, yv_va, yv_te), _ = binarize_top_quartile(vol_tr, vol_va, vol_te)
    else:
        (yv_tr, yv_va), _ = binarize_top_quartile(vol_tr, vol_va)
    for name in ("V", "G", "Z"):
        probe, _ = train_probe(train_feats2[name], yv_tr, val_feats2[name], yv_va,
                                in_dim=train_feats2[name].shape[1], device=device)
        split_X, split_y = (test_feats2[name], yv_te) if has_test else (val_feats2[name], yv_va)
        metrics = evaluate_probe(probe, split_X, split_y, device)
        out[f"task2_{name}_f1"] = metrics["f1"]
        out[f"task2_{name}_acc"] = metrics["accuracy"]

    return out


# ================================================================
# 3. Main: loop over seeds x fusion_types, aggregate, test significance
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pretrained_encoder", default=None)
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", default="./ablation_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ablation] device={device}  seeds={args.seeds}  "
          f"fusion_types=['concat','gated']  epochs<={args.epochs}")

    train_path = os.path.join(args.data_dir, "train.pt")
    val_path = os.path.join(args.data_dir, "val.pt")
    test_path = os.path.join(args.data_dir, "test.pt")

    rho_star = calibrate_from_train(train_path)

    # raw[fusion_type][metric_key] = list of values across seeds
    raw = {"concat": {}, "gated": {}}

    for seed in args.seeds:
        for fusion_type in ("concat", "gated"):
            print(f"\n[ablation] seed={seed}  fusion_type={fusion_type} ...")
            model, best_val = train_one_run(seed, fusion_type, args, rho_star,
                                             train_path, val_path, device)
            print(f"  best_val_loss={best_val:.4f}")
            metrics = evaluate_downstream(model, args, train_path, val_path, test_path, device)
            metrics["val_loss"] = best_val
            for k, v in metrics.items():
                raw[fusion_type].setdefault(k, []).append(v)
            print(f"  task1_Z_f1={metrics['task1_Z_f1']:.4f}  "
                  f"task2_Z_f1={metrics['task2_Z_f1']:.4f}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # --- aggregate: mean +/- std ---
    summary = {}
    for fusion_type in ("concat", "gated"):
        summary[fusion_type] = {}
        for key, vals in raw[fusion_type].items():
            vals = np.array(vals)
            summary[fusion_type][key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}

    print("\n" + "=" * 70)
    print("  Ablation Summary (mean +/- std across seeds)")
    print("=" * 70)
    for task_name, key_prefix in [("Task 1 (regime)", "task1"), ("Task 2 (volatility)", "task2")]:
        print(f"\n{task_name}")
        for view in ("V", "G", "Z"):
            k = f"{key_prefix}_{view}_f1"
            c = summary["concat"][k]
            g = summary["gated"][k]
            print(f"  {view}: concat F1={c['mean']:.4f}+/-{c['std']:.4f}  "
                  f"gated F1={g['mean']:.4f}+/-{g['std']:.4f}  (n={c['n']} seeds)")

    # --- paired significance test: gated vs concat, on Z's F1, matched by seed ---
    print("\n--- Paired significance test (gated vs concat, Z's F1, matched by seed) ---")
    sig_results = {}
    for task_name, key in [("Task 1 (regime)", "task1_Z_f1"), ("Task 2 (volatility)", "task2_Z_f1")]:
        concat_vals = np.array(raw["concat"][key])
        gated_vals = np.array(raw["gated"][key])
        if len(concat_vals) >= 2:
            t_stat, p_val = stats.ttest_rel(gated_vals, concat_vals)
            diff = gated_vals.mean() - concat_vals.mean()
            sig_results[task_name] = {"diff": float(diff), "t_stat": float(t_stat), "p_value": float(p_val)}
            sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
            print(f"  {task_name}: gated - concat = {diff:+.4f}  "
                  f"(paired t={t_stat:.3f}, p={p_val:.4f})  -> {sig}")
        else:
            print(f"  {task_name}: need >=2 seeds for a significance test (got {len(concat_vals)})")

    # --- save everything ---
    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(raw, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "significance": sig_results}, f, indent=2)
    print(f"\n[ablation] saved raw_results.json and summary.json -> {args.out_dir}")

    # --- plot: bar chart with error bars, concat vs gated, both tasks, Z only ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task_name, key) in zip(axes, [("Task 1: Regime (Z)", "task1_Z_f1"),
                                            ("Task 2: Volatility (Z)", "task2_Z_f1")]):
        means = [summary["concat"][key]["mean"], summary["gated"][key]["mean"]]
        stds = [summary["concat"][key]["std"], summary["gated"][key]["std"]]
        ax.bar(["concat", "gated"], means, yerr=stds, capsize=8,
               color=["gray", "mediumseagreen"], alpha=0.85)
        ax.set_ylabel("F1 (test)")
        ax.set_title(task_name)
        ax.set_ylim(0, 1.0)
        if task_name.split(":")[0].strip().replace(" ", "").lower().startswith("task1"):
            pass
    plt.suptitle(f"Fusion Ablation: concat vs gated (n={len(args.seeds)} seeds, "
                 f"mean +/- std)", fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "ablation_fusion_comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[ablation] saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
