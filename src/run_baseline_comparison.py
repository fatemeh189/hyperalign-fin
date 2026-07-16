"""
Baseline Comparison: HyperAlign-Fin vs. TS2Vec
==================================================
Trains TS2Vec for the SAME n seeds already used for HyperAlign-Fin
(run_multiseed_ablation.py), evaluates it on the SAME two downstream
tasks with the SAME linear-probe protocol, and runs a paired
significance test (matched by seed) against HyperAlign-Fin's Z
(read from the already-saved raw_results.json -- no need to retrain
HyperAlign-Fin again).

Usage:
    python run_baseline_comparison.py \
        --data_dir ./cache \
        --hyperalign_raw_results ./ablation_results/raw_results.json \
        --hyperalign_fusion_type concat \
        --n_assets 119 --seeds 0 1 2 3 4 \
        --out_dir ./baseline_results
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
from baseline_ts2vec import pretrain_ts2vec, extract_ts2vec_embedding, check_embedding_health
from downstream_asset_volatility import binarize_top_quartile
from downstream_regime_classification import train_probe, evaluate_probe


# ================================================================
# 1. Extract TS2Vec embeddings + labels for both downstream tasks
# ================================================================

@torch.no_grad()
def extract_regime_from_ts2vec(encoder, loader, device):
    all_Z, all_rho_mean = [], []
    for batch in loader:
        price_window = batch["price_window"]
        rho_mean = batch["rho_mean"]
        emb = extract_ts2vec_embedding(encoder, price_window, device)   # (B,N,d)
        pooled = emb.mean(dim=1).cpu()                                   # market-wide pool
        all_Z.append(pooled)
        all_rho_mean.append(rho_mean)   # thresholded into a label by the caller (needs rho_star)
    return torch.cat(all_Z), torch.cat(all_rho_mean)


@torch.no_grad()
def extract_volatility_from_ts2vec(encoder, loader, device):
    from hyperalign_model import build_node_features
    all_Z, all_vol = [], []
    for batch in loader:
        price_window = batch["price_window"]
        emb = extract_ts2vec_embedding(encoder, price_window, device)    # (B,N,d)
        B, N, d = emb.shape
        all_Z.append(emb.reshape(B * N, d).cpu())
        node_feat = build_node_features(price_window)
        all_vol.append(node_feat[:, :, 1].reshape(B * N))  # std_r
    return torch.cat(all_Z), torch.cat(all_vol)


# ================================================================
# 2. Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--hyperalign_raw_results", required=True,
                    help="raw_results.json from run_multiseed_ablation.py")
    p.add_argument("--hyperalign_fusion_type", default="concat")
    p.add_argument("--n_assets", type=int, required=True)
    p.add_argument("--repr_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask_prob", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", default="./baseline_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[baseline] device={device}  seeds={args.seeds}")

    train_path = os.path.join(args.data_dir, "train.pt")
    val_path = os.path.join(args.data_dir, "val.pt")
    test_path = os.path.join(args.data_dir, "test.pt")
    has_test = os.path.exists(test_path)

    # --- resume support, same pattern as run_multiseed_ablation.py ---
    partial_path = os.path.join(args.out_dir, "ts2vec_raw_partial.json")
    raw = {}
    completed = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            saved = json.load(f)
        raw = saved["raw"]
        completed = set(saved["completed"])
        print(f"[baseline] RESUMING: {len(completed)} seeds already done.")

    def save_partial():
        with open(partial_path, "w") as f:
            json.dump({"raw": raw, "completed": list(completed)}, f, indent=2)

    for seed in args.seeds:
        if seed in completed:
            print(f"[baseline] seed={seed} -- SKIPPED (already done)")
            continue

        print(f"\n[baseline] seed={seed} -- pretraining TS2Vec ...")
        train_loader_pt = make_loader(train_path, args.batch_size, shuffle=True)
        encoder = pretrain_ts2vec(
            train_loader_pt, seed=seed, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, repr_dim=args.repr_dim,
            mask_prob=args.mask_prob, device=device,
        )

        train_loader = make_loader(train_path, args.batch_size, shuffle=False)
        val_loader = make_loader(val_path, args.batch_size, shuffle=False)
        test_loader = make_loader(test_path, args.batch_size, shuffle=False) if has_test else None

        health = check_embedding_health(encoder, train_loader, device)

        metrics = {"embedding_collapsed": health["collapsed"],
                   "embedding_effective_rank": health["effective_rank"]}

        # --- Task 1: regime classification ---
        # rebuild labels exactly like HyperAlignFin's RegimeAwareAlignment:
        # regime = rho_mean > rho_star. Read rho_star from the HyperAlign-Fin
        # results file's calibration record if present, else recompute.
        from train_hyperalign import calibrate_from_train
        rho_star = calibrate_from_train(train_path)

        def pooled_and_labels(loader, enc=encoder):
            Z, rho = extract_regime_from_ts2vec(enc, loader, device)
            y = (rho > rho_star).long()
            return Z, y

        Z_tr, y_tr = pooled_and_labels(train_loader)
        Z_va, y_va = pooled_and_labels(val_loader)
        probe, _ = train_probe(Z_tr, y_tr, Z_va, y_va, in_dim=Z_tr.shape[1], device=device)
        if has_test:
            Z_te, y_te = pooled_and_labels(test_loader)
            m = evaluate_probe(probe, Z_te, y_te, device)
        else:
            m = evaluate_probe(probe, Z_va, y_va, device)
        metrics["task1_f1"] = m["f1"]
        metrics["task1_acc"] = m["accuracy"]

        # --- Task 2: per-asset volatility quartile ---
        Zp_tr, vol_tr = extract_volatility_from_ts2vec(encoder, train_loader, device)
        Zp_va, vol_va = extract_volatility_from_ts2vec(encoder, val_loader, device)
        if has_test:
            Zp_te, vol_te = extract_volatility_from_ts2vec(encoder, test_loader, device)
            (yv_tr, yv_va, yv_te), _ = binarize_top_quartile(vol_tr, vol_va, vol_te)
        else:
            (yv_tr, yv_va), _ = binarize_top_quartile(vol_tr, vol_va)
        probe2, _ = train_probe(Zp_tr, yv_tr, Zp_va, yv_va, in_dim=Zp_tr.shape[1], device=device)
        split_X, split_y = (Zp_te, yv_te) if has_test else (Zp_va, yv_va)
        m2 = evaluate_probe(probe2, split_X, split_y, device)
        metrics["task2_f1"] = m2["f1"]
        metrics["task2_acc"] = m2["accuracy"]

        print(f"  task1_f1={metrics['task1_f1']:.4f}  task2_f1={metrics['task2_f1']:.4f}")
        if health["collapsed"]:
            print(f"  *** WARNING: seed={seed}'s TS2Vec embedding collapsed "
                  f"(effective_rank={health['effective_rank']:.1f}). The F1 scores "
                  f"above are not a fair reflection of TS2Vec's real capability -- "
                  f"do not report this run without fixing training first "
                  f"(try more epochs, lower mask_prob, or lower lr).")

        for k, v in metrics.items():
            raw.setdefault(k, []).append(v)
        completed.add(seed)
        save_partial()
        print(f"  [saved progress: {len(completed)}/{len(args.seeds)} seeds done]")

        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- load HyperAlign-Fin's results for comparison ---
    with open(args.hyperalign_raw_results) as f:
        hyperalign_raw = json.load(f)[args.hyperalign_fusion_type]

    n_collapsed = sum(raw.get("embedding_collapsed", []))
    print("\n" + "=" * 70)
    if n_collapsed > 0:
        print(f"  *** WARNING: {n_collapsed}/{len(args.seeds)} TS2Vec runs showed "
              f"embedding collapse. The comparison below may be UNFAIR to TS2Vec. ***")
        print("  *** Fix training (more epochs / lower mask_prob / lower lr) before "
              "citing these numbers in the paper. ***")
    print("  HyperAlign-Fin (Z) vs. TS2Vec baseline  (paired by seed)")
    print("=" * 70)

    summary = {}
    for task_key, task_name in [("task1", "Task 1: Regime Classification"),
                                 ("task2", "Task 2: Volatility Classification")]:
        hf = np.array(hyperalign_raw[f"{task_key}_Z_f1"])
        ts = np.array(raw[f"{task_key}_f1"])
        n = min(len(hf), len(ts))
        hf, ts = hf[:n], ts[:n]

        print(f"\n{task_name} (n={n} matched seeds)")
        print(f"  HyperAlign-Fin (Z): {hf.mean():.4f} +/- {hf.std(ddof=1) if n>1 else 0:.4f}")
        print(f"  TS2Vec baseline:    {ts.mean():.4f} +/- {ts.std(ddof=1) if n>1 else 0:.4f}")

        if n >= 2:
            t_stat, p_val = stats.ttest_rel(hf, ts)
            try:
                _, w_p = stats.wilcoxon(hf, ts)
            except ValueError:
                w_p = float("nan")
            diff = hf.mean() - ts.mean()
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f"  diff (HyperAlign-Fin - TS2Vec): {diff:+.4f}")
            print(f"  paired t-test: t={t_stat:.3f}  p={p_val:.4f}  ({sig})")
            print(f"  Wilcoxon cross-check: p={w_p:.4f}")
            summary[task_name] = {"hyperalign_mean": float(hf.mean()), "ts2vec_mean": float(ts.mean()),
                                   "diff": float(diff), "p_ttest": float(p_val), "p_wilcoxon": float(w_p)}
        else:
            print("  need >=2 seeds for significance test")

    with open(os.path.join(args.out_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task_key, task_name) in zip(axes, [("task1", "Task 1: Regime"),
                                                  ("task2", "Task 2: Volatility")]):
        hf = np.array(hyperalign_raw[f"{task_key}_Z_f1"])
        ts = np.array(raw[f"{task_key}_f1"])
        n = min(len(hf), len(ts))
        means = [hf[:n].mean(), ts[:n].mean()]
        stds = [hf[:n].std(ddof=1) if n > 1 else 0, ts[:n].std(ddof=1) if n > 1 else 0]
        ax.bar(["HyperAlign-Fin", "TS2Vec"], means, yerr=stds, capsize=8,
               color=["mediumseagreen", "gray"], alpha=0.85)
        ax.set_ylabel("F1 (test)")
        ax.set_title(task_name)
        ax.set_ylim(0, 1.0)
    plt.suptitle(f"HyperAlign-Fin vs. TS2Vec (n={len(args.seeds)} seeds, mean +/- std)",
                 fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "baseline_comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n[baseline] saved figure -> {fig_path}")
    print(f"[baseline] saved comparison_summary.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
