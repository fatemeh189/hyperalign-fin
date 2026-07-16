"""
Corollary 2 Significance Analysis
=====================================
Answers the actual question the paper needs: does the fused Z beat the
best individual view (V or G), with a proper paired test -- not the
concat-vs-gated architecture question already answered by
run_multiseed_ablation.py.

Reads the raw_results.json ALREADY SAVED by run_multiseed_ablation.py
-- no retraining needed, this is pure statistics on existing data.

Usage:
    python analyze_corollary2_significance.py \
        --raw_results ./ablation_results/raw_results.json \
        --fusion_type concat
"""

from __future__ import annotations
import argparse
import json

import numpy as np
from scipy import stats


def paired_test(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str):
    """Paired t-test: is a significantly different from b? (matched by seed)"""
    diff = a - b
    if len(a) < 2:
        print(f"  {name_a} vs {name_b}: need >=2 seeds, got {len(a)}")
        return None
    t_stat, p_val = stats.ttest_rel(a, b)
    mean_diff = diff.mean()
    # Wilcoxon signed-rank as a non-parametric cross-check (robust to small n / non-normality)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    return {
        "mean_a": float(a.mean()), "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()), "std_b": float(b.std(ddof=1)),
        "mean_diff": float(mean_diff), "t_stat": float(t_stat), "p_value_ttest": float(p_val),
        "p_value_wilcoxon": float(w_p),
    }


def report(result, name_a, name_b, task_name):
    if result is None:
        return
    sig = result["p_value_ttest"] < 0.05
    direction = "beats" if result["mean_diff"] > 0 else "underperforms"
    print(f"\n  [{task_name}] {name_a} vs {name_b}:")
    print(f"    {name_a}: {result['mean_a']:.4f} +/- {result['std_a']:.4f}")
    print(f"    {name_b}: {result['mean_b']:.4f} +/- {result['std_b']:.4f}")
    print(f"    diff ({name_a} - {name_b}): {result['mean_diff']:+.4f}")
    print(f"    paired t-test:  t={result['t_stat']:.3f}  p={result['p_value_ttest']:.4f}")
    print(f"    Wilcoxon (non-parametric cross-check): p={result['p_value_wilcoxon']:.4f}")
    if sig:
        print(f"    -> SIGNIFICANT: {name_a} {direction} {name_b} (p<0.05)")
    else:
        print(f"    -> NOT significant (p>=0.05): cannot claim {name_a} {direction} {name_b}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_results", required=True,
                    help="path to raw_results.json from run_multiseed_ablation.py")
    p.add_argument("--fusion_type", default="concat", choices=["concat", "gated"],
                    help="which fusion architecture's seeds to analyze (use whichever "
                         "you decided is the final architecture)")
    args = p.parse_args()

    with open(args.raw_results) as f:
        raw = json.load(f)

    data = raw[args.fusion_type]
    n_seeds = len(data["task1_Z_f1"])
    print("=" * 70)
    print(f"  Corollary 2 Significance Analysis  (fusion_type={args.fusion_type}, "
          f"n={n_seeds} seeds)")
    print("=" * 70)
    print("\nQuestion: does the FUSED Z significantly beat the BEST individual")
    print("view (V or G), per task, with a proper paired test?")

    summary_rows = []
    for task_key, task_name in [("task1", "Task 1: Regime Classification"),
                                 ("task2", "Task 2: Volatility Classification")]:
        print(f"\n{'-'*70}\n{task_name}\n{'-'*70}")
        V = np.array(data[f"{task_key}_V_f1"])
        G = np.array(data[f"{task_key}_G_f1"])
        Z = np.array(data[f"{task_key}_Z_f1"])

        # per-seed "best individual view" (elementwise max of V, G at each seed)
        best_individual = np.maximum(V, G)
        which_is_best = "G" if G.mean() > V.mean() else "V"

        r_zv = paired_test(Z, V, "Z", "V")
        report(r_zv, "Z", "V", task_name)

        r_zg = paired_test(Z, G, "Z", "G")
        report(r_zg, "Z", "G", task_name)

        r_z_best = paired_test(Z, best_individual, "Z", f"max(V,G) [per-seed, mostly {which_is_best}]")
        report(r_z_best, "Z", "max(V,G) [per-seed]", task_name)

        verdict = "PASS" if (r_z_best and r_z_best["p_value_ttest"] < 0.05 and r_z_best["mean_diff"] > 0) else \
                  ("FAIL" if (r_z_best and r_z_best["p_value_ttest"] < 0.05 and r_z_best["mean_diff"] < 0) else "INCONCLUSIVE")
        print(f"\n  ==> Corollary 2 verdict for {task_name}: {verdict}")

        summary_rows.append({
            "task": task_name, "Z_mean": Z.mean(), "Z_std": Z.std(ddof=1),
            "best_individual_mean": best_individual.mean(),
            "p_value": r_z_best["p_value_ttest"] if r_z_best else None,
            "verdict": verdict,
        })

    print("\n" + "=" * 70)
    print("  Citable Summary Table")
    print("=" * 70)
    print(f"{'Task':<35} {'Z (mean+/-std)':<20} {'Best individual':<18} {'p-value':<10} {'Verdict'}")
    for row in summary_rows:
        print(f"{row['task']:<35} {row['Z_mean']:.4f}+/-{row['Z_std']:.4f}    "
              f"{row['best_individual_mean']:<18.4f} "
              f"{row['p_value']:<10.4f} {row['verdict']}")


if __name__ == "__main__":
    main()
