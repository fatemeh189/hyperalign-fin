"""
Master Orchestrator: runs the ENTIRE HyperAlign-Fin pipeline end-to-end
===========================================================================
Runs, in order, every stage already built and tested in this project:

    1. prepare_real_arrays.py    -> raw_arrays/{prices,dates,sectors}.npy
    2. data_pipeline.py          -> cache/{train,val,test}.pt
    3. hyperalign_validation.py -> theory validation figures (Test A/B/C)
    4. pretrain_byol.py          -> checkpoints/gaf_byol.pt
    5. train_hyperalign.py       -> run_final/{best_checkpoint.pt, loss_curve.png}
    6. downstream_regime_classification.py -> downstream_regime/*.png
    7. downstream_asset_volatility.py      -> downstream_volatility/*.png
    8. run_multiseed_ablation.py           -> ablation_results/*.png (n seeds)
    9. run_baseline_comparison.py          -> baseline_results/*.png (n seeds)

Each stage is SKIPPED if its expected output already exists, so this
script is safe to re-run after a Colab disconnect -- it resumes rather
than restarting from scratch. At the end, every .png produced anywhere
under the working directory is copied into ./all_figures/ with a
manifest listing what each one is and which paper section it belongs to.

Usage (from the directory containing all the .py files from this project):
    python run_all.py --n_assets 150 --seq_len 20 --n_seeds 5

This calls the existing, already-tested scripts as subprocesses (not
reimplementations), so a bug fix in any individual script is
automatically picked up here too.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def run_stage(name: str, cmd: list, skip_if_exists: str, log_dir: str) -> bool:
    """
    Runs `cmd` (a list, as for subprocess.run) unless `skip_if_exists`
    (a file or directory path) is already present. Streams output live
    and also saves it to log_dir/<name>.log. Returns True on success.
    """
    if skip_if_exists and os.path.exists(skip_if_exists):
        print(f"\n{'='*70}\n[SKIP] {name} -- found existing output at {skip_if_exists}\n{'='*70}")
        return True

    print(f"\n{'='*70}\n[RUN] {name}\n  $ {' '.join(cmd)}\n{'='*70}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    t0 = time.time()

    with open(log_path, "w") as logf:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
            logf.write(line)
        process.wait()
    dt = time.time() - t0

    if process.returncode != 0:
        print(f"\n[FAIL] {name} exited with code {process.returncode} "
              f"after {dt:.0f}s. See {log_path} for the full log.")
        return False

    print(f"\n[OK] {name} finished in {dt:.0f}s")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_assets", type=int, default=150,
                    help="requested ticker count (actual count after dropna will be lower)")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--train_epochs", type=int, default=100)
    p.add_argument("--byol_epochs", type=int, default=50)
    p.add_argument("--ablation_epochs", type=int, default=60)
    p.add_argument("--ts2vec_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--work_dir", default=".", help="directory containing the .py scripts")
    args = p.parse_args()

    os.chdir(args.work_dir)
    py = sys.executable
    log_dir = "./run_all_logs"
    seeds_arg = [str(s) for s in range(args.n_seeds)]

    stages_ok = {}

    # --- Stage 1: raw arrays from real S&P data ---
    stages_ok["prepare_real_arrays"] = run_stage(
        "prepare_real_arrays",
        [py, "prepare_real_arrays.py", "--n_assets", str(args.n_assets), "--out_dir", "./raw_arrays"],
        skip_if_exists="./raw_arrays/prices.npy",
        log_dir=log_dir,
    )

    # --- Stage 2: build train/val/test cache ---
    stages_ok["data_pipeline"] = run_stage(
        "data_pipeline",
        [py, "data_pipeline.py", "--prices", "./raw_arrays/prices.npy",
         "--dates", "./raw_arrays/dates.npy", "--sectors", "./raw_arrays/sectors.npy",
         "--out_dir", "./cache", "--T", str(args.seq_len), "--tau", "0.5"],
        skip_if_exists="./cache/train.pt",
        log_dir=log_dir,
    )

    # --- Stage 3: theory validation (Test A/B/C figures) ---
    stages_ok["theory_validation"] = run_stage(
        "theory_validation",
        [py, "hyperalign_validation.py"],
        skip_if_exists="./test_C_regime.png",
        log_dir=log_dir,
    )

    # --- Determine actual N after dropna (needed by every stage below) ---
    import numpy as np
    n_actual = args.n_assets
    if os.path.exists("./raw_arrays/prices.npy"):
        n_actual = int(np.load("./raw_arrays/prices.npy").shape[1])
        print(f"\n[INFO] Actual asset count after dropna: N={n_actual} "
              f"(requested {args.n_assets})")

    # --- Stage 4: BYOL pretraining ---
    stages_ok["byol"] = run_stage(
        "byol",
        [py, "pretrain_byol.py", "--data", "./cache/train.pt",
         "--epochs", str(args.byol_epochs), "--out", "./checkpoints/gaf_byol.pt",
         "--seq_len", str(args.seq_len), "--latent_dim", str(args.latent_dim)],
        skip_if_exists="./checkpoints/gaf_byol.pt",
        log_dir=log_dir,
    )

    # --- Stage 5: main HyperAlign-Fin training (final architecture: concat fusion) ---
    stages_ok["train_final"] = run_stage(
        "train_final",
        [py, "train_hyperalign.py", "--data_dir", "./cache", "--out_dir", "./run_final",
         "--n_assets", str(n_actual), "--seq_len", str(args.seq_len),
         "--latent_dim", str(args.latent_dim), "--epochs", str(args.train_epochs),
         "--batch_size", str(args.batch_size), "--weight_decay", str(args.weight_decay),
         "--pretrained_encoder", "./checkpoints/gaf_byol.pt"],
        skip_if_exists="./run_final/best_checkpoint.pt",
        log_dir=log_dir,
    )

    # --- Stage 6: downstream task 1 ---
    stages_ok["downstream_regime"] = run_stage(
        "downstream_regime",
        [py, "downstream_regime_classification.py", "--checkpoint", "./run_final/best_checkpoint.pt",
         "--data_dir", "./cache", "--n_assets", str(n_actual), "--seq_len", str(args.seq_len),
         "--latent_dim", str(args.latent_dim), "--out_dir", "./downstream_regime"],
        skip_if_exists="./downstream_regime/regime_classification_results.png",
        log_dir=log_dir,
    )

    # --- Stage 7: downstream task 2 ---
    stages_ok["downstream_volatility"] = run_stage(
        "downstream_volatility",
        [py, "downstream_asset_volatility.py", "--checkpoint", "./run_final/best_checkpoint.pt",
         "--data_dir", "./cache", "--n_assets", str(n_actual), "--seq_len", str(args.seq_len),
         "--latent_dim", str(args.latent_dim), "--out_dir", "./downstream_volatility"],
        skip_if_exists="./downstream_volatility/volatility_classification_results.png",
        log_dir=log_dir,
    )

    # --- Stage 8: multi-seed ablation (concat vs gated) ---
    stages_ok["ablation"] = run_stage(
        "ablation",
        [py, "run_multiseed_ablation.py", "--data_dir", "./cache",
         "--pretrained_encoder", "./checkpoints/gaf_byol.pt", "--n_assets", str(n_actual),
         "--seq_len", str(args.seq_len), "--latent_dim", str(args.latent_dim),
         "--epochs", str(args.ablation_epochs), "--batch_size", str(args.batch_size),
         "--weight_decay", str(args.weight_decay), "--seeds", *seeds_arg,
         "--out_dir", "./ablation_results"],
        skip_if_exists="./ablation_results/ablation_fusion_comparison.png",
        log_dir=log_dir,
    )

    # --- Stage 9: Corollary 2 significance analysis (cheap, reuses ablation data) ---
    stages_ok["corollary2_analysis"] = run_stage(
        "corollary2_analysis",
        [py, "analyze_corollary2_significance.py",
         "--raw_results", "./ablation_results/raw_results.json", "--fusion_type", "concat"],
        skip_if_exists="",  # always cheap to rerun, no figure produced
        log_dir=log_dir,
    )

    # --- Stage 10: baseline comparison vs TS2Vec ---
    stages_ok["baseline"] = run_stage(
        "baseline",
        [py, "run_baseline_comparison.py", "--data_dir", "./cache",
         "--hyperalign_raw_results", "./ablation_results/raw_results.json",
         "--hyperalign_fusion_type", "concat", "--n_assets", str(n_actual),
         "--epochs", str(args.ts2vec_epochs), "--batch_size", str(args.batch_size),
         "--seeds", *seeds_arg, "--out_dir", "./baseline_results"],
        skip_if_exists="./baseline_results/baseline_comparison.png",
        log_dir=log_dir,
    )

    # --- Collect every figure produced anywhere into one folder ---
    os.makedirs("./all_figures", exist_ok=True)
    manifest = {}
    figure_sections = {
        "test_A_independence.png": "Section 4.2 (Test A)",
        "test_B_orthogonality.png": "Section 4.3 (Test B)",
        "test_B_pca_sensitivity.png": "Section 4.3 (Test B, sensitivity)",
        "test_C_regime.png": "Section 4.4 (Test C)",
        "sensitivity_rho_star.png": "Appendix (optional)",
        "loss_curve.png": "Section 5.1 (Implementation)",
        "regime_classification_results.png": "Section 5.2 (Downstream Task 1)",
        "volatility_classification_results.png": "Section 5.2 (Downstream Task 2)",
        "ablation_fusion_comparison.png": "Section 5.3 (Ablation)",
        "baseline_comparison.png": "Section 5.5 (Baseline Comparison)",
    }
    for png_path in glob.glob("**/*.png", recursive=True):
        if png_path.startswith("all_figures"):
            continue
        fname = os.path.basename(png_path)
        dest = os.path.join("./all_figures", fname)
        # avoid overwriting same-named figures from different runs -- keep the newest
        if os.path.exists(dest) and os.path.getmtime(png_path) <= os.path.getmtime(dest):
            continue
        shutil.copy2(png_path, dest)
        manifest[fname] = {
            "source": png_path,
            "suggested_section": figure_sections.get(fname, "unclassified -- check manually"),
        }

    with open("./all_figures/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Final report ---
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE -- Stage Status")
    print("=" * 70)
    for stage, ok in stages_ok.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {stage}")
    all_ok = all(stages_ok.values())
    print()
    print(f"Figures collected in ./all_figures/ ({len(manifest)} files):")
    for fname, info in manifest.items():
        print(f"  {fname:<45} -> {info['suggested_section']}")
    print()
    if all_ok:
        print("All stages completed successfully.")
    else:
        failed = [s for s, ok in stages_ok.items() if not ok]
        print(f"WARNING: the following stages FAILED: {failed}")
        print("Check ./run_all_logs/<stage>.log for details, fix, and re-run this "
              "script -- completed stages will be skipped automatically.")


if __name__ == "__main__":
    main()
