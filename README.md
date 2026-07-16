# HyperAlign-Fin

Code accompanying the manuscript **"HyperAlign-Fin: Near-Orthogonal Multi-View Alignment for Financial Time Series"** (Lakzaie & Bahiraie).

HyperAlign-Fin pairs a Gramian Angular Field (GAF) encoding of an asset's price window with a hypergraph neural network (HGNN) encoding of its sector and dynamic correlation structure, aligned through a regime-aware contrastive objective. See the manuscript for the theoretical background (Theorems 1–3, Corollaries 1–3) and empirical validation (Section 4).

## Repository structure

All code lives in a single flat folder, `src/`, because several files import each other directly by module name (e.g. `from hyperalign_model import ...`) rather than as a package — keeping everything in one folder means the code runs as-is with no `PYTHONPATH` or `__init__.py` setup required.

```
hyperalign-fin/
├── README.md
├── requirements.txt
└── src/
    ├── hyperalign_validation.py            # Theory validation: Tests A, B, C (Section 4.2)
    ├── prepare_real_arrays.py              # Downloads real S&P 500 data -> raw arrays
    ├── data_pipeline.py                    # Raw arrays -> GAF + hypergraph windows -> train/val/test .pt
    ├── dataset.py                          # PyTorch Dataset / DataLoader, custom collate_fn
    ├── hyperalign_model.py                 # HyperAlign-Fin architecture (encoders, fusion)
    ├── pretrain_byol.py                    # BYOL self-supervised pretraining for the visual encoder
    ├── train_hyperalign.py                 # Main training loop (contrastive alignment, Corollary 1)
    ├── sanity_check.py                     # Post-training embedding-collapse diagnostic
    ├── downstream_regime_classification.py # Downstream Task 1 (Corollary 2)
    ├── downstream_asset_volatility.py      # Downstream Task 2 (Corollary 2, complementary check)
    ├── baseline_ts2vec.py                  # External baseline: TS2Vec (Yue et al., 2022)
    ├── baseline_factorcl.py                # External baseline: FactorCL-inspired fusion (Liang et al., 2023)
    ├── run_multiseed_ablation.py           # Multi-seed concat-vs-gated ablation + significance test
    ├── run_baseline_comparison.py          # Multi-seed HyperAlign-Fin vs. TS2Vec comparison
    ├── run_factorcl_comparison.py          # Multi-seed HyperAlign-Fin vs. FactorCL-inspired comparison
    ├── analyze_corollary2_significance.py  # Paired significance tests on saved ablation results
    ├── sensitivity_regime_percentile.py    # Sensitivity of the regime threshold to calibration percentile
    └── run_all.py                          # Orchestrates every stage above, end-to-end, with resume support
```

## Setup

```bash
git clone https://github.com/<your-username>/hyperalign-fin.git
cd hyperalign-fin
pip install -r requirements.txt
```

Python 3.10+ and a working PyTorch installation (CPU is fine for the dataset sizes used in the paper; GPU recommended for the multi-seed ablation and baseline comparisons) are assumed.

## Quickest path: run everything at once

```bash
cd src
python run_all.py --n_assets 150 --seq_len 20 --n_seeds 5
```

This runs every stage below in order, skips any stage whose output already exists (safe to re-run after an interruption), and collects every figure produced into `./all_figures/` with a manifest mapping each one to its paper section.

## Manual pipeline (if you want to run or inspect stages individually)

Run these from inside `src/`.

**1. Validate the theoretical claims on real data (Section 4.2, Tests A–C)**
```bash
python hyperalign_validation.py
```
Downloads S&P 500 data via `yfinance`, caches sector metadata (`sector_cache.json`) and correlation matrices (`corr_matrices_cache.npy`, `window_end_dates_cache.npy` — also used by `sensitivity_regime_percentile.py` below), and produces the Test A/B/C figures plus a console report.

**2. Prepare raw arrays for the model pipeline**
```bash
python prepare_real_arrays.py --n_assets 150 --out_dir ./raw_arrays
```
Reuses the exact same tickers/sector cache as step 1. Note the printed `n_assets` value (after dropna survivorship filtering) — use it in every step below.

**3. Build windowed GAF + hypergraph tensors and train/val/test splits**
```bash
python data_pipeline.py --prices raw_arrays/prices.npy --dates raw_arrays/dates.npy \
    --sectors raw_arrays/sectors.npy --out_dir ./cache --T 20 --tau 0.5
```

**4. BYOL-pretrain the visual encoder**
```bash
python pretrain_byol.py --data ./cache/train.pt --epochs 50 --out ./checkpoints/gaf_byol.pt
```

**5. Train HyperAlign-Fin**
```bash
python train_hyperalign.py --data_dir ./cache --out_dir ./run1 \
    --n_assets <N from step 2> --seq_len 20 --epochs 100 \
    --pretrained_encoder ./checkpoints/gaf_byol.pt
```
The regime threshold ρ* (Corollary 3) is calibrated automatically from the **training split only** — never hardcoded.

**6. Sanity-check the trained model**
```bash
python sanity_check.py --checkpoint ./run1/best_checkpoint.pt --data_dir ./cache \
    --n_assets <N> --seq_len 20
```
Checks for embedding collapse (a silent contrastive-learning failure mode) via effective rank and per-dimension variance — do this before trusting any downstream number below.

**7. Downstream evaluation (Corollary 2)**
```bash
python downstream_regime_classification.py --checkpoint ./run1/best_checkpoint.pt \
    --data_dir ./cache --n_assets <N> --seq_len 20
python downstream_asset_volatility.py --checkpoint ./run1/best_checkpoint.pt \
    --data_dir ./cache --n_assets <N> --seq_len 20
```
Each compares linear probes trained on V (visual only), G (graph only), and Z (fused).

**8. Multi-seed ablation: concat vs. gated fusion**
```bash
python run_multiseed_ablation.py --data_dir ./cache --pretrained_encoder ./checkpoints/gaf_byol.pt \
    --n_assets <N> --seq_len 20 --seeds 0 1 2 3 4 --epochs 60 --out_dir ./ablation_results
```
Produces `raw_results.json`, consumed by the three scripts below.

**9. Significance of Corollary 2 (fused Z vs. best individual view)**
```bash
python analyze_corollary2_significance.py --raw_results ./ablation_results/raw_results.json --fusion_type gated
```

**10. External baselines (multi-seed, paired significance vs. HyperAlign-Fin)**
```bash
python run_baseline_comparison.py --data_dir ./cache \
    --hyperalign_raw_results ./ablation_results/raw_results.json --hyperalign_fusion_type gated \
    --n_assets <N> --seeds 0 1 2 3 4 --out_dir ./baseline_results

python run_factorcl_comparison.py --data_dir ./cache --pretrained_encoder ./checkpoints/gaf_byol.pt \
    --hyperalign_raw_results ./ablation_results/raw_results.json --hyperalign_fusion_type gated \
    --n_assets <N> --seeds 0 1 2 3 4 --out_dir ./factorcl_results
```
`run_baseline_comparison.py` compares against TS2Vec (no relational access); `run_factorcl_comparison.py` compares against a FactorCL-inspired fusion sharing HyperAlign-Fin's own encoders, isolating the advantage to the alignment mechanism itself.

**11. Sensitivity of the regime threshold to the calibration percentile**
```bash
python sensitivity_regime_percentile.py
```
Requires the cache files produced in step 1. Sweeps percentiles 60–90 and justifies the 75th-percentile choice used in the paper.

## Data

All price data is downloaded on demand from Yahoo Finance via `yfinance` (S&P 500 constituents, 2015–2023) — no raw price data is redistributed in this repository, in keeping with data provider terms.

## Reproducibility notes

- ρ* is **always** calibrated from the training split only (`calibrate_rho_star` / `calibrate_from_train`); results are never computed against a threshold fit on validation or test data.
- `dataset.py` includes a self-check (`_test_collate_padding_is_safe`) verifying that batch-level zero-padding of variable hyperedge counts does not perturb the hypergraph Laplacian.
- `sanity_check.py` and `baseline_ts2vec.py`'s `check_embedding_health` both check for representation collapse before any comparison is reported — a common silent failure mode in contrastive learning.
- Downstream tasks report F1 (not just accuracy) as the deciding metric, since regime and volatility labels can be class-imbalanced.
- All multi-seed comparisons (`run_multiseed_ablation.py`, `run_baseline_comparison.py`, `run_factorcl_comparison.py`) use a paired test (t-test + Wilcoxon cross-check), matched by seed, rather than comparing unpaired means.
- All orchestration scripts (`run_all.py` and the `run_*` scripts) support resuming after an interruption: completed stages/seeds are skipped, not recomputed.

## Citation

If you use this code, please cite:

```bibtex
@article{lakzaie2026hyperalignfin,
  title   = {HyperAlign-Fin: Near-Orthogonal Multi-View Alignment for Financial Time Series},
  author  = {Lakzaie, Fatemeh and Bahiraie, Alireza},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

## License

Add a license (e.g., MIT) before making this repository public, if you intend for others to reuse the code freely.

## Contact

Alireza Bahiraie — alireza.bahiraie@semnan.ac.ir
