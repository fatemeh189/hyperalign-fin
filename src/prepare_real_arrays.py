"""
Bridge script: real S&P-500 data -> arrays ready for data_pipeline.py
========================================================================
Reuses load_real_data() / get_real_sector_labels() from
hyperalign_validation.py (same tickers, same sector cache) so the
model training pipeline uses EXACTLY the same data as the empirical
validation report -- no re-downloading, no risk of drift between the
two.

Usage:
    python prepare_real_arrays.py --n_assets 150 --out_dir ./raw_arrays

Then feed the output into the pipeline you already have:
    python data_pipeline.py --prices raw_arrays/prices.npy \
        --dates raw_arrays/dates.npy --sectors raw_arrays/sectors.npy \
        --out_dir ./cache --T 20 --tau 0.5
    python pretrain_byol.py --data ./cache/train.pt --epochs 50 \
        --out ./checkpoints/gaf_byol.pt
    python train_hyperalign.py --data_dir ./cache --out_dir ./run1 \
        --n_assets <N printed below> --seq_len 20 --epochs 100 \
        --pretrained_encoder ./checkpoints/gaf_byol.pt
"""

from __future__ import annotations
import argparse
import os
import numpy as np

from hyperalign_validation import load_real_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_assets", type=int, default=150)
    p.add_argument("--out_dir", default="./raw_arrays")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    prices, sector_labels, n_assets, dates = load_real_data(args.n_assets)

    np.save(os.path.join(args.out_dir, "prices.npy"), prices)
    np.save(os.path.join(args.out_dir, "dates.npy"), dates)
    np.save(os.path.join(args.out_dir, "sectors.npy"), sector_labels)

    print(f"\n[prepare_real_arrays] Saved to {args.out_dir}/:")
    print(f"  prices.npy   {prices.shape}")
    print(f"  dates.npy    {dates.shape}  ({dates[0]} to {dates[-1]})")
    print(f"  sectors.npy  {sector_labels.shape}  "
          f"({len(np.unique(sector_labels))} unique sectors)")
    print(f"\n[prepare_real_arrays] IMPORTANT: use --n_assets {n_assets} "
          f"(not your original request) in train_hyperalign.py -- this is "
          f"the count AFTER dropna survivorship filtering.")


if __name__ == "__main__":
    main()
