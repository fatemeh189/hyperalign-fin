"""
HyperAlign-Fin — Data Pipeline
================================
Builds a disk-cached, PyTorch-ready dataset from raw price data.

Reuses the EXACT formulas from hyperalign_validation.py (make_gaf,
hyperedge construction) so that numbers computed here are consistent
with the empirical validation report (Section 4).

Split convention (matches your existing production pipeline):
    train : window end-date <  2022-01-01
    val   : 2022-01-01 <= window end-date < 2023-01-01
    test  : window end-date >= 2023-01-01

Output: three files, train.pt / val.pt / test.pt, each a list of dicts:
    {
        "gaf":          FloatTensor (N, T, T)
        "incidence":    FloatTensor (N, E_w)   # E_w varies per window
        "hyperedge_w":  FloatTensor (E_w,)
        "corr":         FloatTensor (N, N)
        "price_window": FloatTensor (T, N)     # raw prices, for build_node_features
        "rho_mean":     float                  # mean |C_ij|, i != j (regime signal)
        "date":         np.datetime64
    }

Usage:
    python data_pipeline.py --prices prices.npy --dates dates.npy \
        --sectors sectors.npy --out_dir ./cache --T 20 --tau 0.5
"""

from __future__ import annotations
import argparse
import os
from typing import List, Dict, Any

import numpy as np
import torch


# ================================================================
# Core transforms — identical formulas to hyperalign_validation.py
# ================================================================

def make_gaf(series: np.ndarray) -> np.ndarray:
    """Gramian Angular Field. series: 1D array, length T. Returns (T,T)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return np.zeros((len(series), len(series)), dtype=np.float32)
    x = np.clip(2 * (series - mn) / (mx - mn) - 1, -1, 1)
    phi = np.arccos(x)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def get_hyperedges(price_window: np.ndarray, sector_labels: np.ndarray,
                    tau: float) -> List[np.ndarray]:
    """
    Returns a list of hyperedges (each an array of asset indices), combining
    static sector hyperedges and dynamic correlation hyperedges — same
    construction as build_hypergraph_laplacian() in hyperalign_validation.py.
    """
    T, N = price_window.shape
    rets = np.diff(np.log(price_window + 1e-8), axis=0)
    C = np.nan_to_num(np.corrcoef(rets.T), nan=0.0)

    edges: List[np.ndarray] = []
    for s in np.unique(sector_labels):
        m = np.where(sector_labels == s)[0]
        if len(m) >= 2:
            edges.append(m)

    visited = set()
    for i in range(N):
        cl = np.where(C[i] >= tau)[0]
        if len(cl) >= 2:
            key = frozenset(cl.tolist())
            if key not in visited:
                visited.add(key)
                edges.append(cl)

    if not edges:
        edges = [np.arange(N)]  # degenerate fallback: one edge containing everyone

    return edges, C


def edges_to_incidence(edges: List[np.ndarray], N: int) -> np.ndarray:
    """List of index arrays -> dense (N, E) incidence matrix."""
    E = len(edges)
    H = np.zeros((N, E), dtype=np.float32)
    for e_idx, members in enumerate(edges):
        H[members, e_idx] = 1.0
    return H


# ================================================================
# Dataset construction
# ================================================================

def build_windows(prices: np.ndarray, dates: np.ndarray, sector_labels: np.ndarray,
                   T: int, tau: float, stride: int = 1,
                   verbose: bool = True) -> List[Dict[str, Any]]:
    """
    prices:  (n_days, N) raw close prices
    dates:   (n_days,)   np.datetime64, aligned with prices rows
    sector_labels: (N,)  integer sector id per asset

    Returns a list of per-window sample dicts (see module docstring).
    """
    n_days, N = prices.shape
    n_w = (n_days - T) // stride
    samples = []

    for k in range(n_w):
        w = k * stride
        price_window = prices[w: w + T, :]                       # (T, N)
        end_date = dates[w + T - 1]

        gaf = np.stack(
            [make_gaf(price_window[:, i]) for i in range(N)], axis=0
        )                                                        # (N, T, T)

        edges, C = get_hyperedges(price_window, sector_labels, tau)
        incidence = edges_to_incidence(edges, N)                 # (N, E_w)
        hyperedge_w = np.ones(len(edges), dtype=np.float32)

        mask = ~np.eye(N, dtype=bool)
        rho_mean = float(np.mean(np.abs(C[mask])))

        samples.append({
            "gaf": torch.from_numpy(gaf),
            "incidence": torch.from_numpy(incidence),
            "hyperedge_w": torch.from_numpy(hyperedge_w),
            "corr": torch.from_numpy(C.astype(np.float32)),
            "price_window": torch.from_numpy(price_window.astype(np.float32)),
            "rho_mean": rho_mean,
            "date": end_date,
        })

        if verbose and k % 200 == 0:
            print(f"  window {k}/{n_w} ({end_date}) ...", end="\r")

    if verbose:
        print(f"\n[data_pipeline] built {len(samples)} windows "
              f"(N={N}, T={T}, tau={tau}, stride={stride})")
    return samples


def split_by_date(samples: List[Dict[str, Any]],
                   train_end: str = "2022-01-01",
                   val_end: str = "2023-01-01"):
    """Chronological split — matches the project's existing convention."""
    train_end = np.datetime64(train_end)
    val_end = np.datetime64(val_end)

    train, val, test = [], [], []
    for s in samples:
        d = np.datetime64(s["date"])
        if d < train_end:
            train.append(s)
        elif d < val_end:
            val.append(s)
        else:
            test.append(s)
    return train, val, test


def save_splits(train, val, test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    torch.save(test, os.path.join(out_dir, "test.pt"))
    print(f"[data_pipeline] saved: train={len(train)}  val={len(val)}  "
          f"test={len(test)}  -> {out_dir}")


# ================================================================
# CLI entry point
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True, help=".npy file, shape (n_days, N)")
    p.add_argument("--dates", required=True, help=".npy file of np.datetime64, shape (n_days,)")
    p.add_argument("--sectors", required=True, help=".npy file, shape (N,), integer sector ids")
    p.add_argument("--out_dir", default="./cache")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train_end", default="2022-01-01")
    p.add_argument("--val_end", default="2023-01-01")
    args = p.parse_args()

    prices = np.load(args.prices)
    dates = np.load(args.dates, allow_pickle=True)
    sectors = np.load(args.sectors)

    assert prices.shape[0] == dates.shape[0], "prices and dates must have the same length"
    assert prices.shape[1] == sectors.shape[0], "sectors length must match number of assets"

    samples = build_windows(prices, dates, sectors, T=args.T, tau=args.tau, stride=args.stride)
    train, val, test = split_by_date(samples, args.train_end, args.val_end)

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty split detected (train={len(train)}, val={len(val)}, "
            f"test={len(test)}). Check --train_end/--val_end against your date range."
        )

    save_splits(train, val, test, args.out_dir)


if __name__ == "__main__":
    main()
