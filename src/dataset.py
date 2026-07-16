"""
HyperAlign-Fin — PyTorch Dataset / DataLoader
================================================
Loads the .pt files produced by data_pipeline.py and batches them.

Why a custom collate_fn:
    Each window has a different number of hyperedges E_w (sector edges are
    fixed, but correlation-cluster edges vary day to day). Rather than
    forcing a global fixed E across the whole dataset (wasteful, and
    brittle if train/test have different cluster counts), we pad only
    WITHIN each batch, to max(E_w) in that batch. Padding columns are
    all-zero, so they contribute exactly zero to the Laplacian
    (Dv, De are computed from row/column sums — an all-zero column
    changes neither), which is verified in test_collate_padding_is_safe()
    at the bottom of this file.
"""

from __future__ import annotations
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader


class HyperAlignDataset(Dataset):
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = torch.load(path, weights_only=False)
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    N, T, _ = batch[0]["gaf"].shape
    E_max = max(item["incidence"].shape[1] for item in batch)

    gaf = torch.stack([item["gaf"] for item in batch], dim=0).unsqueeze(2)  # (B,N,1,T,T)
    corr = torch.stack([item["corr"] for item in batch], dim=0)             # (B,N,N)
    price_window = torch.stack([item["price_window"] for item in batch], dim=0)  # (B,T,N)
    rho_mean = torch.tensor([item["rho_mean"] for item in batch], dtype=torch.float32)  # (B,)

    incidence = torch.zeros(B, N, E_max, dtype=torch.float32)
    hyperedge_w = torch.ones(B, E_max, dtype=torch.float32)  # padding weight irrelevant (see docstring)
    for b, item in enumerate(batch):
        e = item["incidence"].shape[1]
        incidence[b, :, :e] = item["incidence"]
        hyperedge_w[b, :e] = item["hyperedge_w"]
        # padding columns [e:E_max] stay all-zero in `incidence` -> zero contribution

    return {
        "gaf": gaf,
        "incidence": incidence,
        "hyperedge_w": hyperedge_w,
        "corr": corr,
        "price_window": price_window,
        "rho_mean": rho_mean,
    }


def make_loader(path: str, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    """
    num_workers=0 by default: safe on Windows (avoids multiprocessing /
    pickling issues with DataLoader workers). Increase only on Linux/Colab,
    and only inside an `if __name__ == "__main__":` guard.
    """
    ds = HyperAlignDataset(path)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=shuffle,
    )


# ================================================================
# Self-check: padding correctness
# ================================================================

def _test_collate_padding_is_safe():
    """
    Verifies that padding an incidence matrix with all-zero columns does
    NOT change the resulting hypergraph Laplacian, i.e. that batching
    windows with different E is mathematically safe.
    """
    import sys
    sys.path.insert(0, ".")
    from hyperalign_model import HGNNEncoder

    torch.manual_seed(0)
    N, E = 6, 4
    H_small = (torch.rand(1, N, E) > 0.5).float()
    W_small = torch.ones(1, E)

    E_pad = E + 3
    H_padded = torch.zeros(1, N, E_pad)
    H_padded[:, :, :E] = H_small
    W_padded = torch.ones(1, E_pad)

    delta_small = HGNNEncoder.build_laplacian(H_small, W_small)
    delta_padded = HGNNEncoder.build_laplacian(H_padded, W_padded)

    max_diff = (delta_small - delta_padded).abs().max().item()
    assert max_diff < 1e-6, f"Padding changed the Laplacian! max_diff={max_diff}"
    print(f"[OK] zero-column padding is safe (max Laplacian diff = {max_diff:.2e})")


if __name__ == "__main__":
    _test_collate_padding_is_safe()
