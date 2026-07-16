"""
HyperAlign-Fin — Model Architecture (PyTorch)
================================================
Implements every component justified in Section 3 (Theory):

    GAF -> CNN encoder f_theta          -> V_i        (Theorem 1)
    Hypergraph -> HGNN encoder g_phi     -> G_i        (Theorem 2)
    Regime-aware contrastive alignment   -> Z_i        (Corollary 1, 3)
    Linear probe heads for downstream    -> tasks      (Corollary 2)

IMPORTANT — rho* is NOT a universal constant.
    0.417 was the 75th-percentile calibration measured on ONE specific
    dataset (N=119 S&P-500 names, 2015-2023, T=20). It is dataset- and
    split-dependent (Corollary 3). This file therefore does NOT hardcode
    it. Call `calibrate_rho_star(...)` on your TRAIN split only, then
    pass the result explicitly to `HyperAlignFin(rho_star=...)`.
    Recomputing it on val/test data would leak information.

Usage:
    # 1) calibrate on TRAIN split (mirrors Test C in the validation script)
    rho_star, diag = calibrate_rho_star(train_corr_matrices, percentile=75.0)

    # 2) build model with the calibrated threshold — no silent defaults
    model = HyperAlignFin(seq_len=20, n_assets=119, latent_dim=128, rho_star=rho_star)
    out = model(gaf_batch, incidence_batch, hyperedge_weights_batch, rho_mean_batch)
    loss = hyperalign_loss(out)
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Regime threshold calibration (Corollary 3)
# ================================================================

def rho_star_theory(T: int, N: int) -> float:
    """Functional form only (Corollary 3): sqrt(1 - T/N). Reported in the
    paper for its shape, NOT used operationally — it is too conservative
    on real data (see calibrate_rho_star)."""
    return math.sqrt(max(1 - T / N, 0.0))


def calibrate_rho_star(corr_matrices: np.ndarray, percentile: float = 75.0
                        ) -> Tuple[float, dict]:
    """
    Empirically calibrate the regime threshold from TRAIN-split data only.
    Mirrors `test_C` in hyperalign_validation.py exactly: uses the
    MEAN absolute off-diagonal correlation (not max), thresholded at the
    given percentile.

    Args:
        corr_matrices: (n_windows, N, N) correlation matrices from the
                        TRAINING split only.
        percentile:     calibration percentile (75.0 reproduces the
                        Section 4 result of 0.417 on that dataset).

    Returns:
        rho_star:  calibrated threshold (float)
        diagnostics: dict with rho_mean series, crisis_pct, and the
                     (unused) theoretical value for reference.
    """
    N = corr_matrices.shape[-1]
    mask = ~np.eye(N, dtype=bool)
    rho_mean = np.array([
        np.mean(np.abs(corr_matrices[w][mask])) for w in range(len(corr_matrices))
    ])
    rho_star = float(np.percentile(rho_mean, percentile))
    crisis_pct = float(100 * np.mean(rho_mean > rho_star))

    if not (5.0 < crisis_pct < 45.0):
        warnings.warn(
            f"calibrate_rho_star: crisis_pct={crisis_pct:.1f}% is outside "
            f"the sane [5,45]% range used in Section 4. Check `percentile` "
            f"or the input correlation matrices before training."
        )

    diagnostics = {
        "rho_mean_series": rho_mean,
        "crisis_pct": crisis_pct,
        "percentile_used": percentile,
    }
    return rho_star, diagnostics


# ================================================================
# 1. Visual branch — GAF -> CNN encoder  (Theorem 1: K_V valid kernel)
# ================================================================

class GAFEncoder(nn.Module):
    """
    CNN encoder f_theta over Gramian Angular Field matrices.

    Input : (B, N, 1, T, T)   GAF images, one per asset, per window
    Output: (B, N, latent_dim)

    Architecture rationale: GAF matrices are rank-2 (Remark 1), so a
    shallow encoder with small receptive fields is sufficient — this is
    an intentional, theory-motivated choice, not a shortcut.
    """

    def __init__(self, seq_len: int, latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.seq_len = seq_len

        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, gaf: torch.Tensor) -> torch.Tensor:
        # gaf: (B, N, 1, T, T) -> flatten batch+asset dims for conv
        B, N, C, T, _ = gaf.shape
        x = gaf.view(B * N, C, T, T)
        x = self.conv(x)
        x = self.proj(x)
        return x.view(B, N, -1)  # (B, N, latent_dim)


# ================================================================
# 2. Graph branch — Hypergraph Laplacian -> HGNN encoder
#    (Theorem 2: K_G valid kernel on S_+^N)
# ================================================================

class HypergraphConv(nn.Module):
    """
    Single HGNN propagation layer following the normalized hypergraph
    Laplacian of Definition 3:
        Delta = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    Node update: X' = sigma(Delta @ X @ Theta)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim), delta: (B, N, N)
        x = torch.bmm(delta, x)     # propagate: (B,N,N) x (B,N,in_dim)
        return self.theta(x)


class HGNNEncoder(nn.Module):
    """
    Hypergraph neural network g_phi. Consumes the incidence matrix H and
    hyperedge weights W directly and builds the normalized Laplacian
    internally (Eq. for Delta in Definition 3), rather than assuming a
    precomputed Delta — this keeps gradients flowing through the
    hyperedge-construction step if H is made differentiable upstream.

    Input:
        incidence:  (B, N, E)   H^{(t)}
        hyperedge_w:(B, E)      W^{(t)} diagonal (positive weights)
        node_feat:  (B, N, in_dim)  optional raw node features
                    (e.g. last log-return, realized vol); if None,
                    an identity/degree-based feature is used.
    Output:
        (B, N, latent_dim)
    """

    def __init__(self, n_assets: int, latent_dim: int = 128,
                 hidden_dim: int = 64, in_feat_dim: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_assets = n_assets
        self.in_feat_dim = in_feat_dim
        self.input_proj = nn.Linear(in_feat_dim, hidden_dim)

        dims = [hidden_dim] * n_layers + [latent_dim]
        self.layers = nn.ModuleList([
            HypergraphConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(n_layers)])
        self.act = nn.GELU()

    @staticmethod
    def build_laplacian(incidence: torch.Tensor, hyperedge_w: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
        """
        Delta = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2},  trace-normalized.
        incidence:   (B, N, E)
        hyperedge_w: (B, E)
        returns:     (B, N, N)
        """
        H = incidence
        W = torch.diag_embed(hyperedge_w)                       # (B, E, E)
        Dv = H.sum(dim=-1).clamp_min(eps)                        # (B, N)
        De = H.sum(dim=1).clamp_min(eps)                         # (B, E)

        Dv_inv_sqrt = torch.diag_embed(Dv.rsqrt())                # (B, N, N)
        De_inv = torch.diag_embed(1.0 / De)                       # (B, E, E)

        HW = torch.bmm(H, W)                                      # (B, N, E)
        HWDe = torch.bmm(HW, De_inv)                              # (B, N, E)
        HWDeHt = torch.bmm(HWDe, H.transpose(1, 2))               # (B, N, N)
        delta = torch.bmm(torch.bmm(Dv_inv_sqrt, HWDeHt), Dv_inv_sqrt)

        tr = delta.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        delta = delta / tr.clamp_min(eps)
        return delta

    def forward(self, incidence: torch.Tensor, hyperedge_w: torch.Tensor,
                node_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, E = incidence.shape
        if node_feat is None:
            warnings.warn(
                "HGNNEncoder.forward: node_feat=None -> falling back to "
                "vertex-degree features, which carry NO price information. "
                "This is a placeholder for shape-testing only. For actual "
                "training, pass real financial features via "
                "`build_node_features(price_window)` (see below).",
                stacklevel=2,
            )
            deg = incidence.sum(dim=-1, keepdim=True)             # (B, N, 1)
            deg = deg / deg.amax(dim=1, keepdim=True).clamp_min(1e-8)
            node_feat = deg.repeat(1, 1, self.in_feat_dim)

        delta = self.build_laplacian(incidence, hyperedge_w)      # (B, N, N)
        x = self.input_proj(node_feat)
        for conv, norm in zip(self.layers, self.norms):
            x = self.act(norm(conv(x, delta)))
        return x  # (B, N, latent_dim)


def build_node_features(price_window: torch.Tensor) -> torch.Tensor:
    """
    Real financial node features per asset, replacing the degree-based
    placeholder in HGNNEncoder. Use this in the actual training pipeline.

    price_window: (B, T, N) raw prices for the window
    returns:      (B, N, 8) features:
        [mean_return, std_return (realized vol), skew_return,
         last_return, min_return, max_return, mean_abs_return,
         normalized_price_level]

    NOTE: this is a concrete, price-derived feature set — not a
    placeholder — but it is still a design choice, not something
    proven by Section 3. It should be treated as a hyperparameter of
    the architecture and reported/ablated as such.
    """
    rets = torch.log(price_window[:, 1:, :] + 1e-8) - torch.log(price_window[:, :-1, :] + 1e-8)
    # rets: (B, T-1, N)
    mean_r = rets.mean(dim=1)                                   # (B, N)
    std_r = rets.std(dim=1)                                     # (B, N)
    centered = rets - mean_r.unsqueeze(1)
    skew_r = (centered.pow(3).mean(dim=1)) / (std_r.pow(3) + 1e-8)
    last_r = rets[:, -1, :]
    min_r = rets.min(dim=1).values
    max_r = rets.max(dim=1).values
    mean_abs_r = rets.abs().mean(dim=1)
    price_level = price_window[:, -1, :]
    price_level = (price_level - price_level.mean(dim=1, keepdim=True)) / (
        price_level.std(dim=1, keepdim=True) + 1e-8
    )

    feats = torch.stack(
        [mean_r, std_r, skew_r, last_r, min_r, max_r, mean_abs_r, price_level], dim=-1
    )  # (B, N, 8)
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ================================================================
# 3. Regime-aware alignment  (Corollary 1: contrastive not MSE;
#                              Corollary 3: regime gating via rho*_cal)
# ================================================================

class RegimeAwareAlignment(nn.Module):
    """
    InfoNCE-style contrastive alignment between V_i and G_i, gated by
    market regime. Per Corollary 3, the regime signal is the mean
    absolute off-diagonal correlation rho_mean(t), thresholded at a
    threshold calibrated per-dataset via `calibrate_rho_star` (NOT the
    raw theoretical sqrt(1-T/N), which is too conservative — see
    Corollary 3, Remark).

    In Crisis windows (rho_mean > rho*_cal), Assumption A degrades, so
    the alignment temperature is raised (softer, less confident
    alignment) rather than discarding the sample.
    """

    def __init__(self, latent_dim: int, rho_star: float,
                 temperature_normal: float = 0.1,
                 temperature_crisis: float = 0.3):
        """
        rho_star: REQUIRED, no default. Must come from `calibrate_rho_star`
            run on the training split (see module docstring). Passing a
            number here without calibrating it first silently reintroduces
            the exact bug this refactor fixes.
        temperature_normal / temperature_crisis: hyperparameters, NOT
            derived from theory. They must be tuned via the ablation study
            (Section 6) and reported as such — do not present them as
            theory-derived constants in the paper.
        """
        super().__init__()
        if not (0.0 < rho_star < 1.0):
            raise ValueError(
                f"rho_star={rho_star} looks uncalibrated (expected in (0,1)). "
                f"Run calibrate_rho_star(train_corr_matrices) first."
            )
        # buffer, not plain attribute: moves with .to(device) and is saved
        # in state_dict, so a checkpoint always carries the threshold it
        # was calibrated with.
        self.register_buffer("rho_star", torch.tensor(float(rho_star)))
        self.tau_normal = temperature_normal
        self.tau_crisis = temperature_crisis
        self.proj_v = nn.Linear(latent_dim, latent_dim)
        self.proj_g = nn.Linear(latent_dim, latent_dim)

    def forward(self, V: torch.Tensor, G: torch.Tensor,
                rho_mean: torch.Tensor):
        """
        V, G:      (B, N, d)  visual / graph embeddings
        rho_mean:  (B,)       mean |C_ij|, i != j, per window (regime signal)

        Returns:
            z_v, z_g   : (B, N, d) normalized projections
            logits     : (B, N, N) per-batch-item asset-to-asset similarity
            temperature: (B,) per-sample temperature used
            regime     : (B,) bool, True = Crisis
        """
        z_v = F.normalize(self.proj_v(V), dim=-1)
        z_g = F.normalize(self.proj_g(G), dim=-1)

        regime = rho_mean > self.rho_star                          # (B,) bool
        temperature = torch.where(
            regime,
            torch.full_like(rho_mean, self.tau_crisis),
            torch.full_like(rho_mean, self.tau_normal),
        )  # (B,)

        # asset-to-asset similarity within each batch item, temperature-scaled
        logits = torch.einsum("bnd,bmd->bnm", z_v, z_g) / temperature.view(-1, 1, 1)
        return z_v, z_g, logits, temperature, regime


def info_nce_diagonal(logits: torch.Tensor,
                       same_group_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Symmetric InfoNCE with positives on the diagonal (asset i's visual
    embedding should match asset i's graph embedding, and vice versa).
    logits: (B, N, N)

    same_group_mask: optional (B, N, N) bool, True where asset j shares
        at least one hyperedge with asset i (i != j). Motivated directly
        by Definition 3 + Theorem 2: two assets in the same hyperedge
        can receive near-identical G_i after HGNN propagation (their
        neighborhoods in the Laplacian overlap heavily), making them
        unfair, near-indistinguishable negatives. Masking them out
        (setting their logit to -inf, removing them from the softmax
        denominator) does not change what counts as "correct" (the
        diagonal is untouched) -- it only stops penalizing the model
        for failing to separate genuinely confusable pairs, which is
        an artifact of the hypergraph construction, not a modeling
        failure. See build_same_group_mask() below.
    """
    B, N, _ = logits.shape
    if same_group_mask is not None:
        logits = logits.masked_fill(same_group_mask, float("-inf"))

    targets = torch.arange(N, device=logits.device).unsqueeze(0).expand(B, N)
    loss_v2g = F.cross_entropy(logits.reshape(B * N, N), targets.reshape(-1))
    loss_g2v = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), targets.reshape(-1))
    return 0.5 * (loss_v2g + loss_g2v)


def build_same_group_mask(incidence: torch.Tensor) -> torch.Tensor:
    """
    incidence: (B, N, E) hyperedge incidence matrix.
    Returns (B, N, N) bool: True where i != j and assets i, j share at
    least one hyperedge (sector or correlation cluster). Diagonal is
    always False (never mask the true positive).
    """
    B, N, E = incidence.shape
    shared = torch.bmm(incidence, incidence.transpose(1, 2)) > 0   # (B,N,N), True if share >=1 edge
    eye = torch.eye(N, dtype=torch.bool, device=incidence.device).unsqueeze(0)
    return shared & ~eye


# ================================================================
# 3b. Gated fusion — learned, per-sample weighting of V vs G
#     (a more theory-faithful reading of Corollary 2 than flat concat)
# ================================================================

class GatedFusion(nn.Module):
    """
    Replaces flat concat+MLP fusion with a per-dimension learned gate:
        gate = sigmoid(MLP([V, G, rho_mean]))   in [0,1]^d
        Z     = gate * proj_v(V) + (1-gate) * proj_g(G)

    Motivation: Corollary 2's claim (fusion beats either view alone) is
    a STATISTICAL statement about typical tasks, not a guarantee for
    every task in isolation. A flat concat+MLP fusion cannot express
    "trust V more here, G more there" -- it applies the same fixed
    transform everywhere, so on a G-favoring task it dilutes V's
    advantage, and vice versa (exactly what Downstream Tasks 1 and 2
    showed empirically). A gate lets the model learn this per-sample,
    and conditioning it on rho_mean ties the gate to Corollary 3's own
    regime signal -- in Crisis regimes, Assumption A degrades (Section
    3.3), so the model may need to lean more/less on G accordingly;
    this makes that adjustable rather than fixed.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.proj_v = nn.Linear(latent_dim, latent_dim)
        self.proj_g = nn.Linear(latent_dim, latent_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * latent_dim + 1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, V: torch.Tensor, G: torch.Tensor,
                rho_mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V, G: (B, N, d)   rho_mean: (B,)
        Returns: Z (B, N, d), gate (B, N, d) -- gate returned for
        diagnostics/interpretability (e.g. plotting mean gate value by
        regime in the paper).
        """
        B, N, d = V.shape
        rho_expanded = rho_mean.view(B, 1, 1).expand(B, N, 1)
        gate_input = torch.cat([V, G, rho_expanded], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))            # (B, N, d), in [0,1]
        Z = gate * self.proj_v(V) + (1.0 - gate) * self.proj_g(G)
        return self.norm(Z), gate


# ================================================================
# 4. Full model
# ================================================================

@dataclass
class HyperAlignOutput:
    V: torch.Tensor            # (B, N, d) visual embedding
    G: torch.Tensor            # (B, N, d) graph embedding
    Z: torch.Tensor            # (B, N, d) joint / fused latent
    z_v: torch.Tensor          # (B, N, d) projected, normalized visual
    z_g: torch.Tensor          # (B, N, d) projected, normalized graph
    logits: torch.Tensor       # (B, N, N)
    temperature: torch.Tensor  # (B,)
    regime: torch.Tensor       # (B,) bool
    gate: Optional[torch.Tensor] = None  # (B, N, d) only set when fusion_type="gated"


class HyperAlignFin(nn.Module):
    """
    Full HyperAlign-Fin model:
        GAF  -> GAFEncoder  -> V
        Hyp. -> HGNNEncoder -> G
        (V, G) -> RegimeAwareAlignment -> Z (fused, for downstream probes)
    """

    def __init__(self, seq_len: int, n_assets: int, rho_star: float,
                 latent_dim: int = 128, hgnn_in_feat_dim: int = 8,
                 fusion_type: str = "gated"):
        """
        rho_star: REQUIRED. Output of `calibrate_rho_star(train_corr_matrices)`.
            There is intentionally no default — forcing every caller to
            calibrate on their own training split (Corollary 3), rather
            than reusing the 0.417 measured on a different dataset.
        fusion_type: "gated" (default, recommended) or "concat" (the
            original flat concat+MLP, kept for ablation comparison --
            Downstream Tasks 1/2 showed "concat" never beats the best
            individual view; "gated" is expected to do better since it
            can adaptively weight V vs G per sample).
        """
        super().__init__()
        if fusion_type not in ("gated", "concat"):
            raise ValueError(f"fusion_type must be 'gated' or 'concat', got {fusion_type!r}")
        self.fusion_type = fusion_type

        self.visual_encoder = GAFEncoder(seq_len=seq_len, latent_dim=latent_dim)
        self.graph_encoder = HGNNEncoder(
            n_assets=n_assets, latent_dim=latent_dim, in_feat_dim=hgnn_in_feat_dim
        )
        self.alignment = RegimeAwareAlignment(latent_dim=latent_dim, rho_star=rho_star)

        if fusion_type == "gated":
            self.fuse = GatedFusion(latent_dim=latent_dim)
        else:
            self.fuse = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.GELU(),
                nn.LayerNorm(latent_dim),
            )

    def forward(self, gaf: torch.Tensor, incidence: torch.Tensor,
                hyperedge_w: torch.Tensor, rho_mean: torch.Tensor,
                node_feat: Optional[torch.Tensor] = None) -> HyperAlignOutput:
        V = self.visual_encoder(gaf)                                   # (B,N,d)
        G = self.graph_encoder(incidence, hyperedge_w, node_feat)       # (B,N,d)
        z_v, z_g, logits, temperature, regime = self.alignment(V, G, rho_mean)

        gate = None
        if self.fusion_type == "gated":
            Z, gate = self.fuse(z_v, z_g, rho_mean)                     # Corollary 2
        else:
            Z = self.fuse(torch.cat([z_v, z_g], dim=-1))                # Corollary 2 (baseline)

        return HyperAlignOutput(V, G, Z, z_v, z_g, logits, temperature, regime, gate)


def hyperalign_loss(out: HyperAlignOutput,
                     incidence: Optional[torch.Tensor] = None,
                     mask_same_hyperedge_negatives: bool = True) -> torch.Tensor:
    """
    Corollary 1: contrastive (InfoNCE), not MSE, alignment loss.

    mask_same_hyperedge_negatives: if True (default) and `incidence` is
    given, excludes same-hyperedge asset pairs from the negative set
    (see build_same_group_mask). Strongly recommended: without this,
    the InfoNCE loss floor is set by ln(N) - I(V;G), which Theorem 3
    shows is very close to ln(N) itself (near-zero mutual information
    by design) -- the model has almost nothing to learn from the raw
    task. Excluding confusable same-group negatives removes the unfair
    portion of that difficulty.
    """
    mask = None
    if mask_same_hyperedge_negatives and incidence is not None:
        mask = build_same_group_mask(incidence)
    return info_nce_diagonal(out.logits, same_group_mask=mask)


# ================================================================
# 5. Downstream heads (Corollary 2 — linear probing on Z)
# ================================================================

class LinearProbe(nn.Module):
    """Frozen-backbone linear probe, per the PAC-style bound of Corollary 2."""

    def __init__(self, latent_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(latent_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ================================================================
# 6. Smoke test
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    B, N, T, E, d = 4, 32, 20, 12, 128

    # --- Step 1: fake TRAIN-split correlation history -> calibrate rho_star
    #     (in real use: computed once from the actual training windows,
    #     mirroring build_hypergraph_laplacian's C matrix in the
    #     validation script, NOT from val/test data)
    n_train_windows = 300
    fake_corr = np.random.uniform(-0.3, 0.6, size=(n_train_windows, N, N))
    fake_corr = (fake_corr + fake_corr.transpose(0, 2, 1)) / 2
    for w in range(n_train_windows):
        np.fill_diagonal(fake_corr[w], 1.0)
    rho_star, calib_diag = calibrate_rho_star(fake_corr, percentile=75.0)
    print(f"[calibration] rho_star={rho_star:.4f}  crisis_pct={calib_diag['crisis_pct']:.1f}%")

    # --- Step 2: batch of real-shaped inputs
    price_window = torch.cumprod(1 + 0.01 * torch.randn(B, T, N), dim=1) * 100
    gaf = torch.randn(B, N, 1, T, T)  # placeholder GAF tensor (see make_gaf in validation script)
    incidence = (torch.rand(B, N, E) > 0.7).float()
    hyperedge_w = torch.ones(B, E)
    rho_mean = torch.rand(B)  # regime signal per window in this batch

    # --- Step 3: real financial node features (NOT the degree placeholder)
    node_feat = build_node_features(price_window)
    print(f"node_feat: {tuple(node_feat.shape)}")

    # --- Step 4: build model with EXPLICIT, calibrated rho_star
    model = HyperAlignFin(seq_len=T, n_assets=N, rho_star=rho_star, latent_dim=d)
    out = model(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)
    loss_unmasked = hyperalign_loss(out, incidence=None)
    loss_masked = hyperalign_loss(out, incidence=incidence, mask_same_hyperedge_negatives=True)

    print(f"V: {tuple(out.V.shape)}  G: {tuple(out.G.shape)}  Z: {tuple(out.Z.shape)}")
    print(f"logits: {tuple(out.logits.shape)}  regime: {out.regime.tolist()}")
    print(f"loss (unmasked, all negatives):        {loss_unmasked.item():.4f}")
    print(f"loss (same-hyperedge negatives masked): {loss_masked.item():.4f}  "
          f"(should be <= unmasked, since the task got easier)")

    probe = LinearProbe(latent_dim=d, n_classes=3)
    preds = probe(out.Z)
    print(f"probe output: {tuple(preds.shape)}")

    # --- Step 5: confirm the "no default, must calibrate" guard actually works
    try:
        RegimeAwareAlignment(latent_dim=d, rho_star=1.5)  # invalid on purpose
        raise AssertionError("expected ValueError for uncalibrated rho_star")
    except ValueError as e:
        print(f"[guard OK] rejected uncalibrated rho_star: {e}")

    # --- Step 6: gated fusion smoke test + comparison to concat fusion
    print("\n--- fusion_type comparison ---")
    for ft in ("gated", "concat"):
        m = HyperAlignFin(seq_len=T, n_assets=N, rho_star=rho_star, latent_dim=d, fusion_type=ft)
        o = m(gaf, incidence, hyperedge_w, rho_mean, node_feat=node_feat)
        gate_info = ""
        if o.gate is not None:
            gate_info = f"  mean_gate={o.gate.mean().item():.4f} (0=all-G, 1=all-V)"
        print(f"  fusion_type={ft:<8} Z shape={tuple(o.Z.shape)}{gate_info}")
