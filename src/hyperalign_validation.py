"""
HyperAlign-Fin: Empirical Validation v3
==========================================
Fixes over v2 (see review):
    FIX 1 — PCA under-fitting on high-dim G_feats (ld = N(N+1)/2 can be
             >10,000 for N=150). v2 used a fixed 3 components regardless
             of dimensionality; on random data alone this captures <2%
             of variance for G_feats. v3 sweeps multiple component counts
             and reports explained variance so the ratio's stability can
             be checked BEFORE citing it in the paper.
    FIX 2 — Sector labels were arbitrary index blocks (i // chunk_size),
             not real GICS sectors, making "sector hyperedges" meaningless
             for interpretability. v3 fetches real sector metadata via
             yfinance and caches it to disk; falls back to synthetic
             blocks ONLY for USE_REAL_DATA=False, with an explicit label.
    FIX 3 — dropna(axis=1) silently drops tickers with any missing day
             (survivorship bias). v3 reports how many tickers were
             requested vs. kept, so this is visible, not silent.
    FIX 4 — Test B now cross-checks the discretization-based MI against
             sklearn's Kraskov-style k-NN estimator (mutual_info_regression),
             which needs no binning and is more reliable in high dimensions.

Tests:
    A : Independence Assumption (Spearman + Bootstrap CI)               -- unchanged from v2
    B : Informational Orthogonality — Theorem 3
        (i)  PCA-component sweep + explained variance report  [FIX 1]
        (ii) Kraskov k-NN MI cross-check                       [FIX 4]
    C : Regime Threshold rho* — Corollary 3                              -- unchanged from v2

Output files:
    test_A_independence.png
    test_B_orthogonality.png
    test_B_pca_sensitivity.png     <- NEW
    test_C_regime.png
    sensitivity_rho_star.png

Usage:
    pip install yfinance scikit-learn numpy pandas matplotlib scipy
    python hyperalign_validation.py
"""

import json
import math
import os
import time
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PARAMETERS
# ============================================================
USE_REAL_DATA   = True    # True = yfinance | False = synthetic GBM
N_ASSETS        = 150     # number of assets to download
T_WINDOW        = 20      # rolling window length (days)
N_WINDOWS       = 200     # windows (synthetic only)
TAU_CORR        = 0.5     # correlation threshold for hyperedges
N_SECTORS       = 10      # industry sectors (synthetic only)
CI_ALPHA        = 0.95    # confidence interval level
N_BOOTSTRAP     = 1000    # bootstrap resamples
N_SAMPLE_B      = 20      # assets to sample for Test B
PCA_COMPONENTS_SWEEP = [3, 10, 20, 50]   # FIX 1: sweep instead of fixed 3
SECTOR_CACHE_PATH = "sector_cache.json"  # FIX 2: avoid re-fetching every run
# ============================================================


# ================================================================
# 1. DATA
# ================================================================

def load_real_data(n_assets):
    """FIX 3: reports ticker survival explicitly (no silent dropna)."""
    import yfinance as yf
    tickers = [
        'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','JNJ','V',
        'PG','UNH','HD','MA','BAC','XOM','ABBV','PFE','AVGO','COST',
        'CVX','WMT','LLY','TMO','CSCO','ABT','ACN','DHR','NKE','NEE',
        'ADBE','TXN','VZ','PM','RTX','QCOM','HON','ORCL','UPS','IBM',
        'CAT','GE','BA','MMM','GS','MS','AXP','BLK','SPGI','CME',
        'CL','KO','PEP','MCD','SBUX','DIS','CMCSA','T','BDX','SYK',
        'MDT','BSX','EW','ZBH','BAX','COO','IDXX','MTD','WAT','A',
        'IQV','RMD','STE','TFX','REGN','VRTX','BIIB','GILD','AMGN','ILMN',
        'INCY','BMRN','ALNY','SRPT','ARWR','IONS','MRNA','BNTX','NVAX','JAZZ',
        'SNAP','PINS','MTCH','IAC','EXPE','BKNG','ABNB','UBER','LYFT','PYPL',
        'AFRM','SOFI','LC','UPST','TWLO','DDOG','NET','CRWD','ZS','OKTA',
        'SNOW','MDB','HUBS','VEEV','WDAY','TEAM','DOCU','ZM','FIVN','EGHT',
        'ACAD','SGEN','RARE','HALO','NBIX','EXEL','ITCI','FOLD','PTCT',
        'CI','HUM','ELV','MCK','CAH','CVS','WBA','HOLX','BCR',
        'LKQ','AAP','AZO','ORLY','SMP','DAN','BWA','LEA','ALV','APTV',
    ]
    tickers = list(dict.fromkeys(tickers))[:n_assets]
    n_requested = len(tickers)

    raw = yf.download(
        tickers, start='2015-01-01', end='2023-12-31',
        auto_adjust=True, progress=False
    )['Close']
    raw_before = raw.shape[1]
    raw = raw.dropna(axis=1)
    prices = raw.values
    dates = raw.index.values.astype("datetime64[D]")
    n_kept = prices.shape[1]
    surviving_tickers = list(raw.columns)

    print(f"[DATA] Requested {n_requested} tickers -> {raw_before} downloaded "
          f"-> {n_kept} survived dropna (dropped {raw_before - n_kept} for "
          f"missing data: survivorship bias, report this in Limitations).")

    sector_labels = get_real_sector_labels(surviving_tickers)
    print(f"[DATA] Real data loaded: {prices.shape[0]} days x {n_kept} assets, "
          f"{len(np.unique(sector_labels))} real GICS sectors")
    return prices, sector_labels, n_kept, dates


def get_real_sector_labels(tickers, cache_path=SECTOR_CACHE_PATH):
    """
    FIX 2: real GICS sector per ticker via yfinance, not arbitrary index
    blocks. Cached to disk since .info calls are slow (one HTTP request
    per ticker).

    Returns integer sector-id array, aligned with `tickers` order.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    import yfinance as yf
    sectors_raw = []
    for i, t in enumerate(tickers):
        if t in cache:
            sectors_raw.append(cache[t])
            continue
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector', 'Unknown')
        except Exception:
            sector = 'Unknown'
        cache[t] = sector
        sectors_raw.append(sector)
        if i % 20 == 0:
            print(f"  fetching sector metadata {i}/{len(tickers)} ...", end="\r")
        time.sleep(0.05)  # be polite to the API

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    unique_sectors = sorted(set(sectors_raw))
    sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
    labels = np.array([sector_to_id[s] for s in sectors_raw])

    print(f"\n[SECTORS] {len(unique_sectors)} real GICS sectors found: "
          f"{unique_sectors}")
    return labels


def generate_synthetic(n_assets, n_windows, t_window, n_sectors):
    """Synthetic path: block sector labels are a MODELING CHOICE here
    (to control the correlation structure for testing), not a stand-in
    for real GICS data -- only load_real_data's labels are used for any
    claim about real sector interpretability."""
    n_days = n_windows + t_window
    sl = np.array([i // (n_assets // n_sectors) for i in range(n_assets)])
    Sigma = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho = 0.5 if sl[i] == sl[j] else 0.1
            Sigma[i, j] = Sigma[j, i] = rho
    Sigma += np.eye(n_assets) * 0.15
    L = np.linalg.cholesky(Sigma)
    prices = np.cumprod(
        1 + 0.0003 + 0.02 * (np.random.randn(n_days, n_assets) @ L.T),
        axis=0
    ) * 100
    print(f"[DATA] Synthetic GBM: {n_days} days x {n_assets} assets "
          f"({n_sectors} SYNTHETIC block sectors -- not real GICS data)")
    return prices, sl, None


# ================================================================
# 2. CORE TRANSFORMATIONS  (unchanged from v2)
# ================================================================

def make_gaf(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return np.zeros((len(series), len(series)))
    x = np.clip(2 * (series - mn) / (mx - mn) - 1, -1, 1)
    phi = np.arccos(x)
    return np.cos(phi[:, None] + phi[None, :])


def build_hypergraph_laplacian(price_window, sector_labels, tau):
    T, N = price_window.shape
    rets = np.diff(np.log(price_window + 1e-8), axis=0)
    C = np.nan_to_num(np.corrcoef(rets.T), 0)

    edges = []
    for s in np.unique(sector_labels):
        m = np.where(sector_labels == s)[0]
        if len(m) >= 2:
            edges.append(m)
    visited = set()
    for i in range(N):
        cl = np.where(C[i] >= tau)[0]
        if len(cl) >= 2:
            k = frozenset(cl.tolist())
            if k not in visited:
                visited.add(k); edges.append(cl)

    if not edges:
        return np.eye(N) / N

    ne = len(edges)
    H = np.zeros((N, ne)); W = np.ones(ne)
    for ei, m in enumerate(edges):
        H[m, ei] = 1.0

    Dv = H @ W; De = H.T @ np.ones(N)
    Dvi = np.diag(1.0 / np.sqrt(np.maximum(Dv, 1e-8)))
    Dei = np.diag(1.0 / np.maximum(De, 1e-8))
    D = Dvi @ H @ np.diag(W) @ Dei @ H.T @ Dvi
    tr = np.trace(D)
    return D / tr if tr > 0 else D


# ================================================================
# 3. MI ESTIMATOR (unchanged from v2, discretization-based)
# ================================================================

def _optimal_bins(n):
    return max(int(math.ceil(math.log2(n) + 1)), 5)


def disc_mi(x, y):
    nb = _optimal_bins(len(x))
    kbd = KBinsDiscretizer(n_bins=nb, encode='ordinal', strategy='quantile')
    xd = kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
    yd = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    return mutual_info_score(xd, yd)


def disc_h(x):
    """
    Entropy via discretization.

    CRITICAL FIX: quantile (equal-COUNT) binning forces near-uniform bin
    occupancy by construction, so the resulting entropy estimate is
    always close to ln(n_bins) regardless of the true distribution shape
    (verified: four very different synthetic distributions -- uniform,
    normal, exponential, bimodal -- all gave H=2.564945 with n_bins=13
    under 'quantile' strategy, vs. genuinely different values of 2.56,
    1.96, 1.26, 2.18 under 'uniform' (equal-WIDTH) bins). This is why
    Test B's H(V) looked identical across all 20 sampled assets in an
    earlier run: it was measuring bin-count uniformity, not real
    entropy. 'uniform' bins are used here instead, since they actually
    reflect the underlying distribution's shape.
    """
    nb = _optimal_bins(len(x))
    kbd = KBinsDiscretizer(n_bins=nb, encode='ordinal', strategy='uniform')
    xd = kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
    _, c = np.unique(xd, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


# ================================================================
# 4. FEATURE EXTRACTION  (unchanged from v2)
# ================================================================

def extract_features(prices, sector_labels, T, tau):
    n_days, N = prices.shape
    n_w = n_days - T
    tui = np.triu_indices(T)
    tui2 = np.triu_indices(N)
    gd = T * (T + 1) // 2
    ld = N * (N + 1) // 2

    V_feats = np.zeros((n_w, N, gd))
    G_feats = np.zeros((n_w, ld))
    corr_matrices = np.zeros((n_w, N, N))
    returns_all = np.zeros((n_w, T - 1, N))

    print(f"[FEAT] Extracting from {n_w} windows "
          f"(V dim={gd}, G dim={ld}) ...")
    for w in range(n_w):
        win = prices[w: w + T, :]
        for i in range(N):
            V_feats[w, i, :] = make_gaf(win[:, i])[tui]
        D = build_hypergraph_laplacian(win, sector_labels, tau)
        G_feats[w, :] = D[tui2]
        r = np.diff(np.log(win + 1e-8), axis=0)
        corr_matrices[w] = np.nan_to_num(np.corrcoef(r.T), 0)
        returns_all[w] = r
        if w % 200 == 0:
            print(f"  window {w}/{n_w} ...", end="\r")

    print(f"\n[FEAT] Done  V={V_feats.shape}  G={G_feats.shape}  "
          f"(G/n_windows ratio = {ld / n_w:.1f}x -- if >>1, PCA needs "
          f"many components; see Test B sensitivity sweep)")
    return V_feats, G_feats, corr_matrices, returns_all


# ================================================================
# 5. BOOTSTRAP CI (unchanged)
# ================================================================

def bootstrap_ci(data, stat_fn, n_boot=1000, alpha=0.95):
    data = np.asarray(data)
    boots = np.array([
        stat_fn(data[np.random.randint(0, len(data), len(data))])
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, 100 * (1 - alpha) / 2)
    hi = np.percentile(boots, 100 * (1 + alpha) / 2)
    return float(lo), float(hi)


# ================================================================
# 6. TEST A — Independence Assumption (unchanged from v2)
# ================================================================

def test_A(returns_all, corr_matrices, N, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA):
    r_arr = np.zeros(N)
    p_arr = np.zeros(N)
    for i in range(N):
        var_i = np.var(returns_all[:, :, i], axis=1)
        mask = np.arange(N) != i
        co = np.mean(np.abs(corr_matrices[:, i, :][:, mask]), axis=1)
        r_arr[i], p_arr[i] = spearmanr(var_i, co)

    mean_r = float(np.mean(np.abs(r_arr)))
    sig = int(np.sum(p_arr < 0.05))
    ci_lo, ci_hi = bootstrap_ci(np.abs(r_arr), np.mean, n_boot=n_boot, alpha=alpha)
    return r_arr, p_arr, mean_r, sig, ci_lo, ci_hi


def plot_A(r_arr, p_arr, mean_r, sig, N, ci_lo, ci_hi, alpha, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        "Test A — Independence Assumption\n"
        r"$H_0$: intra-asset variance $\perp$ cross-asset correlation",
        fontsize=13, fontweight='bold'
    )
    ax = axes[0]
    ax.hist(r_arr, bins=25, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero (independence)')
    ax.axvline(mean_r, color='orange', lw=2, label=fr'Mean |r| = {mean_r:.3f}')
    ax.axvline(-mean_r, color='orange', lw=2, linestyle=':')
    ax.set_xlabel('Spearman r'); ax.set_ylabel('Number of assets')
    ax.set_title('Distribution of Spearman r', fontsize=10)
    ax.legend(fontsize=8)
    ax.annotate(
        f"Mean |r| = {mean_r:.4f}\n{alpha*100:.0f}% CI  [{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"Significant: {sig}/{N}",
        xy=(0.02, 0.97), xycoords='axes fraction', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    ax = axes[1]
    ax.hist(p_arr, bins=25, color='coral', alpha=0.8, edgecolor='white')
    ax.axvline(0.05, color='red', linestyle='--', lw=1.5, label='p = 0.05')
    ax.set_xlabel('p-value'); ax.set_ylabel('Number of assets')
    ax.set_title(f'p-value Distribution\n({sig}/{N} significant at p < 0.05)', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig('test_A_independence.png', dpi=150, bbox_inches='tight')
        print("[OUT] test_A_independence.png saved")
    plt.show()


# ================================================================
# 7. TEST B — Informational Orthogonality  [FIX 1 + FIX 4]
# ================================================================

def test_B_sweep(V_feats, G_feats, n_sample=N_SAMPLE_B,
                  component_list=PCA_COMPONENTS_SWEEP):
    """
    FIX 1: instead of a single fixed n_components=3, sweep several values
    and report BOTH the MI ratio and the explained variance at each,
    for both V and G. If the ratio is stable and explained variance is
    reasonably high (e.g. >30-50%) across the sweep, the orthogonality
    result is credible. If the ratio changes a lot or explained variance
    stays near-zero even at the largest tested k, the earlier result was
    likely a PCA artifact, not evidence of orthogonality.
    """
    N = V_feats.shape[1]
    n_sample = min(n_sample, N)
    n_windows = V_feats.shape[0]

    results = {}
    for nc in component_list:
        nc_eff = min(nc, n_windows - 1)  # PCA can't exceed n_samples-1 components
        mi_vals = np.zeros(n_sample)
        hv_vals = np.zeros(n_sample)
        ev_v = np.zeros(n_sample)
        ev_g = np.zeros(n_sample)

        pca_g_model = PCA(n_components=nc_eff).fit(G_feats)
        py = pca_g_model.transform(G_feats)
        ev_g_total = pca_g_model.explained_variance_ratio_.sum()

        for i in range(n_sample):
            pca_v_model = PCA(n_components=nc_eff).fit(V_feats[:, i, :])
            px = pca_v_model.transform(V_feats[:, i, :])
            ev_v[i] = pca_v_model.explained_variance_ratio_.sum()
            ev_g[i] = ev_g_total

            mi_vals[i] = np.mean([disc_mi(px[:, k], py[:, k]) for k in range(nc_eff)])
            hv_vals[i] = np.mean([disc_h(px[:, k]) for k in range(nc_eff)])

        ratio_arr = mi_vals / (hv_vals + 1e-10)
        results[nc] = {
            "nc_eff": nc_eff,
            "mean_ratio": float(np.mean(ratio_arr)),
            "std_ratio": float(np.std(ratio_arr)),
            "mean_explained_var_V": float(np.mean(ev_v)),
            "explained_var_G": float(ev_g_total),
        }
        print(f"  [Test B sweep] k={nc_eff:3d}  "
              f"ratio={results[nc]['mean_ratio']:.4f}  "
              f"EV(V)={results[nc]['mean_explained_var_V']*100:5.1f}%  "
              f"EV(G)={results[nc]['explained_var_G']*100:5.1f}%")

    return results


def plot_B_sensitivity(results, save=True):
    ks = sorted(results.keys())
    ratios = [results[k]["mean_ratio"] for k in ks]
    ev_v = [results[k]["mean_explained_var_V"] * 100 for k in ks]
    ev_g = [results[k]["explained_var_G"] * 100 for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Test B Sensitivity — Does the Orthogonality Result Depend on "
                 "PCA Dimensionality?", fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(ks, ratios, 'o-', color='navy')
    ax.axhline(0.10, color='red', linestyle='--', label='Orthogonality threshold')
    ax.set_xlabel('PCA components (k)')
    ax.set_ylabel('Mean I(V;G) / H(V)')
    ax.set_title('Ratio vs. k  (should be stable if result is real)')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ks, ev_v, 'o-', label='Explained variance — V (GAF)', color='steelblue')
    ax.plot(ks, ev_g, 'o-', label='Explained variance — G (Laplacian)', color='coral')
    ax.set_xlabel('PCA components (k)')
    ax.set_ylabel('Cumulative explained variance (%)')
    ax.set_title('PCA coverage — low values mean the MI estimate\n'
                 'is computed on a poor summary of the true signal')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig('test_B_pca_sensitivity.png', dpi=150, bbox_inches='tight')
        print("[OUT] test_B_pca_sensitivity.png saved")
    plt.show()


def test_B_knn_crosscheck(V_feats, G_feats, n_sample=N_SAMPLE_B, nc=10):
    """
    FIX 4: cross-check the discretization-based MI ratio using sklearn's
    Kraskov-style k-NN mutual information estimator, which does not
    require binning and is more reliable in higher dimensions / smaller
    samples. Run on the same PCA-reduced components for comparability.
    """
    N = V_feats.shape[1]
    n_sample = min(n_sample, N)
    n_windows = V_feats.shape[0]
    nc_eff = min(nc, n_windows - 1)

    py = PCA(n_components=nc_eff).fit_transform(G_feats)
    knn_mi_vals = np.zeros(n_sample)

    for i in range(n_sample):
        px = PCA(n_components=nc_eff).fit_transform(V_feats[:, i, :])
        # mutual_info_regression: MI between each V-component and the FIRST
        # G-component (Kraskov estimator, continuous, no binning)
        mi_per_component = [
            mutual_info_regression(px, py[:, k], random_state=0).mean()
            for k in range(nc_eff)
        ]
        knn_mi_vals[i] = np.mean(mi_per_component)

    mean_knn_mi = float(np.mean(knn_mi_vals))
    print(f"[Test B k-NN cross-check] mean Kraskov MI = {mean_knn_mi:.4f} nats "
          f"(k={nc_eff} components; compare trend, not absolute scale, to "
          f"the discretization-based ratio above)")
    return knn_mi_vals, mean_knn_mi


def plot_B(mi_vals, hv_vals, ratio_arr, mean_ratio, ci_lo, ci_hi, alpha,
           n_windows, nc_used, save=True):
    n = len(mi_vals)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Test B — Informational Orthogonality of Two Views  (Theorem 3)\n"
        r"Metric: $I(V_i\,;\,G)\,/\,H(V_i)$"
        f"   Estimator: discretization (Sturges bins, n={n_windows}, k={nc_used} PCA components)",
        fontsize=12, fontweight='bold'
    )
    ax = axes[0]
    x = np.arange(n)
    ax.bar(x, hv_vals, alpha=0.55, label='H(V)  [visual entropy]', color='steelblue')
    ax.bar(x, mi_vals, alpha=0.90, label='I(V;G)  [shared info]', color='coral')
    ax.set_xlabel('Asset index'); ax.set_ylabel('Nats')
    ax.set_title('H(V) vs I(V;G) per asset', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, ratio_arr, color='mediumseagreen', alpha=0.85)
    ax.axhline(0.10, color='red', linestyle='--', lw=1.5, label='Orthogonality threshold (0.10)')
    ax.axhline(mean_ratio, color='navy', lw=2, label=fr'Mean = {mean_ratio:.4f}')
    ax.set_xlabel('Asset index'); ax.set_ylabel('I(V;G) / H(V)')
    ax.set_title('Shared-info ratio per asset\n(lower → more complementary)', fontsize=10)
    ax.legend(fontsize=8)
    ax.annotate(
        f"Mean = {mean_ratio:.4f}\n{alpha*100:.0f}% CI  [{ci_lo:.4f}, {ci_hi:.4f}]",
        xy=(0.02, 0.97), xycoords='axes fraction', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.85)
    )

    ax = axes[2]
    lim = max(hv_vals.max(), mi_vals.max()) * 1.1
    ax.scatter(hv_vals, mi_vals, c='purple', alpha=0.75, s=60, zorder=3)
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='I = H  (fully dependent)')
    ax.plot([0, lim], [0, 0.1 * lim], 'g--', lw=1.5, label='I = 0.1·H  (threshold)')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('H(V) — visual entropy'); ax.set_ylabel('I(V;G) — shared information')
    ax.set_title('Points below green line\n→ complementary views', fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig('test_B_orthogonality.png', dpi=150, bbox_inches='tight')
        print("[OUT] test_B_orthogonality.png saved")
    plt.show()


# ================================================================
# 8. TEST C — Regime Threshold (unchanged from v2)
# ================================================================

def test_C(corr_matrices, T, N, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA):
    mask = ~np.eye(N, dtype=bool)
    rho_mean = np.array([
        np.mean(np.abs(corr_matrices[w][mask])) for w in range(len(corr_matrices))
    ])
    rho_star_theory = float(np.sqrt(max(1 - T / N, 0)))
    rho_star_cal = float(np.percentile(rho_mean, 75))
    regime = (rho_mean > rho_star_cal).astype(int)
    crisis_pct = float(100 * regime.mean())
    ci_lo, ci_hi = bootstrap_ci(rho_mean, np.mean, n_boot=n_boot, alpha=alpha)
    return rho_mean, rho_star_theory, rho_star_cal, regime, crisis_pct, ci_lo, ci_hi


def plot_C(rho_mean, rho_star_theory, rho_star_cal, regime, crisis_pct,
           ci_lo, ci_hi, T, N, alpha, save=True):
    n_w = len(rho_mean)
    w_idx = np.arange(n_w)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(
        "Test C — Market Regime Detection via Calibrated rho*  (Corollary 3)\n"
        fr"Theory: $\rho^*_{{th}} = \sqrt{{1 - T/N}} = {rho_star_theory:.3f}$  "
        fr"|  Calibrated (FULL dataset, not train-only): $\rho^*_{{cal}} = {rho_star_cal:.3f}$",
        fontsize=12, fontweight='bold'
    )
    ax = axes[0]
    ax.plot(w_idx, rho_mean, color='steelblue', lw=0.7, zorder=2,
            label=r'$\rho_{mean}(t)$  — mean |C$_{ij}$|, i≠j')
    ax.axhline(rho_star_cal, color='red', linestyle='--', lw=2,
               label=fr'$\rho^*_{{cal}} = {rho_star_cal:.3f}$')
    ax.axhline(rho_star_theory, color='gray', linestyle=':', lw=1.5,
               label=fr'$\rho^*_{{th}} = {rho_star_theory:.3f}$')
    ax.fill_between(w_idx, ci_lo, ci_hi, alpha=0.15, color='steelblue',
                    label=fr'{alpha*100:.0f}% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.fill_between(w_idx, rho_star_cal, rho_mean, where=(rho_mean > rho_star_cal),
                    alpha=0.35, color='red', label='Crisis regime')
    ax.set_ylabel(r'$\rho_{mean}(t)$')
    ax.set_title('Mean Cross-Asset Correlation Over Time', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, min(1.05, rho_mean.max() * 1.25))

    ax = axes[1]
    ax.fill_between(w_idx, 0, regime, color='coral', alpha=0.85,
                    label='Crisis (1) / Normal (0)')
    ax.set_xlabel('Window index'); ax.set_ylabel('Regime')
    ax.set_title(f'Detected Regime  —  {crisis_pct:.1f}% of windows in Crisis', fontsize=10)
    ax.set_ylim(-0.15, 1.5); ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig('test_C_regime.png', dpi=150, bbox_inches='tight')
        print("[OUT] test_C_regime.png saved")
    plt.show()


def check_known_crisis_periods(rho_mean: np.ndarray, window_end_dates,
                                rho_star_cal: float) -> Optional[bool]:
    """
    FIX 5 (this review round): the naive "crisis_pct=25%" claim is a
    tautology of percentile calibration -- it is ALWAYS ~25% by
    construction, regardless of the data. The only real test of
    Corollary 3 is whether the flagged windows coincide with actual
    historical stress periods.

    Returns:
        True  if crisis periods clearly separate from normal periods (gap > 20 pts)
        False if they do not
        None  if the check could not be run (e.g. no real dates)
    """
    if window_end_dates is None:
        print("[Test C temporal check] SKIPPED -- no real dates available "
              "(synthetic data path). This check requires real market dates.")
        return None

    regime = rho_mean > rho_star_cal
    known_periods = {
        "COVID crash (2020-02-15 to 2020-04-30)": ("2020-02-15", "2020-04-30"),
        "2022 selloff (2022-01-01 to 2022-10-31)": ("2022-01-01", "2022-10-31"),
        "Normal period, e.g. 2017 (2017-01-01 to 2017-12-31)": ("2017-01-01", "2017-12-31"),
    }
    print("\n[Test C temporal check] Does 'Crisis' align with REAL events, "
          "or is 25% just the calibration percentile everywhere?")
    print(f"{'Period':<50} {'n_windows':>10} {'%Crisis':>10}")
    print("-" * 72)
    rows = []
    for name, (start, end) in known_periods.items():
        mask = (window_end_dates >= np.datetime64(start)) & (window_end_dates <= np.datetime64(end))
        if mask.sum() == 0:
            print(f"{name:<50} {'(no data in range)':>10}")
            continue
        pct_crisis = 100 * regime[mask].mean()
        rows.append((name, pct_crisis))
        print(f"{name:<50} {mask.sum():>10} {pct_crisis:>9.1f}%")

    if not rows:
        return None

    covid_or_2022 = [p for n, p in rows if "COVID" in n or "2022" in n]
    normal = [p for n, p in rows if "Normal" in n]
    if covid_or_2022 and normal:
        gap = min(covid_or_2022) - max(normal)
        passed = gap > 20
        if passed:
            print(f"\n  -> PASS: crisis periods show {min(covid_or_2022):.0f}%+ "
                  f"vs. normal period {max(normal):.0f}% -- threshold carries "
                  f"real temporal information (gap={gap:.0f} pts).")
        else:
            print(f"\n  -> FAIL: gap between crisis and normal periods is only "
                  f"{gap:.0f} points -- the 25% figure may carry little real "
                  f"temporal information. Do not claim Corollary 3 is "
                  f"'confirmed' without resolving this.")
        return passed
    return None


def test_C_out_of_sample(corr_matrices: np.ndarray, window_end_dates,
                          N: int, train_end: str = "2022-01-01") -> Optional[bool]:
    """
    The strongest version of Corollary 3's validation: calibrate rho*
    using ONLY pre-train_end data (exactly as calibrate_rho_star() does
    in hyperalign_model.py / train_hyperalign.py), then check whether
    this threshold -- which never saw the 2022 selloff during
    calibration -- still flags it as Crisis. This is the deployment-
    relevant question; check_known_crisis_periods() above answers a
    weaker, in-sample version of it.
    """
    if window_end_dates is None:
        print("[Test C out-of-sample check] SKIPPED -- no real dates available.")
        return None

    mask_offdiag = ~np.eye(N, dtype=bool)
    rho_mean_all = np.array([
        np.mean(np.abs(corr_matrices[w][mask_offdiag])) for w in range(len(corr_matrices))
    ])

    train_mask = window_end_dates < np.datetime64(train_end)
    if train_mask.sum() < 30:
        print(f"[Test C out-of-sample check] SKIPPED -- only {train_mask.sum()} "
              f"train windows before {train_end}, too few to calibrate.")
        return None

    rho_star_oos = float(np.percentile(rho_mean_all[train_mask], 75))
    regime_oos = rho_mean_all > rho_star_oos

    print(f"\n[Test C out-of-sample check] Calibrating rho* on TRAIN ONLY "
          f"(before {train_end}, {train_mask.sum()} windows) -> "
          f"rho*_oos = {rho_star_oos:.4f}")
    print("Does this TRAIN-ONLY threshold still catch the UNSEEN 2022 selloff?")

    periods = {
        "2022 selloff (unseen by calibration)": ("2022-01-01", "2022-10-31"),
        "2017 (unseen, normal)": ("2017-01-01", "2017-12-31"),
    }
    print(f"{'Period':<45} {'n_windows':>10} {'%Crisis':>10}")
    print("-" * 67)
    rows = []
    for name, (start, end) in periods.items():
        mask = (window_end_dates >= np.datetime64(start)) & (window_end_dates <= np.datetime64(end))
        if mask.sum() == 0:
            continue
        pct = 100 * regime_oos[mask].mean()
        rows.append((name, pct))
        print(f"{name:<45} {mask.sum():>10} {pct:>9.1f}%")

    if len(rows) < 2:
        return None
    crisis_pct, normal_pct = rows[0][1], rows[1][1]
    gap = crisis_pct - normal_pct
    passed = gap > 20
    if passed:
        print(f"\n  -> PASS: a threshold calibrated WITHOUT seeing 2022 still "
              f"separates it from a normal year (gap={gap:.0f} pts). "
              f"Corollary 3 generalizes out-of-sample.")
    else:
        print(f"\n  -> FAIL: out-of-sample threshold does not separate 2022 from "
              f"normal (gap={gap:.0f} pts). The in-sample result above may not "
              f"generalize to deployment (train_hyperalign.py's calibration).")
    return passed


def plot_sensitivity(T_current, N_current, save=True):
    T_vals = [10, 15, 20, 30, 50]
    N_vals = [50, 100, 200, 500, 1000]
    table = np.array([[np.sqrt(max(1 - T / N, 0)) for N in N_vals] for T in T_vals])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(table, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(N_vals))); ax.set_yticks(range(len(T_vals)))
    ax.set_xticklabels([f'N={n}' for n in N_vals])
    ax.set_yticklabels([f'T={t}' for t in T_vals])
    ax.set_xlabel('Number of assets (N)'); ax.set_ylabel('Window length (T)')
    ax.set_title(r'Theoretical Regime Threshold  $\rho^* = \sqrt{1 - T/N}$'
                 '\nGreen = stronger orthogonality guarantee')
    for i in range(len(T_vals)):
        for j in range(len(N_vals)):
            ax.text(j, i, f'{table[i, j]:.3f}', ha='center', va='center',
                    fontsize=9, fontweight='bold')
    if T_current in T_vals and N_current in N_vals:
        ti, ni = T_vals.index(T_current), N_vals.index(N_current)
        ax.add_patch(plt.Rectangle((ni-0.5, ti-0.5), 1, 1, fill=False,
                     edgecolor='blue', linewidth=3))
        ax.text(ni, ti + 0.42, 'current', ha='center', fontsize=7, color='blue')
    plt.colorbar(im, label=r'$\rho^*$')
    plt.tight_layout()
    if save:
        plt.savefig('sensitivity_rho_star.png', dpi=150, bbox_inches='tight')
        print("[OUT] sensitivity_rho_star.png saved")
    plt.show()


# ================================================================
# 10. FINAL REPORT
# ================================================================

def print_report(mean_r, ci_A, sig, N, sweep_results, knn_mean_mi,
                 rho_star_th, rho_star_cal, crisis_pct, temporal_check_result,
                 oos_check_result, T, tau, alpha, n_windows):
    sep = '=' * 70
    print(f'\n{sep}')
    print('   HyperAlign-Fin — Empirical Validation Report  v3')
    print(sep)
    print(f'   T={T}  N={N}  tau={tau}  CI={int(alpha*100)}%  n_windows={n_windows}')
    print()

    ok_A = mean_r < 0.5
    print("Test A — Independence Assumption")
    print(f"  Mean |r| = {mean_r:.4f}  [{int(alpha*100)}% CI: {ci_A[0]:.4f}-{ci_A[1]:.4f}]  "
          f"Significant: {sig}/{N}")
    print("  -> Partial support." if ok_A else "  -> Violated.")
    print()

    print("Test B — Informational Orthogonality (PCA sensitivity sweep)")
    largest_k = max(sweep_results.keys())
    stable = (max(r["mean_ratio"] for r in sweep_results.values()) -
              min(r["mean_ratio"] for r in sweep_results.values())) < 0.05
    for k, r in sweep_results.items():
        print(f"    k={r['nc_eff']:3d}  ratio={r['mean_ratio']:.4f}  "
              f"EV(V)={r['mean_explained_var_V']*100:5.1f}%  "
              f"EV(G)={r['explained_var_G']*100:5.1f}%")
    print(f"  k-NN (Kraskov) cross-check MI = {knn_mean_mi:.4f} nats")
    print(f"  Ratio stability across k: {'STABLE' if stable else 'UNSTABLE -- investigate before citing'}")
    ok_B = sweep_results[largest_k]["mean_ratio"] < 0.10 and stable
    print("  -> Confirmed (stable + below threshold)." if ok_B else
          "  -> NOT reliable yet -- do not cite the ratio without resolving instability.")
    print()

    print("Test C — Regime Threshold (Corollary 3)")
    print(f"  rho*_theory={rho_star_th:.4f}  rho*_calibrated={rho_star_cal:.4f} "
          f"(FULL dataset -- recalibrate on TRAIN split only before training)")
    print(f"  Crisis windows = {crisis_pct:.1f}%  "
          f"(NOTE: this is mechanically ~25% by construction of the 75th-"
          f"percentile calibration -- see the temporal check above for the "
          f"real test of whether rho*_cal is meaningful)")
    if temporal_check_result is None:
        print("  -> Temporal check not available (no real dates) -- "
              "Corollary 3 NOT confirmed, only the calibration mechanics ran.")
        ok_C = False
    else:
        ok_C = temporal_check_result
        print("  -> Confirmed, in-sample (crisis windows coincide with real stress periods)."
              if ok_C else
              "  -> NOT confirmed -- crisis label does not separate from normal periods.")

    if oos_check_result is None:
        print("  -> Out-of-sample check not available.")
    else:
        print(f"  -> Out-of-sample (train-only calibration): "
              f"{'CONFIRMED -- generalizes' if oos_check_result else 'FAILS TO GENERALIZE -- see warning above'}")
        ok_C = ok_C and oos_check_result  # require BOTH in-sample and out-of-sample to count as a pass
    print()

    passed = sum([ok_A, ok_B, ok_C])
    print('-' * 70)
    print(f"Overall: {passed}/3 tests passed")
    print(sep)


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print('\n' + '=' * 70)
    print('   HyperAlign-Fin — Empirical Validation  v3  (methodology fixes)')
    print('=' * 70 + '\n')

    if USE_REAL_DATA:
        try:
            prices, sector_labels, N_ASSETS, dates = load_real_data(N_ASSETS)
        except Exception as e:
            print(f"[WARN] yfinance: {e}\n -> switching to synthetic.")
            prices, sector_labels, dates = generate_synthetic(N_ASSETS, N_WINDOWS, T_WINDOW, N_SECTORS)
    else:
        prices, sector_labels, dates = generate_synthetic(N_ASSETS, N_WINDOWS, T_WINDOW, N_SECTORS)

    print()
    V_feats, G_feats, corr_matrices, returns_all = extract_features(
        prices, sector_labels, T_WINDOW, TAU_CORR)
    n_windows_actual = len(returns_all)

    print("\n[TEST A] Independence Assumption ...")
    r_arr, p_arr, mean_r, sig, ci_A_lo, ci_A_hi = test_A(
        returns_all, corr_matrices, N_ASSETS, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA)
    plot_A(r_arr, p_arr, mean_r, sig, N_ASSETS, ci_A_lo, ci_A_hi, CI_ALPHA)

    print("\n[TEST B] Informational Orthogonality -- PCA sensitivity sweep ...")
    sweep_results = test_B_sweep(V_feats, G_feats, n_sample=N_SAMPLE_B)
    plot_B_sensitivity(sweep_results)

    print("\n[TEST B] k-NN (Kraskov) cross-check ...")
    knn_vals, knn_mean_mi = test_B_knn_crosscheck(V_feats, G_feats, n_sample=N_SAMPLE_B)

    # main Test B plot uses the LARGEST swept k (most reliable estimate)
    best_k = max(sweep_results.keys())
    nc_eff = sweep_results[best_k]["nc_eff"]
    py = PCA(n_components=nc_eff).fit_transform(G_feats)
    mi_vals = np.zeros(N_SAMPLE_B); hv_vals = np.zeros(N_SAMPLE_B)
    for i in range(N_SAMPLE_B):
        px = PCA(n_components=nc_eff).fit_transform(V_feats[:, i, :])
        mi_vals[i] = np.mean([disc_mi(px[:, k], py[:, k]) for k in range(nc_eff)])
        hv_vals[i] = np.mean([disc_h(px[:, k]) for k in range(nc_eff)])
    ratio_arr = mi_vals / (hv_vals + 1e-10)
    mean_ratio = float(np.mean(ratio_arr))
    ci_B_lo, ci_B_hi = bootstrap_ci(ratio_arr, np.mean, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA)
    plot_B(mi_vals, hv_vals, ratio_arr, mean_ratio, ci_B_lo, ci_B_hi,
           CI_ALPHA, n_windows_actual, nc_eff)

    print("\n[TEST C] Regime Detection ...")
    rho_mean, rho_star_th, rho_star_cal, regime, crisis_pct, ci_C_lo, ci_C_hi = test_C(
        corr_matrices, T_WINDOW, N_ASSETS, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA)
    plot_C(rho_mean, rho_star_th, rho_star_cal, regime, crisis_pct,
           ci_C_lo, ci_C_hi, T_WINDOW, N_ASSETS, CI_ALPHA)

    # window-end dates aligned with rho_mean/corr_matrices, for the temporal check
    window_end_dates = dates[T_WINDOW - 1: T_WINDOW - 1 + len(rho_mean)] if dates is not None else None
    temporal_check_result = check_known_crisis_periods(rho_mean, window_end_dates, rho_star_cal)
    oos_check_result = test_C_out_of_sample(corr_matrices, window_end_dates, N_ASSETS)

    # Cache corr_matrices + dates to disk so downstream analysis scripts
    # (e.g. sensitivity_regime_percentile.py) can reuse them without
    # re-downloading data or re-extracting features from scratch.
    np.save('corr_matrices_cache.npy', corr_matrices)
    if window_end_dates is not None:
        np.save('window_end_dates_cache.npy', window_end_dates)
    print(f"\n[CACHE] Saved corr_matrices_cache.npy (shape {corr_matrices.shape}) "
          f"and window_end_dates_cache.npy for reuse by other scripts.")

    print("\n[SENS] Sensitivity Table ...")
    plot_sensitivity(T_WINDOW, N_ASSETS)

    print_report(mean_r, (ci_A_lo, ci_A_hi), sig, N_ASSETS, sweep_results,
                 knn_mean_mi, rho_star_th, rho_star_cal, crisis_pct,
                 temporal_check_result, oos_check_result,
                 T_WINDOW, TAU_CORR, CI_ALPHA, n_windows_actual)
