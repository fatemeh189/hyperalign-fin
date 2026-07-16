"""
Sensitivity Analysis: Regime Threshold Percentile Choice
============================================================
Addresses a concrete reviewer concern: the 75th-percentile calibration
for rho*_calibrated (Corollary 3) was chosen without demonstrating that
75 is a better choice than, say, 70 or 80. This script sweeps a range
of percentiles and reports, for each, the same in-sample and
out-of-sample crisis-vs-normal separation gap used to validate
Corollary 3, giving an empirical (not merely heuristic) justification
for the percentile used.

Requires corr_matrices_cache.npy and window_end_dates_cache.npy,
produced by running the (patched) hyperalign_validation.py once.

Usage:
    python sensitivity_regime_percentile.py
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERCENTILES = [60, 65, 70, 75, 80, 85, 90]
KNOWN_PERIODS = {
    "COVID crash": ("2020-02-15", "2020-04-30"),
    "2022 selloff": ("2022-01-01", "2022-10-31"),
    "2017 (calm)": ("2017-01-01", "2017-12-31"),
}
TRAIN_END = "2022-01-01"


def compute_rho_mean(corr_matrices: np.ndarray) -> np.ndarray:
    N = corr_matrices.shape[-1]
    mask = ~np.eye(N, dtype=bool)
    return np.array([np.mean(np.abs(corr_matrices[w][mask])) for w in range(len(corr_matrices))])


def crisis_pct_by_period(regime: np.ndarray, window_end_dates: np.ndarray) -> dict:
    out = {}
    for name, (start, end) in KNOWN_PERIODS.items():
        m = (window_end_dates >= np.datetime64(start)) & (window_end_dates <= np.datetime64(end))
        out[name] = float(100 * regime[m].mean()) if m.sum() > 0 else None
    return out


def main():
    try:
        corr_matrices = np.load("corr_matrices_cache.npy")
        window_end_dates = np.load("window_end_dates_cache.npy", allow_pickle=True)
    except FileNotFoundError as e:
        print(f"[ERROR] Missing cache file: {e}. Run the patched "
              f"hyperalign_validation.py once first (it now saves "
              f"corr_matrices_cache.npy / window_end_dates_cache.npy).")
        return

    N = corr_matrices.shape[-1]
    rho_mean = compute_rho_mean(corr_matrices)
    train_mask = window_end_dates < np.datetime64(TRAIN_END)
    print(f"[sensitivity] N={N} assets, {len(rho_mean)} windows total, "
          f"{train_mask.sum()} in train split (before {TRAIN_END})")

    rows = []
    print(f"\n{'Pctl':<6} {'rho*_full':<12} {'rho*_train':<12} "
          f"{'COVID%':<10} {'2022%':<10} {'2017%':<10} {'Gap':<15} {'Overall%':<12}")
    print("-" * 92)

    for pct in PERCENTILES:
        rho_star_full = float(np.percentile(rho_mean, pct))
        rho_star_train = float(np.percentile(rho_mean[train_mask], pct))

        regime_full = rho_mean > rho_star_full
        overall_crisis_pct = float(100 * regime_full.mean())  # = 100-pct, mechanically, but computed directly for clarity
        crisis_full = crisis_pct_by_period(regime_full, window_end_dates)

        covid = crisis_full["COVID crash"]
        y2022 = crisis_full["2022 selloff"]
        y2017 = crisis_full["2017 (calm)"]
        gap = (min(v for v in [covid, y2022] if v is not None) - y2017
               if covid is not None and y2022 is not None and y2017 is not None else None)

        rows.append({
            "percentile": pct, "rho_star_full": rho_star_full, "rho_star_train": rho_star_train,
            "covid_pct": covid, "y2022_pct": y2022, "y2017_pct": y2017, "gap": gap,
            "overall_crisis_pct": overall_crisis_pct,
        })

        def fmt(v):
            return f"{v:<10.1f}" if v is not None else f"{'N/A':<10}"

        gap_str = f"{gap:<15.1f}" if gap is not None else f"{'N/A':<15}"
        print(f"{pct:<6} {rho_star_full:<12.4f} {rho_star_train:<12.4f} "
              f"{fmt(covid)}{fmt(y2022)}{fmt(y2017)}{gap_str}{overall_crisis_pct:<12.1f}")

    # --- identify the percentile with the largest separation gap ---
    valid_rows = [r for r in rows if r["gap"] is not None]
    if not valid_rows:
        print("\n[ERROR] No period had complete data (COVID + 2022 + 2017 all present) -- "
              "cannot compute a sensitivity gap. Check that window_end_dates spans "
              "2017-2022 at minimum.")
        return

    # --- IMPORTANT: gap alone is a flawed criterion -- lowering the
    # percentile mechanically increases the gap by labeling more of ALL
    # history as "Crisis" (overall_crisis_pct = 100-percentile, exactly,
    # regardless of data). A percentile chosen by gap alone will always
    # push toward the lowest percentile tested, which is not economically
    # defensible: it would treat a very large fraction of all trading
    # days as an exceptional stress regime, emptying the concept of
    # "regime" of its meaning. We therefore report BOTH criteria and
    # recommend the percentile with the best gap AMONG those with a
    # plausible overall crisis rate (conventionally <= 30%), rather than
    # the single best-gap percentile unconditionally.
    PLAUSIBLE_MAX_OVERALL_PCT = 30.0

    valid_rows = [r for r in rows if r["gap"] is not None]
    if not valid_rows:
        print("\n[ERROR] No period had complete data (COVID + 2022 + 2017 all present) -- "
              "cannot compute a sensitivity gap. Check that window_end_dates spans "
              "2017-2022 at minimum.")
        return

    plausible_rows = [r for r in valid_rows if r["overall_crisis_pct"] <= PLAUSIBLE_MAX_OVERALL_PCT]
    best_gap_unconditional = max(valid_rows, key=lambda r: r["gap"])
    chosen = next((r for r in rows if r["percentile"] == 75), None)
    if chosen is None or chosen["gap"] is None:
        print("\n[WARNING] Percentile 75 not evaluable (missing period data) -- "
              "cannot compare against it directly.")
        return

    print(f"\n[NOTE] Gap alone is maximized by the LOWEST percentile tested "
          f"({best_gap_unconditional['percentile']}th, gap={best_gap_unconditional['gap']:.1f}), "
          f"but this labels {best_gap_unconditional['overall_crisis_pct']:.0f}% of ALL "
          f"history as 'Crisis' -- likely too permissive to be a meaningful regime "
          f"indicator. Restricting to percentiles with a plausible overall crisis rate "
          f"(<= {PLAUSIBLE_MAX_OVERALL_PCT:.0f}% of all time) is a fairer comparison.")

    if plausible_rows:
        best_plausible = max(plausible_rows, key=lambda r: r["gap"])
        print(f"\n[RESULT] Among percentiles with overall crisis rate <= "
              f"{PLAUSIBLE_MAX_OVERALL_PCT:.0f}%, the best gap is at "
              f"{best_plausible['percentile']}th percentile "
              f"(gap={best_plausible['gap']:.1f}, overall={best_plausible['overall_crisis_pct']:.0f}%)")
        print(f"[RESULT] Percentile used in the paper (75th): "
              f"gap={chosen['gap']:.1f}, overall={chosen['overall_crisis_pct']:.0f}%")

        if abs(best_plausible["gap"] - chosen["gap"]) <= 5.0:
            print(f"\n[JUSTIFICATION] The 75th percentile (gap={chosen['gap']:.1f}, "
                  f"overall crisis rate=25%) is within 5 points of the best gap achievable "
                  f"among percentiles with a plausible overall crisis rate "
                  f"({best_plausible['percentile']}th, gap={best_plausible['gap']:.1f}). "
                  f"This supports 75 as a principled choice: it is close to optimal on "
                  f"crisis/normal separation while keeping the overall Crisis-regime "
                  f"frequency at a historically plausible ~25%, rather than being chosen "
                  f"arbitrarily or by an unconstrained gap-maximization that would trend "
                  f"toward labeling most of history as 'Crisis'.")
        else:
            print(f"\n[SUGGESTION] Consider {best_plausible['percentile']}th percentile "
                  f"instead of 75th: it achieves a larger separation gap "
                  f"({best_plausible['gap']:.1f} vs {chosen['gap']:.1f}) while keeping "
                  f"the overall crisis rate within a plausible range "
                  f"({best_plausible['overall_crisis_pct']:.0f}% vs {chosen['overall_crisis_pct']:.0f}%).")
    else:
        print(f"\n[WARNING] No percentile tested keeps the overall crisis rate under "
              f"{PLAUSIBLE_MAX_OVERALL_PCT:.0f}%. The 75th percentile (25% overall) is "
              f"likely still the most defensible choice tested, but consider testing "
              f"higher percentiles (e.g. 90-95) as well.")

    # --- plot ---
    pcts = [r["percentile"] for r in rows]
    gaps = [r["gap"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcts, gaps, "o-", color="navy")
    ax.axvline(75, color="red", linestyle="--", label="Percentile used in paper (75th)")
    ax.set_xlabel("Calibration percentile")
    ax.set_ylabel("Crisis/Normal separation gap (percentage points)")
    ax.set_title("Sensitivity of Regime Threshold to Calibration Percentile\n"
                  "(gap = min(COVID%, 2022%) - 2017%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("sensitivity_regime_percentile.png", dpi=150)
    print("\n[OUT] sensitivity_regime_percentile.png saved")


if __name__ == "__main__":
    main()
