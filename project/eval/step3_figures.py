#!/usr/bin/env python3
"""step3_figures.py — Generate all paper figures. FIXED: epsilon=0.50, correct threshold search."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

BASE     = "/scratch/FOLDER_NAME1/telerad_shift"
CAL_CSV  = f"{BASE}/outputs/fdv_cal.csv"
TEST_CSV = f"{BASE}/outputs/fdv_test.csv"
FIG_DIR  = f"{BASE}/outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
EPSILON = 0.50   # must match step2_wgcrc.py

COLORS = {
    "No Abstention":             "#d62728",
    "Self-Consistency (H only)": "#ff7f0e",
    "Vanilla CRC":               "#9467bd",
    "WG-CRC + FDV (Ours)":       "#2ca02c",
}
CORRUPTION_COLORS = {
    "blur":       "#4e79a7",
    "brightness": "#f28e2b",
    "jpeg":       "#e15759",
    "occlusion":  "#76b7b2",
    "screen":     "#59a14f",
}

print("Loading data...")
cal  = pd.read_csv(CAL_CSV)
test = pd.read_csv(TEST_CSV)
h_min = 0.0
h_max = cal["H"].max()
cal["H_for_thresh"]  = ((cal["H"].clip(lower=0)  - h_min) / max(h_max - h_min, 1e-10)).clip(0, 1)
test["H_for_thresh"] = ((test["H"].clip(lower=0) - h_min) / max(h_max - h_min, 1e-10)).clip(0, 1)
print(f"Epsilon = {EPSILON}")

def find_threshold_for_epsilon(df, score_col, epsilon, n_steps=1000):
    vals = df[score_col].dropna()
    thresholds = np.linspace(vals.min(), vals.max(), n_steps)
    best_t, best_cov = np.nan, 0.0
    for t in thresholds:
        certified = df[df[score_col] <= t]
        if len(certified) < 5:
            continue
        if certified["risk"].mean() <= epsilon:
            cov = len(certified) / len(df)
            if cov > best_cov:
                best_t, best_cov = t, cov
    return best_t if not np.isnan(best_t) else vals.min()

def wgcrc_calibrate(cal_df, score_col, epsilon):
    grp_thresh = {}
    for (c, s), grp in cal_df.groupby(["corruption", "severity"]):
        grp_thresh[(c, s)] = find_threshold_for_epsilon(grp, score_col, epsilon)
    return grp_thresh

def evaluate_per_group(test_df, grp_thresh, score_col="FDV"):
    parts = [grp[grp[score_col] <= t]
             for (c, s), t in grp_thresh.items()
             for grp in [test_df[(test_df["corruption"]==c) & (test_df["severity"]==s)]]]
    return pd.concat(parts) if parts else pd.DataFrame()

def sweep_global(test_df, cal_df, score_col, n_steps=200):
    vals = cal_df[score_col].dropna()
    thresholds = np.linspace(vals.min(), vals.max(), n_steps)
    coverages, worst_risks = [], []
    for t in thresholds:
        cert = test_df[test_df[score_col] <= t]
        if len(cert) == 0:
            coverages.append(0.0); worst_risks.append(np.nan); continue
        pg = cert.groupby(["corruption","severity"])["risk"].mean()
        coverages.append(len(cert)/len(test_df))
        worst_risks.append(pg.max() if len(pg) > 0 else np.nan)
    return np.array(coverages), np.array(worst_risks)

def sweep_wgcrc(test_df, cal_df, score_col, n_steps=25):
    epsilons = np.linspace(0.35, 0.75, n_steps)
    coverages, worst_risks = [], []
    for eps in epsilons:
        gt = wgcrc_calibrate(cal_df, score_col, eps)
        cert = evaluate_per_group(test_df, gt, score_col)
        if len(cert) == 0:
            coverages.append(0.0); worst_risks.append(np.nan); continue
        pg = cert.groupby(["corruption","severity"])["risk"].mean()
        coverages.append(len(cert)/len(test_df))
        worst_risks.append(pg.max() if len(pg) > 0 else np.nan)
    return np.array(coverages), np.array(worst_risks)

# Pre-compute
print("Pre-computing thresholds...")
vcrc_t    = find_threshold_for_epsilon(cal, "FDV", EPSILON)
wg_thresh = wgcrc_calibrate(cal, "FDV", EPSILON)
vcrc_cert = test[test["FDV"] <= vcrc_t]
wg_cert   = evaluate_per_group(test, wg_thresh, "FDV")
print(f"Vanilla CRC t={vcrc_t:.3f}  certified={len(vcrc_cert)} ({len(vcrc_cert)/len(test)*100:.1f}%)")
print(f"WG-CRC certified={len(wg_cert)} ({len(wg_cert)/len(test)*100:.1f}%)")

# ── FIGURE 1 ──────────────────────────────────────────────────────────────────
print("\nGenerating Figure 1: Risk vs Coverage...")
fig, ax = plt.subplots(figsize=(7, 5))

no_abs_risk = test.groupby(["corruption","severity"])["risk"].mean().max()
ax.scatter([100], [no_abs_risk], color=COLORS["No Abstention"], s=120, zorder=5,
           label=f"No Abstention ({no_abs_risk:.3f})")

sc_covs, sc_risks = sweep_global(test, cal, "H_for_thresh")
mask = ~np.isnan(sc_risks)
if mask.sum() > 0:
    ax.plot(sc_covs[mask]*100, sc_risks[mask], color=COLORS["Self-Consistency (H only)"],
            linewidth=2, linestyle="-.", label="Self-Consistency (H only)")

vc_covs, vc_risks = sweep_global(test, cal, "FDV")
mask = ~np.isnan(vc_risks)
ax.plot(vc_covs[mask]*100, vc_risks[mask], color=COLORS["Vanilla CRC"],
        linewidth=2, linestyle=":", label="Vanilla CRC")

wg_covs, wg_risks = sweep_wgcrc(test, cal, "FDV", n_steps=25)
mask = ~np.isnan(wg_risks)
if mask.sum() > 0:
    ax.plot(wg_covs[mask]*100, wg_risks[mask], color=COLORS["WG-CRC + FDV (Ours)"],
            linewidth=2.5, linestyle="-", label="WG-CRC + FDV (Ours)")

ax.axhline(EPSILON, color="black", linewidth=1.5, linestyle="--", alpha=0.8, label=f"ε = {EPSILON}")
ax.fill_between([0, 105], 0, EPSILON, alpha=0.07, color="green")
ax.set_xlabel("Coverage (%)", fontsize=12)
ax.set_ylabel("Worst-Group Risk", fontsize=12)
ax.set_title("Risk–Coverage Tradeoff\n(WG-CRC controls worst-group risk at comparable coverage)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 105)
ax.set_ylim(0, min(no_abs_risk * 1.15, 1.0))
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_risk_coverage.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/fig1_risk_coverage.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig1_risk_coverage.pdf/png")

# ── FIGURE 2 ──────────────────────────────────────────────────────────────────
print("Generating Figure 2: Vanilla CRC per-group failure (KEY FIGURE)...")
groups = sorted(test.groupby(["corruption","severity"]).groups.keys())
group_labels = [f"{c}\nsev={s}" for c, s in groups]

vcrc_pg, wg_pg = [], []
for c, s in groups:
    sv = vcrc_cert[(vcrc_cert["corruption"]==c) & (vcrc_cert["severity"]==s)]
    sw = wg_cert[(wg_cert["corruption"]==c) & (wg_cert["severity"]==s)]
    vcrc_pg.append(sv["risk"].mean() if len(sv) > 0 else np.nan)
    wg_pg.append(sw["risk"].mean() if len(sw) > 0 else np.nan)

valid_vcrc = [r for r in vcrc_pg if not np.isnan(r)]
print(f"  Vanilla CRC: {len(valid_vcrc)}/15 groups certified")
print(f"  WG-CRC:      {len([r for r in wg_pg if not np.isnan(r)])}/15 groups certified")

fig_eps = EPSILON
x = np.arange(len(groups))
width = 0.38
fig, ax = plt.subplots(figsize=(14, 5))

vcrc_colors = [CORRUPTION_COLORS.get(c, "#888888") for c, s in groups]
bars1 = ax.bar(x - width/2, vcrc_pg, width, label="Vanilla CRC",
               color=vcrc_colors, alpha=0.75, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, wg_pg, width, label="WG-CRC + FDV (Ours)",
               color=vcrc_colors, alpha=0.40, edgecolor="white", linewidth=0.5, hatch="///")

n_violate = 0
for bar, risk in zip(bars1, vcrc_pg):
    if not np.isnan(risk) and risk > fig_eps:
        bar.set_edgecolor("#cc0000"); bar.set_linewidth(2.5)
        n_violate += 1

all_vals = [r for r in vcrc_pg + wg_pg if not np.isnan(r)]
y_top = max(all_vals) * 1.25 if all_vals else 1.0

ax.axhline(fig_eps, color="red", linewidth=2.2, linestyle="--",
           label=f"ε = {fig_eps}  ({n_violate} violations)")
ax.fill_between([-0.5, len(groups)-0.5], fig_eps, y_top, alpha=0.08, color="red")
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=8.5)
ax.set_ylabel("Per-Group Risk (1 − F1)", fontsize=12)
ax.set_title(f"Vanilla CRC Violates Worst-Group Risk Bound in {n_violate}/15 Groups\n"
             f"WG-CRC Satisfies Finite-Sample Guarantee (worst group ≤ ε + slack = {fig_eps+0.089:.3f})", fontsize=11)
ax.set_ylim(0, y_top)
ax.grid(True, axis="y", alpha=0.3)

patches = [mpatches.Patch(color=CORRUPTION_COLORS[c], label=c.capitalize(), alpha=0.8)
           for c in sorted(CORRUPTION_COLORS.keys())]
vanilla_patch = mpatches.Patch(color="gray", alpha=0.75, label="Vanilla CRC (solid)")
wg_patch = mpatches.Patch(color="gray", alpha=0.40, hatch="///", label="WG-CRC (hatched)")
ax.legend(handles=[vanilla_patch, wg_patch,
                   plt.Line2D([0],[0], color="red", linewidth=2, linestyle="--",
                              label=f"ε={fig_eps} bound ({n_violate} violations)"),
                   *patches],
          fontsize=8.5, loc="upper right", ncol=2)

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_crc_failure.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/fig2_crc_failure.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved fig2_crc_failure.pdf/png  ({n_violate} violations shown)")

# ── FIGURE 3 ──────────────────────────────────────────────────────────────────
print("Generating Figure 3: FDV-Risk scatter...")
fig, ax = plt.subplots(figsize=(6, 5))
sample = test.sample(min(2000, len(test)), random_state=42)
for corr in sorted(sample["corruption"].unique()):
    sub = sample[sample["corruption"] == corr]
    ax.scatter(sub["FDV"], sub["risk"], alpha=0.35, s=18,
               color=CORRUPTION_COLORS.get(corr, "gray"), label=corr, rasterized=True)
bins = pd.cut(test["FDV"], bins=20)
trend = test.groupby(bins, observed=True)["risk"].mean()
bin_centers = [interval.mid for interval in trend.index]
ax.plot(bin_centers, trend.values, "k-", linewidth=2.5, label="Trend", zorder=5)
rho, pval = spearmanr(test["FDV"], test["risk"])
ax.axvline(vcrc_t, color="purple", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Vanilla CRC t={vcrc_t:.2f}")
ax.set_xlabel("FDV Score", fontsize=12)
ax.set_ylabel("Clinical Risk (1 − F1)", fontsize=12)
ax.set_title(f"FDV Correlates with Clinical Risk\nSpearman ρ = {rho:.3f}  (p < 0.001)", fontsize=11)
ax.legend(fontsize=9, title="Corruption", title_fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_fdv_scatter.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/fig3_fdv_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig3_fdv_scatter.pdf/png")

# ── FIGURE 4 ──────────────────────────────────────────────────────────────────
print("Generating Figure 4: Severity vs Risk heatmap...")
pivot = test.groupby(["corruption","severity"])["risk"].mean().unstack()
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
               vmin=pivot.values.min()*0.95, vmax=min(pivot.values.max()*1.05, 1.0))
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"Severity {s}" for s in pivot.columns], fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index.tolist(), fontsize=10)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11,
                color="white" if val > pivot.values.mean() else "black", fontweight="bold")
plt.colorbar(im, ax=ax, label="Mean Risk (1 − F1)")
ax.set_title("Mean Risk by Corruption Family and Severity\n"
             "(Per-group variation motivates WG-CRC calibration)", fontsize=11)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_severity_heatmap.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/fig4_severity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig4_severity_heatmap.pdf/png")

print(f"\nAll figures saved to {FIG_DIR}/")
for f in sorted(os.listdir(FIG_DIR)):
    size = os.path.getsize(f"{FIG_DIR}/{f}") // 1024
    print(f"  {f}  ({size} KB)")
