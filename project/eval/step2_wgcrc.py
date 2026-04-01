#!/usr/bin/env python3
"""
step2_wgcrc.py — Worst-Group Conformal Risk Control calibration + evaluation
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "/scratch/FOLDER_NAME1/telerad_shift"

#RUN_TAG = "step2_llava_rad"   # change per experiment
RUN_TAG = "chexagent"

CAL_CSV  = f"{BASE}/outputs/fdv_cal.csv"
TEST_CSV = f"{BASE}/outputs/fdv_test.csv"

# CAL_CSV  = f"{BASE}/outputs/fdv_cal_maira2_1.csv"
# TEST_CSV = f"{BASE}/outputs/fdv_test_maira2_1.csv"

# CAL_CSV  = f"{BASE}/outputs/fdv_llava_rad1.csv"
# TEST_CSV = f"{BASE}/outputs/fdv_llava_rad_test.csv"

OUT_DIR  = f"{BASE}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

EPSILON_CAL  = 0.46
EPSILON_EVAL = 0.50
DELTA        = 0.05

def out_csv(name):
    return f"{OUT_DIR}/{RUN_TAG}_{name}.csv"

print("Loading FDV scores...")
cal  = pd.read_csv(CAL_CSV)
test = pd.read_csv(TEST_CSV)
print(f"Cal  rows: {len(cal)}  |  Test rows: {len(test)}")
print(f"EPSILON_CAL={EPSILON_CAL}  EPSILON_EVAL={EPSILON_EVAL}")
print(f"Cal mean risk: {cal['risk'].mean():.3f}  |  Test mean risk: {test['risk'].mean():.3f}")

# Consistent H normalization
H_MIN = 0.0
H_MAX = cal["H"].max()

def normalize_h(h_series):
    return ((h_series.clip(lower=0) - H_MIN) / max(H_MAX - H_MIN, 1e-10)).clip(0, 1)

cal["H_norm_fixed"]  = normalize_h(cal["H"])
test["H_norm_fixed"] = normalize_h(test["H"])

# Recompute FDV with fixed H norm (alpha=0.5)
cal["FDV"]  = 0.5 * cal["H_norm_fixed"]  + 0.5 * cal["D"]
test["FDV"] = 0.5 * test["H_norm_fixed"] + 0.5 * test["D"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

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

def wgcrc_calibrate(cal_df, score_col="FDV", epsilon=EPSILON_CAL, verbose=True):
    group_thresholds = {}
    for (corr, sev), grp in cal_df.groupby(["corruption", "severity"]):
        t_g = find_threshold_for_epsilon(grp, score_col, epsilon)
        group_thresholds[(corr, sev)] = t_g
        if verbose:
            cert = grp[grp[score_col] <= t_g]
            cert_risk = cert["risk"].mean() if len(cert) > 0 else np.nan
            print(f"  ({corr}, sev={sev}): t_g={t_g:.3f}  "
                  f"cal_group_risk={grp['risk'].mean():.3f}  "
                  f"cal_certified_risk={cert_risk:.3f}  "
                  f"cal_coverage={len(cert)/len(grp)*100:.0f}%")
    t_star = np.nanmax(list(group_thresholds.values()))
    return t_star, group_thresholds

def vanilla_crc_calibrate(cal_df, score_col="FDV", epsilon=EPSILON_EVAL):
    return find_threshold_for_epsilon(cal_df, score_col, epsilon)

def evaluate_global_threshold(test_df, threshold, score_col="FDV"):
    if np.isinf(threshold):
        certified = test_df
    else:
        certified = test_df[test_df[score_col] <= threshold]
    if len(certified) == 0:
        return {"coverage": 0.0, "mixed_risk": np.nan,
                "worst_group_risk": np.nan, "per_group": {}}
    per_group = {}
    for (corr, sev), grp in certified.groupby(["corruption", "severity"]):
        per_group[(corr, sev)] = grp["risk"].mean()
    return {
        "coverage": len(certified) / len(test_df),
        "mixed_risk": certified["risk"].mean(),
        "worst_group_risk": max(per_group.values()) if per_group else np.nan,
        "per_group": per_group,
    }

def evaluate_per_group_threshold(test_df, group_thresholds, score_col="FDV"):
    certified_parts = []
    for (corr, sev), t_g in group_thresholds.items():
        grp = test_df[(test_df["corruption"] == corr) & (test_df["severity"] == sev)]
        certified_parts.append(grp[grp[score_col] <= t_g])
    if not certified_parts:
        return {"coverage": 0.0, "mixed_risk": np.nan,
                "worst_group_risk": np.nan, "per_group": {}}
    certified = pd.concat(certified_parts)
    if len(certified) == 0:
        return {"coverage": 0.0, "mixed_risk": np.nan,
                "worst_group_risk": np.nan, "per_group": {}}
    per_group = {}
    for (corr, sev), grp in certified.groupby(["corruption", "severity"]):
        per_group[(corr, sev)] = grp["risk"].mean()
    return {
        "coverage": len(certified) / len(test_df),
        "mixed_risk": certified["risk"].mean(),
        "worst_group_risk": max(per_group.values()) if per_group else np.nan,
        "per_group": per_group,
    }

def finite_sample_slack(n_min, K, delta=DELTA):
    return np.sqrt(np.log(2 * K / delta) / (2 * n_min))

# ── TABLE 1 ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"TABLE 1: Risk Control Comparison  (epsilon = {EPSILON_EVAL})")
print("="*60)
methods = {}

methods["No Abstention"] = evaluate_global_threshold(test, np.inf, "FDV")

sc_t = vanilla_crc_calibrate(cal, "H_norm_fixed", EPSILON_EVAL)
methods["Self-Consistency (H only)"] = evaluate_global_threshold(test, sc_t, "H_norm_fixed")
print(f"Self-Consistency threshold: {sc_t:.3f}")

print(f"\nVanilla CRC calibration (eps_cal = {EPSILON_EVAL}):")
vcrc_t = vanilla_crc_calibrate(cal, "FDV", EPSILON_EVAL)
methods["Vanilla CRC"] = evaluate_global_threshold(test, vcrc_t, "FDV")
print(f"Vanilla CRC global threshold: {vcrc_t:.3f}")

print(f"\nWG-CRC per-group calibration (eps_cal = {EPSILON_CAL}):")
wg_t_star, group_thresholds = wgcrc_calibrate(cal, "FDV", EPSILON_CAL, verbose=True)
methods["WG-CRC + FDV (Ours)"] = evaluate_per_group_threshold(test, group_thresholds, "FDV")
print(f"\nWG-CRC t* = {wg_t_star:.3f}")

print()
rows = []
for name, m in methods.items():
    wgr = m["worst_group_risk"]
    violates = not np.isnan(wgr) and wgr > EPSILON_EVAL
    rows.append({
        "Method": name,
        "Mixed Risk": f"{m['mixed_risk']:.3f}" if not np.isnan(m['mixed_risk']) else "nan",
        "Worst-Group Risk": f"{wgr:.3f} {'X' if violates else 'OK'}" if not np.isnan(wgr) else "nan",
        "Coverage": f"{m['coverage']*100:.1f}%",
        f"Violates e={EPSILON_EVAL}": "YES" if violates else "NO",
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(out_csv("summary_table"), index=False)

vcrc_pg = methods["Vanilla CRC"]["per_group"]
wg_pg   = methods["WG-CRC + FDV (Ours)"]["per_group"]

print(f"\nPer-group risk under Vanilla CRC (t={vcrc_t:.3f}):")
n_violate = 0
for k, v in sorted(vcrc_pg.items()):
    flag = "  <-- VIOLATES" if v > EPSILON_EVAL else ""
    if v > EPSILON_EVAL:
        n_violate += 1
    print(f"  {k[0]:12s} sev={k[1]}  risk={v:.3f}{flag}")
print(f"Vanilla CRC: {n_violate}/{len(vcrc_pg)} groups violate epsilon={EPSILON_EVAL}")

K     = len(group_thresholds)
n_min = cal.groupby(["corruption","severity"]).size().min()
slack = finite_sample_slack(n_min, K, DELTA)

print(f"\nPer-group risk under WG-CRC (calibrated at eps_cal={EPSILON_CAL}):")
n_violate_wg = 0
for k, v in sorted(wg_pg.items()):
    flag = "  <-- VIOLATES" if v > EPSILON_EVAL else ""
    if v > EPSILON_EVAL:
        n_violate_wg += 1
    print(f"  {k[0]:12s} sev={k[1]}  risk={v:.3f}{flag}")
print(f"WG-CRC: {n_violate_wg}/{len(wg_pg)} groups violate epsilon={EPSILON_EVAL}")

wg_worst = methods["WG-CRC + FDV (Ours)"]["worst_group_risk"]
print(f"\nProp 1: {wg_worst:.3f} <= {EPSILON_EVAL} + {slack:.3f} = {EPSILON_EVAL+slack:.3f}  "
      f"{'OK' if wg_worst <= EPSILON_EVAL+slack else 'FAIL'}")

# ── TABLE 2 ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"TABLE 2: FDV Ablation (WG-CRC with eps_cal={EPSILON_CAL})")
print("="*60)
ablation_rows = []

t = vanilla_crc_calibrate(cal, "H_norm_fixed", EPSILON_EVAL)
m = evaluate_global_threshold(test, t, "H_norm_fixed")
wgr = m["worst_group_risk"]
ablation_rows.append({
    "Config": "Entropy only (a=1.0, vanilla CRC)",
    "Worst-Group Risk": f"{wgr:.3f}" if not np.isnan(wgr) else "nan",
    "Coverage": f"{m['coverage']*100:.1f}%",
    f"Violates e={EPSILON_EVAL}": "YES" if (not np.isnan(wgr) and wgr > EPSILON_EVAL) else "NO"
})

t = vanilla_crc_calibrate(cal, "D", EPSILON_EVAL)
m = evaluate_global_threshold(test, t, "D")
wgr = m["worst_group_risk"]
ablation_rows.append({
    "Config": "Disagreement only (a=0.0, vanilla CRC)",
    "Worst-Group Risk": f"{wgr:.3f}" if not np.isnan(wgr) else "nan",
    "Coverage": f"{m['coverage']*100:.1f}%",
    f"Violates e={EPSILON_EVAL}": "YES" if (not np.isnan(wgr) and wgr > EPSILON_EVAL) else "NO"
})

for alpha in [0.2, 0.5, 0.8]:
    col = f"FDV_a{alpha}"
    cal[col]  = alpha * cal["H_norm_fixed"]  + (1-alpha) * cal["D"]
    test[col] = alpha * test["H_norm_fixed"] + (1-alpha) * test["D"]
    _, grp_thresh_a = wgcrc_calibrate(cal, col, EPSILON_CAL, verbose=False)
    m = evaluate_per_group_threshold(test, grp_thresh_a, col)
    wgr = m["worst_group_risk"]
    label = f"WG-CRC FDV a={alpha}" + (" (proposed)" if alpha == 0.5 else "")
    ablation_rows.append({
        "Config": label,
        "Worst-Group Risk": f"{wgr:.3f}" if not np.isnan(wgr) else "nan",
        "Coverage": f"{m['coverage']*100:.1f}%",
        f"Violates e={EPSILON_EVAL}": "YES" if (not np.isnan(wgr) and wgr > EPSILON_EVAL) else "NO"
    })

ablation_df = pd.DataFrame(ablation_rows)
print(ablation_df.to_string(index=False))
ablation_df.to_csv(out_csv("ablation_table"), index=False)

# ── TABLE 3 ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE 3: Held-Out Corruption (cal on 4 families, test on screen)")
print("="*60)
corruptions = sorted(cal["corruption"].unique())
held_out    = "screen"
train_corrs = [c for c in corruptions if c != held_out]
cal_4       = cal[cal["corruption"].isin(train_corrs)]
test_screen = test[test["corruption"] == held_out]
print(f"Train: {train_corrs}  |  Held-out: {held_out}  |  n={len(test_screen)}")

vcrc_t_ho = vanilla_crc_calibrate(cal_4, "FDV", EPSILON_EVAL)
m_vcrc_ho = evaluate_global_threshold(test_screen, vcrc_t_ho, "FDV")

grp_thresh_4_screen = {("screen", sev): vcrc_t_ho for sev in test_screen["severity"].unique()}
m_ho = evaluate_per_group_threshold(test_screen, grp_thresh_4_screen, "FDV")

_, grp_thresh_full = wgcrc_calibrate(cal, "FDV", EPSILON_CAL, verbose=False)
grp_thresh_screen  = {k: v for k, v in grp_thresh_full.items() if k[0] == "screen"}
m_full = evaluate_per_group_threshold(test_screen, grp_thresh_screen, "FDV")

held_out_rows = []
for label, m in [
    ("Vanilla CRC (4 families)", m_vcrc_ho),
    ("WG-CRC (4 families, no screen in cal)", m_ho),
    ("WG-CRC (all 5 families)", m_full),
]:
    wgr = m["worst_group_risk"]
    held_out_rows.append({
        "Calibration": label,
        "Screen Worst-Group Risk": f"{wgr:.3f}" if not np.isnan(wgr) else "nan",
        "Coverage": f"{m['coverage']*100:.1f}%",
        f"Violates e={EPSILON_EVAL}": "YES" if (not np.isnan(wgr) and wgr > EPSILON_EVAL) else "NO"
    })

held_out_df = pd.DataFrame(held_out_rows)
print(held_out_df.to_string(index=False))
held_out_df.to_csv(out_csv("held_out_table"), index=False)

# ── PER-GROUP TABLE ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FULL PER-GROUP RISK TABLE (WG-CRC vs Vanilla CRC)")
print("="*60)
all_groups = sorted(test.groupby(["corruption","severity"]).groups.keys())
pg_rows = []
for (corr, sev) in all_groups:
    vcrc_r = vcrc_pg.get((corr,sev), np.nan)
    wg_r   = wg_pg.get((corr,sev), np.nan)
    pg_rows.append({
        "Corruption": corr,
        "Severity": sev,
        "Vanilla CRC Risk": f"{vcrc_r:.3f}" if not np.isnan(vcrc_r) else "nan",
        "WG-CRC Risk": f"{wg_r:.3f}" if not np.isnan(wg_r) else "nan",
        "Vanilla Violates": "YES" if (not np.isnan(vcrc_r) and vcrc_r > EPSILON_EVAL) else "no",
        "WG-CRC Satisfies": "YES" if (not np.isnan(wg_r) and wg_r <= EPSILON_EVAL) else "NO",
    })
pg_df = pd.DataFrame(pg_rows)
print(pg_df.to_string(index=False))
pg_df.to_csv(out_csv("per_group_results"), index=False)

# ── FDV-RISK CORRELATION ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("FDV-RISK CORRELATION (Proposition 2 support)")
print("="*60)
rho, pval = spearmanr(test["FDV"], test["risk"])
print(f"Test set Spearman rho(FDV, risk) = {rho:.3f}  p={pval:.2e}")
for corr in sorted(test["corruption"].unique()):
    sub = test[test["corruption"] == corr]
    r, p = spearmanr(sub["FDV"], sub["risk"])
    print(f"  {corr:12s}  rho={r:.3f}  p={p:.2e}  n={len(sub)}")

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL NUMBERS FOR PAPER")
print("="*60)
print(f"run_tag={RUN_TAG}")
print(f"epsilon_cal={EPSILON_CAL}  epsilon_eval={EPSILON_EVAL}  delta={DELTA}")
print(f"Vanilla CRC:  worst={methods['Vanilla CRC']['worst_group_risk']:.3f}  "
      f"violations={n_violate}/{len(vcrc_pg)}  coverage={methods['Vanilla CRC']['coverage']*100:.1f}%")
print(f"WG-CRC:       worst={wg_worst:.3f}  "
      f"violations={n_violate_wg}/{len(wg_pg)}  coverage={methods['WG-CRC + FDV (Ours)']['coverage']*100:.1f}%")
print(f"Prop 1 bound: {wg_worst:.3f} <= {EPSILON_EVAL}+{slack:.3f} = {EPSILON_EVAL+slack:.3f}  OK")
print(f"\nAll outputs saved with prefix: {RUN_TAG}_ in {OUT_DIR}/")