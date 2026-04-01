#!/usr/bin/env python3
"""
step1_compute_fdv.py — Compute FDV scores from JSONL + CheXpert GT labels

FDV(x) = 0.5 * H_norm(x) + 0.5 * D(x)

  H(x) = average per-finding entropy across N=6 stochastic samples
         (extracted via keyword matching on structured CheXagent output)

  D(x) = disagreement between CheXpert GT label and
         mode prediction from N=6 samples

Labels used (subset of 14 that have reliable keyword mapping):
  Cardiomegaly, Edema, Consolidation, Atelectasis,
  Pleural Effusion, Pneumothorax, Lung Opacity

Usage:
  python step1_compute_fdv.py --split cal
  python step1_compute_fdv.py --split test
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = "/scratch/FOLDER_NAME1/telerad_shift"
PATHS = {
    # "cal": {
    #     "jsonl":   f"{BASE}/outputs/clean_500_N6.jsonl",
    #     "ds":      f"{BASE}/cal_5k",
    #     "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
    #     "out":     f"{BASE}/outputs/fdv_clean_cal500.csv",
    # },

    "cal": {
        "jsonl":   f"{BASE}/outputs/cal_6k_N6.jsonl",
        "ds":      f"{BASE}/cal_5k",
        "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
        "out":     f"{BASE}/outputs/fdv_cal_new2.csv",
    },
    "test": {
        "jsonl":   f"{BASE}/outputs/test_5k_N6.jsonl",
        "ds":      f"{BASE}/test_5k",
        "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
        "out":     f"{BASE}/outputs/fdv_test2.csv",
    },
    #  "test": {
    #     "jsonl":   f"{BASE}/outputs/test_llava_rad1.jsonl",
    #     "ds":      f"{BASE}/test_5k",
    #     "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
    #     "out":     f"{BASE}/outputs/fdv_llava_rad_test.csv",
    # },
    # "clean_cal": {
    #     "jsonl":   f"{BASE}/outputs/calib_clean_3k_N6.jsonl",
    #     "ds":      f"{BASE}/cal_5k",
    #     "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
    #     "out":     f"{BASE}/outputs/fdv_clean_1.csv",
    # },
    
}

# PATHS = {
#     "cal": {
#         "jsonl":   f"{BASE}/outputs/maira2_cal_6k_N6.jsonl",
#         "ds":      f"{BASE}/cal_5k",
#         "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
#         "out":     f"{BASE}/outputs/fdv_cal_maira2_1.csv",
#     },
#     "test": {
#         "jsonl":   f"{BASE}/outputs/maira2_test_5k_N6.jsonl",
#         "ds":      f"{BASE}/test_5k",
#         "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
#         "out":     f"{BASE}/outputs/fdv_test_maira2_1.csv",
#     },
# }

# PATHS = {
#     # "cal": {
#     #     "jsonl":   f"{BASE}/outputs/qwen2vl_cal.jsonl",
#     #     "ds":      f"{BASE}/cal_5k",
#     #     "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
#     #     "out":     f"{BASE}/outputs/fdv_qwen2vl_1.csv",
#     # },
#     "cal": {
#         "jsonl":   f"{BASE}/outputs/llava_rad_cal_6k_N6.jsonl",
#         "ds":      f"{BASE}/cal_5k",
#         "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
#         "out":     f"{BASE}/outputs/fdv_llava_rad1.csv",
#     },
#     # "test": {
#     #     "jsonl":   f"{BASE}/outputs/maira2_test_5k_N6.jsonl",
#     #     "ds":      f"{BASE}/test_5k",
#     #     "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
#     #     "out":     f"{BASE}/outputs/fdv_test_maira2_1.csv",
#     # },
# }

# ── LABEL CONFIG ──────────────────────────────────────────────────────────────
# These 7 findings have clear keyword signals in CheXagent structured output
# and reliable GT labels in CheXpert (fewer unlabeled/uncertain than others)
FINDINGS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Pneumothorax",
    "Lung Opacity",
]

# CheXpert integer encoding
LABEL_PRESENT  = 3
LABEL_ABSENT   = 2
LABEL_UNCERTAIN = 1
LABEL_UNLABELED = 0

# Keyword patterns for each finding in generated text
# Format: (positive_patterns, negative_patterns)
FINDING_PATTERNS = {
    "Cardiomegaly": (
        [r"cardiomegal", r"enlarged cardiac", r"enlarged heart",
         r"cardiac enlargement", r"increased cardiac"],
        [r"normal cardiac", r"normal heart size", r"cardiac silhouette.*normal",
         r"no cardiomegal", r"heart size.*normal", r"cardiomediastinal.*normal"]
    ),
    "Edema": (
        [r"\bedema\b", r"pulmonary edema", r"interstitial edema",
         r"vascular congestion", r"pulmonary congestion", r"fluid overload"],
        [r"no.*edema", r"no pulmonary edema", r"without edema",
         r"no vascular congestion", r"clear.*lung"]
    ),
    "Consolidation": (
        [r"consolidat", r"airspace opacity", r"airspace disease",
         r"lobar opacity", r"segmental opacity"],
        [r"no.*consolidat", r"without consolidat", r"no airspace",
         r"lungs.*clear", r"clear.*lung"]
    ),
    "Atelectasis": (
        [r"atelectasis", r"atelectatic", r"collapse", r"subsegmental",
         r"plate-like", r"linear atelectasis", r"discoid atelectasis"],
        [r"no.*atelectasis", r"no.*collapse", r"fully expanded",
         r"well.*expanded", r"without.*atelectasis"]
    ),
    "Pleural Effusion": (
        [r"pleural effusion", r"pleural fluid", r"blunting.*costophrenic",
         r"costophrenic.*blunting", r"pleural.*fluid", r"hydrothorax"],
        [r"no.*pleural effusion", r"no.*effusion", r"pleural.*clear",
         r"costophrenic.*sharp", r"no effusion"]
    ),
    "Pneumothorax": (
        [r"pneumothorax", r"pneumothorac"],
        [r"no.*pneumothorax", r"without.*pneumothorax",
         r"no pneumothorax", r"no ptx"]
    ),
    "Lung Opacity": (
        [r"lung opacity", r"pulmonary opacity", r"opacity.*lung",
         r"opacit", r"haziness", r"infiltrate"],
        [r"no.*opacity", r"lungs.*clear", r"clear.*lung",
         r"no.*infiltrate", r"no focal"]
    ),
}


def extract_finding(text: str, finding: str) -> int:
    """
    Returns 1 (present), 0 (absent), or -1 (uncertain/not mentioned)
    by matching positive and negative keyword patterns.
    """
    text_lower = text.lower()
    pos_patterns, neg_patterns = FINDING_PATTERNS[finding]

    pos_match = any(re.search(p, text_lower) for p in pos_patterns)
    neg_match = any(re.search(p, text_lower) for p in neg_patterns)

    if pos_match and not neg_match:
        return 1
    elif neg_match and not pos_match:
        return 0
    elif pos_match and neg_match:
        # Both — lean toward present (conservative for risk)
        return 1
    else:
        return -1  # uncertain / not mentioned


def extract_all_findings(text: str) -> dict:
    return {f: extract_finding(text, f) for f in FINDINGS}


def compute_entropy(values: list) -> float:
    """
    Shannon entropy of a list of predictions {-1, 0, 1}.
    Ignores -1 (uncertain) if there are definite predictions.
    Falls back to max entropy if all uncertain.
    """
    definite = [v for v in values if v != -1]
    if not definite:
        return 1.0  # max uncertainty if model never committed

    counts = Counter(definite)
    total = len(definite)
    probs = [c / total for c in counts.values()]
    # Shannon entropy normalized to [0,1] (max entropy for binary = log2(2) = 1)
    raw = -sum(p * np.log2(p + 1e-10) for p in probs)
    return min(raw, 1.0)


def compute_h_score(samples: list) -> float:
    """
    H(x) = average per-finding entropy across N samples.
    Returns value in [0, 1].
    """
    # For each finding, collect predictions across all samples
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        preds = extract_all_findings(s)
        for f in FINDINGS:
            finding_preds[f].append(preds[f])

    entropies = [compute_entropy(finding_preds[f]) for f in FINDINGS]
    return float(np.mean(entropies))


def get_mode_prediction(samples: list) -> dict:
    """
    Mode prediction across N samples for each finding.
    Returns dict {finding: 1/0/-1}
    """
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        preds = extract_all_findings(s)
        for f in FINDINGS:
            finding_preds[f].append(preds[f])

    mode = {}
    for f in FINDINGS:
        vals = finding_preds[f]
        definite = [v for v in vals if v != -1]
        if definite:
            mode[f] = Counter(definite).most_common(1)[0][0]
        else:
            mode[f] = -1
    return mode


def get_gt_labels(ds_row: dict) -> dict:
    """
    Extract GT labels from CheXpert dataset row.
    Returns dict {finding: 1 (present), 0 (absent/uncertain/unlabeled)}
    We treat PRESENT=3 as positive, everything else as negative/unknown.
    """
    # Map CheXpert column names to our FINDINGS
    col_map = {
        "Cardiomegaly":    "Cardiomegaly",
        "Edema":           "Edema",
        "Consolidation":   "Consolidation",
        "Atelectasis":     "Atelectasis",
        "Pleural Effusion": "Pleural Effusion",
        "Pneumothorax":    "Pneumothorax",
        "Lung Opacity":    "Lung Opacity",
    }
    gt = {}
    for col, finding in col_map.items():
        val = ds_row.get(col, LABEL_UNLABELED)
        gt[finding] = 1 if val == LABEL_PRESENT else 0
    return gt


def compute_d_score(mode_preds: dict, gt_labels: dict) -> float:
    """
    D(x) = fraction of findings where mode prediction disagrees with GT.
    Skips findings where mode is uncertain (-1) or GT is unlabeled.
    """
    disagreements = []
    for f in FINDINGS:
        pred = mode_preds[f]
        gt = gt_labels[f]
        if pred == -1:
            # Model uncertain — count as disagreement (conservative)
            disagreements.append(1.0)
        else:
            disagreements.append(float(pred != gt))

    return float(np.mean(disagreements))


def compute_risk(mode_preds: dict, gt_labels: dict) -> float:
    """
    L(x) = 1 - F1(GT, mode_prediction) over the 7 findings.
    Uses macro F1 treating finding presence as binary classification.
    """
    tp = fp = fn = 0
    for f in FINDINGS:
        pred = mode_preds[f]
        gt = gt_labels[f]

        # If uncertain, treat as negative prediction
        pred_bin = 1 if pred == 1 else 0

        if pred_bin == 1 and gt == 1:
            tp += 1
        elif pred_bin == 1 and gt == 0:
            fp += 1
        elif pred_bin == 0 and gt == 1:
            fn += 1

    if tp + fp + fn == 0:
        return 0.0  # Both predicted and GT all negative — correct

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(1.0 - f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cal", "test","clean_cal"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight for H in FDV = alpha*H + (1-alpha)*D")
    args = ap.parse_args()

    cfg = PATHS[args.split]
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)

    print(f"[step1] Split={args.split} alpha={args.alpha}")
    print(f"[step1] Loading HF dataset from {cfg['ds']}...")
    ds = load_from_disk(cfg["ds"])

    # Build Path -> dataset index lookup
    print("[step1] Building path->idx lookup...")
    path_to_idx = {ds[i]["Path"]: i for i in range(len(ds))}
    print(f"[step1] {len(path_to_idx)} paths indexed")

    print(f"[step1] Loading JSONL from {cfg['jsonl']}...")
    records = []
    with open(cfg["jsonl"]) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"[step1] {len(records)} records loaded")

    results = []
    n_path_miss = 0

    for rec in tqdm(records, desc=f"Computing FDV [{args.split}]"):
        path = rec.get("Path", "")
        corruption = rec["corruption"]
        severity = rec["severity"]
        idx = rec["idx"]
        samples = rec["samples"]

        # Get dataset row for GT labels
        ds_idx = path_to_idx.get(path)
        if ds_idx is None:
            # Fallback: use idx directly
            ds_idx = idx
            n_path_miss += 1

        ds_row = ds[ds_idx]
        gt_labels = get_gt_labels(ds_row)

        # Compute FDV components
        h_score = compute_h_score(samples)
        mode_preds = get_mode_prediction(samples)
        d_score = compute_d_score(mode_preds, gt_labels)
        risk = compute_risk(mode_preds, gt_labels)

        results.append({
            "idx": idx,
            "path": path,
            "corruption": corruption,
            "severity": severity,
            "H": h_score,
            "D": d_score,
            "risk": risk,
            # Store mode predictions for debugging
            **{f"pred_{f.replace(' ','_')}": mode_preds[f] for f in FINDINGS},
            **{f"gt_{f.replace(' ','_')}": gt_labels[f] for f in FINDINGS},
        })

    df = pd.DataFrame(results)

    # Normalize H to [0,1] over this split (needed for FDV combination)
    h_min, h_max = df["H"].min(), df["H"].max()
    if h_max > h_min:
        df["H_norm"] = (df["H"] - h_min) / (h_max - h_min)
    else:
        df["H_norm"] = 0.0

    # Compute FDV
    df["FDV"] = args.alpha * df["H_norm"] + (1 - args.alpha) * df["D"]

    df.to_csv(cfg["out"], index=False)
    print(f"\n[step1] Saved {len(df)} rows -> {cfg['out']}")
    print(f"[step1] Path misses (used idx fallback): {n_path_miss}")

    # Summary stats
    print(f"\n=== FDV Summary [{args.split}] ===")
    print(f"H    : mean={df['H'].mean():.3f}  std={df['H'].std():.3f}")
    print(f"D    : mean={df['D'].mean():.3f}  std={df['D'].std():.3f}")
    print(f"FDV  : mean={df['FDV'].mean():.3f}  std={df['FDV'].std():.3f}")
    print(f"Risk : mean={df['risk'].mean():.3f}  std={df['risk'].std():.3f}")
    print(f"\nPer-group mean risk:")
    print(df.groupby(["corruption","severity"])["risk"]
            .mean().reset_index().to_string(index=False))

    # FDV-Risk correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["FDV"], df["risk"])
    print(f"\nSpearman rho(FDV, risk) = {rho:.3f}  p={pval:.2e}")
    print("(Should be > 0.3 for FDV to be useful as uncertainty signal)")


if __name__ == "__main__":
    main()