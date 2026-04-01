#!/usr/bin/env python3
"""
step1_compute_fdv_nih.py — Compute FDV scores for NIH ChestX-ray14

Reads NIH inference JSONL (from chexagent_runner.py) and NIH labels CSV,
computes FDV = alpha*H_norm + (1-alpha)*D per (idx, corruption, severity),
outputs fdv_nih.csv compatible with step2 WG-CRC evaluation.

Usage:
  python3 step1_compute_fdv_nih.py \
    --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/nih_1k_N6.jsonl \
    --labels_csv /scratch/FOLDER_NAME1/telerad_shift/data/nih_labels_valid.csv \
    --out_csv /scratch/FOLDER_NAME1/telerad_shift/outputs/fdv_nih.csv

NIH findings (13 used, Hernia excluded due to <3 positives):
  Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
  Emphysema, Fibrosis, Infiltration, Mass, Nodule,
  Pleural_Thickening, Pneumonia, Pneumothorax
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── NIH FINDINGS ──────────────────────────────────────────────────────────────
NIH_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Keywords for each finding (case-insensitive)
FINDING_KEYWORDS = {
    "Atelectasis":       ["atelectasis", "atelectatic", "subsegmental", "discoid"],
    "Cardiomegaly":      ["cardiomegaly", "enlarged heart", "cardiac enlargement",
                          "cardiomegaly", "enlarged cardiac"],
    "Consolidation":     ["consolidation", "consolidative", "airspace opacity",
                          "lobar opacity"],
    "Edema":             ["edema", "oedema", "pulmonary edema", "interstitial edema",
                          "vascular congestion", "congestion"],
    "Effusion":          ["effusion", "pleural fluid", "pleural effusion",
                          "blunting", "costophrenic"],
    "Emphysema":         ["emphysema", "emphysematous", "hyperinflation",
                          "hyperinflated", "air trapping"],
    "Fibrosis":          ["fibrosis", "fibrotic", "scarring", "scar", "fibrous"],
    "Infiltration":      ["infiltrate", "infiltration", "opacity", "opacities",
                          "haziness", "hazy"],
    "Mass":              ["mass", "masses", "lesion", "lesions", "nodular mass"],
    "Nodule":            ["nodule", "nodules", "nodular"],
    "Pleural_Thickening":["pleural thickening", "thickened pleura",
                          "pleural scarring"],
    "Pneumonia":         ["pneumonia", "pneumonic", "infection", "infectious"],
    "Pneumothorax":      ["pneumothorax", "pneumothoraces", "collapsed lung",
                          "lung collapse"],
}

NEGATIVE_KEYWORDS = [
    "no ", "without", "absent", "clear", "normal", "unremarkable",
    "negative", "not seen", "no evidence", "free of", "well expanded"
]


def extract_findings(text: str) -> dict:
    """
    Extract binary findings from free-text radiology report.
    Returns dict {finding: 0 or 1}.
    Heuristic: finding present if keyword found AND no negation in same sentence.
    """
    text_lower = text.lower()
    result = {}

    for finding, keywords in FINDING_KEYWORDS.items():
        found = 0
        for kw in keywords:
            if kw in text_lower:
                # Check for negation in surrounding context (±80 chars)
                idx = text_lower.find(kw)
                context = text_lower[max(0, idx-80):idx+80]
                negated = any(neg in context for neg in NEGATIVE_KEYWORDS)
                if not negated:
                    found = 1
                    break
        result[finding] = found

    return result


def compute_entropy(predictions: list) -> float:
    """
    Shannon entropy of finding predictions across N samples.
    predictions: list of dicts {finding: 0/1}
    Returns normalized entropy H in [0, 1].
    """
    if not predictions:
        return 1.0  # max uncertainty if no predictions

    n = len(predictions)
    if n < 2:
        return 0.0

    findings = list(predictions[0].keys())
    H_total = 0.0

    for f in findings:
        votes = [p[f] for p in predictions]
        p1 = sum(votes) / n
        p0 = 1 - p1
        # Binary entropy
        h = 0.0
        if p1 > 0: h -= p1 * math.log2(p1)
        if p0 > 0: h -= p0 * math.log2(p0)
        H_total += h

    # Normalize by max possible entropy (1 bit per finding)
    H_norm = H_total / len(findings) if findings else 0.0
    return float(np.clip(H_norm, 0, 1))


def compute_disagreement(predictions: list) -> float:
    """
    Cross-modal disagreement D = mean pairwise disagreement across findings.
    D = 0 if all samples agree, D = 1 if maximum disagreement.
    """
    if len(predictions) < 2:
        return 0.0

    findings = list(predictions[0].keys())
    n = len(predictions)
    D_total = 0.0

    for f in findings:
        votes = [p[f] for p in predictions]
        # Fraction of pairs that disagree
        disagreements = 0
        pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                pairs += 1
                if votes[i] != votes[j]:
                    disagreements += 1
        D_f = disagreements / pairs if pairs > 0 else 0.0
        D_total += D_f

    return float(D_total / len(findings)) if findings else 0.0


def compute_risk(pred_findings: dict, gt_findings: dict) -> float:
    """
    Risk = 1 - macro F1 between predicted and ground-truth findings.
    pred_findings: dict {finding: 0/1} — mode prediction
    gt_findings:   dict {finding: 0/1} — ground truth
    Returns float in [0, 1].
    """
    findings = [f for f in NIH_FINDINGS if f in gt_findings]
    if not findings:
        return 1.0

    f1_scores = []
    for f in findings:
        pred = pred_findings.get(f, 0)
        gt   = gt_findings.get(f, 0)

        tp = int(pred == 1 and gt == 1)
        fp = int(pred == 1 and gt == 0)
        fn = int(pred == 0 and gt == 1)
        tn = int(pred == 0 and gt == 0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt == 0 else 0.0)
        rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if gt == 0 else 0.0)

        # Edge case: both pred and gt are 0 → F1 = 1
        if gt == 0 and pred == 0:
            f1 = 1.0
        elif (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)
    return float(1.0 - macro_f1)


def mode_prediction(predictions: list) -> dict:
    """Majority vote across N samples per finding."""
    if not predictions:
        return {f: 0 for f in NIH_FINDINGS}
    findings = list(predictions[0].keys())
    result = {}
    n = len(predictions)
    for f in findings:
        votes = sum(p[f] for p in predictions)
        result[f] = int(votes > n / 2)
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",       required=True,
                    help="NIH inference JSONL from chexagent_runner.py")
    ap.add_argument("--labels_csv",  required=True,
                    help="NIH labels CSV with columns: idx, Atelectasis, ...")
    ap.add_argument("--out_csv",     required=True,
                    help="Output FDV CSV path")
    ap.add_argument("--alpha",       type=float, default=0.5,
                    help="FDV = alpha*H + (1-alpha)*D (default 0.5)")
    ap.add_argument("--min_valid_samples", type=int, default=2,
                    help="Min non-empty samples required (else skip row)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load ground truth labels
    print(f"[step1] Loading labels from {args.labels_csv}", flush=True)
    labels_df = pd.read_csv(args.labels_csv)

    # Normalize label column names
    labels_df.columns = [c.strip() for c in labels_df.columns]

    # Build Path -> gt dict (join on image filename)
    gt_map = {}
    for _, row in labels_df.iterrows():
        image_key = str(row["Image"]).strip()
        gt = {}
        for f in NIH_FINDINGS:
            if f in row:
                gt[f] = int(row[f]) if not pd.isna(row[f]) else 0
            else:
                gt[f] = 0
        gt_map[image_key] = gt

    print(f"[step1] Loaded GT for {len(gt_map)} images", flush=True)
    print(f"[step1] Findings: {NIH_FINDINGS}", flush=True)

    # Process JSONL
    records = []
    skipped = 0
    H_max_seen = 0.0

    print(f"[step1] Processing {args.jsonl} ...", flush=True)

    # First pass: collect all H values for normalization
    all_H = []
    with open(args.jsonl) as f:
        for line in tqdm(f, desc="Pass 1 (H values)"):
            r = json.loads(line.strip())
            valid = [s for s in r["samples"] if s and s.strip()]
            if len(valid) < args.min_valid_samples:
                continue
            preds = [extract_findings(s) for s in valid]
            H = compute_entropy(preds)
            all_H.append(H)

    H_max = max(all_H) if all_H else 1.0
    print(f"[step1] H_max={H_max:.4f} (used for normalization)", flush=True)

    # Second pass: compute FDV and risk
    with open(args.jsonl) as f:
        for line in tqdm(f, desc="Pass 2 (FDV + risk)"):
            r = json.loads(line.strip())

            idx        = int(r["idx"])
            path       = str(r.get("Path", "")).strip()
            corruption = str(r["corruption"])
            severity   = int(r["severity"])

            if path not in gt_map:
                skipped += 1
                continue

            valid = [s for s in r["samples"] if s and s.strip()]
            if len(valid) < args.min_valid_samples:
                skipped += 1
                continue

            preds    = [extract_findings(s) for s in valid]
            H        = compute_entropy(preds)
            D        = compute_disagreement(preds)
            H_norm   = float(np.clip(H / H_max, 0, 1))
            FDV      = args.alpha * H_norm + (1 - args.alpha) * D

            # Risk = 1 - macro F1 of mode prediction vs GT
            mode_pred = mode_prediction(preds)
            risk      = compute_risk(mode_pred, gt_map[path])

            records.append({
                "idx":        idx,
                "corruption": corruption,
                "severity":   severity,
                "H":          H,
                "H_norm":     H_norm,
                "D":          D,
                "FDV":        FDV,
                "risk":       risk,
                "n_valid":    len(valid),
                "n_samples":  len(r["samples"]),
            })

    print(f"[step1] Records: {len(records)} | Skipped: {skipped}", flush=True)

    df = pd.DataFrame(records)

    # Summary stats
    print(f"\n[step1] Summary:")
    print(f"  FDV:  mean={df['FDV'].mean():.3f}  std={df['FDV'].std():.3f}")
    print(f"  H:    mean={df['H'].mean():.3f}   std={df['H'].std():.3f}")
    print(f"  D:    mean={df['D'].mean():.3f}   std={df['D'].std():.3f}")
    print(f"  risk: mean={df['risk'].mean():.3f}  std={df['risk'].std():.3f}")

    # Spearman correlation FDV vs risk
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["FDV"], df["risk"])
    print(f"\n[step1] Spearman rho(FDV, risk) = {rho:.3f}  p={pval:.2e}")

    rho_d, pval_d = spearmanr(df["D"], df["risk"])
    rho_h, pval_h = spearmanr(df["H_norm"], df["risk"])
    print(f"[step1] Spearman rho(D, risk)   = {rho_d:.3f}  p={pval_d:.2e}")
    print(f"[step1] Spearman rho(H, risk)   = {rho_h:.3f}  p={pval_h:.2e}")

    # Per-corruption breakdown
    print(f"\n[step1] Per-corruption Spearman rho(FDV, risk):")
    for corr in sorted(df["corruption"].unique()):
        sub = df[df["corruption"] == corr]
        r2, p2 = spearmanr(sub["FDV"], sub["risk"])
        print(f"  {corr:15s}: rho={r2:.3f}  p={p2:.2e}  n={len(sub)}")

    # Save
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[step1] Saved {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()