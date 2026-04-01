#!/usr/bin/env python3
"""
step1_compute_fdv_qwen.py — Compute FDV from Qwen2-VL structured output

Qwen2-VL outputs clean yes/no per finding:
  Cardiomegaly: Present
  Pleural Effusion: Absent
  ...

So parsing is trivial — no keyword matching needed.

Usage:
  python3 step1_compute_fdv_qwen.py --split cal
  python3 step1_compute_fdv_qwen.py --split cal --alpha 0.5
"""

import argparse
import json
import os
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.stats import spearmanr
from tqdm import tqdm

# ── PATHS ──────────────────────────────────────────────────────────────────
BASE = "/scratch/FOLDER_NAME1/telerad_shift"
PATHS = {
    "cal": {
        "jsonl":    f"{BASE}/outputs/qwen2vl_cal.jsonl",
        "ds":       f"{BASE}/cal_5k",
        "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
        "out":      f"{BASE}/outputs/fdv_qwen_cal.csv",
    },
    "test": {
        "jsonl":    f"{BASE}/outputs/qwen2vl_test.jsonl",
        "ds":       f"{BASE}/test_5k",
        "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
        "out":      f"{BASE}/outputs/fdv_qwen_test.csv",
    },
}

FINDINGS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Consolidation",
    "Atelectasis",
    "Edema",
    "Pneumothorax",
    "Lung Opacity",
]

LABEL_PRESENT = 3  # CheXpert encoding


# ── PARSER for Qwen structured output ─────────────────────────────────────
# Expected format:
#   Cardiomegaly: Present
#   Pleural Effusion: Absent
#   Consolidation: Present
#   ...

def parse_structured_output(text: str) -> dict:
    """
    Parse Qwen2-VL yes/no structured output.
    Returns {finding: 1/0/-1} for each finding.
    1 = Present, 0 = Absent, -1 = not found / unclear
    """
    if not text or len(text.strip()) < 5:
        return {f: -1 for f in FINDINGS}

    result = {f: -1 for f in FINDINGS}
    t = text.lower()

    for finding in FINDINGS:
        # Match "Finding: Present/Absent" or "Finding: Yes/No"
        # Also handle partial matches and minor formatting variations
        pattern = re.escape(finding.lower()) + r"\s*[:\-]\s*(\w+)"
        m = re.search(pattern, t)
        if m:
            answer = m.group(1).lower()
            if answer in ("present", "yes", "true", "positive", "1"):
                result[finding] = 1
            elif answer in ("absent", "no", "false", "negative", "0", "not"):
                result[finding] = 0
            else:
                result[finding] = -1
        else:
            # Fallback: check if finding is mentioned at all
            if finding.lower() in t:
                # Check context around the mention
                idx = t.find(finding.lower())
                context = t[idx:idx+40]
                if any(w in context for w in ["present", "yes", "positive"]):
                    result[finding] = 1
                elif any(w in context for w in ["absent", "no", "negative"]):
                    result[finding] = 0

    return result


def is_empty(text: str) -> bool:
    return not text or len(text.strip()) < 5


# ── H SCORE ────────────────────────────────────────────────────────────────

def compute_entropy(values: list) -> float:
    definite = [v for v in values if v != -1]
    if not definite:
        return 1.0
    counts = Counter(definite)
    total  = len(definite)
    probs  = [c / total for c in counts.values()]
    raw    = -sum(p * np.log2(p + 1e-10) for p in probs)
    return min(raw, 1.0)


def compute_h_score(samples: list) -> float:
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        preds = parse_structured_output(s)
        for f in FINDINGS:
            finding_preds[f].append(preds[f])
    return float(np.mean([compute_entropy(finding_preds[f]) for f in FINDINGS]))


# ── D SCORE (pairwise) ─────────────────────────────────────────────────────

def compute_d_score(samples: list) -> float:
    valid = [s for s in samples if not is_empty(s)]
    if len(valid) < 2:
        return 1.0

    all_preds = [parse_structured_output(s) for s in valid]
    pair_disagreements = []

    for (pa, pb) in combinations(all_preds, 2):
        disags = []
        for f in FINDINGS:
            a, b = pa[f], pb[f]
            if a == -1 or b == -1:
                disags.append(0.5)
            else:
                disags.append(float(a != b))
        pair_disagreements.append(np.mean(disags))

    return float(np.mean(pair_disagreements))


# ── MODE + RISK ────────────────────────────────────────────────────────────

def get_mode_prediction(samples: list) -> dict:
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        if not is_empty(s):
            preds = parse_structured_output(s)
            for f in FINDINGS:
                finding_preds[f].append(preds[f])

    mode = {}
    for f in FINDINGS:
        definite = [v for v in finding_preds[f] if v != -1]
        mode[f] = Counter(definite).most_common(1)[0][0] if definite else -1
    return mode


def get_gt_labels(ds_row: dict) -> dict:
    col_map = {f: f for f in FINDINGS}
    gt = {}
    for col, finding in col_map.items():
        val = ds_row.get(col, 0)
        gt[finding] = 1 if val == LABEL_PRESENT else 0
    return gt


def compute_risk(mode_preds: dict, gt_labels: dict) -> float:
    tp = fp = fn = 0
    for f in FINDINGS:
        pred = 1 if mode_preds[f] == 1 else 0
        gt   = gt_labels[f]
        if   pred == 1 and gt == 1: tp += 1
        elif pred == 1 and gt == 0: fp += 1
        elif pred == 0 and gt == 1: fn += 1
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(1.0 - f1)


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cal", "test"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    cfg = PATHS[args.split]
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)

    print(f"[step1_qwen] split={args.split}  alpha={args.alpha}")
    print(f"[step1_qwen] Loading HF dataset from {cfg['ds']} ...")
    ds = load_from_disk(cfg["ds"])
    path_to_idx = {ds[i]["Path"]: i for i in range(len(ds))}
    print(f"[step1_qwen] {len(path_to_idx)} paths indexed")

    # Handle chunked output files
    jsonl_path = cfg["jsonl"]
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        # Try loading from chunks
        chunk_files = sorted(Path(os.path.dirname(jsonl_path))
                             .glob(f"{Path(jsonl_path).stem}_chunk*.jsonl"))
        print(f"[step1_qwen] Loading from {len(chunk_files)} chunk files ...")
        for cf in chunk_files:
            with open(cf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    print(f"[step1_qwen] {len(records)} records loaded")

    results = []
    n_path_miss = n_empty = 0

    for rec in tqdm(records, desc="Computing FDV [qwen]"):
        path       = rec.get("Path", "")
        corruption = rec["corruption"]
        severity   = rec["severity"]
        idx        = rec["idx"]
        samples    = rec.get("samples", [])

        valid = [s for s in samples if not is_empty(s)]
        if len(valid) == 0:
            n_empty += 1

        ds_idx = path_to_idx.get(path)
        if ds_idx is None:
            ds_idx = idx
            n_path_miss += 1

        ds_row    = ds[ds_idx]
        gt_labels = get_gt_labels(ds_row)

        h_score    = compute_h_score(samples)
        d_score    = compute_d_score(samples)
        mode_preds = get_mode_prediction(samples)
        risk       = compute_risk(mode_preds, gt_labels)

        results.append({
            "idx":        idx,
            "path":       path,
            "corruption": corruption,
            "severity":   severity,
            "n_valid":    len(valid),
            "H":          h_score,
            "D":          d_score,
            "risk":       risk,
            **{f"pred_{f.replace(' ','_')}": mode_preds[f] for f in FINDINGS},
            **{f"gt_{f.replace(' ','_')}":   gt_labels[f]  for f in FINDINGS},
        })

    df = pd.DataFrame(results)

    h_min, h_max = df["H"].min(), df["H"].max()
    df["H_norm"] = (df["H"] - h_min) / (h_max - h_min) if h_max > h_min else 0.0
    df["FDV"]    = args.alpha * df["H_norm"] + (1 - args.alpha) * df["D"]

    df.to_csv(cfg["out"], index=False)
    print(f"\n[step1_qwen] Saved {len(df)} rows → {cfg['out']}")
    print(f"[step1_qwen] Path misses : {n_path_miss}")
    print(f"[step1_qwen] Empty rows  : {n_empty}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"FDV Summary [{args.split}]  (Qwen2-VL)")
    print(f"{'='*50}")
    print(f"  n_valid mean : {df['n_valid'].mean():.1f} / {N_SAMPLES} samples")
    print(f"  H            : mean={df['H'].mean():.3f}  std={df['H'].std():.3f}")
    print(f"  D            : mean={df['D'].mean():.3f}  std={df['D'].std():.3f}")
    print(f"  FDV          : mean={df['FDV'].mean():.3f}  std={df['FDV'].std():.3f}")
    print(f"  Risk         : mean={df['risk'].mean():.3f}  std={df['risk'].std():.3f}")

    print(f"\nPer-group mean risk:")
    print(df.groupby(["corruption","severity"])["risk"]
            .mean().reset_index().to_string(index=False))

    rho,   pval  = spearmanr(df["FDV"], df["risk"])
    rho_h, _     = spearmanr(df["H"],   df["risk"])
    rho_d, _     = spearmanr(df["D"],   df["risk"])

    print(f"\nSpearman rho(FDV, risk) = {rho:.3f}   p = {pval:.2e}")
    print(f"Spearman rho(H,   risk) = {rho_h:.3f}")
    print(f"Spearman rho(D,   risk) = {rho_d:.3f}")
    print(f"\n(CheXagent reference:  rho_H=0.19  rho_D=0.61  rho_FDV=0.49)")

    # Prediction rate sanity check
    print(f"\n=== Prediction rates ===")
    for f in FINDINGS:
        col = f"pred_{f.replace(' ','_')}"
        if col in df.columns:
            n1  = (df[col] == 1).sum()
            n0  = (df[col] == 0).sum()
            nm1 = (df[col] == -1).sum()
            gt_col = f"gt_{f.replace(' ','_')}"
            gt_rate = df[gt_col].mean() * 100 if gt_col in df.columns else -1
            print(f"  {f:20s}  pred={n1/len(df)*100:.1f}%  absent={n0/len(df)*100:.1f}%"
                  f"  uncertain={nm1/len(df)*100:.1f}%  GT={gt_rate:.1f}%")


N_SAMPLES = 6  # for reporting only

if __name__ == "__main__":
    main()