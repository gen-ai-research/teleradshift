#!/usr/bin/env python3
"""
step1_compute_fdv_maira2.py — Compute FDV scores from MAIRA-2 JSONL output

MAIRA-2 usually generates more report-like findings text than LLaVA-Med, but
its wording can still vary across samples. This script extracts CheXpert-style
findings from the generated FINDINGS text and computes:

FDV(x) = alpha * H_norm(x) + (1-alpha) * D(x)

  H(x) = average per-finding entropy across N stochastic samples
  D(x) = mean pairwise disagreement across sample pairs

Usage:
  python step1_compute_fdv_maira2.py --split cal
  python step1_compute_fdv_maira2.py --split test
  python step1_compute_fdv_maira2.py --split cal --alpha 0.5 --min_samples 2
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

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = "/scratch/FOLDER_NAME1/telerad_shift"
PATHS = {
    "cal": {
        "jsonl":    f"{BASE}/outputs/maira2_cal_6k_N6.jsonl",
        "ds":       f"{BASE}/cal_5k",
        "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
        "out":      f"{BASE}/outputs/fdv_maira2_cal.csv",
    },
    "test": {
        "jsonl":    f"{BASE}/outputs/maira2_test_5k_N6.jsonl",
        "ds":       f"{BASE}/test_5k",
        "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
        "out":      f"{BASE}/outputs/fdv_maira2_test.csv",
    },
}

# ── FINDINGS ──────────────────────────────────────────────────────────────────
FINDINGS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Pneumothorax",
    "Lung Opacity",
]

LABEL_PRESENT = 3  # CheXpert encoding


# ── TEXT NORMALIZATION ────────────────────────────────────────────────────────

def sample_to_text(sample):
    """
    Robust conversion of MAIRA-2 sample to plain text.
    Usually samples are strings, but keep this safe.
    """
    if sample is None:
        return ""

    if isinstance(sample, str):
        return sample.strip()

    if isinstance(sample, dict):
        for k in ["text", "report", "findings", "output"]:
            if k in sample and isinstance(sample[k], str):
                return sample[k].strip()
        parts = []
        for _, v in sample.items():
            t = sample_to_text(v)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()

    if isinstance(sample, list):
        parts = []
        for x in sample:
            t = sample_to_text(x)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()

    return str(sample).strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def is_empty(text: str) -> bool:
    text = normalize_text(text)
    return not text or len(text.strip()) < 5


def split_sentences(text: str) -> list:
    text = normalize_text(text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


# ── CONTEXT RULES ─────────────────────────────────────────────────────────────

NEG_PATTERNS_GENERIC = [
    r"\bno\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bno evidence of\b",
    r"\bno definite\b",
    r"\bno focal\b",
    r"\bunremarkable\b",
    r"\bclear\b",
    r"\bnormal\b",
]

UNCERTAIN_PATTERNS = [
    r"\bcould represent\b",
    r"\bmay represent\b",
    r"\bmay reflect\b",
    r"\bcould reflect\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bcannot exclude\b",
    r"\bnot excluded\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bsuggestive of\b",
    r"\bconcerning for\b",
]


def has_local_negation(sentence: str, match_start: int, window: int = 80) -> bool:
    s = sentence.lower()
    left = s[max(0, match_start - window):match_start + window]
    return any(re.search(p, left) for p in NEG_PATTERNS_GENERIC)


def has_uncertain_context(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(p, s) for p in UNCERTAIN_PATTERNS)


# ── PATTERNS ──────────────────────────────────────────────────────────────────

FINDING_PATTERNS = {
    "Cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\benlarged heart\b",
        r"\bcardiac enlargement\b",
        r"\benlarged cardiac silhouette\b",
        r"\bcardiac silhouette is enlarged\b",
        r"\bheart size is enlarged\b",
    ],
    "Edema": [
        r"\bedema\b",
        r"\bpulmonary edema\b",
        r"\binterstitial edema\b",
        r"\bvascular congestion\b",
        r"\bpulmonary vascular congestion\b",
        r"\bfluid overload\b",
        r"\bcongestion\b",
    ],
    "Consolidation": [
        r"\bconsolidation\b",
        r"\bconsolidative\b",
        r"\bairspace opacity\b",
        r"\bairspace opacities\b",
        r"\bairspace disease\b",
        r"\blobar opacity\b",
        r"\bfocal opacity\b",
    ],
    "Atelectasis": [
        r"\batelectasis\b",
        r"\batelectatic\b",
        r"\bsubsegmental atelectasis\b",
        r"\blinear atelectasis\b",
        r"\bdiscoid atelectasis\b",
        r"\bplate[- ]like atelectasis\b",
    ],
    "Pleural Effusion": [
        r"\bpleural effusion\b",
        r"\bpleural effusions\b",
        r"\bpleural fluid\b",
        r"\bcostophrenic blunting\b",
        r"\bblunting of the costophrenic\b",
        r"\bbilateral effusions\b",
        r"\bsmall effusion\b",
    ],
    "Pneumothorax": [
        r"\bpneumothorax\b",
        r"\bpneumothoraces\b",
        r"\bptx\b",
        r"\bair in the pleural space\b",
    ],
    "Lung Opacity": [
        r"\bopacity\b",
        r"\bopacities\b",
        r"\blung opacity\b",
        r"\bpulmonary opacity\b",
        r"\bpatchy opacity\b",
        r"\bpatchy opacities\b",
        r"\bhazy opacity\b",
        r"\bhazy opacities\b",
        r"\bhaziness\b",
        r"\binfiltrate\b",
        r"\binfiltrates\b",
        r"\binfiltration\b",
        r"\bground[- ]glass opacity\b",
        r"\bground[- ]glass opacities\b",
    ],
}


# ── PARSING ───────────────────────────────────────────────────────────────────

def extract_finding(text: str, finding: str) -> int:
    """
    Returns:
      1  = present
      0  = absent
     -1  = uncertain / not mentioned

    Strategy:
    - sentence-level positive pattern matching
    - local negation check around match
    - uncertain hedging => -1
    """
    text = normalize_text(text)
    if is_empty(text):
        return -1

    pats = FINDING_PATTERNS[finding]
    sentences = split_sentences(text)

    found_uncertain = False

    for sentence in sentences:
        s = sentence.lower()

        for pat in pats:
            for m in re.finditer(pat, s):
                if has_local_negation(s, m.start()):
                    return 0
                if has_uncertain_context(s):
                    found_uncertain = True
                    continue
                return 1

    if found_uncertain:
        return -1
    return -1


def extract_all_findings(text: str) -> dict:
    return {f: extract_finding(text, f) for f in FINDINGS}


# ── H SCORE (entropy) ─────────────────────────────────────────────────────────

def compute_entropy(values: list) -> float:
    """
    Shannon entropy of predictions, ignoring uncertain (-1). [0,1]
    """
    definite = [v for v in values if v != -1]
    if not definite:
        return 1.0
    counts = Counter(definite)
    total = len(definite)
    probs = [c / total for c in counts.values()]
    raw = -sum(p * np.log2(p + 1e-10) for p in probs)
    return min(raw, 1.0)


def compute_h_score(samples: list) -> float:
    """
    H(x) = mean per-finding entropy across N samples. [0,1]
    """
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        txt = sample_to_text(s)
        preds = extract_all_findings(txt)
        for f in FINDINGS:
            finding_preds[f].append(preds[f])

    entropies = [compute_entropy(finding_preds[f]) for f in FINDINGS]
    return float(np.mean(entropies))


# ── D SCORE (pairwise disagreement) ───────────────────────────────────────────

def compute_d_score(samples: list) -> float:
    """
    D(x) = mean pairwise disagreement across all sample pairs.

    For each pair (i,j) and each finding:
      - disagree if one says 1 and the other says 0
      - if either is uncertain (-1), count partial disagreement 0.5
    """
    valid_samples = [sample_to_text(s) for s in samples if not is_empty(sample_to_text(s))]
    if len(valid_samples) < 2:
        return 1.0

    all_preds = [extract_all_findings(s) for s in valid_samples]

    pair_disagreements = []
    for pa, pb in combinations(all_preds, 2):
        finding_disag = []
        for f in FINDINGS:
            a, b = pa[f], pb[f]
            if a == -1 or b == -1:
                finding_disag.append(0.5)
            else:
                finding_disag.append(float(a != b))
        pair_disagreements.append(np.mean(finding_disag))

    return float(np.mean(pair_disagreements))


# ── RISK (requires GT) ────────────────────────────────────────────────────────

def get_mode_prediction(samples: list) -> dict:
    """
    Mode prediction across valid samples for each finding.
    """
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        txt = sample_to_text(s)
        if not is_empty(txt):
            preds = extract_all_findings(txt)
            for f in FINDINGS:
                finding_preds[f].append(preds[f])

    mode = {}
    for f in FINDINGS:
        vals = finding_preds[f]
        definite = [v for v in vals if v != -1]
        mode[f] = Counter(definite).most_common(1)[0][0] if definite else -1
    return mode


def get_gt_labels(ds_row: dict) -> dict:
    """
    Extract GT binary labels from CheXpert dataset row.
    """
    col_map = {
        "Cardiomegaly":     "Cardiomegaly",
        "Edema":            "Edema",
        "Consolidation":    "Consolidation",
        "Atelectasis":      "Atelectasis",
        "Pleural Effusion": "Pleural Effusion",
        "Pneumothorax":     "Pneumothorax",
        "Lung Opacity":     "Lung Opacity",
    }
    gt = {}
    for col, finding in col_map.items():
        val = ds_row.get(col, 0)
        gt[finding] = 1 if val == LABEL_PRESENT else 0
    return gt


def compute_risk(mode_preds: dict, gt_labels: dict) -> float:
    """
    Risk = 1 - macro_F1 over 7 findings.
    """
    tp = fp = fn = 0
    for f in FINDINGS:
        pred = 1 if mode_preds[f] == 1 else 0
        gt   = gt_labels[f]

        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1

    if tp + fp + fn == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(1.0 - f1)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cal", "test"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="FDV = alpha*H + (1-alpha)*D  (default 0.5)")
    ap.add_argument("--min_samples", type=int, default=2,
                    help="Min non-empty samples required; else mark as uncertain")
    args = ap.parse_args()

    cfg = PATHS[args.split]
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)

    print(f"[step1_maira2] split={args.split}  alpha={args.alpha}")
    print(f"[step1_maira2] Loading HF dataset from {cfg['ds']} ...")
    ds = load_from_disk(cfg["ds"])

    print("[step1_maira2] Building path→idx lookup ...")
    path_to_idx = {ds[i]["Path"]: i for i in range(len(ds))}
    print(f"[step1_maira2] {len(path_to_idx)} paths indexed")

    print(f"[step1_maira2] Loading JSONL from {cfg['jsonl']} ...")
    records = []
    with open(cfg["jsonl"]) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[step1_maira2] {len(records)} records loaded")

    results = []
    n_path_miss = 0
    n_low_samples = 0

    for rec in tqdm(records, desc=f"Computing FDV [{args.split}]"):
        path       = rec.get("Path", "")
        corruption = rec["corruption"]
        severity   = rec["severity"]
        idx        = rec["idx"]
        samples    = rec.get("samples", [])

        valid = [sample_to_text(s) for s in samples if not is_empty(sample_to_text(s))]
        if len(valid) < args.min_samples:
            n_low_samples += 1

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

    df["FDV"] = args.alpha * df["H_norm"] + (1 - args.alpha) * df["D"]

    df.to_csv(cfg["out"], index=False)
    print(f"\n[step1_maira2] Saved {len(df)} rows → {cfg['out']}")
    print(f"[step1_maira2] Path misses  : {n_path_miss}")
    print(f"[step1_maira2] Low-sample rows (<{args.min_samples} valid): {n_low_samples}")

    print(f"\n{'='*50}")
    print(f"FDV Summary [{args.split}]  (MAIRA-2)")
    print(f"{'='*50}")
    print(f"  n_valid mean : {df['n_valid'].mean():.1f} / {len(df['n_valid'])} samples")
    print(f"  H            : mean={df['H'].mean():.3f}  std={df['H'].std():.3f}")
    print(f"  D            : mean={df['D'].mean():.3f}  std={df['D'].std():.3f}")
    print(f"  FDV          : mean={df['FDV'].mean():.3f}  std={df['FDV'].std():.3f}")
    print(f"  Risk         : mean={df['risk'].mean():.3f}  std={df['risk'].std():.3f}")

    print(f"\nPer-group mean risk:")
    print(df.groupby(["corruption", "severity"])["risk"]
            .mean().reset_index().to_string(index=False))

    rho, pval = spearmanr(df["FDV"], df["risk"])
    print(f"\nSpearman rho(FDV, risk) = {rho:.3f}   p = {pval:.2e}")

    rho_h, _ = spearmanr(df["H"], df["risk"])
    rho_d, _ = spearmanr(df["D"], df["risk"])
    print(f"Spearman rho(H,   risk) = {rho_h:.3f}")
    print(f"Spearman rho(D,   risk) = {rho_d:.3f}")

    empty_rate = n_low_samples / len(df) * 100
    print(f"\nEmpty/low-sample rate: {empty_rate:.1f}%")
    if empty_rate > 20:
        print("  ⚠  High empty rate — check MAIRA-2 output quality")


if __name__ == "__main__":
    main()