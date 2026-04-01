#!/usr/bin/env python3
"""
step1_compute_fdv_llava.py — Compute FDV scores from LLaVA-Med JSONL output

LLaVA-Med generates descriptive/educational text, numbered lists, and
sometimes truncated mid-sentence outputs. This script handles all of these.

FDV(x) = alpha * H_norm(x) + (1-alpha) * D(x)

  H(x) = average per-finding entropy across N stochastic samples
         (how often does each finding flip across samples?)

  D(x) = mean pairwise disagreement across sample pairs
         (how many pairs of samples contradict each other?)
         NOTE: pure sample-to-sample, does NOT require GT labels

Usage:
  python step1_compute_fdv_llava.py --split cal
  python step1_compute_fdv_llava.py --split test
  python step1_compute_fdv_llava.py --split cal --alpha 0.5 --min_samples 2
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
        "jsonl":    f"{BASE}/outputs/llava_rerun_cal.jsonl",
        "ds":       f"{BASE}/cal_5k",
        "manifest": f"{BASE}/manifests/cal_manifest_sub6k.csv",
        "out":      f"{BASE}/outputs/fdv_llava_cal.csv",
    },
    "test": {
        "jsonl":    f"{BASE}/outputs/llava_rerun_test.jsonl",
        "ds":       f"{BASE}/test_5k",
        "manifest": f"{BASE}/manifests/test_manifest_sub5k.csv",
        "out":      f"{BASE}/outputs/fdv_llava_test.csv",
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

# ── KEYWORD PATTERNS (tuned for LLaVA-Med descriptive output) ─────────────────
# LLaVA-Med uses:
#   - numbered lists: "3. Pleural effusion: An abnormal accumulation..."
#   - descriptive sentences: "also shows a right-sided pleural effusion"
#   - educational phrasing: "indicates that there is an area of the lung where..."
#   - negations: "no evidence of", "no signs of", "without"
#
# Strategy: match the MENTION of a finding as positive, unless clearly negated.
# Truncated samples that mention a finding mid-sentence still count.

FINDING_PATTERNS = {
    "Cardiomegaly": (
        [
            r"cardiomegal",
            r"omegaly",                              # truncated: "cardiomegaly"
            r"enlarged\s+(cardiac|heart)",
            r"an\s+enlarged\s+heart",
            r"cardiac\s+enlargement",
            r"increased\s+cardiac\s+(size|shadow)",
            r"enlarged\s+cardiac\s+silhouette",
            r"heart\s+is\s+enlarged",
            r"cardiac\s+silhouette\s+is\s+enlarged",
            r"enlargement\s+of\s+the\s+heart",
            r"indicates\s+an\s+enlarged\s+heart",
            r"sign\s+of.*heart\s+condition",
            r"heart.related\s+abnormalit",
        ],
        [
            r"no\s+cardiomegal",
            r"no\s+cardiac\s+enlargement",
            r"normal\s+(cardiac|heart)\s+(size|silhouette)",
            r"heart\s+size\s+is\s+normal",
            r"cardiac\s+silhouette\s+is\s+normal",
            r"no\s+enlarged\s+heart",
        ],
    ),
    "Edema": (
        [
            r"\bedema\b",
            r"pulmonary\s+edema",
            r"interstitial\s+edema",
            r"pulmonary\s+congestion",
            r"vascular\s+congestion",
            r"fluid\s+overload",
            r"pulmonary\s+vascular\s+congestion",
            r"abnormal\s+accumulation\s+of\s+fluid\s+in\s+the\s+lung",
            r"fluid\s+accumulation.*lung",
        ],
        [
            r"no\s+(pulmonary\s+)?edema",
            r"no\s+vascular\s+congestion",
            r"no\s+pulmonary\s+congestion",
            r"lungs\s+are\s+clear",
            r"clear\s+lungs",
            r"no\s+fluid\s+overload",
        ],
    ),
    "Consolidation": (
        [
            r"consolidat",
            r"airspace\s+(opacity|disease)",
            r"air\s+spaces\s+are\s+filled",
            r"area\s+of\s+the\s+lung\s+where\s+the\s+air\s+spaces",
            r"filled\s+with\s+(fluid|pus|material)",
            r"lobar\s+(opacity|pneumonia)",
            r"segmental\s+opacity",
            r"lung\s+tissue\s+due\s+to\s+the\s+presence\s+of\s+fluid",  # truncated
            r"related\s+to\s+pneumonia",                                  # truncated suffix
            r"infection.*inflammation.*lung",
            r"pneumonia.*bronchitis",
        ],
        [
            r"no\s+consolidat",
            r"without\s+consolidat",
            r"no\s+airspace\s+(opacity|disease)",
            r"lungs\s+are\s+clear",
            r"clear\s+lung",
            r"no\s+focal\s+opacity",
        ],
    ),
    "Atelectasis": (
        [
            r"atelectasis",
            r"atelectatic",
            r"collapse\s+(of\s+)?(the\s+)?(lung|lobe|segment)",
            r"lung\s+collapse",
            r"subsegmental\s+(atelectasis|collapse)",
            r"plate.like\s+atelectasis",
            r"linear\s+atelectasis",
            r"discoid\s+atelectasis",
            r"partial\s+collapse",
            r"collapsed\s+lung",                     # truncated: "collapsed lung, in this image"
        ],
        [
            r"no\s+atelectasis",
            r"no\s+collapse",
            r"fully\s+expanded",
            r"well.?expanded",
            r"lungs\s+are\s+(fully|well)\s+expanded",
        ],
    ),
    "Pleural Effusion": (
        [
            r"pleural\s+effusion",
            r"pleural\s+fluid",
            r"fluid\s+in\s+the\s+pleural",
            r"abnormal\s+accumulation\s+of\s+fluid\s+in\s+the\s+pleural",
            r"accumulation\s+of\s+fluid\s+in\s+the\s+pleural",          # truncated prefix
            r"ulation.*pleural\s+space",                                  # truncated: "accumulation in pleural"
            r"ulation\s+of\s+fluid\s+in\s+the\s+pleural",               # truncated
            r"fluid.*pleural\s+space\s+surrounding",
            r"blunting\s+of\s+the\s+costophrenic",
            r"costophrenic\s+(angle\s+)?blunting",
            r"hydrothorax",
            r"(right|left|bilateral).sided\s+pleural\s+effusion",
            r"(right|left|bilateral)\s+pleural\s+effusion",
            r"fluid\s+accumulation\s+in\s+the\s+(chest|pleural)",
            r"line\s+the\s+lungs\s+and\s+the\s+chest",                  # "pleural space that lines..."
            r"lungs\s+and\s+the\s+chest\s+cavity",                      # truncated pleural description
            r"between\s+the\s+lung\s+and\s+the\s+chest\s+wall",
            r"area\s+between\s+the\s+lung\s+and\s+the\s+chest",
        ],
        [
            r"no\s+pleural\s+effusion",
            r"no\s+effusion",
            r"pleural\s+space\s+is\s+clear",
            r"no\s+pleural\s+fluid",
            r"costophrenic\s+angles\s+are\s+sharp",
        ],
    ),
    "Pneumothorax": (
        [
            r"pneumothorax",
            r"pneumothorac",
            r"neumothorax",                           # truncated: "pneumothorax"
            r"air\s+in\s+the\s+pleural\s+space",
            r"collapsed\s+lung\s+due\s+to\s+air",
            r"presence\s+of\s+air\s+in\s+the\s+pleural",
            r"is\s+a\s+collapsed\s+lung",             # truncated: "which is a collapsed lung"
        ],
        [
            r"no\s+pneumothorax",
            r"without\s+pneumothorax",
            r"no\s+ptx",
            r"no\s+air\s+in\s+the\s+pleural",
        ],
    ),
    "Lung Opacity": (
        [
            r"lung\s+opacity",
            r"pulmonary\s+opacity",
            r"opacity\s+(in|of|at)\s+the\s+lung",
            r"opacit",
            r"haziness",
            r"infiltrat",
            r"ground.glass\s+opacity",
            r"patchy\s+opacity",
            r"increased\s+opacity",
            r"areas?\s+of\s+opacity",
            r"cause\s+of\s+the\s+opacity",            # truncated suffix seen in data
            r"cause\s+and\s+significance\s+of\s+these\s+opacities",
            r"fluid\s+or\s+other\s+material",         # "lung tissue due to fluid or material"
            r"fluid.*inflammation.*lung",
            r"lung\s+tissue\s+due\s+to",
        ],
        [
            r"no\s+(lung\s+)?opacity",
            r"no\s+opacit",
            r"no\s+infiltrat",
            r"lungs\s+are\s+clear",
            r"clear\s+lung",
            r"no\s+haziness",
        ],
    ),
}


# ── PARSING ───────────────────────────────────────────────────────────────────

def is_empty(text: str) -> bool:
    """True if sample is empty or whitespace only. Even short fragments
    may contain truncated finding keywords so we keep them."""
    return not text or len(text.strip()) < 5



# Phrases that indicate LLaVA is listing differentials / explaining concepts
# rather than asserting a confirmed finding in THIS patient
DIFFERENTIAL_PATTERNS = [
    r"could\s+be\s+(due\s+to|caused\s+by)",
    r"may\s+(indicate|suggest|be\s+due\s+to|be\s+caused)",
    r"might\s+be",
    r"such\s+as\s+\w",           # "such as pneumonia or..."
    r"or\s+other\s+lung",        # "infection, inflammation, or other lung"
    r"or\s+other\s+(condition|disease|infection|cause)",
    r"like\s+pneumonia",
    r"(including|such\s+as)\s+(pneumonia|bronchitis|infection)",
    r"various\s+(condition|cause|reason|disease)",
    r"underlying\s+(condition|cause|issue|medical)",
    r"further\s+evaluation",
    r"requires?\s+further",
    r"consult.*healthcare",
    r"healthcare\s+professional",
    r"determine\s+the\s+cause",
    r"appropriate\s+treatment",
]

def is_differential_context(sentence: str) -> bool:
    """True if sentence is listing possibilities rather than confirming a finding."""
    t = sentence.lower()
    return any(re.search(p, t) for p in DIFFERENTIAL_PATTERNS)

def split_sentences(text: str) -> list:
    """Split text into sentences for context-aware matching."""
    # Split on period, but keep short fragments together
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def extract_finding(text: str, finding: str) -> int:
    """
    Returns 1 (present), 0 (absent), -1 (uncertain/not mentioned).
    
    Key improvement: only counts a finding as PRESENT if it appears in
    an assertive sentence, not in a differential/educational context.
    e.g. "Consolidation in the right lower lobe" → present=1
         "could be due to consolidation or infection" → uncertain=-1
    """
    if is_empty(text):
        return -1

    t = text.lower()
    pos_pats, neg_pats = FINDING_PATTERNS[finding]

    # Check negation first (always trust negations)
    neg = any(re.search(p, t) for p in neg_pats)
    if neg:
        return 0

    # For positive matches — check sentence context
    sentences = split_sentences(text)
    assertive_positive = False

    for sentence in sentences:
        s = sentence.lower()
        pos_match = any(re.search(p, s) for p in pos_pats)
        if pos_match:
            # Only count if NOT a differential/educational context
            if not is_differential_context(sentence):
                assertive_positive = True
                break

    if assertive_positive:
        return 1

    return -1  # not mentioned assertively


def extract_all_findings(text: str) -> dict:
    return {f: extract_finding(text, f) for f in FINDINGS}


# ── H SCORE (entropy) ─────────────────────────────────────────────────────────

def compute_entropy(values: list) -> float:
    """Shannon entropy of predictions, ignoring uncertain (-1). [0,1]"""
    definite = [v for v in values if v != -1]
    if not definite:
        return 1.0   # all uncertain → max entropy
    counts = Counter(definite)
    total = len(definite)
    probs = [c / total for c in counts.values()]
    raw = -sum(p * np.log2(p + 1e-10) for p in probs)
    return min(raw, 1.0)


def compute_h_score(samples: list) -> float:
    """H(x) = mean per-finding entropy across N samples. [0,1]"""
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        preds = extract_all_findings(s)
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
      - skip if either is uncertain (-1)
    Returns value in [0,1]. Higher = more contradictory = more uncertain.
    """
    valid_samples = [s for s in samples if not is_empty(s)]
    if len(valid_samples) < 2:
        return 1.0  # only 0 or 1 usable sample → max uncertainty

    # Extract predictions for all valid samples
    all_preds = [extract_all_findings(s) for s in valid_samples]

    pair_disagreements = []
    for (pa, pb) in combinations(all_preds, 2):
        finding_disag = []
        for f in FINDINGS:
            a, b = pa[f], pb[f]
            if a == -1 or b == -1:
                # Both uncertain → slight disagreement (0.5), not full
                finding_disag.append(0.5)
            else:
                finding_disag.append(float(a != b))
        pair_disagreements.append(np.mean(finding_disag))

    return float(np.mean(pair_disagreements))


# ── RISK (requires GT) ────────────────────────────────────────────────────────

def get_mode_prediction(samples: list) -> dict:
    """Mode prediction across valid samples for each finding."""
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        if not is_empty(s):
            preds = extract_all_findings(s)
            for f in FINDINGS:
                finding_preds[f].append(preds[f])

    mode = {}
    for f in FINDINGS:
        vals = finding_preds[f]
        definite = [v for v in vals if v != -1]
        mode[f] = Counter(definite).most_common(1)[0][0] if definite else -1
    return mode


def get_gt_labels(ds_row: dict) -> dict:
    """Extract GT binary labels from CheXpert dataset row."""
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
    """Risk = 1 - macro_F1 over 7 findings."""
    tp = fp = fn = 0
    for f in FINDINGS:
        pred = 1 if mode_preds[f] == 1 else 0
        gt = gt_labels[f]
        if pred == 1 and gt == 1:   tp += 1
        elif pred == 1 and gt == 0: fp += 1
        elif pred == 0 and gt == 1: fn += 1

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

    print(f"[step1_llava] split={args.split}  alpha={args.alpha}")
    print(f"[step1_llava] Loading HF dataset from {cfg['ds']} ...")
    ds = load_from_disk(cfg["ds"])

    print("[step1_llava] Building path→idx lookup ...")
    path_to_idx = {ds[i]["Path"]: i for i in range(len(ds))}
    print(f"[step1_llava] {len(path_to_idx)} paths indexed")

    print(f"[step1_llava] Loading JSONL from {cfg['jsonl']} ...")
    records = []
    with open(cfg["jsonl"]) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[step1_llava] {len(records)} records loaded")

    results = []
    n_path_miss = 0
    n_low_samples = 0

    for rec in tqdm(records, desc=f"Computing FDV [{args.split}]"):
        path       = rec.get("Path", "")
        corruption = rec["corruption"]
        severity   = rec["severity"]
        idx        = rec["idx"]
        samples    = rec.get("samples", [])

        # Count usable samples
        valid = [s for s in samples if not is_empty(s)]
        if len(valid) < args.min_samples:
            n_low_samples += 1

        # GT labels
        ds_idx = path_to_idx.get(path)
        if ds_idx is None:
            ds_idx = idx
            n_path_miss += 1
        ds_row    = ds[ds_idx]
        gt_labels = get_gt_labels(ds_row)

        # Scores
        h_score    = compute_h_score(samples)
        d_score    = compute_d_score(samples)   # pure sample-to-sample
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

    # Normalise H → [0,1] over this split
    h_min, h_max = df["H"].min(), df["H"].max()
    df["H_norm"] = (df["H"] - h_min) / (h_max - h_min) if h_max > h_min else 0.0

    # FDV
    df["FDV"] = args.alpha * df["H_norm"] + (1 - args.alpha) * df["D"]

    df.to_csv(cfg["out"], index=False)
    print(f"\n[step1_llava] Saved {len(df)} rows → {cfg['out']}")
    print(f"[step1_llava] Path misses  : {n_path_miss}")
    print(f"[step1_llava] Low-sample rows (<{args.min_samples} valid): {n_low_samples}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"FDV Summary [{args.split}]  (LLaVA-Med)")
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
    print(f"\n(CheXagent reference: rho_H=0.19  rho_D=0.61  rho_FDV=0.49)")

    # Empty rate
    empty_rate = n_low_samples / len(df) * 100
    print(f"\nEmpty/low-sample rate: {empty_rate:.1f}%  (was 6% in LLaVA run)")
    if empty_rate > 20:
        print("  ⚠  High empty rate — check LLaVA output quality")


if __name__ == "__main__":
    main()