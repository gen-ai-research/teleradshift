#!/usr/bin/env python3
"""
step1_fdv_v2.py — FDV_v2 with VLED (embedding-based disagreement)

Replaces binary keyword-parse D with Visual-Linguistic Embedding Divergence
(VLED): mean pairwise cosine distance between BioViL-T / CXR-BERT embeddings
of the N raw text samples per image.

VLED is backbone-agnostic:
  - No keyword parsing → works on any VLM output format
  - Measures semantic disagreement in clinical meaning space
  - Captures pathology-level inconsistency even when surface tokens differ
  - Works on Maira-2, RaDialog, LLaVA, CheXagent equally

Outputs BOTH original D_kw (keyword) AND VLED so you can compare directly.
FDV    = alpha * H_norm + (1-alpha) * D_kw   (original)
FDV_v2 = alpha * H_norm + (1-alpha) * VLED   (new)

Usage:
  # Maira-2
  python step1_fdv_v2.py --split cal --model_tag maira2

  # CheXagent (compare D_kw vs VLED)
  python step1_fdv_v2.py --split cal --model_tag chexagent \\
      --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/chexagent_cal_6k_N6.jsonl

  # RaDialog (after running radialog_runner.py)
  python step1_fdv_v2.py --split cal --model_tag radialog \\
      --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/radialog_cal_6k_N6.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.stats import spearmanr
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = "/scratch/FOLDER_NAME1/telerad_shift"

JSONL_TEMPLATE = f"{BASE}/outputs/{{model_tag}}_{{split}}_6k_N6.jsonl"
DS_TEMPLATE    = f"{BASE}/{{split}}_5k"
OUT_TEMPLATE   = f"{BASE}/outputs/fdv_v2_{{model_tag}}_{{split}}.csv"

# ── FINDINGS (identical to step1) ─────────────────────────────────────────────
FINDINGS = [
    "Cardiomegaly", "Edema", "Consolidation", "Atelectasis",
    "Pleural Effusion", "Pneumothorax", "Lung Opacity",
]
LABEL_PRESENT = 3

# ── EMBEDDING MODELS (priority order) ─────────────────────────────────────────
EMBED_MODELS = {
    "biovilt": "microsoft/BioViL-T",
    "cxrbert": "microsoft/BiomedVLP-CXR-BERT-specialized",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Clinical text encoder (BioViL-T / CXR-BERT)
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalTextEncoder:
    """
    Frozen, eval-mode encoder for mean-pooled clinical text embeddings.
    Used only for VLED — never trained, never updated.
    """

    def __init__(self, model_id: str, device: torch.device, cache_dir: str = None):
        print(f"[Encoder] Loading {model_id} on {device} ...", flush=True)
        kw = dict(cache_dir=cache_dir) if cache_dir else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)
        self.model     = AutoModel.from_pretrained(model_id, **kw).eval().to(device)
        self.device    = device
        print(f"[Encoder] Ready.", flush=True)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Returns L2-normalised embeddings (N, d).
        Empty strings → zero vector (treated as missing in VLED).
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            # Prevent tokenizer crash on empty strings
            batch_safe = [t if t.strip() else "[PAD]" for t in batch]
            enc = self.tokenizer(
                batch_safe, return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            ).to(self.device)

            out  = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            # Zero out embeddings for originally-empty inputs
            for j, t in enumerate(batch):
                if not t.strip():
                    emb[j] = 0.0

            # L2 normalise
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            all_embs.append(emb.cpu().float().numpy())

        return np.concatenate(all_embs, axis=0)


def load_encoder(choice: str, device: torch.device, cache_dir: str = None) -> ClinicalTextEncoder:
    if choice == "auto":
        for key, mid in EMBED_MODELS.items():
            try:
                return ClinicalTextEncoder(mid, device, cache_dir)
            except Exception as e:
                print(f"[Encoder] {key} failed ({e}), trying next ...", flush=True)
        raise RuntimeError("No embedding model loaded. Check HF access / transformers install.")
    return ClinicalTextEncoder(EMBED_MODELS[choice], device, cache_dir)


def compute_vled(texts: List[str], encoder: ClinicalTextEncoder, batch_size: int = 64) -> float:
    """
    VLED(x) = mean pairwise cosine distance across N sample embeddings.

    cos_distance(a,b) = 1 - a·b  (vectors are L2-normalised)
    Pairs where either embedding is zero (empty sample) are excluded.
    Returns 1.0 (max uncertainty) if fewer than 2 non-empty samples.
    """
    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return 1.0

    embs  = encoder.encode(texts, batch_size=batch_size)       # (N, d)
    norms = np.linalg.norm(embs, axis=1)                       # (N,)
    sim   = np.clip(embs @ embs.T, -1.0, 1.0)                  # (N, N)
    dist  = np.clip(1.0 - sim, 0.0, 1.0)

    N = len(texts)
    valid_dists = [
        dist[i, j]
        for i in range(N) for j in range(i + 1, N)
        if norms[i] > 1e-6 and norms[j] > 1e-6
    ]
    return float(np.mean(valid_dists)) if valid_dists else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Text processing (verbatim from step1 — do NOT change)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_to_text(sample) -> str:
    if sample is None:
        return ""
    if isinstance(sample, str):
        return sample.strip()
    if isinstance(sample, dict):
        for k in ["text", "report", "findings", "output"]:
            if k in sample and isinstance(sample[k], str):
                return sample[k].strip()
        return "\n".join(sample_to_text(v) for v in sample.values()).strip()
    if isinstance(sample, list):
        return "\n".join(sample_to_text(x) for x in sample).strip()
    return str(sample).strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def is_empty(text: str) -> bool:
    return not normalize_text(text) or len(normalize_text(text)) < 5


def split_sentences(text: str) -> list:
    text = normalize_text(text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


NEG_PATTERNS = [
    r"\bno\b", r"\bwithout\b", r"\babsent\b", r"\bnegative for\b",
    r"\bfree of\b", r"\bno evidence of\b", r"\bno definite\b",
    r"\bno focal\b", r"\bunremarkable\b", r"\bclear\b", r"\bnormal\b",
]
UNCERTAIN_PATTERNS = [
    r"\bcould represent\b", r"\bmay represent\b", r"\bmay reflect\b",
    r"\bcould reflect\b", r"\bpossibly\b", r"\bpossible\b",
    r"\bcannot exclude\b", r"\bnot excluded\b", r"\blikely\b",
    r"\bprobably\b", r"\bsuggestive of\b", r"\bconcerning for\b",
]
FINDING_PATTERNS = {
    "Cardiomegaly":     [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bcardiac enlargement\b",
                         r"\benlarged cardiac silhouette\b", r"\bheart size is enlarged\b"],
    "Edema":            [r"\bedema\b", r"\bpulmonary edema\b", r"\binterstitial edema\b",
                         r"\bvascular congestion\b", r"\bfluid overload\b", r"\bcongestion\b"],
    "Consolidation":    [r"\bconsolidation\b", r"\bconsolidative\b", r"\bairspace opacity\b",
                         r"\bairspace disease\b", r"\blobar opacity\b", r"\bfocal opacity\b"],
    "Atelectasis":      [r"\batelectasis\b", r"\batelectatic\b", r"\bsubsegmental atelectasis\b",
                         r"\blinear atelectasis\b", r"\bdiscoid atelectasis\b"],
    "Pleural Effusion": [r"\bpleural effusion\b", r"\bpleural fluid\b", r"\bcostophrenic blunting\b",
                         r"\bbilateral effusions\b", r"\bsmall effusion\b"],
    "Pneumothorax":     [r"\bpneumothorax\b", r"\bpneumothoraces\b", r"\bptx\b"],
    "Lung Opacity":     [r"\bopacity\b", r"\bopacities\b", r"\bhaziness\b", r"\binfiltrate\b",
                         r"\binfiltrates\b", r"\bground[- ]glass opacity\b"],
}


def has_local_negation(s: str, start: int, window: int = 80) -> bool:
    ctx = s[max(0, start - window): start + window]
    return any(re.search(p, ctx) for p in NEG_PATTERNS)


def has_uncertain(s: str) -> bool:
    return any(re.search(p, s.lower()) for p in UNCERTAIN_PATTERNS)


def extract_finding(text: str, finding: str) -> int:
    text = normalize_text(text)
    if is_empty(text):
        return -1
    found_uncertain = False
    for sentence in split_sentences(text):
        s = sentence.lower()
        for pat in FINDING_PATTERNS[finding]:
            for m in re.finditer(pat, s):
                if has_local_negation(s, m.start()):
                    return 0
                if has_uncertain(s):
                    found_uncertain = True
                    continue
                return 1
    return -1 if found_uncertain else -1


def extract_all(text: str) -> dict:
    return {f: extract_finding(text, f) for f in FINDINGS}


def compute_h_score(samples: list) -> float:
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        preds = extract_all(sample_to_text(s))
        for f in FINDINGS:
            finding_preds[f].append(preds[f])
    entropies = []
    for f in FINDINGS:
        definite = [v for v in finding_preds[f] if v != -1]
        if not definite:
            entropies.append(1.0)
            continue
        counts = Counter(definite)
        total  = len(definite)
        probs  = [c / total for c in counts.values()]
        h = -sum(p * np.log2(p + 1e-10) for p in probs)
        entropies.append(min(h, 1.0))
    return float(np.mean(entropies))


def compute_d_kw(samples: list) -> float:
    """Original keyword-parse D — kept unchanged for comparison."""
    valid = [sample_to_text(s) for s in samples if not is_empty(sample_to_text(s))]
    if len(valid) < 2:
        return 1.0
    all_preds = [extract_all(s) for s in valid]
    pair_disag = []
    for pa, pb in combinations(all_preds, 2):
        fd = []
        for f in FINDINGS:
            a, b = pa[f], pb[f]
            fd.append(0.5 if (a == -1 or b == -1) else float(a != b))
        pair_disag.append(np.mean(fd))
    return float(np.mean(pair_disag))


def get_mode_prediction(samples: list) -> dict:
    finding_preds = {f: [] for f in FINDINGS}
    for s in samples:
        txt = sample_to_text(s)
        if not is_empty(txt):
            for f in FINDINGS:
                finding_preds[f].append(extract_finding(txt, f))
    mode = {}
    for f in FINDINGS:
        definite = [v for v in finding_preds[f] if v != -1]
        mode[f] = Counter(definite).most_common(1)[0][0] if definite else -1
    return mode


def get_gt_labels(ds_row: dict) -> dict:
    return {f: (1 if ds_row.get(f, 0) == LABEL_PRESENT else 0) for f in FINDINGS}


def compute_risk(mode_preds: dict, gt_labels: dict) -> float:
    tp = fp = fn = 0
    for f in FINDINGS:
        pred = 1 if mode_preds[f] == 1 else 0
        gt   = gt_labels[f]
        if pred == 1 and gt == 1:   tp += 1
        elif pred == 1 and gt == 0: fp += 1
        elif pred == 0 and gt == 1: fn += 1
    if tp + fp + fn == 0:
        return 0.0
    p  = tp / (tp + fp + 1e-10)
    r  = tp / (tp + fn + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    return float(1.0 - f1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",     choices=["cal", "test"], required=True)
    ap.add_argument("--model_tag", required=True,
                    help="maira2 / chexagent / radialog / llava16")
    ap.add_argument("--jsonl",    default=None, help="Override JSONL path")
    ap.add_argument("--ds_path",  default=None, help="Override HF dataset path")
    ap.add_argument("--out_csv",  default=None, help="Override output CSV path")
    ap.add_argument("--alpha",       type=float, default=0.5)
    ap.add_argument("--min_samples", type=int,   default=2)
    ap.add_argument("--embed_model",  default="auto",
                    choices=["auto", "biovilt", "cxrbert"])
    ap.add_argument("--embed_batch",  type=int,   default=64)
    ap.add_argument("--embed_device", default="auto",
                    help="auto / cpu / cuda / cuda:0")
    ap.add_argument("--embed_cache",  default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    jsonl_path = args.jsonl   or JSONL_TEMPLATE.format(model_tag=args.model_tag, split=args.split)
    ds_path    = args.ds_path or DS_TEMPLATE.format(split=args.split)
    out_csv    = args.out_csv or OUT_TEMPLATE.format(model_tag=args.model_tag, split=args.split)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    tag = f"[step1_fdv_v2/{args.model_tag}]"
    print(f"{tag} split={args.split}  alpha={args.alpha}  embed={args.embed_model}")

    # ── Embedding model ────────────────────────────────────────────────────────
    if args.embed_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.embed_device)
    encoder = load_encoder(args.embed_model, device, args.embed_cache)

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"{tag} Loading HF dataset from {ds_path} ...", flush=True)
    ds = load_from_disk(ds_path)
    path_to_idx = {ds[i]["Path"]: i for i in range(len(ds))}
    print(f"{tag} {len(path_to_idx)} paths indexed", flush=True)

    # ── JSONL ──────────────────────────────────────────────────────────────────
    print(f"{tag} Loading JSONL from {jsonl_path} ...", flush=True)
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"{tag} {len(records)} records loaded", flush=True)

    # ── Main loop ──────────────────────────────────────────────────────────────
    results       = []
    n_path_miss   = 0
    n_low_samples = 0

    for rec in tqdm(records, desc=f"Computing FDV_v2 [{args.model_tag}/{args.split}]"):
        path       = rec.get("Path", "")
        corruption = rec["corruption"]
        severity   = rec["severity"]
        idx        = rec["idx"]
        samples    = rec.get("samples", [])

        texts = [sample_to_text(s) for s in samples]
        valid = [t for t in texts if not is_empty(t)]
        if len(valid) < args.min_samples:
            n_low_samples += 1

        ds_idx = path_to_idx.get(path)
        if ds_idx is None:
            ds_idx = idx
            n_path_miss += 1

        ds_row     = ds[ds_idx]
        gt_labels  = get_gt_labels(ds_row)
        mode_preds = get_mode_prediction(samples)
        risk       = compute_risk(mode_preds, gt_labels)

        h_score = compute_h_score(samples)
        d_kw    = compute_d_kw(samples)
        vled    = compute_vled(texts, encoder, batch_size=args.embed_batch)

        results.append({
            "idx":        idx,
            "path":       path,
            "corruption": corruption,
            "severity":   severity,
            "n_valid":    len(valid),
            "H":          h_score,
            "D_kw":       d_kw,
            "VLED":       vled,
            "risk":       risk,
        })

    # ── DataFrame + scores ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    h_min, h_max = df["H"].min(), df["H"].max()
    df["H_norm"] = (df["H"] - h_min) / (h_max - h_min) if h_max > h_min else 0.0
    df["FDV"]    = args.alpha * df["H_norm"] + (1 - args.alpha) * df["D_kw"]
    df["FDV_v2"] = args.alpha * df["H_norm"] + (1 - args.alpha) * df["VLED"]

    df.to_csv(out_csv, index=False)

    # ── Print summary (matches step1 format) ───────────────────────────────────
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"FDV_v2 Summary [{args.split}]  ({args.model_tag})")
    print(sep)
    print(f"  n_records    : {len(df)}")
    print(f"  n_valid mean : {df['n_valid'].mean():.1f} / {len(df)} samples")
    print(f"  H            : mean={df['H'].mean():.3f}  std={df['H'].std():.3f}")
    print(f"  D_kw         : mean={df['D_kw'].mean():.3f}  std={df['D_kw'].std():.3f}")
    print(f"  VLED         : mean={df['VLED'].mean():.3f}  std={df['VLED'].std():.3f}")
    print(f"  FDV          : mean={df['FDV'].mean():.3f}  std={df['FDV'].std():.3f}")
    print(f"  FDV_v2       : mean={df['FDV_v2'].mean():.3f}  std={df['FDV_v2'].std():.3f}")
    print(f"  Risk         : mean={df['risk'].mean():.3f}  std={df['risk'].std():.3f}")

    print(f"\nPer-group mean risk:")
    print(df.groupby(["corruption", "severity"])["risk"]
            .mean().reset_index().to_string(index=False))

    print(f"\n{'Signal':<10}  {'rho':>6}  {'p-value':>12}")
    print("-" * 34)
    for col in ["H", "D_kw", "VLED", "FDV", "FDV_v2"]:
        rho, pval = spearmanr(df[col], df["risk"])
        marker = "  ← NEW" if col in ("VLED", "FDV_v2") else ""
        print(f"  {col:<8}  {rho:>6.3f}  {pval:>12.2e}{marker}")

    print(f"\nPath misses     : {n_path_miss}")
    print(f"Low-sample rows : {n_low_samples} ({n_low_samples/len(df)*100:.1f}%)")

    # ── Operability check ──────────────────────────────────────────────────────
    vled_std = df["VLED"].std()
    rho_vled, _ = spearmanr(df["VLED"], df["risk"])
    print(f"\n{sep}")
    print(f"VLED OPERABILITY CHECK  ({args.model_tag})")
    print(sep)
    print(f"  Var(VLED)    = {vled_std:.4f}  "
          f"{'✓ signal exists' if vled_std > 0.01 else '✗ too uniform — embedding D also fails'}")
    print(f"  rho(VLED,risk)= {rho_vled:.3f}  "
          f"{'✓ discriminative' if rho_vled > 0.35 else '✗ weak — VLM not visually grounded'}")
    operable = vled_std > 0.01 and rho_vled > 0.35
    print(f"  FDV_v2 operable: {'YES ✓' if operable else 'NO ✗'}")
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()


"""
python step1_fdv_v2.py \
    --split cal \
    --model_tag maira2 \
    --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/maira2_cal_6k_N6.jsonl \
    --embed_model auto   # tries BioViL-T first, falls back to CXR-BERT


python step1_fdv_v2.py \
    --split cal \
    --model_tag llavarerun \
    --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/llava_rerun_cal.jsonl \
    --embed_model auto # tries BioViL-T first, falls back to CXR-BERT

python step1_fdv_v2.py \
    --split cal \
    --model_tag chexagent \
    --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/cal_6k_N6.jsonl \
    --embed_model auto # tries BioViL-T first, falls back to CXR-BERT

python step1_fdv_v2.py \
    --split cal \
    --model_tag qwen2vl \
    --jsonl /scratch/FOLDER_NAME1/telerad_shift/outputs/qwen2vl_cal.jsonl.jsonl \
    --embed_model auto # tries BioViL-T first, falls back to CXR-BERT

# Check what JSONL the original step1 used vs what you're passing now
head -1 /scratch/FOLDER_NAME1/telerad_shift/outputs/cal_6k_N6.jsonl | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print('idx:', r['idx'])
print('Path:', r.get('Path','MISSING'))
print('corruption:', r['corruption'])
print('n_samples:', len(r.get('samples',[])))
print('sample[0][:100]:', str(r.get('samples',[''])[0])[:100])
"

"""