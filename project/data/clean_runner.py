#!/usr/bin/env python3
"""
clean_runner.py — Run CheXagent inference on CLEAN (uncorrupted) images.
Skips apply_corruption entirely — passes original image directly to model.

Usage:
  CUDA_VISIBLE_DEVICES=3 python clean_runner.py

Output:
  /scratch/FOLDER_NAME1/telerad_shift/outputs/clean_500_N6.jsonl
"""

import os, sys, json, time, random, atexit, shutil, hashlib, argparse, traceback
from typing import Any, Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE        = "/scratch/FOLDER_NAME1/telerad_shift"
DS_PATH     = f"{BASE}/test_5k"          # use test_5k images (clean versions)
OUT_JSONL   = f"{BASE}/outputs/clean_500_N6.jsonl"
ERR_JSONL   = f"{BASE}/outputs/clean_500_N6.err.jsonl"
TMP_DIR     = f"{BASE}/tmp_clean"
N_IMAGES    = 500                         # number of clean images to run
N_SAMPLES   = 6                           # stochastic samples per image
SEED        = 42
SAMPLE_BATCH = 6
MODEL_ID    = "StanfordAIMI/CheXagent-2-3b-srrg-findings"
PROMPT      = (
    "Generate a chest X-ray radiology report. "
    "Output ONLY the FINDINGS section. "
    "Do not output IMPRESSION or any other text."
)
TEMPERATURE  = 0.8
TOP_P        = 0.9
MAX_NEW_TOKENS = 160
SEED_STRIDE  = 1000
DTYPE_STR    = "auto"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(s: str) -> torch.dtype:
    if s == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def load_model(model_id: str, dtype_str: str = "auto"):
    torch_dtype = resolve_dtype(dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if getattr(cfg, "pad_token_id", None) is None:
        pad = getattr(cfg, "eos_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
        cfg.pad_token_id = pad

    rope = getattr(cfg, "rope_scaling", None)
    if isinstance(rope, dict) and ("type" not in rope or "factor" not in rope):
        rtype = rope.get("type", rope.get("rope_type", "linear"))
        if rtype not in ("linear", "dynamic"): rtype = "linear"
        cfg.rope_scaling = {**rope, "type": rtype, "factor": rope.get("factor", 1.0)}

    from transformers import AutoModel, AutoProcessor
    _orig_am = AutoModel.from_pretrained.__func__

    @classmethod
    def _am_patch(cls, *a, **kw):
        kw["device_map"] = None
        kw["low_cpu_mem_usage"] = False
        try:
            if torch._C._get_default_device() == "meta":
                torch._C._set_default_device("cpu")
        except Exception:
            pass
        return _orig_am(cls, *a, **kw)

    AutoModel.from_pretrained = _am_patch

    _orig_ap = AutoProcessor.from_pretrained.__func__

    @classmethod
    def _ap_patch(cls, *a, **kw):
        try:
            return _orig_ap(cls, *a, **kw)
        except Exception as e:
            from transformers import AutoImageProcessor
            ip = AutoImageProcessor.from_pretrained(*a, **kw)
            class _S: image_processor = ip
            return _S()

    AutoProcessor.from_pretrained = _ap_patch

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=cfg, device_map="auto",
        torch_dtype=torch_dtype, trust_remote_code=True,
    )
    model.eval()

    AutoModel.from_pretrained = classmethod(_orig_am)
    AutoProcessor.from_pretrained = classmethod(_orig_ap)

    try:
        model.model.visual.to(dtype=torch_dtype)
    except AttributeError:
        pass

    gen = model.generation_config
    if getattr(gen, "pad_token_id", None) is None:
        gen.pad_token_id = cfg.pad_token_id
    if getattr(gen, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None):
        gen.eos_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    print(f"[load_model] Ready | dtype={torch_dtype} | device={device}", flush=True)
    return model, tokenizer, device


def save_pil_temp(img: Image.Image, tmp_dir: str, key: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    p = os.path.join(tmp_dir, hashlib.md5(key.encode()).hexdigest() + ".jpg")
    if not os.path.exists(p):
        img.save(p, format="JPEG", quality=95)
    return p


@torch.inference_mode()
def generate_n_batched(
    model, tokenizer, device,
    image_path: str,
    prompt: str,
    base_seed: int,
    N: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed_stride: int = 1000,
    sample_batch: int = -1,
) -> Tuple[List[str], List[int]]:
    base_seed = int(base_seed)
    set_all_seeds(base_seed)

    try:
        query = tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])
        conv = [{"from": "system", "value": "You are a helpful radiology assistant."},
                {"from": "human",  "value": query}]
        ids = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    except AttributeError:
        ids = tokenizer(f"Image: {image_path}\n{prompt}",
                        return_tensors="pt")["input_ids"].to(device)

    kw: Dict[str, Any] = dict(
        do_sample=True, num_beams=1,
        temperature=float(temperature),
        top_p=float(top_p),
        use_cache=True,
        max_new_tokens=int(max_new_tokens),
    )

    B = N if sample_batch <= 0 else min(sample_batch, N)
    samples: List[str] = []
    seeds:   List[int] = []
    remaining = N
    call_i = 0

    pbar = tqdm(total=N, desc="  samples", leave=False, position=1)
    while remaining > 0:
        b = min(B, remaining)
        call_seed = base_seed + seed_stride * call_i
        set_all_seeds(call_seed)
        ids_b = ids.repeat(b, 1)
        out = model.generate(ids_b, **kw)
        texts = tokenizer.batch_decode(out[:, ids.size(1):], skip_special_tokens=True)
        for t in texts:
            samples.append(t.strip())
            seeds.append(call_seed)
        remaining -= b
        call_i += 1
        pbar.update(b)
    pbar.close()
    return samples, seeds


def load_done_keys(path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                done.add(int(r["idx"]))
            except Exception:
                pass
    return done


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSONL)), exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))

    print(f"[clean_runner] Loading dataset from {DS_PATH}...")
    ds = load_from_disk(DS_PATH)
    print(f"[clean_runner] Dataset size: {len(ds)}")
    print(f"[clean_runner] Running on first {N_IMAGES} images, N={N_SAMPLES} samples each")

    model, tokenizer, device = load_model(MODEL_ID, DTYPE_STR)

    done = load_done_keys(OUT_JSONL)
    print(f"[clean_runner] Already done: {len(done)} images")

    with open(OUT_JSONL, "a") as fout, open(ERR_JSONL, "a") as ferr:
        for i in tqdm(range(N_IMAGES), desc="Clean inference"):
            if i in done:
                continue
            try:
                row = ds[i]
                # ── KEY DIFFERENCE: no apply_corruption, use image directly ──
                img = row["image"]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                img = img.convert("RGB")

                img_path = save_pil_temp(img, TMP_DIR, f"clean_{i}")

                samples, call_seeds = generate_n_batched(
                    model, tokenizer, device,
                    image_path=img_path,
                    prompt=PROMPT,
                    base_seed=SEED + i * SEED_STRIDE,
                    N=N_SAMPLES,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed_stride=SEED_STRIDE,
                    sample_batch=SAMPLE_BATCH,
                )

                rec = {
                    "model_id":   MODEL_ID,
                    "prompt":     PROMPT,
                    "Path":       row.get("Path", ""),
                    "idx":        i,
                    "corruption": "clean",
                    "severity":   0,
                    "seed":       SEED + i * SEED_STRIDE,
                    "samples":    samples,
                    "sample_seeds": call_seeds,
                    "ts":         time.time(),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                done.add(i)

            except Exception as e:
                ferr.write(json.dumps({
                    "idx": i,
                    "corruption": "clean",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }) + "\n")
                ferr.flush()

    # Verify output
    count = sum(1 for _ in open(OUT_JSONL))
    print(f"\n[clean_runner] Done — {count} records written")
    print(f"  out: {OUT_JSONL}")
    print(f"  err: {ERR_JSONL}")

    # Quick sample check
    with open(OUT_JSONL) as f:
        sample_rec = json.loads(f.readline())
    print(f"\nSample record:")
    print(f"  idx={sample_rec['idx']}  corruption={sample_rec['corruption']}")
    print(f"  Path={sample_rec['Path']}")
    print(f"  N samples={len(sample_rec['samples'])}")
    print(f"  sample[0][:150]: {sample_rec['samples'][0][:150]}")


if __name__ == "__main__":
    main()