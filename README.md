# TeleRadShift 🩻

> Worst-Group Conformal Risk Control for Safe VLM Triage Under Telemedicine Acquisition Shift
>
> *ACM International Conference on Multimedia (MM '26)*

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-blue?style=flat-square)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=flat-square)](https://gen-ai-research.github.io/teleradshift)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#license)

---

## Overview

Chest radiographs in low-resource telemedicine settings often arrive as display-recaptured smartphone images rather than native digital studies. These images carry structured acquisition artifacts — blur, brightness shifts, JPEG compression, occlusion, and screen distortions — that create clinically significant distribution shift for radiology vision-language models (VLMs).

Standard Conformal Risk Control (CRC) enforces average-risk guarantees but can silently violate worst-group safety on harder corruption subgroups. We address this with:

- **FDV (Findings-Disagreement-Variance)** — an uncertainty score combining entropy and cross-sample disagreement to capture cross-modal decoupling under acquisition shift.
- **WG-CRC (Worst-Group Conformal Risk Control)** — a post-hoc calibration framework that enforces subgroup-level safety guarantees without retraining.
- **TeleRadShift** — a structured 15-group benchmark with 1,800 real smartphone captures for evaluating VLM reliability under telemedicine conditions.

---

## Key Results

| Method | Worst-Group Risk ↓ | Coverage ↑ | Violations |
|---|---|---|---|
| No Abstention | 0.696 | 100.0% | 15/15 ✗ |
| Vanilla CRC | 0.528 | 33.6% | 3/15 ✗ |
| Bonferroni | 0.467 | 2.7% | 0/15 ✓ |
| **WG-CRC + FDV (ours)** | **0.500** | **23.9%** | **0/15 ✓** |

WG-CRC achieves **8.9× improvement in coverage** over Bonferroni at matched safety, and transfers to 1,800 real smartphone captures with 0/9 violations.

---

## Installation

```bash
git clone https://github.com/gen-ai-research/teleradshift.git
cd teleradshift
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, `transformers`, `numpy`, `scipy`, `scikit-learn`, `pillow`, `opencv-python`

---

## Quick Start

```python
from teleradshift.vlm import CheXagentWrapper
from teleradshift.uncertainty import compute_fdv, parse_findings
from teleradshift.calibration import WGCRC

# 1. Generate stochastic VLM samples
model = CheXagentWrapper(model_id="CheXagent-8B")
reports = model.generate_stochastic(image_path="xray.jpg", n_samples=6, temperature=0.8)

# 2. Compute FDV uncertainty
findings = [parse_findings(r) for r in reports]
fdv_score = compute_fdv(findings, alpha=0.5)

# 3. Calibrate and predict
wg_crc = WGCRC(epsilon=0.50, base_margin=0.04)
wg_crc.fit(fdv_scores=cal_fdv, risks=cal_risks, groups=cal_groups)
decision = wg_crc.predict(fdv_score=fdv_score, group="blur_s2")
# Returns: "certify" or "defer"
```

---

## Citation

```bibtex
@inproceedings{teleradshift2026,
  title     = {TeleRadShift: Worst-Group Conformal Risk Control for Safe VLM Triage Under Telemedicine Acquisition Shift},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License.