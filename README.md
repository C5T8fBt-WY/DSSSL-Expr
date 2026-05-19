# DSSSL-Expr — NTK-based Data Shapley Pseudo-Labeling

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/mryo00/ntk-data-shapley-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Demo and reference code for the JSAI 2026 paper:

> **Fundamental Study on Data Shapley-based Pseudo-Labeling via Neural Tangent Kernel**
> R. Mitoma, T. Mukaeda, K. Shima — Yokohama National University

---

## Overview

Standard pseudo-labeling relies on a trained model to estimate labels for unlabeled data.
This work shows that the Data Shapley value of an unlabeled point can be computed **without any training**, using the analytical Neural Tangent Kernel (NTK) of a randomly initialized network.

Key result: the ensemble average of Data Shapley values over many random initializations converges to the deterministic analytical NTK as a Monte Carlo approximation, enabling **training-free**, deterministic pseudo-label selection in a fraction of the time.

### Core formula

Given labeled data $\mathcal{D}_L = \{(x_i, y_i)\}$ and an unlabeled point $x_U$ with candidate label $\tilde{y}$:

$$\text{DS}(x_U, \tilde{y}) \approx \sum_{(x_i, y_i) \in \mathcal{D}_L} \text{coef}(\tilde{y}, y_i) \cdot \Theta_{\text{ana}}(x_U, x_i)$$

where $\Theta_{\text{ana}}$ is the analytical NTK and

$$\text{coef}(\tilde{y}, y_i) = \begin{cases} (C-1)/C & \text{if } \tilde{y} = y_i \\ -1/C & \text{otherwise} \end{cases}$$

The candidate label with the highest score is selected as the pseudo-label.

---

## Quick start

```bash
# Install dependencies
uv sync                  # or:  pip install -r requirements.txt

# Run the minimal example (≈ 61% MNIST accuracy on 1000 unlabeled samples)
uv run python ntk_ds.py

# Launch the interactive Gradio demo
uv run python demo/app.py
```

First launch takes ~30–60 s (MNIST download + analytical-NTK JIT trace); subsequent runs are fast.

---

## Files

| File | Description |
|------|-------------|
| [`ntk_ds.py`](ntk_ds.py) | **Copy-paste-friendly core utility.** Three functions: `build_mlp_kernel_fn`, `compute_ntk_scores`, `select_pseudo_labels`. No private dependencies. |
| [`demo/app.py`](demo/app.py) | Gradio web demo: browse MNIST unlabeled samples and inspect their DS score distributions. |
| [`pyproject.toml`](pyproject.toml) / [`requirements.txt`](requirements.txt) | Dependencies (JAX, neural-tangents, Gradio, PyTorch for data loading). |

---

## Usage as a library

```python
import numpy as np
from ntk_ds import build_mlp_kernel_fn, compute_ntk_scores, select_pseudo_labels

# labeled_x:   (n_L, d)   labeled features
# labeled_y:   (n_L,)     integer class labels
# unlabeled_x: (n_U, d)   unlabeled features

kernel_fn     = build_mlp_kernel_fn(hidden_size=256)
scores        = compute_ntk_scores(labeled_x, labeled_y, unlabeled_x, kernel_fn)
pseudo_labels = select_pseudo_labels(scores)   # shape (n_U,)
```

With 5 labeled examples per class and 1000 unlabeled samples, this achieves **≈ 61% MNIST pseudo-label accuracy** (vs. the 10% random baseline), computed in well under one second after JIT warm-up.

---

## Requirements

- Python 3.12+
- JAX (CPU build is sufficient; GPU optional)
- `neural-tangents==0.6.5` (for closed-form NTK of MLP with Erf activation)

---

## Citation

```bibtex
@inproceedings{mitoma2026jsai,
  title     = {Neural Tangent Kernel を導入した Data Shapley 型疑似ラベル法に関する基礎検討},
  author    = {Mitoma, Ryo and Mukaeda, Takayuki and Shima, Keisuke},
  booktitle = {The 40th Annual Conference of the Japanese Society for Artificial Intelligence},
  year      = {2026},
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

This work is supported by JST BOOST (grant JPMJBS2427).
