# DSSSL-Expr — NTK-based Data Shapley Pseudo-Labeling

Demo and experiment code for the JSAI 2026 paper:

> **Fundamental Study on Data Shapley-based Pseudo-Labeling via Neural Tangent Kernel**  
> R. Mitoma, T. Mukaeda, K. Shima — Yokohama National University

---

## Overview

Standard pseudo-labeling relies on a trained model to estimate labels for unlabeled data.
This work shows that the Data Shapley value of an unlabeled point can be computed **without any training**, using the analytical Neural Tangent Kernel (NTK) of a randomly initialized network.

The key result: the ensemble average of Data Shapley values computed over many random initializations converges to the deterministic analytical NTK as a Monte Carlo approximation.
This enables **training-free**, deterministic pseudo-label selection in a fraction of the time.

### Core formula

Given labeled data $\mathcal{D}_L = \{(x_i, y_i)\}$ and an unlabeled point $x_U$ with candidate label $\tilde{y}$:

$$\text{DS}(x_U, \tilde{y}) \approx \sum_{(x_i, y_i) \in \mathcal{D}_L} \text{coef}(\tilde{y}, y_i) \cdot \Theta_{\text{ana}}(x_U, x_i)$$

where $\Theta_{\text{ana}}$ is the analytical NTK and

$$\text{coef}(\tilde{y}, y_i) = \begin{cases} (C-1)/C & \text{if } \tilde{y} = y_i \\ -1/C & \text{otherwise} \end{cases}$$

The candidate label with the highest score is selected as the pseudo-label.

---

## Quick start

```bash
# Install (Linux / macOS)
uv sync

# Windows — install JAX manually first:
#   pip install "jax[cpu]"
#   then: uv sync --no-build-isolation

# Run the minimal example
uv run python ntk_ds.py

# Launch the interactive Gradio demo
uv run python demo/app.py
```

---

## Files

| File | Description |
|------|-------------|
| `ntk_ds.py` | **Copy-paste-friendly core utility.** Three functions: `build_mlp_kernel_fn`, `compute_ntk_scores`, `select_pseudo_labels`. No private dependencies. |
| `demo/app.py` | Gradio web demo: browse MNIST unlabeled samples and inspect DS score distributions. |
| `pyproject.toml` | Dependencies (JAX, neural-tangents, Gradio, PyTorch for data loading). |

---

## Usage as a library

```python
import numpy as np
from ntk_ds import build_mlp_kernel_fn, compute_ntk_scores, select_pseudo_labels

# labeled_x: (n_L, d)  labeled features
# labeled_y: (n_L,)    integer class labels
# unlabeled_x: (n_U, d) unlabeled features

kernel_fn     = build_mlp_kernel_fn(hidden_size=256)
scores        = compute_ntk_scores(labeled_x, labeled_y, unlabeled_x, kernel_fn)
pseudo_labels = select_pseudo_labels(scores)   # shape (n_U,)
```

Achieves **~61% pseudo-label accuracy on MNIST** with only 5 labeled examples per class
(vs. 10% random baseline and ~47% for a single-model ensemble), computed in under 0.1 s.

---

## Requirements

- Python 3.12+
- JAX (CPU build is sufficient; GPU optional)
- `neural-tangents==0.6.5` (for closed-form NTK of MLP with Erf activation)

---

## Citation

```bibtex
@inproceedings{mitoma2026jsai,
  title  = {Neural Tangent Kernel を導入した Data Shapley 型疑似ラベル法に関する基礎検討},
  author = {Mitoma, Ryo and Mukaeda, Takayuki and Shima, Keisuke},
  booktitle = {The 40th Annual Conference of the Japanese Society for Artificial Intelligence},
  year   = {2026},
}
```

---

## Acknowledgements

This work is supported by JST BOOST (grant JPMJBS2427).
