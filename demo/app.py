"""
Gradio demo: NTK-based Data Shapley Pseudo-Labeling on MNIST.

Precomputes DS scores for all unlabeled samples on startup, then lets you
browse them interactively. For each selected sample you see:
  - The digit image.
  - A bar chart of DS scores per class label (0-9).
  - Predicted vs. ground-truth label.

Run:
    uv run python demo/app.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for Gradio
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from torchvision import datasets, transforms

# --- locate ntk_ds.py regardless of where this script is launched from -----
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from ntk_ds import build_mlp_kernel_fn, compute_ntk_scores, select_pseudo_labels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LABELS_PER_CLASS    = 5
UNLABELED_PER_CLASS = 10   # keep small so the gallery is readable
SEED                = 42
N_CLASSES           = 10

# ---------------------------------------------------------------------------
# Load MNIST and split into labeled / unlabeled
# ---------------------------------------------------------------------------
print("Loading MNIST…")
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
_dataset = datasets.MNIST(str(_ROOT / "data"), train=True, download=True, transform=_transform)

_rng     = np.random.default_rng(SEED)
_targets = np.array(_dataset.targets)

_labeled_idx, _unlabeled_idx = [], []
for _cls in range(N_CLASSES):
    _cls_idx = np.where(_targets == _cls)[0]
    _rng.shuffle(_cls_idx)
    _labeled_idx.extend(_cls_idx[:LABELS_PER_CLASS].tolist())
    _unlabeled_idx.extend(
        _cls_idx[LABELS_PER_CLASS:LABELS_PER_CLASS + UNLABELED_PER_CLASS].tolist()
    )


def _get_data(indices):
    flat_imgs, raw_imgs, labels = [], [], []
    for idx in indices:
        img, lbl = _dataset[idx]
        arr = img.numpy()
        flat_imgs.append(arr.flatten())
        raw_imgs.append(arr.squeeze())      # (28, 28) for display
        labels.append(lbl)
    return np.stack(flat_imgs), np.array(labels), raw_imgs


labeled_x,   labeled_y,   labeled_imgs   = _get_data(_labeled_idx)
unlabeled_x, true_labels, unlabeled_imgs = _get_data(_unlabeled_idx)

# ---------------------------------------------------------------------------
# Precompute NTK DS scores
# ---------------------------------------------------------------------------
print("Building analytical NTK kernel function…")
kernel_fn = build_mlp_kernel_fn(hidden_size=256)

print("Computing NTK Data Shapley scores for all unlabeled samples…")
scores        = compute_ntk_scores(labeled_x, labeled_y, unlabeled_x, kernel_fn)
pseudo_labels = select_pseudo_labels(scores)

n_unlabeled = len(unlabeled_x)
accuracy    = (pseudo_labels == true_labels).mean() * 100
print(f"Pseudo-label accuracy: {accuracy:.1f}%  ({n_unlabeled} samples)")

# ---------------------------------------------------------------------------
# Helper: build labeled-data reference grid (static)
# ---------------------------------------------------------------------------
def _make_labeled_grid():
    fig, axes = plt.subplots(2, 5, figsize=(6, 3))
    for cls in range(N_CLASSES):
        ax  = axes[cls // 5][cls % 5]
        idx = np.where(labeled_y == cls)[0][0]
        ax.imshow(labeled_imgs[idx], cmap="gray", interpolation="nearest")
        ax.set_title(f"Class {cls}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"Labeled data — {LABELS_PER_CLASS} per class (showing 1 each)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Helper: build per-sample DS visualization
# ---------------------------------------------------------------------------
def _make_sample_plot(idx: int):
    idx  = int(idx)
    pred = int(pseudo_labels[idx])
    true = int(true_labels[idx])

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 3.5))

    # --- Digit image ---------------------------------------------------------
    ax_img.imshow(unlabeled_imgs[idx], cmap="gray", interpolation="nearest")
    result_str = "CORRECT" if pred == true else "WRONG"
    ax_img.set_title(
        f"Unlabeled sample #{idx}\n"
        f"Predicted: {pred}   True: {true}   [{result_str}]",
        fontsize=11,
        color="green" if pred == true else "red",
    )
    ax_img.axis("off")

    # --- DS score bar chart --------------------------------------------------
    sample_scores = scores[idx]

    colors = ["#4a90d9"] * N_CLASSES          # default blue
    if pred == true:
        colors[pred] = "#2ecc71"              # green when correct
    else:
        colors[pred] = "#e67e22"              # orange for predicted (wrong)
        colors[true] = "#2ecc71"              # green for true label

    ax_bar.bar(range(N_CLASSES), sample_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(range(N_CLASSES))
    ax_bar.set_xlabel("Candidate label", fontsize=10)
    ax_bar.set_ylabel("Data Shapley score", fontsize=10)
    ax_bar.set_title(
        "NTK Data Shapley scores per class\n"
        "(green = true label,  orange = predicted if wrong)",
        fontsize=9,
    )
    ax_bar.axhline(y=0, color="black", linewidth=0.6, linestyle="--")
    ax_bar.margins(x=0.02)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_DESCRIPTION = f"""\
## How it works

Given **{LABELS_PER_CLASS} labeled examples per class** (total: {len(labeled_x)}),  
the analytical NTK of a 2-layer MLP (Erf activation, width 256) is used to assign  
Data Shapley scores to every *(unlabeled point, candidate label)* pair — **without any training**.

The candidate label with the highest score is selected as the pseudo-label.  
Achieved accuracy on this split: **{accuracy:.1f}%** ({n_unlabeled} unlabeled samples)
— compare to 10% random baseline and ~47% for a single randomly-initialized model.

Use the slider to browse unlabeled samples and inspect the DS score distribution.
"""

with gr.Blocks(title="NTK Data Shapley Demo") as demo:
    gr.Markdown("# NTK-based Data Shapley Pseudo-Labeling")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Labeled reference data")
            labeled_grid_plot = gr.Plot(label="One example per class")

        with gr.Column(scale=2):
            gr.Markdown("### Unlabeled sample inspector")
            slider = gr.Slider(
                minimum=0,
                maximum=n_unlabeled - 1,
                step=1,
                value=0,
                label="Sample index",
            )
            sample_plot = gr.Plot(label="DS score breakdown")

    # Render static labeled grid on page load
    demo.load(fn=_make_labeled_grid, outputs=labeled_grid_plot)

    # Render sample plot on slider change and on page load
    demo.load(fn=lambda: _make_sample_plot(0), outputs=sample_plot)
    slider.change(fn=_make_sample_plot, inputs=slider, outputs=sample_plot)


if __name__ == "__main__":
    demo.launch()
