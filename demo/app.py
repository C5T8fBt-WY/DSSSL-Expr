"""
Gradio demo: NTK-based Data Shapley Pseudo-Labeling on MNIST.

On startup, MNIST is downloaded and analytical NTK Data Shapley scores are
precomputed for 1000 unlabeled samples. The slider then lets you browse them
interactively.

Run:
    uv run python demo/app.py

First launch takes ~30–60 s (MNIST download + analytical-NTK JIT trace).
Subsequent runs are fast.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from torchvision import datasets, transforms

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from ntk_ds import build_mlp_kernel_fn, compute_ntk_scores, select_pseudo_labels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LABELS_PER_CLASS    = 5
UNLABELED_PER_CLASS = 100    # 1000 samples — stable accuracy estimate
SEED                = 42
N_CLASSES           = 10

# Globals filled by _initialize() on first UI load.
state: dict = {"ready": False}


def _load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        str(_ROOT / "data"), train=True, download=True, transform=transform
    )

    rng     = np.random.default_rng(SEED)
    targets = np.array(dataset.targets)

    labeled_idx, unlabeled_idx = [], []
    for cls in range(N_CLASSES):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        labeled_idx.extend(cls_idx[:LABELS_PER_CLASS].tolist())
        unlabeled_idx.extend(
            cls_idx[LABELS_PER_CLASS:LABELS_PER_CLASS + UNLABELED_PER_CLASS].tolist()
        )

    def get_data(indices):
        flat, raw, lbls = [], [], []
        for idx in indices:
            img, lbl = dataset[idx]
            arr = img.numpy()
            flat.append(arr.flatten())
            raw.append(arr.squeeze())
            lbls.append(lbl)
        return np.stack(flat), np.array(lbls), raw

    labeled = get_data(labeled_idx)
    unlabeled = get_data(unlabeled_idx)
    return labeled, unlabeled


def _initialize(progress=gr.Progress()):
    """Heavy startup: download MNIST, build NTK kernel, score all unlabeled."""
    if state["ready"]:
        return _description_md(), _labeled_grid_fig(), _sample_plot_fig(0)

    progress(0.05, desc="Loading MNIST…")
    labeled, unlabeled = _load_mnist()
    state["labeled_x"], state["labeled_y"], state["labeled_imgs"] = labeled
    state["unlabeled_x"], state["true_labels"], state["unlabeled_imgs"] = unlabeled

    progress(0.30, desc="Building analytical NTK kernel function…")
    kernel_fn = build_mlp_kernel_fn(hidden_size=256)

    progress(0.55, desc="Computing NTK Data Shapley scores (JIT-tracing, ~30s)…")
    state["scores"] = compute_ntk_scores(
        state["labeled_x"], state["labeled_y"], state["unlabeled_x"], kernel_fn
    )
    state["pseudo_labels"] = select_pseudo_labels(state["scores"])
    state["accuracy"] = float(
        (state["pseudo_labels"] == state["true_labels"]).mean() * 100
    )
    state["n_unlabeled"] = len(state["unlabeled_x"])
    state["ready"] = True

    progress(1.0, desc="Done.")
    return _description_md(), _labeled_grid_fig(), _sample_plot_fig(0)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _labeled_grid_fig():
    fig, axes = plt.subplots(2, 5, figsize=(6, 3))
    for cls in range(N_CLASSES):
        ax  = axes[cls // 5][cls % 5]
        idx = np.where(state["labeled_y"] == cls)[0][0]
        ax.imshow(state["labeled_imgs"][idx], cmap="gray", interpolation="nearest")
        ax.set_title(f"Class {cls}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"Labeled data — {LABELS_PER_CLASS} per class (showing 1 of {LABELS_PER_CLASS})",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def _sample_plot_fig(idx):
    idx  = int(idx)
    pred = int(state["pseudo_labels"][idx])
    true = int(state["true_labels"][idx])

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax_img.imshow(state["unlabeled_imgs"][idx], cmap="gray", interpolation="nearest")
    result_str = "CORRECT" if pred == true else "WRONG"
    ax_img.set_title(
        f"Unlabeled sample #{idx}\n"
        f"Predicted: {pred}   True: {true}   [{result_str}]",
        fontsize=11,
        color="green" if pred == true else "red",
    )
    ax_img.axis("off")

    sample_scores = state["scores"][idx]
    colors = ["#4a90d9"] * N_CLASSES
    if pred == true:
        colors[pred] = "#2ecc71"
    else:
        colors[pred] = "#e67e22"
        colors[true] = "#2ecc71"

    ax_bar.bar(range(N_CLASSES), sample_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(range(N_CLASSES))
    ax_bar.set_xlabel("Candidate label", fontsize=10)
    ax_bar.set_ylabel("Data Shapley score", fontsize=10)
    ax_bar.set_title(
        "NTK Data Shapley scores per class\n"
        "(green = true label, orange = predicted if wrong)",
        fontsize=9,
    )
    ax_bar.axhline(y=0, color="black", linewidth=0.6, linestyle="--")
    ax_bar.margins(x=0.02)
    fig.tight_layout()
    return fig


def _slider_callback(idx):
    fig = _sample_plot_fig(idx)
    try:
        return fig
    finally:
        plt.close(fig)


def _description_md():
    return (
        f"## How it works\n\n"
        f"Given **{LABELS_PER_CLASS} labeled examples per class** "
        f"(total: {LABELS_PER_CLASS * N_CLASSES}), the analytical NTK of a "
        "2-hidden-layer MLP (Erf activation, width 256) assigns Data Shapley "
        "scores to every *(unlabeled point, candidate label)* pair — "
        "**without any training**.\n\n"
        "The candidate label with the highest score is selected as the pseudo-label. "
        f"Accuracy on this split: **{state['accuracy']:.1f}%** over "
        f"{state['n_unlabeled']} unlabeled samples (compare to the 10% random baseline).\n\n"
        "Use the slider to browse unlabeled samples and inspect the DS score distribution."
    )


# ---------------------------------------------------------------------------
# Gradio UI — heavy work is deferred to demo.load(), so the page renders fast
# ---------------------------------------------------------------------------
_INITIAL_MD = (
    "## Loading…\n\n"
    "Downloading MNIST and computing analytical NTK Data Shapley scores. "
    "First-time setup takes ~30–60 s on a free CPU instance."
)

with gr.Blocks(title="NTK Data Shapley Demo") as demo:
    gr.Markdown("# NTK-based Data Shapley Pseudo-Labeling")
    description = gr.Markdown(_INITIAL_MD)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Labeled reference data")
            labeled_plot = gr.Plot(label="One example per class")

        with gr.Column(scale=2):
            gr.Markdown("### Unlabeled sample inspector")
            slider = gr.Slider(
                minimum=0,
                maximum=UNLABELED_PER_CLASS * N_CLASSES - 1,
                step=1,
                value=0,
                label="Sample index",
            )
            sample_plot = gr.Plot(label="DS score breakdown")

    demo.load(
        fn=_initialize,
        outputs=[description, labeled_plot, sample_plot],
    )
    slider.change(fn=_slider_callback, inputs=slider, outputs=sample_plot)


if __name__ == "__main__":
    demo.queue().launch()
