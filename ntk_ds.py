"""
NTK-based Data Shapley pseudo-labeling — minimal copy-paste utility.

Key formula (from the paper):
    DS(x_U, ỹ) = Σ_{(x_i, y_i)} (p(x_U) - ỹ)ᵀ Θ(x_U, x_i) (p(x_i) - y_i)

Under one approximation valid at random initialization:
  - Uniform softmax: p(x) ≈ 1/C for all classes

…and using the fact that the scalar NTK of a fully-connected network is
automatically diagonal-in-class (neural_tangents reduces output channels
analytically), the formula reduces to:

    DS(x_U, ỹ) ≈ Σ_i coef(ỹ, y_i) * Θ_scalar(x_U, x_i)

where:
    coef = (C-1)/C  if ỹ == y_i  (pseudo-label matches labeled class)
    coef =    -1/C  if ỹ != y_i  (pseudo-label mismatches)

The scalar NTK is computed analytically via `neural_tangents` (Novak et al. 2020).
For an MLP with Erf activation this admits a closed form, enabling training-free
pseudo-label selection in O(n_U × n_L) kernel evaluations.

References:
    Mitoma et al. 2026 (JSAI)
    Ghorbani & Zou 2019 — Data Shapley
    Wang et al. 2025 — gradient-based DS approximation
    Jacot et al. 2018 — Neural Tangent Kernel
    Novak et al. 2020 — neural_tangents library
"""

from typing import Callable

import numpy as np
from neural_tangents import stax


def build_mlp_kernel_fn(
    hidden_size: int = 256,
    n_hidden_layers: int = 2,
    n_classes: int = 10,
) -> Callable:
    """
    Build an MLP with Erf activation and return its analytical NTK kernel function.

    Erf is used instead of Tanh because it admits a closed-form analytical NTK
    in neural_tangents. The two activations yield nearly identical pseudo-label
    accuracy in practice (within ±1 pp on MNIST, see paper §4).

    The final readout width does not affect the scalar NTK value for stax
    fully-connected networks (neural_tangents reduces output channels
    analytically), so `n_classes` here is just metadata for the readout.

    Args:
        hidden_size:     Width of each hidden layer.
        n_hidden_layers: Number of hidden layers, not counting the readout.
        n_classes:       Number of output classes (readout width).

    Returns:
        kernel_fn: Callable — kernel_fn(X1, X2, 'ntk') → (n1, n2) NTK matrix.
    """
    layers = []
    for _ in range(n_hidden_layers):
        layers += [stax.Dense(hidden_size), stax.Erf()]
    layers.append(stax.Dense(n_classes))
    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn


def compute_ntk_scores(
    labeled_x: np.ndarray,
    labeled_y: np.ndarray,
    unlabeled_x: np.ndarray,
    kernel_fn: Callable,
    n_classes: int = 10,
) -> np.ndarray:
    """
    Compute Data Shapley scores for every (unlabeled point, candidate label) pair.

    Args:
        labeled_x:   Shape (n_L, d)  — labeled features (flattened).
        labeled_y:   Shape (n_L,)    — labeled integer class indices.
        unlabeled_x: Shape (n_U, d)  — unlabeled features (flattened).
        kernel_fn:   Output of build_mlp_kernel_fn().
        n_classes:   Number of classes (10 for MNIST). Must match build_mlp_kernel_fn.

    Returns:
        scores: Shape (n_U, n_classes). scores[i, c] is the (mean over labeled set)
                DS score for assigning pseudo-label c to unlabeled point i.
                The predicted pseudo-label is argmax over the class axis.
    """
    labeled_y = np.asarray(labeled_y).astype(int).ravel()
    n_labeled = labeled_x.shape[0]

    ntk_matrix = np.asarray(kernel_fn(unlabeled_x, labeled_x, "ntk"))
    assert ntk_matrix.ndim == 2, (
        f"Expected scalar NTK of shape (n_U, n_L); got {ntk_matrix.shape}. "
        "This utility supports only fully-connected stax models."
    )

    # Coefficients derived from the uniform-softmax approximation.
    match_coef    =  (n_classes - 1) / n_classes
    mismatch_coef = -1.0              / n_classes

    # Vectorized form of:
    #   for c in range(n_classes):
    #       coefs = where(labeled_y == c, match, mismatch)
    #       scores[:, c] = (ntk_matrix * coefs).sum(axis=1) / n_labeled
    onehot = np.eye(n_classes)[labeled_y]                                # (n_L, C)
    coef_matrix = onehot * (match_coef - mismatch_coef) + mismatch_coef  # (n_L, C)
    scores = ntk_matrix @ coef_matrix / n_labeled                        # (n_U, C)
    return scores


def select_pseudo_labels(scores: np.ndarray) -> np.ndarray:
    """
    Select pseudo-labels as argmax of DS scores.

    Args:
        scores: Shape (n_U, n_classes) — output of compute_ntk_scores().

    Returns:
        pseudo_labels: Shape (n_U,) — integer class predictions.
    """
    return np.argmax(scores, axis=1)


# ---------------------------------------------------------------------------
# Minimal end-to-end example (run as a script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from torchvision import datasets, transforms

    LABELS_PER_CLASS    = 5
    UNLABELED_PER_CLASS = 100
    SEED                = 42

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    rng     = np.random.default_rng(SEED)
    targets = np.array(dataset.targets)

    labeled_idx, unlabeled_idx = [], []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        labeled_idx.extend(cls_idx[:LABELS_PER_CLASS].tolist())
        unlabeled_idx.extend(
            cls_idx[LABELS_PER_CLASS:LABELS_PER_CLASS + UNLABELED_PER_CLASS].tolist()
        )

    def get_data(indices):
        imgs, lbls = [], []
        for i in indices:
            img, lbl = dataset[i]
            imgs.append(img.numpy().flatten())
            lbls.append(lbl)
        return np.stack(imgs), np.array(lbls)

    labeled_x,   labeled_y   = get_data(labeled_idx)
    unlabeled_x, true_labels = get_data(unlabeled_idx)

    print("Building NTK kernel function…")
    kernel_fn = build_mlp_kernel_fn(hidden_size=256)

    print("Computing NTK DS scores…")
    t0     = time.time()
    scores = compute_ntk_scores(labeled_x, labeled_y, unlabeled_x, kernel_fn)
    elapsed = time.time() - t0

    pseudo_labels = select_pseudo_labels(scores)
    accuracy      = (pseudo_labels == true_labels).mean() * 100

    print(f"Pseudo-label accuracy : {accuracy:.1f}%")
    print(f"Computation time      : {elapsed:.2f}s")
    print(f"(Labeled: {len(labeled_x)}, Unlabeled: {len(unlabeled_x)})")
