"""
NTK-based Data Shapley pseudo-labeling — minimal copy-paste utility.

Key formula (from the paper):
    DS(x_U, ỹ) = Σ_{(x_i, y_i)} (p(x_U) - ỹ)ᵀ Θ(x_U, x_i) (p(x_i) - y_i)

Under two approximations valid at random initialization:
  1. Uniform softmax: p(x) ≈ 1/C for all classes
  2. Diagonal NTK:   Θ(x, x') ≈ scalar × I  (holds exactly in the infinite-width limit)

…the formula reduces to:
    DS(x_U, ỹ) ≈ Σ_i coef(ỹ, y_i) * NTK_scalar(x_U, x_i)

where:
    coef = (C-1)/C  ≈ +0.9   if ỹ == y_i   (pseudo-label matches)
    coef =    -1/C  ≈ -0.1   if ỹ != y_i   (pseudo-label mismatches)

The scalar NTK is computed analytically via `neural_tangents` (Novak et al. 2020).
For a 2-layer MLP with Erf activation, the closed-form kernel is available,
enabling training-free pseudo-label selection in O(n_U × n_L) kernel evaluations.

References:
    Mitoma et al. 2026 (JSAI) — https://github.com/C5T8fBt-WY/DSSSL-Expr
    Ghorbani & Zou 2019 — Data Shapley
    Wang et al. 2025 — gradient-based DS approximation
    Jacot et al. 2018 — Neural Tangent Kernel
    Novak et al. 2020 — neural_tangents library
"""

import numpy as np
from neural_tangents import stax


def build_mlp_kernel_fn(hidden_size: int = 256, n_hidden_layers: int = 2):
    """
    Build an MLP with Erf activation and return its analytical NTK kernel function.

    Erf is used instead of Tanh because it admits a closed-form analytical NTK
    in neural_tangents. In practice the two activations yield nearly identical
    pseudo-label accuracy (within ±1 pp on MNIST, see paper §4).

    Args:
        hidden_size:     Width of each hidden layer.
        n_hidden_layers: Number of hidden layers (default 2, matching the paper).

    Returns:
        kernel_fn: Callable — kernel_fn(X1, X2, 'ntk') → (n1, n2) NTK matrix.
    """
    layers = []
    for _ in range(n_hidden_layers):
        layers += [stax.Dense(hidden_size), stax.Erf()]
    layers.append(stax.Dense(10))           # 10-class output
    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn


def compute_ntk_scores(
    labeled_x: np.ndarray,
    labeled_y: np.ndarray,
    unlabeled_x: np.ndarray,
    kernel_fn,
    n_classes: int = 10,
) -> np.ndarray:
    """
    Compute Data Shapley scores for every (unlabeled point, candidate label) pair.

    Args:
        labeled_x:   Shape (n_L, d)  — labeled features (flattened).
        labeled_y:   Shape (n_L,)    — labeled class indices.
        unlabeled_x: Shape (n_U, d)  — unlabeled features (flattened).
        kernel_fn:   Output of build_mlp_kernel_fn().
        n_classes:   Number of classes (10 for MNIST).

    Returns:
        scores: Shape (n_U, n_classes).  scores[i, c] is the DS score for
                assigning pseudo-label c to unlabeled point i.
                The predicted pseudo-label is argmax over the class axis.
    """
    n_labeled = labeled_x.shape[0]

    # Analytical NTK matrix — shape (n_U, n_L)
    ntk_matrix = np.array(kernel_fn(unlabeled_x, labeled_x, 'ntk'))

    # Coefficients derived from the uniform-softmax approximation
    match_coef    =  (n_classes - 1) / n_classes   # (C-1)/C  e.g. 0.9 for C=10
    mismatch_coef = -1.0              / n_classes   #    -1/C  e.g. -0.1 for C=10

    scores = np.zeros((unlabeled_x.shape[0], n_classes))
    for y_tilde in range(n_classes):
        coefs = np.where(labeled_y == y_tilde, match_coef, mismatch_coef)  # (n_L,)
        scores[:, y_tilde] = (ntk_matrix * coefs).sum(axis=1) / n_labeled

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

    # --- Load MNIST ----------------------------------------------------------
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

    # --- NTK-DS pseudo-labeling ----------------------------------------------
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
