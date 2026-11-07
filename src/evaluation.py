"""
Evaluation module.

Functions to test the model, compute the anomaly scores (reconstruction MSE,
total intensity) and display the performance reports: ROC, precision-recall
and confusion matrix.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_auc_score, roc_curve)

CLASS_LABELS = ["Normal", "Anomaly"]


# ====================================
# Data checks
# ====================================

def sanity_check_dataset(name, dataset, expected_shape=(740, 2)):
    """Print basic diagnostics (shape, NaN/Inf, intensity range) of a dataset."""
    print(f"--- Diagnostic: {name} ---")

    print(f"Expected shape : (N, {expected_shape[0]}, {expected_shape[1]})")
    print(f"Actual shape   : {dataset.shape}")
    if dataset.shape[1:] != expected_shape:
        print("CRITICAL ERROR: wrong number of points or columns!")

    n_nans = np.isnan(dataset).sum()
    n_infs = np.isinf(dataset).sum()
    if n_nans > 0 or n_infs > 0:
        print(f"CRITICAL ERROR: contains {n_nans} NaNs and {n_infs} Infs!")
    else:
        print("Clean data (0 NaN, 0 Inf)")

    # Scaling check (on the intensity column: index 1)
    print(f"Min intensity  : {np.min(dataset[:, :, 1]):.4f}")
    print(f"Max intensity  : {np.max(dataset[:, :, 1]):.4f}")
    print("\n")


def visual_check(dataset, name="Random spectrum"):
    """Plot one random spectrum of the dataset."""
    idx = np.random.randint(0, dataset.shape[0])
    spectrum = dataset[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(spectrum[:, 0], spectrum[:, 1], color="blue")
    plt.title(f"Visual check: {name} (index {idx})")
    plt.xlabel("m/z")
    plt.ylabel("Intensity (scaled)")
    plt.grid(True, alpha=0.3)
    plt.show()


# ====================================
# Scores
# ====================================

def compute_scores(model, X):
    """
    Compute, for each spectrum of X:
      - the reconstruction MSE (shape error),
      - the total intensity (signal volume).
    """
    reconstruction = model.predict(X, verbose=0)
    mse = np.mean(np.square(X[:, :, 1:] - reconstruction), axis=(1, 2))
    total_intensity = np.sum(X[:, :, 1], axis=1)
    return mse, total_intensity


def _collect_scores(model, X_norm, X_mass, X_low):
    """Scores per group, plus the pooled labels (0 = normal, 1 = anomaly) and scores."""
    mse_norm, int_norm = compute_scores(model, X_norm)
    mse_mass, int_mass = compute_scores(model, X_mass)
    mse_low, int_low = compute_scores(model, X_low)

    return {
        "mse": {"norm": mse_norm, "mass": mse_mass, "low": mse_low},
        "int": {"norm": int_norm, "mass": int_mass, "low": int_low},
        "y_true": np.array([0] * len(mse_norm) + [1] * (len(mse_mass) + len(mse_low))),
        "all_mse": np.concatenate([mse_norm, mse_mass, mse_low]),
        "all_int": np.concatenate([int_norm, int_mass, int_low]),
    }


def _min_max(x):
    """Rescale an array to [0, 1]."""
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)


# ====================================
# Plots shared by both dashboards
# ====================================

def _plot_top_row(axes, s, mse_threshold, int_threshold):
    """Top row: MSE histogram, intensity-vs-MSE scatter plot, MSE boxplot."""
    mse, tot = s["mse"], s["int"]

    # 1. MSE histogram
    ax = axes[0]
    ax.hist(mse["norm"], bins=30, alpha=0.6, color="blue", label="Healthy")
    ax.hist(mse["low"], bins=30, alpha=0.6, color="red", label="Low Intensity")
    ax.hist(mse["mass"], bins=30, alpha=0.7, color="green", label="Mass Shift")
    ax.axvline(mse_threshold, color="black", lw=2, linestyle="--", label="MSE threshold")
    ax.set_title("1. Error distribution (MSE)")
    ax.set_xlabel("MSE (log scale)")
    ax.set_ylabel("Count")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Scatter plot: intensity vs MSE
    ax = axes[1]
    ax.scatter(tot["norm"], mse["norm"], color="blue", alpha=0.5, label="Healthy", s=15)
    ax.scatter(tot["low"], mse["low"], color="red", alpha=0.7, label="Low Intensity", s=15)
    ax.scatter(tot["mass"], mse["mass"], color="green", alpha=0.7, label="Mass Shift", s=15)
    ax.axhline(mse_threshold, color="black", linestyle="--", label="AI filter (MSE)")
    ax.axvline(int_threshold, color="grey", linestyle="-.", label=f"Intensity filter ({int_threshold})")
    ax.set_title("2. Scatter plot (intensity vs MSE)")
    ax.set_xlabel("Total intensity")
    ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. MSE boxplot
    ax = axes[2]
    ax.boxplot([mse["norm"], mse["low"], mse["mass"]])
    ax.set_xticklabels(["Healthy", "Low Int.", "Mass Shift"])
    ax.axhline(mse_threshold, color="black", linestyle="--")
    ax.set_title("3. Error boxplot (MSE)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def _plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS).plot(
        cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(title)


def _print_report(title, y_true, y_pred):
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))


# ====================================
# Dashboards
# ====================================

def evaluate_hybrid(X_norm, X_mass, X_low, model, mse_threshold=0.0004, int_threshold=310):
    """
    Evaluate the hybrid detector: a spectrum is flagged as an anomaly if its
    reconstruction MSE is above `mse_threshold` OR its total intensity is
    below `int_threshold`.

    Shows a 6-panel dashboard comparing the AI score, the intensity score and
    the hybrid score on healthy, mass-shift and low-intensity spectra.
    """
    print("Computing predictions (MSE and intensity)...")
    s = _collect_scores(model, X_norm, X_mass, X_low)
    y_true, all_mse, all_int = s["y_true"], s["all_mse"], s["all_int"]

    y_pred = ((all_mse > mse_threshold) | (all_int < int_threshold)).astype(int)
    _print_report(f"HYBRID MODEL RESULTS (MSE threshold = {mse_threshold}, "
                  f"intensity threshold = {int_threshold})", y_true, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Multiclass Evaluation Dashboard - Hybrid model", fontsize=16, fontweight="bold")
    _plot_top_row(axes[0], s, mse_threshold, int_threshold)

    # Continuous scores in [0, 1] (1 = anomaly). Low intensity means anomaly,
    # so the intensity is negated; the hybrid score is the maximum of the two.
    score_ai = _min_max(all_mse)
    score_int = _min_max(-all_int)
    score_hybrid = np.maximum(score_ai, score_int)
    scores = [
        ("AI", score_ai, "green", 2, 0.7),
        ("Intensity", score_int, "red", 2, 0.7),
        ("Hybrid", score_hybrid, "blue", 3, 1.0),
    ]

    # 4. ROC curves
    ax = axes[1, 0]
    for name, score, color, lw, alpha in scores:
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, color=color, lw=lw, alpha=alpha, label=f"{name} (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title("4. ROC curves comparison")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 5. Precision-recall curves
    ax = axes[1, 1]
    for name, score, color, lw, alpha in scores:
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, color=color, lw=lw, alpha=alpha, label=f"{name} (AP = {ap:.2f})")
    ax.set_title("5. Precision-recall comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # 6. Confusion matrix
    _plot_confusion_matrix(axes[1, 2], y_true, y_pred, "6. Confusion matrix (hybrid model)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def evaluate_autoencoder(X_norm, X_mass, X_low, model, mse_threshold=0.0004, int_threshold=310):
    """
    Evaluate the autoencoder alone: a spectrum is flagged as an anomaly if its
    reconstruction MSE is above `mse_threshold`.

    `int_threshold` is only used to draw the intensity filter on the scatter
    plot; it does not influence the predictions.
    """
    print("Computing predictions (MSE and intensity)...")
    s = _collect_scores(model, X_norm, X_mass, X_low)
    y_true, all_mse = s["y_true"], s["all_mse"]

    y_pred = (all_mse > mse_threshold).astype(int)
    _print_report(f"AI-ONLY RESULTS (MSE threshold = {mse_threshold})", y_true, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Multiclass Evaluation Dashboard - AI only", fontsize=16, fontweight="bold")
    _plot_top_row(axes[0], s, mse_threshold, int_threshold)

    # 4. ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_true, all_mse)
    auc = roc_auc_score(y_true, all_mse)
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title("4. Overall ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Precision-recall curve
    ax = axes[1, 1]
    precision, recall, _ = precision_recall_curve(y_true, all_mse)
    ax.plot(recall, precision, color="darkorange", lw=2)
    ax.set_title("5. Precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)

    # 6. Confusion matrix
    _plot_confusion_matrix(axes[1, 2], y_true, y_pred, "6. Confusion matrix (AI only)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
