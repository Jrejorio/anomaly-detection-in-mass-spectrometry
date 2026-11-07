"""
Data preparation module.

Functions to load, align, filter and scale mass spectra.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def load_spectra(folder):
    """
    Load every CSV file of a folder into a 3D numpy tensor.

    Each CSV must have the columns `mz` and `spectrum`. Intensities are
    transformed with log1p. Files whose layout differs from the first one
    (different number of columns or rows) are skipped.

    Returns an array of shape (n_spectra, n_points, 2), where the last axis
    holds (m/z, log-intensity).
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise ValueError(f"No CSV file found in folder {folder}")

    # The first file defines the reference layout
    reference = pd.read_csv(files[0])
    n_rows, n_cols = reference.shape

    print(f"Loading folder: {folder.name}")

    spectra = []
    for file in tqdm(files, desc="Reading CSV files", unit="file"):
        try:
            df = pd.read_csv(file)
            if df.shape != (n_rows, n_cols):
                raise ValueError(f"unexpected shape {df.shape}, expected {(n_rows, n_cols)}")
            spectra.append(np.stack((df["mz"].values, np.log1p(df["spectrum"].values)), axis=-1))
        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    return np.stack(spectra, axis=0)


def read_windows(windows_file):
    """
    Read the text file listing the m/z windows of interest.

    Each line has the form `start;end`. Malformed lines are ignored.
    Returns a list of (start, end) tuples.
    """
    windows = []
    with open(windows_file, "r") as f:
        for line in f:
            line = line.strip()
            if ";" in line and not line.startswith("["):
                parts = line.split(";")
                try:
                    windows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return windows


def align_and_filter_dataset(dataset, mz_reference, windows_file):
    """
    1. Align each spectrum on the `mz_reference` axis by interpolation.
    2. Keep only the points that fall inside the m/z windows of interest.

    Returns a fixed-size tensor (n_spectra, n_filtered_points, 2).
    """
    n_spectra = dataset.shape[0]
    n_points_ref = len(mz_reference)

    aligned = np.zeros((n_spectra, n_points_ref, 2))
    aligned[:, :, 0] = mz_reference

    for i in range(n_spectra):
        interpolator = interp1d(dataset[i, :, 0], dataset[i, :, 1],
                                bounds_error=False, fill_value=0)
        aligned[i, :, 1] = interpolator(mz_reference)

    mask = np.zeros(n_points_ref, dtype=bool)
    for start, end in read_windows(windows_file):
        mask |= (mz_reference >= start) & (mz_reference <= end)

    return aligned[:, mask, :]


def scale_datasets(X_train, *other_datasets):
    """
    Fit a MinMaxScaler on the training set, then apply it to the other sets.

    Handles the 3D -> 2D -> 3D reshape required by scikit-learn.
    Returns (X_train_scaled, [other scaled datasets], scaler).
    """
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 2)).reshape(X_train.shape)

    others_scaled = [
        scaler.transform(dataset.reshape(-1, 2)).reshape(dataset.shape)
        for dataset in other_datasets
    ]

    return X_train_scaled, others_scaled, scaler
