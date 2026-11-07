# ====================================
# CASSIOPEE PROJECT - Main script
# ====================================

import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation import (evaluate_autoencoder, evaluate_hybrid,
                            sanity_check_dataset, visual_check)
from src.models import autoencoder_v4
from src.preprocessing import (align_and_filter_dataset, load_spectra,
                               scale_datasets)
from src.simulation import simulate_mass_shifts

# ====================================
# 0. GLOBAL PARAMETERS
# ====================================

EPOCHS = 50
BATCH_SIZE = 8
RANDOM_STATE = 42

WINDOWS_FILE = "config/windows.txt"
WEIGHTS_FILE = "models_saved/ae_v4_weights.h5"
CACHE_DIR = Path("data/processed")

MSE_THRESHOLD = 0.0001
INTENSITY_THRESHOLD = 310

# Show the data checks (set to True to inspect the preprocessed data)
RUN_SANITY_CHECKS = False


def load_with_cache(folder, cache_name):
    """Load the spectra of `folder`, using a .npy cache since reading the CSVs is slow."""
    cache_file = CACHE_DIR / cache_name
    if cache_file.exists():
        print(f"Loading {cache_name} from the NumPy cache...")
        return np.load(cache_file)

    print(f"First read of {folder} (this can take a while)...")
    data = load_spectra(folder)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, data)
    print("Cache saved!")
    return data


def main():
    # ====================================
    # 1. DATA PREPARATION
    # ====================================
    print("Loading and preparing the data...")

    # Raw data: 3D tensors (n_spectra, n_points, 2), cached as .npy
    ref_a = load_with_cache("data/data/reference_spectrum", "reference_spectrum_cache.npy")
    ref_b = load_with_cache("data/data2/reference spectrum", "reference_spectrum2_cache.npy")
    low_a = load_with_cache("data/data/low_intensity", "low_intensity_cache.npy")
    low_b = load_with_cache("data/data2/low intensity", "low_intensity2_cache.npy")
    mass_shift_raw = load_with_cache("data/data/mass_shift", "mass_shift_cache.npy")
    others_raw = load_with_cache("data/data/other", "others_cache.npy")

    ref_raw = np.concatenate([ref_a, ref_b], axis=0)
    low_int_raw = np.concatenate([low_a, low_b], axis=0)

    # The m/z axis of the first reference spectrum is the master axis
    mz_master = ref_raw[0, :, 0]

    # Train/test split (on normal spectra only)
    X_train_raw, X_test_norm_raw = train_test_split(ref_raw, test_size=0.2, random_state=RANDOM_STATE)

    # Alignment on the master axis + keep only the m/z windows of interest
    def prepare(dataset):
        return align_and_filter_dataset(dataset, mz_master, WINDOWS_FILE)

    X_train = prepare(X_train_raw)
    X_test_norm = prepare(X_test_norm_raw)
    low_int = prepare(low_int_raw)
    mass_shift = prepare(mass_shift_raw)
    others = prepare(others_raw)

    # Global scaling (fitted on the training set only)
    X_train_scaled, (X_test_norm_scaled, low_int_scaled, mass_shift_scaled, others_scaled), _ = scale_datasets(
        X_train, X_test_norm, low_int, mass_shift, others
    )

    # Extra artificial mass shifts, built from the normal test spectra
    X_fake_mass_shift = simulate_mass_shifts(X_test_norm_scaled, min_severity=0.2, max_severity=0.8,
                                             seed=RANDOM_STATE)
    mass_shift_combined = np.concatenate((mass_shift_scaled, X_fake_mass_shift), axis=0)

    print("Training data shape:", X_train_scaled.shape)

    if RUN_SANITY_CHECKS:
        sanity_check_dataset("Train (Reference)", X_train_scaled)
        sanity_check_dataset("Test (Reference)", X_test_norm_scaled)
        sanity_check_dataset("Anomalies (Low Intensity)", low_int_scaled)
        sanity_check_dataset("Anomalies (Mass Shift)", mass_shift_combined)
        sanity_check_dataset("Anomalies (Others)", others_scaled)
        visual_check(X_train_scaled, "Train Normal")
        visual_check(mass_shift_combined, "Mass Shift")

    # ====================================
    # 2. AUTOENCODER TRAINING
    # ====================================
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = autoencoder_v4(input_shape=input_shape, output_channels=1)

    # Weights are saved to avoid retraining at every run
    if os.path.exists(WEIGHTS_FILE):
        print("Weights found! Loading the model...")
        model.load_weights(WEIGHTS_FILE)
    else:
        print("No weights found. Starting training...")
        os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
        model.fit(
            X_train_scaled,
            X_train_scaled[:, :, 1:],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test_norm_scaled, X_test_norm_scaled[:, :, 1:]),
            verbose=1,
        )
        model.save_weights(WEIGHTS_FILE)
        print("Weights saved!")

    # ====================================
    # 3. EVALUATION
    # ====================================
    print("Evaluating the model...")

    evaluate_hybrid(X_test_norm_scaled, mass_shift_combined, low_int_scaled, model,
                    mse_threshold=MSE_THRESHOLD, int_threshold=INTENSITY_THRESHOLD)
    evaluate_autoencoder(X_test_norm_scaled, mass_shift_combined, low_int_scaled, model,
                         mse_threshold=MSE_THRESHOLD, int_threshold=INTENSITY_THRESHOLD)


if __name__ == "__main__":
    main()
