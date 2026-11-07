"""
Simulation module.

Generates artificial anomalies to complement the real ones.
"""

import numpy as np


def simulate_mass_shifts(normal_spectra, min_severity=0.2, max_severity=0.8, seed=None):
    """
    Apply a random m/z shift (mass shift) to a block of normal spectra.

    Each spectrum is shifted by a random amount between `min_severity` and
    `max_severity`, in a random direction. Returns a new block of the same
    shape containing the artificial anomalies.
    """
    print(f"Simulating mass shifts (shift between {min_severity} and {max_severity} m/z)...")

    rng = np.random.default_rng(seed)
    simulated = np.copy(normal_spectra)
    n_spectra = simulated.shape[0]

    shifts = rng.uniform(min_severity, max_severity, size=n_spectra)
    directions = rng.choice([-1, 1], size=n_spectra)
    simulated[:, :, 0] += (shifts * directions)[:, None]

    print(f"{n_spectra} mass shift anomalies generated!")
    return simulated
