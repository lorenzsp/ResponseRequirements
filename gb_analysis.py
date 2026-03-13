# ==================== Imports ====================
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from segwo_utils import (InterpolatedOrbits, compute_strain2x, compute_covariance, compute_violation_ratios)

# Configure JAX for 64-bit precision (required for GW phase accuracy)
jax.config.update("jax_enable_x64", True)

# Add local paths for JaxGB
sys.path.insert(0, str(Path.cwd() / "tests"))
sys.path.insert(0, str(Path.cwd() / "src"))

from jaxgb.jaxgb import JaxGB

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {jax.devices()}")

# ==================== Observation Configuration ====================

# Observation parameters
T_OBS_DAYS = 360                        # Observation time in days
TMAX = T_OBS_DAYS * 24 * 3600          # Observation time in seconds
N_FREQ_BINS = 128                       # Number of frequency bins for heterodyned response
T0 = 0                                  # Start time (seconds)
DF = 1.0 / TMAX                         # Frequency resolution (Hz)

# Search region parameters
F0_CENTER = 4.0e-3                      # Central frequency of search (Hz)
F0_WIDTH = 1e-6                         # Half-width of frequency search region (Hz)

# Amplitude bounds
A_MIN = 1e-23                           # Minimum amplitude (strain)
A_MAX = 1e-22                           # Maximum amplitude (strain)

# Frequency derivative bounds
FDOT_MAX = 1e-15                        # Maximum fdot (Hz/s)

print(f"Observation time: {T_OBS_DAYS:.1f} days ({TMAX:.2e} seconds)")
print(f"Frequency resolution: {DF:.2e} Hz")
print(f"Search region: {F0_CENTER - F0_WIDTH:.6e} - {F0_CENTER + F0_WIDTH:.6e} Hz")
print(f"Amplitude range: {A_MIN:.2e} - {A_MAX:.2e}")

# ==================== LISA Orbits Setup ====================
with h5py.File("processed_trajectories.h5", "r") as ds:
    t_orb_dataset   = ds["t_interp"][()]
    x_orb_dataset   = ds["spacecraft_positions"][()]
    v_orb_dataset   = ds["spacecraft_velocities"][()]
    ltts_dataset    = ds['owlt_12_23_31_13_32_21'][()]

t_orb        = t_orb_dataset
x_orb        = np.median(x_orb_dataset, axis=0)
v_orb        = np.median(v_orb_dataset, axis=0)
ltts_median  = np.median(ltts_dataset,  axis=0)
realizations = x_orb_dataset.shape[0]
N            = realizations
print(f"Number of realizations: {realizations}")
orbits = InterpolatedOrbits(t_orb, x_orb,
                            spacecraft_velocities=v_orb,
                            ltts=ltts_median,
                            interp_order=3)
i = 0
perturbed_orbits = InterpolatedOrbits(
                t_orb_dataset,
                x_orb_dataset[i],
                spacecraft_velocities=v_orb_dataset[i],
                ltts=ltts_dataset[i],
                interp_order=3,
            )

jaxgb_nom = JaxGB(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
jaxgb_perturb  = JaxGB(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)

print(f"JaxGB initialized with n={N_FREQ_BINS} frequency bins")

# ==================== GB signal ====================

# Source 1:
source_params = np.array([
    3.995e-3,                          # f0 (Hz)
    1e-15,                           # fdot (Hz/s) - no evolution
    1e-22,                         # amplitude (strain)
    0.5,                           # ecliptic latitude (rad)
    2.0,                           # ecliptic longitude (rad)
    1.2,                           # polarization (rad)
    0.8,                           # inclination (rad)
    0.3,                           # initial phase (rad)
], dtype=float)

