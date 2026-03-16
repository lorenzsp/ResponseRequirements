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
T_OBS_DAYS = 360 * 1                      # Observation time in days
TMAX = T_OBS_DAYS * 24 * 3600          # Observation time in seconds
N_FREQ_BINS = 128                       # Number of frequency bins for heterodyned response
T0 = 0                                  # Start time (seconds)
DF = 1.0 / TMAX                         # Frequency resolution (Hz)

print(f"Observation time: {T_OBS_DAYS:.1f} days ({TMAX:.2e} seconds)")
print(f"Frequency resolution: {DF:.2e} Hz")
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
jaxgb_perturb  = JaxGB(orbits=perturbed_orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)

print(f"JaxGB initialized with n={N_FREQ_BINS} frequency bins")

# ==================== GB signal ====================
import healpy as hp
nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

mismatch = []
f0_vec = np.logspace(-4,0.0, num=20)

for ff in f0_vec:
    print(f"Processing f0 = {ff:.2e} Hz")

    # Source 1:
    source_params = np.array([
        ff,                          # f0 (Hz)
        1e-15,                           # fdot (Hz/s) - no evolution
        1e-22,                         # amplitude (strain)
        0.5,                           # ecliptic latitude (rad)
        2.0,                           # ecliptic longitude (rad)
        0.0,                           # polarization (rad)
        0.5,                           # inclination (rad)
        0.0,                           # initial phase (rad)
    ], dtype=float)
    
    
    source_params = np.repeat(source_params[None, :], len(betas), axis=0)  # Shape (N, 8)
    source_params[:, 3] = betas  # Update latitudes
    source_params[:, 4] = lambs  # Update longitudes
    
    A_nom, E1_nom, T1_nom = jaxgb_nom.get_tdi(jnp.array(source_params),tdi_generation=2.0,tdi_combination="AET")
    A_perturb, E1_perturb, T1_perturb = jaxgb_perturb.get_tdi(jnp.array(source_params),tdi_generation=2.0,tdi_combination="AET")
    kmin = int(np.array(jaxgb_nom.get_kmin(source_params[0,0],source_params[0,1])))

    df = 1.0 / TMAX
    freqs = df * (np.arange(N_FREQ_BINS) + kmin)
    
    cov_AET = compute_covariance(freqs, ltts_median).mean(axis=0)
    inv_cov_AET = np.linalg.inv(cov_AET)
    
    d_nom = np.stack([A_nom, E1_nom, T1_nom], axis=0)
    d_perturb = np.stack([A_perturb, E1_perturb, T1_perturb], axis=0)
    
    # val_per_sky = np.einsum('csf,fcd,dsf->s', d_perturb.conj(), inv_cov_AET, d_nom)
    nom_pert = 4 * np.einsum('csf,fcd,dsf->s', d_perturb.conj(), inv_cov_AET, d_nom).real * df
    nom_nom = 4 * np.einsum('csf,fcd,dsf->s', d_nom.conj(), inv_cov_AET, d_nom).real * df
    pert_pert = 4 * np.einsum('csf,fcd,dsf->s', d_perturb.conj(), inv_cov_AET, d_perturb).real * df
    mismatch.append(1 - nom_pert / (nom_nom * pert_pert)**0.5)

mismatch = np.array(mismatch)
median_m = np.median(mismatch,axis=1)
upper_m = np.percentile(mismatch, 84, axis=1)
lower_m = np.percentile(mismatch, 16, axis=1)

plt.figure(figsize=(3.25, 2.44))
plt.errorbar(f0_vec, median_m, yerr=[median_m - lower_m, upper_m - median_m], fmt='-')
plt.loglog()
plt.xlabel('Frequency $f_0$ [Hz]')
plt.ylabel('Mismatch')
plt.tight_layout()
plt.savefig("gb_mismatch_vs_frequency.png", dpi=300)
plt.show()

# plt.figure(figsize=(10, 4))
# plt.semilogy(freqs, np.abs(A_nom)/np.abs(A_perturb), label='my sum')
# # plt.plot(freqs, np.abs(A_perturb), '--', label='sum_tdi')
# plt.title('Individual Source A Channel Magnitudes')
# plt.xlabel('Frequency Bin')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.show()