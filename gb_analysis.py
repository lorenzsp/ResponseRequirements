# ==================== Imports ====================
import os
import sys
import tempfile
from pathlib import Path
import healpy as hp

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
from boosted_jaxgb import JaxGBFull

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {jax.devices()}")

# ==================== Observation Configuration ====================

# Observation parameters
T_OBS_DAYS = 365.25 * 1.0                      # Observation time in days
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
i = 5
perturbed_orbits = InterpolatedOrbits(
                t_orb_dataset,
                x_orb_dataset[i],
                spacecraft_velocities=v_orb_dataset[i],
                ltts=ltts_dataset[i],
                interp_order=3,
            )


nonrel_nominal_orbit = JaxGB(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
rel_nominal_orbit = JaxGBFull(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
test_rel_nominal_orbit = JaxGBFull(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
test_rel_nominal_orbit.velocity *= 0.0

nonrel_perturbed_orbit  = JaxGB(orbits=perturbed_orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
rel_perturbed_orbit = JaxGBFull(orbits=perturbed_orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)


template_generators = {}
template_generators["perturbed_vs_nominal_with_nonrel"] = [nonrel_nominal_orbit, nonrel_perturbed_orbit]
template_generators["perturbed_vs_nominal_with_rel"] = [rel_nominal_orbit, rel_perturbed_orbit]

template_generators["nonrel_vs_rel_with_perturbed"] = [nonrel_perturbed_orbit, rel_perturbed_orbit]
template_generators["nonrel_vs_rel_with_nominal"] = [nonrel_nominal_orbit, rel_nominal_orbit]
template_generators["nonrel_vs_rel_test"] = [nonrel_nominal_orbit, test_rel_nominal_orbit]


list_analysis = [key for key in template_generators.keys()]
mismatch = {key : [] for key in list_analysis}
print(list_analysis)

# ==================== GB signal ====================
nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis
f0_vec = np.logspace(-4,0.0, num=10)
output_file = "gb_mismatch_results.h5"

if os.path.exists(output_file):
    mismatch = {}
    with h5py.File(output_file, "r") as f:
        f0_vec = f["f0_vec"][()]
        for key in f.keys():
            if key.startswith("mismatch_"):
                mismatch[key] = f[key][()]
else:
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
        
        for key in list_analysis:
            print(f"  Analyzing {key}...")
            A_nom, E1_nom, T1_nom = template_generators[key][0].get_tdi(jnp.array(source_params), tdi_generation=2.0, tdi_combination="AET")
            A_perturb, E1_perturb, T1_perturb = template_generators[key][1].get_tdi(jnp.array(source_params), tdi_generation=2.0, tdi_combination="AET")
            
            kmin = int(np.array(template_generators[key][0].get_kmin(source_params[0,0],source_params[0,1])))

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
            temp_mism = 1 - nom_pert / (nom_nom * pert_pert)**0.5
            mismatch[key].append(temp_mism)
            print(f"    Mismatch: {temp_mism.mean():.2e} (mean), {temp_mism.min():.2e} (min), {temp_mism.max():.2e} (max)")
    
    for key in list_analysis:
        mismatch[key] = np.asarray(mismatch[key])
    
    # ==================== Save Results ====================
    with h5py.File(output_file, "w") as f:
        f.create_dataset("f0_vec", data=f0_vec)
        for key in mismatch.keys():
            f.create_dataset(f"mismatch_{key}", data=mismatch[key])
    print(f"Results saved to {output_file}")

plt.figure()
for key in mismatch.keys():
    if 'test' in key:
        continue
    plt.errorbar(f0_vec, mismatch[key].mean(axis=1), yerr=[mismatch[key].min(axis=1), mismatch[key].max(axis=1)], label=key, fmt='o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mismatch")
plt.title("Mismatch between GB templates")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("gb_mismatch_plot.png")
plt.show()

