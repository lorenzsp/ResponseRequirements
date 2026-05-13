# ==================== Test: JaxGB vs JaxGBFull (non-relativistic) ====================
"""
Self-contained script to test mismatch between:
  - nonrel_nominal_orbit_old: JaxGB (old model)
  - nonrel_nominal_orbit: JaxGBFull (new model)
Both use non-relativistic orbits (v=0).
"""

import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from segwo_utils import InterpolatedOrbits, compute_covariance

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path.cwd() / "tests"))
sys.path.insert(0, str(Path.cwd() / "src"))

from jaxgb.jaxgb import JaxGB
from boosted_jaxgb import JaxGBFull

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {jax.devices()}")

# ==================== Configuration ====================

T_OBS_DAYS  = 15.0
TMAX        = T_OBS_DAYS * 24 * 3600
N_FREQ_BINS = 128
T0          = 0.0
DF          = 1.0 / TMAX

print(f"Observation time: {T_OBS_DAYS:.1f} days ({TMAX:.2e} seconds)")
print(f"Frequency resolution: {DF:.2e} Hz")

# ==================== Load orbits ====================

with h5py.File("data/processed_trajectories.h5", "r") as ds:
    t_orb_dataset = ds["t_interp"][()]
    x_orb_dataset = ds["spacecraft_positions"][()]
    v_orb_dataset = ds["spacecraft_velocities"][()]
    ltts_dataset  = ds["owlt_12_23_31_13_32_21"][()]

    mask_time     = t_orb_dataset < TMAX * 1.1
    t_orb_dataset = t_orb_dataset[mask_time]
    x_orb_dataset = x_orb_dataset[:, mask_time, :]
    v_orb_dataset = v_orb_dataset[:, mask_time, :]
    ltts_dataset  = ltts_dataset[:, mask_time]

t_orb       = t_orb_dataset
x_orb       = np.median(x_orb_dataset, axis=0)
v_orb       = np.median(v_orb_dataset, axis=0)
ltts_median = np.median(ltts_dataset, axis=0)

print(f"Number of orbit realizations: {x_orb_dataset.shape[0]}")

# Non-relativistic orbits (v=0)
orbits_v0 = InterpolatedOrbits(t_orb, x_orb,
                               spacecraft_velocities=v_orb * 0.0,
                               ltts=ltts_median,
                               interp_order=3)

# ==================== Create models ====================

print("\nCreating models...")
nonrel_nominal_orbit_old = JaxGB(orbits=orbits_v0, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
nonrel_nominal_orbit = JaxGBFull(orbits=orbits_v0, t_obs=TMAX, t0=T0, n=N_FREQ_BINS, old_implementation=True)

print(f"  Old model (JaxGB):      {type(nonrel_nominal_orbit_old).__name__}")
print(f"  New model (JaxGBFull):  {type(nonrel_nominal_orbit).__name__}")

# ==================== Sky grid ====================

nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

print(f"\nSky grid: nside={nside}, npix={npix}")

# ==================== Test across frequencies ====================

f0_vec = np.logspace(-4, 0.0, num=10)
n_freq = len(f0_vec)

results = {
    "f0_vec": f0_vec,
    "mismatch_mean": np.zeros(n_freq),
    "mismatch_min": np.zeros(n_freq),
    "mismatch_max": np.zeros(n_freq),
}

print(f"\n{'='*70}")
print(f"Testing mismatch: JaxGB vs JaxGBFull (non-relativistic)")
print(f"{'='*70}\n")

for i_f, f0 in enumerate(f0_vec):
    print(f"[{i_f+1}/{n_freq}]  f0 = {f0:.4e} Hz")

    # Source parameters for all sky locations
    source_params = np.array([
        f0,                          # f0 (Hz)
        0.0,                         # fdot (Hz/s)
        1e-20,                       # amplitude (strain)
        0.5,                         # beta (placeholder)
        2.0,                         # lambda (placeholder)
        np.pi/3,                     # psi
        np.pi/3,                     # iota
        np.pi/3,                     # phi0
    ], dtype=float)

    # Replicate across sky and update beta, lambda
    source_params = np.repeat(source_params[None, :], npix, axis=0)
    source_params[:, 3] = betas
    source_params[:, 4] = lambs

    # Get frequency bins
    kmin = int(np.array(nonrel_nominal_orbit_old.get_kmin(f0, 0.0)))
    kmax = kmin + N_FREQ_BINS
    freqs = DF * (np.arange(N_FREQ_BINS) + kmin)

    # Covariance
    cov_AET = compute_covariance(freqs, ltts_median).mean(axis=0)
    inv_cov_AET = np.linalg.inv(cov_AET)

    # Generate TDI with old model
    A_old, E_old, T_old = nonrel_nominal_orbit_old.get_tdi(
        jnp.array(source_params), tdi_generation=2.0, tdi_combination="AET"
    )
    d_old = np.stack([A_old, E_old, T_old], axis=1)  # (npix, 3, N_FREQ_BINS)

    # Generate TDI with new model
    A_new, E_new, T_new = nonrel_nominal_orbit.get_tdi(
        jnp.array(source_params), tdi_generation=2.0, tdi_combination="AET"
    )
    d_new = np.stack([A_new, E_new, T_new], axis=1)  # (npix, 3, N_FREQ_BINS)

    # Compute mismatch
    mismatch = np.asarray(
        nonrel_nominal_orbit.mismatch(d_old, d_new, inv_cov_AET, maximise_phase=False)
    )

    results["mismatch_mean"][i_f] = mismatch.mean()
    results["mismatch_min"][i_f] = mismatch.min()
    results["mismatch_max"][i_f] = mismatch.max()

    print(f"  Mismatch: mean={mismatch.mean():.2e}, min={mismatch.min():.2e}, max={mismatch.max():.2e}\n")

# ==================== Summary & Plot ====================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'f0 (Hz)':>12}  {'Mismatch (mean)':>16}  {'Mismatch (min)':>16}  {'Mismatch (max)':>16}")
print("-"*70)
for i_f in range(n_freq):
    print(
        f"{f0_vec[i_f]:>12.4e}  "
        f"{results['mismatch_mean'][i_f]:>16.4e}  "
        f"{results['mismatch_min'][i_f]:>16.4e}  "
        f"{results['mismatch_max'][i_f]:>16.4e}"
    )

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.fill_between(f0_vec, results["mismatch_min"], results["mismatch_max"], alpha=0.3, label="min-max range")
ax.plot(f0_vec, results["mismatch_mean"], "-o", linewidth=2, markersize=6, label="mean")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency $f_0$ (Hz)", fontsize=12)
ax.set_ylabel("Mismatch (JaxGB vs JaxGBFull)", fontsize=12)
ax.set_title("Mismatch between Old and New Non-Relativistic Models", fontsize=13)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
out_fig = "test_nonrel_models_mismatch.png"
plt.savefig(out_fig, dpi=150)
print(f"\nPlot saved to {out_fig}")

# Save data
out_h5 = "test_nonrel_models_mismatch.h5"
with h5py.File(out_h5, "w") as f:
    for key, val in results.items():
        f.create_dataset(key, data=val)
    f.attrs["description"] = "Mismatch test: JaxGB (old) vs JaxGBFull (new), non-relativistic"
    f.attrs["T_OBS_DAYS"] = T_OBS_DAYS

print(f"Data saved to {out_h5}")
