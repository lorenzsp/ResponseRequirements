"""
Compute-only script: response evolution over an orbital segment.

Loads trajectory realisations, computes the strain-to-TDI mixing matrix for
the median (nominal) orbit and for realization 1 at each time point, then
saves everything to an HDF5 file.  No figures are produced here;
run plot_evolution.py afterwards.

Usage
-----
    python run_evolution.py
"""

import os

import h5py
import healpy as hp
import numpy as np
from lisaorbits import InterpolatedOrbits
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

from segwo_utils import compute_strain2x

np.random.seed(2601)

# ---------------------------------------------------------------------------
# TDI combinations
# ---------------------------------------------------------------------------
A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Base logspace grid
f_base = np.logspace(-4, 0., 300)
# Extra frequencies used in the time-evolution plot (may not land on logspace grid)
f_extra = np.array([1e-3, 5e-3, 1e-2, 5e-2])
f = np.unique(np.sort(np.concatenate([f_base, f_extra])))

# Time axis: 120 points over one month (~1/12 year)
array_ltts = np.linspace(0, 365 * 86400 / 12, 120)

# HEALPix sky grid
nside = 6
npix  = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

output_dir = "segwo_results/"
os.makedirs(output_dir, exist_ok=True)
hdf5_path = os.path.join(output_dir, "evolution_data.h5")

# ---------------------------------------------------------------------------
# Load trajectories
# ---------------------------------------------------------------------------
print("Loading trajectories from processed_trajectories.h5 …")
with h5py.File("processed_trajectories.h5", "r") as ds:
    t_orb_dataset = ds["t_interp"][()]
    x_orb_dataset = ds["spacecraft_positions"][()]
    v_orb_dataset = ds["spacecraft_velocities"][()]

t_orb        = t_orb_dataset
x_orb_median = np.median(x_orb_dataset, axis=0)
v_orb_median = np.median(v_orb_dataset, axis=0)
realizations = x_orb_dataset.shape[0]
print(f"  Realizations : {realizations}")
print(f"  t shape      : {t_orb.shape}")
print(f"  x shape      : {x_orb_median.shape}")

dist = np.linalg.norm(x_orb_median[:, 0] - x_orb_median[:, 1], axis=-1) / 1e9
print(f"  SC1–SC2 distance: mean={dist.mean():.3f} Gm, std={dist.std():.3f} Gm")

# ---------------------------------------------------------------------------
# Nominal (median) orbit
# ---------------------------------------------------------------------------
print("Building nominal orbits …")
orbits_nominal = InterpolatedOrbits(t_orb, x_orb_median, v_orb_median, interp_order=3)
ltts_nom      = orbits_nominal.compute_ltt(t=array_ltts)
positions_nom = orbits_nominal.compute_position(t=array_ltts)
print(f"  ltts shape : {ltts_nom.shape}  |  positions shape : {positions_nom.shape}")

print("Computing strain2x for nominal orbit …")
strain2x_nominal = compute_strain2x(f, betas, lambs, ltts_nom, positions_nom,
                                    orbits_nominal, A, E, T)
print(f"  strain2x_nominal shape: {strain2x_nominal.shape}")

# ---------------------------------------------------------------------------
# Realization 1
# ---------------------------------------------------------------------------
print("Building realization-1 orbits …")
orbits_real1   = InterpolatedOrbits(t_orb, x_orb_dataset[1], v_orb_dataset[1], interp_order=3)
ltts_real1     = orbits_real1.compute_ltt(t=array_ltts)
positions_real1 = orbits_real1.compute_position(t=array_ltts)

print("Computing strain2x for realization 1 …")
strain2x_real1 = compute_strain2x(f, betas, lambs, ltts_real1, positions_real1,
                                  orbits_real1, A, E, T)
print(f"  strain2x_real1 shape: {strain2x_real1.shape}")

# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------
print(f"Saving to {hdf5_path} …")
with h5py.File(hdf5_path, "w") as hf:
    hf.create_dataset("frequencies",  data=f)
    hf.create_dataset("time_points",  data=array_ltts)
    hf.create_dataset("betas",        data=betas)
    hf.create_dataset("lambs",        data=lambs)
    hf.attrs["nside"] = nside
    hf.attrs["npix"]  = npix

    nom = hf.create_group("nominal")
    nom.create_dataset("ltts",           data=ltts_nom)
    nom.create_dataset("positions",      data=positions_nom)
    nom.create_dataset("strain2x_real",  data=np.real(strain2x_nominal))
    nom.create_dataset("strain2x_imag",  data=np.imag(strain2x_nominal))

    r1 = hf.create_group("realization_1")
    r1.create_dataset("ltts",            data=ltts_real1)
    r1.create_dataset("positions",       data=positions_real1)
    r1.create_dataset("strain2x_real",   data=np.real(strain2x_real1))
    r1.create_dataset("strain2x_imag",   data=np.imag(strain2x_real1))

size_gb = (strain2x_nominal.nbytes + strain2x_real1.nbytes) / 1e9
print(f"Results saved to {hdf5_path}  (approx. {size_gb:.2f} GB complex data)")
