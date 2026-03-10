"""
Compute-only script: perturbation analysis for the SEGWO response.

Loads or generates orbits, draws N perturbed realisations, computes the
strain-to-TDI mixing matrix and its errors, then saves everything to an
HDF5 file.  No figures are produced here; run plot_analysis.py afterwards.

Usage
-----
    python run_analysis.py --run_flag static
    python run_analysis.py --run_flag evolving --time_eval 15
    python run_analysis.py --run_flag periodic_dev --time_eval 0
"""

import argparse
import os

import h5py
import healpy as hp
import numpy as np
from tqdm import trange

from lisaorbits import StaticConstellation
from lisaconstants import c
from lisaconstants.indexing import LINKS
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

from segwo_utils import (InterpolatedOrbits, compute_strain2x,
                         relative_errors_sky, compute_violation_ratios)
from perturbation_utils import get_static_variation, create_orbit_with_periodic_dev

np.random.seed(2601)

# ---------------------------------------------------------------------------
# TDI combinations
# ---------------------------------------------------------------------------
A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------
f = np.logspace(-4, 0., 100)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SEGWO Analysis — Compute Only")
parser.add_argument('--run_flag', type=str, default='static',
                    choices=['static', 'periodic_dev', 'evolving'],
                    help="Orbit type to use.")
parser.add_argument('--time_eval', type=float, default=0.0,
                    help="Evaluation epoch (days); relevant for evolving runs.")
args = parser.parse_args()

run_flag = args.run_flag
array_ltts = np.asarray([args.time_eval * 86400])
print(f"run_flag: {run_flag}  |  time_eval: {args.time_eval} days")

# ---------------------------------------------------------------------------
# Orbit setup
# ---------------------------------------------------------------------------
if run_flag == 'static':
    N = 1000
    orbits = StaticConstellation.from_armlengths(2.5e9, 2.5e9, 2.5e9)

if run_flag == 'periodic_dev':
    N = 10
    _orb = create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=False,
                                          armlength_error=0.0,
                                          rotation_error=0.0,
                                          translation_error=0.0,
                                          period=15 * 86400,
                                          equal_armlength=False)
    t_orb, x_orb, v_orb = _orb.t, _orb.x, _orb.v
    orbits = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)

if run_flag == 'evolving':
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

# ---------------------------------------------------------------------------
# Nominal orbit quantities
# ---------------------------------------------------------------------------
ltts      = orbits.compute_ltt(t=array_ltts)
positions = orbits.compute_position(t=array_ltts)
print(f"Nominal ltts shape: {ltts.shape}  |  positions shape: {positions.shape}")

# ---------------------------------------------------------------------------
# HEALPix sky grid
# ---------------------------------------------------------------------------
nside = 6
npix  = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

# ---------------------------------------------------------------------------
# Nominal strain-to-TDI matrix
# ---------------------------------------------------------------------------
print("Computing nominal strain2x …")
strain2x_nominal = compute_strain2x(f, betas, lambs, ltts, positions, orbits, A, E, T)
print(f"strain2x_nominal shape: {strain2x_nominal.shape}")

# ---------------------------------------------------------------------------
# Perturbation parameters (one case; extend the list to run multiple)
# ---------------------------------------------------------------------------
perturbation_params = [
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9],
     "armlength_error": 1,
     "rotation_error": 5e3,
     "translation_error": 50e3},
]

if run_flag == 'evolving':
    perturbation_params = perturbation_params[:1]
    std_time   = f"{array_ltts[0] / 86400}days"
    output_dirs = [f"segwo_results/{std_time}"]
else:
    output_dirs = ["segwo_results/"]

# ---------------------------------------------------------------------------
# Main loop over perturbation cases
# ---------------------------------------------------------------------------
for output_dir, params in zip(output_dirs, perturbation_params):
    output_dir += run_flag + "/"
    print("=" * 60)
    print(f"Perturbation params : {params}")
    print(f"Output directory    : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Draw N perturbed realisations ---
    perturbed_ltt       = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))

    for i in trange(N, desc="Perturbations"):
        if run_flag == 'static':
            po = get_static_variation(
                arm_lengths=params["arm_lengths"],
                armlength_error=params["armlength_error"],
                rotation_error=params["rotation_error"],
                translation_error=params["translation_error"],
            )

        if run_flag == 'periodic_dev':
            _po = create_orbit_with_periodic_dev(
                fpath="new_orbits.h5", use_gpu=False,
                armlength_error=params["armlength_error"],
                rotation_error=params["rotation_error"],
                translation_error=params["translation_error"],
                period=15 * 86400, equal_armlength=False,
            )
            po = InterpolatedOrbits(_po.t, _po.x, _po.v, interp_order=3)

        if run_flag == 'evolving':
            po = InterpolatedOrbits(
                t_orb_dataset,
                x_orb_dataset[i],
                spacecraft_velocities=v_orb_dataset[i],
                ltts=ltts_dataset[i],
                interp_order=3,
            )

        perturbed_ltt[i]       = po.compute_ltt(t=array_ltts)
        perturbed_positions[i] = po.compute_position(t=array_ltts)

    ltt_residuals      = perturbed_ltt - ltts
    position_residuals = perturbed_positions - positions

    for i in range(6):
        print(f"  ltt std  link {LINKS[i]}: {np.std(ltt_residuals[:, i]) * c:.2f} m")
    for i in range(3):
        print(f"  pos std  SC{i + 1}      : {np.std(position_residuals[:, i]) / 1e3:.2f} km")

    # --- Compute perturbed strain2x and error metrics ---
    print("Computing perturbed strain2x …")
    strain2x_perturbed = compute_strain2x(
        f, betas, lambs, perturbed_ltt, perturbed_positions, orbits, A, E, T
    )

    thr = 1e-12

    rel_err_sky = relative_errors_sky(np.abs(strain2x_perturbed), np.abs(strain2x_nominal))
    rel_err_sky[:, np.abs(strain2x_nominal[0]) < thr] = 0.0
    rel_err_sky[np.abs(strain2x_perturbed) < thr] = 0.0

    denom = np.sqrt(
        np.abs(strain2x_perturbed * np.conj(strain2x_perturbed))
        * np.abs(strain2x_nominal * np.conj(strain2x_nominal))
    )
    product = np.where(denom > thr,
                       strain2x_nominal * np.conj(strain2x_perturbed) / denom,
                       1.0 + 0j)
    mism = np.abs(1 - product)
    mism[:, np.abs(strain2x_nominal[0]) < thr] = 0.0
    mism[np.abs(strain2x_perturbed) < thr] = 0.0

    rel_err_sky = np.nan_to_num(rel_err_sky, nan=0.0, posinf=0.0)
    mism        = np.nan_to_num(mism,        nan=0.0, posinf=0.0)

    strain2x_abs_error   = np.max(rel_err_sky, axis=0)   # (F, P, 3, 2)
    strain2x_angle_error = np.max(mism,        axis=0)   # (F, P, 3, 2)

    amp_violation_ratio, phase_violation_ratio = compute_violation_ratios(
        strain2x_abs_error, strain2x_angle_error,
        amp_req=1e-4, phase_req=1e-2,
    )
    print(f"  Amplitude violation ratio : {amp_violation_ratio:.4f}")
    print(f"  Phase    violation ratio  : {phase_violation_ratio:.4f}")

    # --- Save to HDF5 ---
    hdf5_path = os.path.join(output_dir, "results.h5")
    print(f"Saving results to {hdf5_path} …")
    with h5py.File(hdf5_path, "w") as hf:
        # Metadata
        meta = hf.create_group("metadata")
        meta.attrs["run_flag"]         = run_flag
        meta.attrs["time_eval_days"]   = args.time_eval
        meta.attrs["nside"]            = nside
        meta.attrs["npix"]             = npix
        meta.attrs["N_perturbations"]  = N
        meta.attrs["armlength_error"]  = params["armlength_error"]
        meta.attrs["rotation_error"]   = params["rotation_error"]
        meta.attrs["translation_error"]= params["translation_error"]
        meta.create_dataset("frequencies", data=f)
        meta.create_dataset("betas",       data=betas)
        meta.create_dataset("lambs",       data=lambs)

        # Nominal
        nom = hf.create_group("nominal")
        nom.create_dataset("ltts",           data=ltts)
        nom.create_dataset("positions",      data=positions)
        nom.create_dataset("strain2x_real",  data=np.real(strain2x_nominal))
        nom.create_dataset("strain2x_imag",  data=np.imag(strain2x_nominal))

        # Perturbed samples
        pert = hf.create_group("perturbed")
        pert.create_dataset("ltts",               data=perturbed_ltt)
        pert.create_dataset("positions",          data=perturbed_positions)
        pert.create_dataset("ltt_residuals",      data=ltt_residuals)
        pert.create_dataset("position_residuals", data=position_residuals)

        # Error metrics
        err = hf.create_group("errors")
        err.create_dataset("strain2x_abs_error",   data=strain2x_abs_error)
        err.create_dataset("strain2x_angle_error", data=strain2x_angle_error)
        err.attrs["amp_violation_ratio"]   = amp_violation_ratio
        err.attrs["phase_violation_ratio"] = phase_violation_ratio

    print(f"Results saved to {hdf5_path}")

    # Keep the legacy text summary for quick inspection
    np.savetxt(
        os.path.join(output_dir, "strain2x_errors.txt"),
        np.array([phase_violation_ratio, amp_violation_ratio]),
        header="Phase violation ratio | Amplitude violation ratio",
    )
