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
import matplotlib.pyplot as plt
import h5py
import healpy as hp
import numpy as np
from tqdm import trange

from lisaorbits import StaticConstellation
from lisaconstants import c
from lisaconstants.indexing import LINKS
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

from segwo_utils import (InterpolatedOrbits, compute_strain2x, compute_covariance, compute_violation_ratios, get_static_variation)

np.random.seed(2601)


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------
f = np.logspace(-4, -2., 10)
f = np.append(f, np.logspace(-2., 0., 140))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SEGWO Analysis — Compute Only")
parser.add_argument('--run_flag', type=str, default='static',
                    choices=['static', 'periodic_dev', 'evolving'],
                    help="Orbit type to use.")
parser.add_argument('--time_eval', type=float, default=0.0,
                    help="Evaluation epoch (days); relevant for evolving runs.")
parser.add_argument('--boost_flag', type=int, default=1, choices=[0, 1],
                    help="Whether to include the velocity boost in the response computation (1) or not (0).")
args = parser.parse_args()

run_flag = args.run_flag
array_ltts = np.asarray([args.time_eval * 86400])
boost_flag = float(args.boost_flag)
print(f"run_flag: {run_flag}  |  time_eval: {args.time_eval} days  |  boost_flag: {boost_flag}")

# ---------------------------------------------------------------------------
# Orbit setup
# ---------------------------------------------------------------------------
if run_flag == 'static':
    N = 1000
    orbits = StaticConstellation.from_armlengths(2.5e9, 2.5e9, 2.5e9)

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
velocities = orbits.compute_velocity(t=array_ltts)
print(f"Nominal ltts shape: {ltts.shape}  |  positions shape: {positions.shape}  |  velocities shape: {velocities.shape}")

# ---------------------------------------------------------------------------
# HEALPix sky grid
# ---------------------------------------------------------------------------
nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

# ---------------------------------------------------------------------------
# Nominal strain-to-TDI matrix
# ---------------------------------------------------------------------------
print("Computing nominal strain2x …")
strain2x_nominal = compute_strain2x(f, betas, lambs, ltts, positions, velocities=velocities)
print(f"strain2x_nominal shape: {strain2x_nominal.shape}")
cov_AET = compute_covariance(f, ltts)[:,:,np.newaxis,:,:] # added an axis for sky pixels
print(f"cov_AET shape: {cov_AET.shape}")
strain2x_check_v0 = compute_strain2x(f, betas, lambs, ltts, positions, velocities=velocities*0.0)
strain2x_check_baghi = compute_strain2x(f, betas, lambs, ltts, positions, velocities=None)
check = np.abs(1-np.abs(strain2x_check_v0.conj() * strain2x_check_baghi)/(np.abs(strain2x_check_baghi) * np.abs(strain2x_check_v0)))
print("Test against Baghi:", check[~np.isnan(check)].max(),check[~np.isnan(check)].mean())
rel_diff_resp = np.abs(1-np.abs(strain2x_check_v0.conj() * strain2x_nominal)/(np.abs(strain2x_nominal) * np.abs(strain2x_check_v0)))
print("Relative difference in response from boosted:", rel_diff_resp[~np.isnan(rel_diff_resp)].max(),rel_diff_resp[~np.isnan(rel_diff_resp)].mean())

pol = 0
mismatch_boost = []
for pol in range(2):
    # check shapes for np.linalg.solve
    x = np.linalg.solve(cov_AET, strain2x_nominal[...,pol][...,np.newaxis])[...,0]
    a_star = np.conj(strain2x_check_v0[...,pol])

    A_B = 4 * np.real(np.einsum("ijkl,ijkl->ijk", a_star, x))
    A_A = 4 * np.einsum("ijkl,ijkl->ijk", a_star, np.linalg.solve(cov_AET, strain2x_check_v0[...,pol][...,np.newaxis])[...,0]).real
    B_B = 4 * np.einsum("ijkl,ijkl->ijk", strain2x_nominal[...,pol].conj(), x).real
    mismatch_boost.append(np.abs(1 - A_B / (B_B * A_A)**0.5))
mismatch_boost = np.stack(mismatch_boost,axis=-1)

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
    std_time   = f"{array_ltts[0] / 86400}days_"
    output_dirs = [f"segwo_results/{std_time}"]
else:
    output_dirs = ["segwo_results/"]

# ---------------------------------------------------------------------------
# Main loop over perturbation cases
# ---------------------------------------------------------------------------
for output_dir, params in zip(output_dirs, perturbation_params):
    output_dir += run_flag + "_boost" + str(boost_flag) + "/"
    print("=" * 60)
    print(f"Perturbation params : {params}")
    print(f"Output directory    : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Draw N perturbed realisations ---
    perturbed_ltt       = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))
    perturbed_velocities = np.zeros((N, 3, 3))

    for i in trange(N, desc="Perturbations"):
        if run_flag == 'static':
            po = get_static_variation(
                arm_lengths=params["arm_lengths"],
                armlength_error=params["armlength_error"],
                rotation_error=params["rotation_error"],
                translation_error=params["translation_error"],
            )

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
        perturbed_velocities[i] = po.compute_velocity(t=array_ltts)

    ltt_residuals      = perturbed_ltt - ltts
    position_residuals = perturbed_positions - positions
    velocity_residuals = perturbed_velocities - velocities
    for i in range(6):
        print(f"  ltt std  link {LINKS[i]}: {np.std(ltt_residuals[:, i]) * c:.2e} m")
    for i in range(3):
        print(f"  pos std  SC{i + 1}      : {np.std(position_residuals[:, i]) / 1e3:.2e} km")
    for i in range(3):
        print(f"  vel std  SC{i + 1}      : {np.std(velocity_residuals[:, i]) / 1e3:.2e} km")

    # --- Compute perturbed strain2x and error metrics ---
    print("Computing perturbed strain2x …")
    strain2x_perturbed = compute_strain2x(f, betas, lambs, perturbed_ltt, perturbed_positions, velocities=perturbed_velocities*boost_flag)
    
    print(cov_AET.shape, strain2x_nominal.shape, strain2x_perturbed.shape)
    # --- Compute Mismatch Metric ---
    # a^T C^-1 b = a^T x where x is obtained from C x = b
    pol = 0
    mismatch = []
    for pol in range(2):
        # check shapes for np.linalg.solve
        x = np.linalg.solve(cov_AET, strain2x_nominal[...,pol][...,np.newaxis])[...,0]
        a_star = np.conj(strain2x_perturbed[...,pol])

        A_B = 4 * np.real(np.einsum("ijkl,ijkl->ijk", a_star, x))
        A_A = 4 * np.einsum("ijkl,ijkl->ijk", a_star, np.linalg.solve(cov_AET, strain2x_perturbed[...,pol][...,np.newaxis])[...,0]).real
        B_B = 4 * np.einsum("ijkl,ijkl->ijk", strain2x_nominal[...,pol].conj(), x).real
        mismatch.append(np.abs(1 - A_B / (B_B * A_A)**0.5))
    mismatch = np.stack(mismatch,axis=-1)

    # --- Compute Relative Difference on Amplitude abd Phase Difference Metric ---
    denom = np.sqrt(
        np.abs(strain2x_perturbed * np.conj(strain2x_perturbed))
        * np.abs(strain2x_nominal * np.conj(strain2x_nominal))
    )
    mask = denom > 0.0
    rel_amp = np.zeros_like(strain2x_perturbed, dtype=float)
    phase_diff = np.zeros_like(strain2x_perturbed, dtype=float)
    phase_diff[mask] += np.abs(1-np.real(strain2x_perturbed * np.conj(strain2x_nominal))[mask]/denom[mask])
    rel_amp[mask] += np.abs(np.abs(strain2x_perturbed) - np.abs(strain2x_nominal))[mask] / np.sqrt(denom[mask])
    
    strain2x_abs_error   = np.max(rel_amp, axis=0)   # (F, P, 3, 2)
    strain2x_angle_error = np.max(phase_diff,        axis=0)   # (F, P, 3, 2)

    amp_violation_ratio, phase_violation_ratio = compute_violation_ratios(
        strain2x_abs_error, strain2x_angle_error,
        amp_req=1e-4, phase_req=1e-2,
    )
    # print(f"  Amplitude violation ratio : {amp_violation_ratio:.4f}")
    # print(f"  Phase    violation ratio  : {phase_violation_ratio:.4f}")
    
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
        nom.create_dataset("mismatch_boost", data=mismatch_boost)

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
        err.create_dataset("mismatch", data=mismatch)
        err.attrs["amp_violation_ratio"]   = amp_violation_ratio
        err.attrs["phase_violation_ratio"] = phase_violation_ratio

    print(f"Results saved to {hdf5_path}")

    plt.figure(figsize=(12, 5))
    plt.contourf(np.arange(npix), f, np.log10(mismatch_boost[0,:,:,0]))
    plt.yscale('log')
    plt.colorbar(label='log10 Mismatch due to boost hplus')
    plt.xlabel('Sky pixel index')
    plt.ylabel('Frequency (Hz)')
    plt.title('Mismatch between boosted and non-boosted response')
    
    plt.savefig(os.path.join(output_dir, "mismatch_boost.png"))
    # Keep the legacy text summary for quick inspection
    np.savetxt(
        os.path.join(output_dir, "strain2x_errors.txt"),
        np.array([phase_violation_ratio, amp_violation_ratio]),
        header="Phase violation ratio | Amplitude violation ratio",
    )
