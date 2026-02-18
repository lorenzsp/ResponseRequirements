import h5py
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits import StaticConstellation, ResampledOrbits, InterpolatedOrbits

from lisaconstants import c
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from segwo_utils import *
import os
from perturbation_utils import get_static_variation, create_orbit_with_periodic_dev, plot_orbit_3d
import sys
import argparse

A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# Choices to run
# Define a frequency axis
N = 10
# random time array for testing
array_ltts = np.asarray([0.0])
array_ltts = np.random.uniform(0, 365*86400, size=1)  # 100 random times over a year
array_ltts = np.asarray([0.0])  # test at half a year

f = np.logspace(-4, 0., 100)

parser = argparse.ArgumentParser(description="SEGWO Analysis")
parser.add_argument('--run_flag', type=str, default='static', choices=['static', 'periodic_dev', 'waldemar'],
                    help="Type of run: static, periodic_dev, or waldemar")
args = parser.parse_args()
run_flag = args.run_flag

print(f"run_flag set to: {run_flag}")

if run_flag == 'static':
    # Define static combination with equal arms
    orbits = StaticConstellation.from_armlengths(2.5e9, 2.5e9, 2.5e9)

if run_flag == 'periodic_dev':
    # periodic deviation
    orbits = create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=False, 
                                                        armlength_error=0.0, 
                                                        rotation_error=0.0, 
                                                        translation_error=0.0, 
                                                        period=15*86400, equal_armlength=False)
    t_orb = orbits.t
    x_orb = orbits.x
    v_orb = orbits.v
    print(t_orb.shape, x_orb.shape, v_orb.shape)
    distance = np.linalg.norm(x_orb[:,0] - x_orb[:,1],axis=-1)/1e9
    print(f"Distance between SC1 and SC2: {distance.mean()} [m], std: {distance.std()} [Mm]")
    orbits = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)

if run_flag == 'waldemar':
    # waldemar's orbits
    with h5py.File("processed_trajectories.h5", "r") as dset:
        t_orb_dataset = dset["t_interp"][()]
        x_orb_dataset = dset["spacecraft_positions"][()]
        v_orb_dataset = dset["spacecraft_velocities"][()]
    
    t_orb = t_orb_dataset
    x_orb = np.median(x_orb_dataset, axis=0)  # Use the median over all realizations
    v_orb = np.median(v_orb_dataset, axis=0)  # Use the median over all realizations
    print(t_orb.shape, x_orb.shape, v_orb.shape)
    distance = np.linalg.norm(x_orb[:,0] - x_orb[:,1],axis=-1)/1e9
    print(f"Distance between SC1 and SC2: {distance.mean()} [m], std: {distance.std()} [Mm]")
    realizations = x_orb_dataset.shape[0]
    print(f"Number of realizations: {realizations}")
    orbits = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)

# Compute the positions of the spacecraft and the light travel times at t=0.0
ltts = orbits.compute_ltt(t=array_ltts)
print("Light travel times shape", ltts.shape)
positions = orbits.compute_position(t=array_ltts)
print("Positions shape", positions.shape)
# Compute the signal covariance matrix for eta variables

# We use healpy to define a grid of points on the sky
nside = 6

npix = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
# Conversion from colatitude to latitude
betas, lambs = np.pi / 2 - thetas, phis

# Example usage
strain2x = compute_strain2x(f, betas, lambs, ltts, positions, orbits, A, E, T)

##########################
# Generalized Error Analysis
##########################
output_dirs = ["segwo_results/rotations/", "segwo_results/translations/", "segwo_results/armlengths/", "segwo_results/all/"]

# Define perturbation parameters for each case
perturbation_params = [
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 0, "rotation_error": 5e3, "translation_error": 0},
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 0, "rotation_error": 0, "translation_error": 50e3},
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 1, "rotation_error": 0, "translation_error": 0},
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 1, "rotation_error": 5e3, "translation_error": 50e3},
]

if run_flag == 'waldemar':
    perturbation_params = perturbation_params[:1]  # only do one realization for waldemar's orbits, since they already include different variations
    output_dirs = ['segwo_results/']
    N = realizations - 1

for output_dir, params in zip(output_dirs, perturbation_params):
    output_dir += run_flag + "/"
    print("="*40)
    print("Perturbation parameters:", params)
    print(f"Running analysis for {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    perturbed_ltt = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))
    
    for i in range(0,N):
        print(f"  Perturbation {i+1}/{N}")
        # old: perturbed_orbit = perturbed_static_orbits(**params)
        
        if run_flag == 'static':
            # new static variation
            perturbed_orbit = get_static_variation(arm_lengths=params["arm_lengths"], armlength_error=params["armlength_error"], rotation_error=params["rotation_error"], translation_error=params["translation_error"])
        
        if run_flag == 'periodic_dev':
            # periodic deviation
            perturbed_orbit = create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=False, 
                                                            armlength_error=params["armlength_error"], 
                                                            rotation_error=params["rotation_error"], 
                                                            translation_error=params["translation_error"], 
                                                            period=15*86400, equal_armlength=False)
            t_orb = perturbed_orbit.t
            x_orb = perturbed_orbit.x
            v_orb = perturbed_orbit.v
            perturbed_orbit = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)
        
        if run_flag == 'waldemar':
            # waldemar's orbits
            perturbed_orbit = InterpolatedOrbits(t_orb_dataset, x_orb_dataset[i], v_orb_dataset[i], interp_order=3)
        
        perturbed_ltt[i] = perturbed_orbit.compute_ltt(t=array_ltts)
        perturbed_positions[i] = perturbed_orbit.compute_position(t=array_ltts)
    
    ltt_residuals = perturbed_ltt - ltts
    for i in range(6):
        print(f"ltt std for link {orbits.LINKS[i]}", np.std(ltt_residuals[:, i])*c, "meters")
    # create histograms of the residuals
    plt.figure()
    for i in range(6):
        plt.hist(ltt_residuals[:, i], bins=20, alpha=0.5, label=f"Link {orbits.LINKS[i]}")
    plt.xlabel("Light travel time residuals [s]")
    plt.ylabel("Count")
    plt.title("Histogram of light travel time residuals")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ltt_residuals_histogram.png"))
    plt.close()
    
    position_residuals = perturbed_positions - positions
    for i in range(3):
        print(f"Position std for sc {i}:", np.std(position_residuals[:,i])/1e3, "kilometers")
    # create histograms of the position residuals
    plt.figure()
    for i in range(3):
        plt.hist(position_residuals[:, i, 0]/1e3, bins=20, alpha=0.5, label=f"SC {i} x")
        plt.hist(position_residuals[:, i, 1]/1e3, bins=20, alpha=0.5, label=f"SC {i} y")
        plt.hist(position_residuals[:, i, 2]/1e3, bins=20, alpha=0.5, label=f"SC {i} z")
    plt.xlabel("Position residuals (kilometers)")
    plt.ylabel("Count")
    plt.title("Histogram of position residuals")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "position_residuals_histogram.png"))
    plt.close()
    
    strain2x_perturbed = compute_strain2x(f, betas, lambs, perturbed_ltt, perturbed_positions, orbits, A, E, T)
    rel_err_sky = relative_errors_sky(np.abs(strain2x_perturbed), np.abs(strain2x))
    thr = 1e-10
    # mask where the denominator is zero
    mask = np.abs(strain2x[0]) < thr
    # count the number zeros
    count_zeros = np.sum(mask)
    print(f"Number of zeros in strain2x: {count_zeros}")
    rel_err_sky[:,mask] = 0.0
    mask = np.abs(strain2x_perturbed) < thr
    # count the number of zeros
    count_zeros = np.sum(mask)
    print(f"Number of zeros in strain2x_perturbed: {count_zeros}")
    rel_err_sky[mask] = 0.0
    
    # the following is proportional to exp(i * delta phi)
    product = strain2x * np.conj(strain2x_perturbed) / np.sqrt(np.abs(strain2x_perturbed * np.conj(strain2x_perturbed)) * np.abs(strain2x * np.conj(strain2x)))
    mism = np.abs(1 - product)
    mask = np.abs(strain2x[0]) < thr
    mism[:,mask] = 0.0
    mask = np.abs(strain2x_perturbed) < thr
    mism[mask] = 0.0

    # check for NaN values
    if np.isnan(rel_err_sky).any():
        print("NaN values in strain2x_abs_error")
        # remove NaN values
        rel_err_sky = np.nan_to_num(rel_err_sky, posinf=0)

    if np.isnan(mism).any():
        print("NaN values in strain2x_angle_error")
        # remove NaN values
        mism = np.nan_to_num(mism, posinf=0)
    
    strain2x_abs_error = np.max(rel_err_sky, axis=0)
    # abs_phase_err = absolute_errors(np.angle(strain2x_perturbed), np.angle(strain2x))
    # strain2x_angle_error = np.max(abs_phase_err, axis=0)
    strain2x_angle_error = np.max(mism, axis=0)
    
    
    # Save the plots
    plot_strain_errors(
        f, strain2x_abs_error, strain2x_angle_error, 
        output_file=os.path.join(output_dir, "strain2x_errors_frequency.png")
    )

    plot_gw_response_maps(
        strain2x_abs_error, f, npix, 
        folder=output_dir + 'amplitude_errors',
    )

    plot_gw_response_maps(
        strain2x_angle_error, f, npix, 
        folder=output_dir + 'phase_errors',
    )

    # Compute violation ratios
    amp_violation_ratio, phase_violation_ratio = compute_violation_ratios(
        strain2x_abs_error, strain2x_angle_error, amp_req=1e-4, phase_req=1e-2
    )

    # Print the results
    print(f"{output_dir} - Amplitude Violation Ratio:", amp_violation_ratio)
    print(f"{output_dir} - Phase Violation Ratio:", phase_violation_ratio)
    np.savetxt(
        os.path.join(output_dir, "strain2x_errors.txt"), 
        np.array([phase_violation_ratio, amp_violation_ratio]),
        header="Max relative error (amplitude) | Max absolute error (phase)"
    )
