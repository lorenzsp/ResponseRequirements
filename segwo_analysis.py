import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits import StaticConstellation
from lisaconstants import c
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from segwo.response import compute_strain2link
from segwo.cov import construct_mixing_from_pytdi, compose_mixings
from segwo_utils import *
import os

A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# Define a frequency axis
f = np.logspace(-4, 0., 500)

# Define static combination with equal arms
orbits = StaticConstellation.from_armlengths(2.5e9, 2.5e9, 2.5e9)

# Compute the positions of the spacecraft and the light travel times at t=0.0
ltts = orbits.compute_ltt(t=[0.0])
positions = orbits.compute_position(t=[0.0])

# Compute the signal covariance matrix for eta variables

# We use healpy to define a grid of points on the sky
nside = 14

npix = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
# Conversion from colatitude to latitude
betas, lambs = np.pi / 2 - thetas, phis

# Example usage
strain2x = compute_strain2x(f, betas, lambs, ltts, positions, orbits, A, E, T)

##########################
# Generalized Error Analysis
##########################
output_dirs = ["rotations/", "translations/", "armlengths/"]
N = 100

# Define perturbation parameters for each case
perturbation_params = [
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 0, "rotation_error": 5e3, "translation_error": 0},
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 0, "rotation_error": 0, "translation_error": 50e3},
    {"arm_lengths": [2.5e9, 2.5e9, 2.5e9], "armlength_error": 1, "rotation_error": 0, "translation_error": 0},
]

for output_dir, params in zip(output_dirs, perturbation_params):
    print(f"Running analysis for {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    perturbed_ltt = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))

    for i in range(N):
        perturbed_orbit = perturbed_static_orbits(**params)
        perturbed_ltt[i] = perturbed_orbit.compute_ltt(t=[0.0])
        perturbed_positions[i] = perturbed_orbit.compute_position(t=[0.0])

    strain2x_perturbed = compute_strain2x(f, betas, lambs, ltts, perturbed_positions, orbits, A, E, T)
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
