"""
Plot script: loads results produced by run_analysis.py and generates all figures.

No computation is done here beyond what is needed to produce the plots.

Usage
-----
    python plot_analysis.py --results_dir segwo_results/static/
    python plot_analysis.py --results_dir segwo_results/15.0daysevolving/
    python plot_analysis.py --results_dir segwo_results/0.0daysevolving/
"""

import argparse
import os

import h5py
import numpy as np

from plot_utils import (
    plot_response,
    plot_strain_errors,
    plot_gw_response_maps,
    plot_ltt_residuals_histogram,
    plot_position_residuals_histogram,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SEGWO Analysis — Plot Only")
parser.add_argument('--results_dir', type=str, required=True,
                    help="Directory containing results.h5 (output of run_analysis.py).")
args = parser.parse_args()

output_dir = args.results_dir.rstrip('/') + '/'
hdf5_path  = os.path.join(output_dir, "results.h5")

if not os.path.exists(hdf5_path):
    raise FileNotFoundError(
        f"Results file not found: {hdf5_path}\n"
        "Run run_analysis.py first."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading results from {hdf5_path} …")
with h5py.File(hdf5_path, "r") as hf:
    f         = hf["metadata/frequencies"][()]
    betas     = hf["metadata/betas"][()]
    lambs     = hf["metadata/lambs"][()]
    nside     = int(hf["metadata"].attrs["nside"])
    npix      = int(hf["metadata"].attrs["npix"])
    run_flag  = str(hf["metadata"].attrs["run_flag"])

    strain2x_nominal = (hf["nominal/strain2x_real"][()] +
                        1j * hf["nominal/strain2x_imag"][()])

    ltt_residuals      = hf["perturbed/ltt_residuals"][()]
    position_residuals = hf["perturbed/position_residuals"][()]

    strain2x_abs_error   = hf["errors/strain2x_abs_error"][()]
    strain2x_angle_error = hf["errors/strain2x_angle_error"][()]
    amp_violation_ratio   = float(hf["errors"].attrs["amp_violation_ratio"])
    phase_violation_ratio = float(hf["errors"].attrs["phase_violation_ratio"])

print(f"  run_flag                 : {run_flag}")
print(f"  Amplitude violation ratio: {amp_violation_ratio:.4f}")
print(f"  Phase    violation ratio : {phase_violation_ratio:.4f}")
print(f"  strain2x shape           : {strain2x_nominal.shape}")
print(f"  abs error shape          : {strain2x_abs_error.shape}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# 1. Nominal response
print("Plotting nominal response …")
plot_response(f, npix, np.abs(strain2x_nominal[0]),
              folder=output_dir,
              output_file="initial_strain2x.png",
              metric="mean")

# 2. Residual histograms
print("Plotting LTT residual histogram …")
plot_ltt_residuals_histogram(
    ltt_residuals,
    os.path.join(output_dir, "ltt_residuals_histogram.png"),
)

print("Plotting position residual histogram …")
plot_position_residuals_histogram(
    position_residuals,
    os.path.join(output_dir, "position_residuals_histogram.png"),
)

# 3. Frequency-domain error plots + sky maps
for metric in ("mean", "max"):
    print(f"Plotting strain errors [{metric}] …")
    plot_strain_errors(
        f, strain2x_abs_error, strain2x_angle_error,
        output_file=os.path.join(output_dir,
                                 f"{metric}_strain2x_errors_frequency.png"),
        metric=metric,
    )

    print(f"Plotting amplitude error sky maps [{metric}] …")
    plot_gw_response_maps(
        strain2x_abs_error, f, npix,
        folder=os.path.join(output_dir, f"{metric}_amplitude_errors"),
        metric=metric,
    )

    print(f"Plotting phase error sky maps [{metric}] …")
    plot_gw_response_maps(
        strain2x_angle_error, f, npix,
        folder=os.path.join(output_dir, f"{metric}_phase_errors"),
        metric=metric,
    )

print(f"\nAll plots saved to {output_dir}")
