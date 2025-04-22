import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["font.size"] = 16
# ticks size
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
# plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16

import glob
import healpy as hp
from utils import plot_orbit_3d, plot_sigma_vs_mismatch, plot_frequency_vs_mismatch
fpath = "new_orbits.h5"
T = 1.0

# Define constants
results_dir = "results"
list_h5 = glob.glob(os.path.join(results_dir, "*.h5"))
print(list_h5)
labels = ["A", "T"]
index_labels = np.array([0, 2],dtype=int)  # Assuming you want to plot channels A and T

result_dict = {}
for h5file_name in list_h5:
    # Load data from the HDF5 file
    with h5py.File(h5file_name, "r") as h5file:
        h5file_name = h5file_name.split("/")[-1].split(".h5")[0]
        # obtain frequency from name
        frequency = [float(h5file_name.split("_f")[1].split("_")[0])]
        sigma_vec = h5file["sigma_vec"][:]
        parameters = h5file["parameters"][:]  # Shape: (Ndraws, 8), assuming last two are lambda and beta
        Ndraws = len(parameters)
        time = h5file["time"][:]

        result_dict[h5file_name] = {"parameters": parameters, "sigma_vec": sigma_vec, "frequency": frequency[0], "Ndraws": Ndraws, "time": time, "mismatch_upp": [], "mismatch_low": [], "mismatch_median": []}
        # Process each delta_x
        for delta_x in sigma_vec:
            group_name = f"sigma_{int(delta_x)}"
            rms_data = h5file[f"{group_name}/rms"][:]  # Shape: (Ndraws, 3, time)
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

            # Compute mean and standard deviation for RMS and mismatch
            mismatch_upp = np.quantile(mismatch_data, 0.975, axis=0)[index_labels]  # Shape: (3, time)
            mismatch_low = np.quantile(mismatch_data, 0.025, axis=0)[index_labels]  # Shape: (3, time)
            mismatch_median = np.quantile(mismatch_data, 0.5, axis=0)[index_labels]  # Shape: (3, time)
            result_dict[h5file_name]["mismatch_upp"].append(mismatch_upp)
            result_dict[h5file_name]["mismatch_low"].append(mismatch_low)
            result_dict[h5file_name]["mismatch_median"].append(mismatch_median)

######################################################
# make plots
plot_sigma_vs_mismatch(result_dict, results_dir, labels)
plot_frequency_vs_mismatch(result_dict, results_dir, labels)

# specific run
filter_freq = 1e-4
filter_sigma = 1.0
for h5file_name, data in result_dict.items():
    # obtain frequency from name
    frequency = data["frequency"]
    sigma_vec = data["sigma_vec"]
    if frequency != filter_freq:
        continue

    print("Processing file:", h5file_name)
    Ndraws = data["Ndraws"]
    time = data["time"]
    mismatch_upp = data["mismatch_upp"]
    mismatch_low = data["mismatch_low"]
    mismatch_median = data["mismatch_median"]
    parameters = data["parameters"]

    Ndraws = len(parameters)

    # Process each delta_x
    for inp in zip(sigma_vec,mismatch_upp, mismatch_low, mismatch_median):
        
        delta_x, mismatch_upp, mismatch_low, mismatch_median = inp
        if filter_sigma != delta_x:
            continue
        # Initialize combined plots for RMS and mismatch
        fig_mismatch, ax_mismatch = plt.subplots(len(labels), 1, sharex=True, figsize=(5, 8))
        # fig_mismatch.suptitle("Mismatch 95 percent upper bound")

        # Plot mismatch
        for i, lab in enumerate(labels):
            ax_mismatch[i].plot(time[5:]/86400, mismatch_median[i][5:], alpha=0.8)
            ax_mismatch[i].fill_between(time[5:]/86400, mismatch_low[i][5:], mismatch_upp[i][5:], alpha=0.3, label=f"{int(delta_x)}-sigma")
            ax_mismatch[i].set_yscale("log")
            [ax_mismatch[i].axvline(el, linestyle="--", color="k", alpha=0.3) for el in [14]]#

        # Finalize mismatch plot
        for i, lab in enumerate(labels):
            ax_mismatch[i].set_ylabel(f"Mismatch {lab}")
            ax_mismatch[i].legend(ncol=2)#fontsize="small", loc="upper right")
            ax_mismatch[i].grid()
        ax_mismatch[-1].set_xlabel("Time [days]")
        plt.xlim(0, 30)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"mismatch_upper95_{h5file_name.split('.h5')[0]}.png"))
        plt.close(fig_mismatch)


        # Extract sky location parameters (lambda and beta)
        sky_lambda = parameters[:, 6]  # Assuming column 6 is lambda
        sky_beta = parameters[:, 7]    # Assuming column 7 is beta
        ind_channel = 0  # Assuming channel A is at index 0
        final_mism = np.log10(mismatch_data[:, ind_channel, -1])

        # convert color
        # Normalize final_mism to a 0-1 range for color mapping
        norm = plt.Normalize(vmin=np.nanmin(final_mism), vmax=np.nanmax(final_mism))
        color_mism = plt.cm.viridis(norm(final_mism))
        # Convert sky_lambda and sky_beta to theta and phi for healpy
        theta = sky_lambda + np.pi / 2  # Convert beta to colatitude
        phi = sky_beta  # Lambda is longitude

        scatter_points = [(1.0, sky_lambda[el], phi[el], color_mism[el]) for el in range(len(phi))]
        output_ = os.path.join(results_dir, f"3d_mismatch_sigma_{int(delta_x)}_{h5file_name.split('.h5')[0]}.png")
        plot_orbit_3d(fpath, T, scatter_points=scatter_points, output_file=output_)
        # print range theta phi
        print("phi min max", phi.min(),phi.max())
        print("theta min max", theta.min(),theta.max())

        # Create a healpy map
        nside = 8  # Healpy resolution parameter
        npix = hp.nside2npix(nside)
        healpy_map = np.full(npix, np.nan)

        # Assign mismatch values to the healpy map
        pix_indices = hp.ang2pix(nside, theta, phi)
        for pix, mism in zip(pix_indices, final_mism):
            healpy_map[pix] = mism

        # Plot the healpy map using mollview
        hp.mollview(
        healpy_map,
        title=f"Sky Location vs Mismatch for {int(delta_x)}-sigma",
        unit="log10 Mismatch",
        cmap="viridis",
        min=np.nanmin(final_mism),
        max=np.nanmax(final_mism),
        )
        plt.savefig(os.path.join(results_dir, f"mollview_mismatch_sigma_{int(delta_x)}_channel{ind_channel}_{h5file_name.split('.h5')[0]}.png"))
        plt.close()

        # Extract inclination and compute cos(inclination)
        inclination = parameters[:, 3]  # Assuming column 3 is inclination

        # Initialize plot for inclination vs mismatch
        plt.figure(figsize=(10, 6))
        # Compute the mean mismatch over time for each realization
        mean_mismatch = np.mean(mismatch_data[:, 0, :], axis=1)  # Assuming channel 0 (A)
        # Scatter plot for inclination vs mismatch
        plt.scatter(inclination, mean_mismatch, label=f"{int(delta_x)}-sigma", alpha=0.7)

        # Finalize the plot
        # plt.xlabel("Cos(Inclination)")
        plt.xlabel("Inclination")
        plt.ylabel("Mean Mismatch (Channel A)")
        plt.yscale("log")
        plt.title("Inclination vs Mismatch Across Different Sigma Values")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(results_dir, f"inclination_vs_mismatch_{h5file_name.split('.h5')[0]}.png")
        plt.savefig(output_file)
        plt.close()
        print("Sky location mismatch scatter plots saved in the 'results' directory.")