import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import healpy as hp
from utils import plot_orbit_3d
fpath = "new_orbits.h5"
T = 1.0  # years
plot_orbit_3d(fpath, T)

# Define constants
labels = ["A", "E", "T"]
results_dir = "results"
h5file_name = "tdi_deviation.h5"  # Change this to the desired HDF5 file name
list_h5 = glob.glob(os.path.join(results_dir, "*.h5"))
print(list_h5)

fig_mismatch, ax_mismatch = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
for h5file_name in list_h5:
    # Load data from the HDF5 file
    with h5py.File(h5file_name, "r") as h5file:
        h5file_name = h5file_name.split("/")[-1].split(".h5")[0]
        # obtain frequency from name
        frequency = [float(h5file_name.split("_f")[1].split("_")[0])]

        sigma_vec = h5file["sigma_vec"][:]
        Ndraws = len(h5file["parameters"][:])
        time = h5file["time"][:]
        # Initialize combined plots for RMS and mismatch
        
        # Process each delta_x
        for delta_x in sigma_vec[1:2]:
            group_name = f"sigma_{int(delta_x)}"
            rms_data = h5file[f"{group_name}/rms"][:]  # Shape: (Ndraws, 3, time)
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

            # Compute mean and standard deviation for RMS and mismatch
            mismatch_upp = np.quantile(mismatch_data, 0.975, axis=0)  # Shape: (3, time)
            mismatch_low = np.quantile(mismatch_data, 0.025, axis=0)  # Shape: (3, time)
            mismatch_median = np.quantile(mismatch_data, 0.5, axis=0)  # Shape: (3, time)
            # Plot mismatch
            for i, lab in enumerate(labels):
                ax_mismatch[i].loglog(frequency, mismatch_median[i][-1], 'o', alpha=0.8)
                ax_mismatch[i].fill_between(frequency, mismatch_low[i][-1], mismatch_upp[i][-1], alpha=0.3, label=f"{int(delta_x)}-sigma")

        # Finalize mismatch plot
        for i, lab in enumerate(labels):
            ax_mismatch[i].set_ylabel(f"Mismatch {lab}")
            ax_mismatch[i].legend(ncol=2)#fontsize="small", loc="upper right")
            ax_mismatch[i].grid()
        ax_mismatch[-1].set_xlabel("Frequency [Hz]")
        
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"freq_mismatch_{h5file_name.split('.h5')[0]}.png"))
plt.close(fig_mismatch)

breakpoint()
######################################################
for h5file_name in list_h5:
    
    # Load data from the HDF5 file
    with h5py.File(h5file_name, "r") as h5file:
        h5file_name = h5file_name.split("/")[-1].split(".h5")[0]
        sigma_vec = h5file["sigma_vec"][:]
        Ndraws = len(h5file["parameters"][:])
        time = h5file["time"][:]
        # Initialize combined plots for RMS and mismatch
        fig_mismatch, ax_mismatch = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
        fig_mismatch.suptitle("Mismatch 95 percent upper bound")

        # Process each delta_x
        for delta_x in sigma_vec:
            group_name = f"sigma_{int(delta_x)}"
            rms_data = h5file[f"{group_name}/rms"][:]  # Shape: (Ndraws, 3, time)
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

            # Compute mean and standard deviation for RMS and mismatch
            mismatch_upp = np.quantile(mismatch_data, 0.975, axis=0)  # Shape: (3, time)
            mismatch_low = np.quantile(mismatch_data, 0.025, axis=0)  # Shape: (3, time)
            mismatch_median = np.quantile(mismatch_data, 0.5, axis=0)  # Shape: (3, time)
            # Plot mismatch
            for i, lab in enumerate(labels):
                ax_mismatch[i].plot(time[5:]/86400, mismatch_median[i][5:], alpha=0.8)
                ax_mismatch[i].fill_between(time[5:]/86400, mismatch_low[i][5:], mismatch_upp[i][5:], alpha=0.3, label=f"{int(delta_x)}-sigma")
                ax_mismatch[i].set_yscale("log")
                [ax_mismatch[i].axvline(el, linestyle="--", color="k", alpha=0.3) for el in [14]]#
                # ax_mismatch[i].axvline(28, linestyle="--", color="k")
                # ax_mismatch[i].set_xscale("log")

        # Finalize mismatch plot
        for i, lab in enumerate(labels):
            ax_mismatch[i].set_ylabel(f"Mismatch {lab}")
            ax_mismatch[i].legend(ncol=2)#fontsize="small", loc="upper right")
            ax_mismatch[i].grid()
        ax_mismatch[-1].set_xlabel("Time [days]")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"mismatch_upper95_{h5file_name.split('.h5')[0]}.png"))
        plt.close(fig_mismatch)

        print("Mean RMS and mismatch plots with error bars saved in the 'results' directory.")

        # Load data from the HDF5 file
        sigma_vec = h5file["sigma_vec"][:]
        parameters = h5file["parameters"][:]  # Shape: (Ndraws, 8), assuming last two are lambda and beta
        Ndraws = len(parameters)

        # Extract sky location parameters (lambda and beta)
        sky_lambda = parameters[:, 6]  # Assuming column 6 is lambda
        sky_beta = parameters[:, 7]    # Assuming column 7 is beta
        
        # Initialize mollview plot for mismatch
        for delta_x in sigma_vec:
            group_name = f"sigma_{int(delta_x)}"
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)
            # obtain FFT of mismatch
            # fft_mims = np.fft.rfft(mismatch_data)
            # Compute the mean mismatch over time for each realization
            ind_channel = 0  # Assuming we are interested in the first channel
            final_mism = np.log10(mismatch_data[:, ind_channel, -1])

            # convert color
            # Normalize final_mism to a 0-1 range for color mapping
            norm = plt.Normalize(vmin=np.nanmin(final_mism), vmax=np.nanmax(final_mism))
            color_mism = plt.cm.viridis(norm(final_mism))
            # Convert sky_lambda and sky_beta to theta and phi for healpy
            theta = sky_lambda + np.pi / 2  # Convert beta to colatitude
            phi = sky_beta  # Lambda is longitude
            # scatter_points = [(phi)]
            # [(radius, lam, beta, color_function), ...]
            # x = radius * np.cos(beta_s) * np.cos(lam_s)
            # y = radius * np.cos(beta_s) * np.sin(lam_s)
            # z = radius * np.sin(beta_s)
            el = 0
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
        for delta_x in sigma_vec:
            group_name = f"sigma_{int(delta_x)}"
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

            # Compute the mean mismatch over time for each realization
            mean_mismatch = np.mean(mismatch_data[:, 0, :], axis=1)  # Assuming channel 0 (A)

            # Scatter plot for inclination vs mismatch
            plt.scatter(inclination, mean_mismatch, label=f"{int(delta_x)}-sigma", alpha=0.7)

            # find the max mismatch and define parameters 

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