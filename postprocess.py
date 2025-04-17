import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
# Define constants
labels = ["A", "E", "T"]
results_dir = "results"
h5file_name = "tdi_deviation.h5"  # Change this to the desired HDF5 file name
list_h5 = glob.glob(os.path.join(results_dir, "*.h5"))
print(list_h5)
for h5file_name in list_h5:
    
    # Load data from the HDF5 file
    with h5py.File(h5file_name, "r") as h5file:
        h5file_name = h5file_name.split("/")[-1].split(".h5")[0]
        sigma_vec = h5file["sigma_vec"][:]
        Ndraws = len(h5file["parameters"][:])
        time = h5file["time"][:]
        # Initialize combined plots for RMS and mismatch
        fig_mismatch, ax_mismatch = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        fig_mismatch.suptitle("Mismatch 95 percent upper bound")

        # Process each delta_x
        for delta_x in sigma_vec:
            group_name = f"sigma_{int(delta_x)}"
            rms_data = h5file[f"{group_name}/rms"][:]  # Shape: (Ndraws, 3, time)
            mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

            # Compute mean and standard deviation for RMS and mismatch
            mismatch_mean = np.quantile(mismatch_data, 0.95, axis=0)  # Shape: (3, time)
            mismatch_std = np.std(mismatch_data, axis=0)   # Shape: (3, time)

            # Plot mismatch
            for i, lab in enumerate(labels):
                ax_mismatch[i].semilogy(time[5:]/86400, mismatch_mean[i][5:], label=f"{int(delta_x)}-sigma", alpha=0.8)
                # ax_mismatch[i].fill_between(time, mismatch_mean[i] - mismatch_std[i], mismatch_mean[i] + mismatch_std[i], alpha=0.3)

        # Finalize mismatch plot
        for i, lab in enumerate(labels):
            ax_mismatch[i].set_ylabel(f"Mismatch {lab}")
            ax_mismatch[i].legend()#fontsize="small", loc="upper right")
        ax_mismatch[-1].set_xlabel("Time [days]")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"mismatch_upper95_{h5file_name.split('.h5')[0]}.png"))
        plt.close(fig_mismatch)

    print("Mean RMS and mismatch plots with error bars saved in the 'results' directory.")

    # Load data from the HDF5 file
    # with h5py.File(h5file_path, "r") as h5file:
    #     sigma_vec = h5file["sigma_vec"][:]
    #     parameters = h5file["parameters"][:]  # Shape: (Ndraws, 8), assuming last two are lambda and beta
    #     Ndraws = len(parameters)

    #     # Extract sky location parameters (lambda and beta)
    #     sky_lambda = parameters[:, 6]  # Assuming column 6 is lambda
    #     sky_beta = parameters[:, 7]    # Assuming column 7 is beta

    #     # Initialize scatter plot for mismatch
    #     for delta_x in sigma_vec:
    #         group_name = f"sigma_{int(delta_x)}"
    #         mismatch_data = h5file[f"{group_name}/mismatch"][:]  # Shape: (Ndraws, 3, time)

    #         # Compute the mean mismatch over time for each realization
    #         ind_channel = 0  # Assuming we are interested in the first channel
    #         final_mism = mismatch_data[:,ind_channel,-1]
    #         # Create scatter plot
    #         plt.figure(figsize=(8, 6))
    #         scatter = plt.scatter(sky_lambda, sky_beta, c=final_mism, cmap="viridis", s=50, edgecolor="k")
    #         plt.colorbar(scatter, label="Mismatch")
    #         plt.xlabel("Sky Location Parameter λ (radians)")
    #         plt.ylabel("Sky Location Parameter β (radians)")
    #         plt.title(f"Sky Location vs Mismatch for {int(delta_x)}-sigma")
    #         plt.grid(alpha=0.3)
    #         plt.savefig(os.path.join(results_dir, f"{h5file_name.split('.h5')[0]}_sky_location_mismatch_sigma_{int(delta_x)}_channel{ind_channel}.png"))
    #         plt.close()

    print("Sky location mismatch scatter plots saved in the 'results' directory.")