import h5py 
import numpy as np
import os
import matplotlib.pyplot as plt

with h5py.File("processed_trajectories.h5", "r") as dset:
    pars_names = dset['sample_parameter_names'][()]
    pars = dset['sample_parameters'][()]
    mean_par = np.mean(dset['sample_parameters'][()],axis=0)
    std_par = np.std(dset['sample_parameters'][()],axis=0)

os.makedirs("simulation_parameters", exist_ok=True)

# create histogram of parameters and save to file
for i, name in enumerate(pars_names):
    plt.figure()
    plt.hist(pars[:, i], bins=50, density=True, alpha=0.7)
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"Histogram of {name}")
    plt.savefig(f"simulation_parameters/{name}_histogram.png")
    plt.close()
    