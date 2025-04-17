import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import sawtooth
from scipy.signal.windows import tukey
import h5py
from fastlisaresponse import pyResponseTDI, ResponseWrapper
from astropy import units as un
from lisatools.utils.constants import *
# make nice plots
np.random.seed(2601)

from utils import *
import os

fpath = "new_orbits.h5"
T = 1/12  # years
plot_orbit_3d(fpath, T)

################################################################
use_gpu = False
gb = GBWave(use_gpu=use_gpu )
dt = 10.0


def get_response(orbit):
    # default settings
    # order of the langrangian interpolation
    order = 25
    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"
    index_lambda = 6
    index_beta = 7
    t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)
    tdi_kwargs_esa = dict(order=order, tdi=tdi_gen, tdi_chan="AET",)
    return ResponseWrapper(
    gb,
    T,
    dt,
    index_lambda,
    index_beta,
    t0=t0,
    flip_hx=False,  # set to True if waveform is h+ - ihx
    use_gpu=use_gpu,
    remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=True,  # False if using polar angle (theta)
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    orbits=orbit,
    **tdi_kwargs_esa,
    )

def get_variation(time_vec, t_initial=0, period=14*86400, rho=1.0):
    """
    Generate a periodic variation in the local orbital frame
    :param time_vec: time vector
    :param t_initial: initial time
    :param period: period of the variation
    :param rho: amplitude of the variation
    :return: 2D array of shape (len(time_vec), 3)
    """
    periodic = sawtooth(2 * np.pi * (time_vec-t_initial)/period)
    return random_vectors_on_sphere(size=size)*rho*(1 + periodic[:,None])/2
        

# Create orbit objects
fpath = "new_orbits.h5"
orb_default = ESAOrbits(fpath)
deviation = {which: np.zeros_like(getattr(orb_default, which + "_base")) for which in ["ltt", "x", "n", "v"]}
orb_default.configure(linear_interp_setup=True, deviation=deviation)
# default orbit
gb_lisa_esa = get_response(orb_default)

# define variations
# 10 km, 100 km, 50 km (along-track, cross-track, radial) from message
# worst from fig 24 file:///Users/lorenzo.speri/Downloads/s40295-021-00263-2.pdf
# division by 3 because of the three sigma and multiply by 1e3 to convert to meters
# radial
sigma_radial = 1e3 * 1e3 /3 #50e3
# along-track
sigma_along = 1e3 * 1e3 /3#10e3
# cross-track
sigma_cross =  1e3 * 1e3 /3#100e3
list_sigma = [sigma_radial, sigma_along, sigma_cross]

change = "x"
sigma_vec = [1, 3, 10, 100]
orbit_list = []

for delta_x in sigma_vec:
    # create the deviation dictionary
    orb_dev = ESAOrbits(fpath)
    deviation = {which: np.zeros_like(getattr(orb_dev, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    
    # time
    time_vec = orb_dev.t_base
    size = len(orb_dev.t_base)
    xbase = orb_dev.x_base
    local_orbital_frame_pos = np.sum(xbase,axis=1)/3
    
    # loop over spacecraft
    for sc in range(3):
        # deviation in the local orbital frame
        deviation_lof = get_variation(time_vec, t_initial=0, period=14*86400, rho=delta_x * list_sigma[0])
        deviation["x"][:, sc, :] += deviation_lof
        deviation["v"][:, sc, :] += np.gradient(deviation_lof,time_vec, axis=0)

    orb_dev.configure(linear_interp_setup=True, deviation=deviation)
    orbit_list.append(orb_dev)

plt.figure()
sc = 0
plt.title(f"Deviation in the orbit of the spacecraft {sc}")
for delta_x, orb_dev in zip(sigma_vec, orbit_list):
    # plot the deviation
    time_vec = orb_dev.t_base
    deviation_lof = orb_dev.deviation["x"][:, sc, :]
    # plot deviation
    plt.semilogy(time_vec/YRSID_SI, np.linalg.norm(deviation_lof,axis=1)/1e3 ,label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
plt.ylim([1.0, 1e5])
plt.xlabel("Time [years]")
plt.legend()
plt.ylabel("Deviation Radius [km]")
plt.savefig("radius_deviation.png")
coord_color = [(r"$x_{\rm ref} - x$ [km]", "g"), (r"$y_{\rm ref} - y$ [km]","b"), (r"$z_{\rm ref}- z$ [km]","r")]

fig, ax = plt.subplots(3, 1, sharex=True)
for orb_dev in orbit_list[:1]:
    arr = getattr(orb_dev, change)
    arr_def = getattr(orb_default, change)
    for ii in range(3):
        for sc in range(1):
            ax[ii].plot(np.arange(arr.shape[0]) * dt / YRSID_SI, (arr_def[:, sc, ii]-arr[:, sc, ii])/1e3, 
                        label=f"SC{sc}",color=coord_color[sc][1], alpha=0.3)
        ax[ii].axhline(list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].axhline(-list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].set_ylabel(coord_color[ii][0])
ax[2].set_xlabel("Time [years]")
ax[0].set_title(f"SC Coordinates")
plt.tight_layout()
plt.savefig("orbits_deviation.png")

# define GB default parameters
A = 1.084702251e-22
f = 2.35962078e-3
fdot = 1.47197271e-17

# decie how many variations
channel_generator = [get_response(orb_dev) for orb_dev in orbit_list]
# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
# randomly draw the sky coordinates
Ndraws = 10
for f in [1e-4, 1e-3, 1e-2]:
    par_list = np.asarray([draw_parameters(A=A, f=f, fdot=fdot) for i in range(Ndraws)])
    fname = f"results/tdi_deviation_A{A}_f{f}_fdot{fdot}.h5"
    with h5py.File(fname, "w") as h5file:
        h5file.create_dataset("sigma_vec", data=sigma_vec)
        h5file.create_dataset("parameters", data=par_list)
        rms_dict = {}
        mismatch_dict = {}
        for realization in range(Ndraws):
            print("------------------------------")
            print(realization, par_list[realization])
            chans = [channel_generator[i](*par_list[realization]) for i in range(len(channel_generator))]
            chans_default = gb_lisa_esa(*par_list[realization])
            
            # Save deviations for each delta_x
            for delta_x, chan in zip(sigma_vec, chans):

                # Compute and save RMS
                rms_list = []
                for i, lab in enumerate(["A", "E", "T"]):
                    rms = np.abs(chan[i] - chans_default[i]) / (2 * np.mean(chans_default[i]**2))**0.5
                    window = 1000
                    rms = np.convolve(rms, np.ones(window) / window, mode='same')
                    rms_list.append(rms)
                rms = np.array(rms_list)
                rms_dict[f"rms_real{realization}_sigma{delta_x}"] = rms
                
                # Compute and save mismatch
                mismatch_list = []
                for i, lab in enumerate(["A", "E", "T"]):
                    overlap = np.cumsum(chan[i] * chans_default[i])
                    overlap /= (np.cumsum(chans_default[i]**2) * np.cumsum(chan[i]**2))**0.5
                    mismatch = np.abs(1 - overlap)
                    mismatch_list.append(mismatch)
                mismatch = np.array(mismatch_list)
                mismatch_dict[f"mismatch_real{realization}_sigma{delta_x}"] = mismatch

        # create delta_x group
        for delta_x in sigma_vec:
            # create rms and mismatch datasets
            rms_to_save = np.asarray([rms_dict[f"rms_real{realization}_sigma{delta_x}"] for realization in range(Ndraws)])
            mism_to_save = np.asarray([mismatch_dict[f"mismatch_real{realization}_sigma{delta_x}"] for realization in range(Ndraws)])
            h5file.create_group(f"sigma_{int(delta_x)}")
            h5file.create_dataset(f"sigma_{int(delta_x)}/rms", data=rms_to_save)
            h5file.create_dataset(f"sigma_{int(delta_x)}/mismatch", data=mism_to_save)
        
        # Save plots as before
        if realization == 0:
            ###################################################
            # deviation orbit
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].set_title("Deviations from Default Orbit")
            # plot them
            for i, lab in enumerate(["A", "E", "T"]):
                ax[i].plot(np.arange(len(chans_default[0])) * dt / YRSID_SI, chans_default[i], 'k')
                ax[i].set_ylabel(lab)
                for delta_x, chan in zip(sigma_vec, chans):
                    ax[i].plot(np.arange(len(chan[0])) * dt / YRSID_SI, chan[i], ':', label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
            ax[2].set_xlabel("Time [years]")
            ax[2].legend()
            plt.savefig(fname[:-3] + "_deviation_orbit.png")

            ###################################################
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].set_title("Root Mean Square from Default Orbit")
            # plot them
            for i, lab in enumerate(["A", "E", "T"]):
                ax[i].set_ylabel("Relative Error " + lab)
                for delta_x, chan in zip(sigma_vec, chans):
                    rms = np.abs(chan[i] - chans_default[i]) / (2 * np.mean(chans_default[i]**2))**0.5
                    window = 1000
                    rms = np.convolve(rms, np.ones(window) / window, mode='same')
                    new_time = np.arange(rms.shape[0]) * dt / YRSID_SI
                    ax[i].semilogy(new_time, rms, ':', label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
            ax[2].set_xlabel("Time [years]")
            ax[2].legend()
            plt.legend()
            plt.savefig(fname[:-3] + "_tdi_deviation.png")

            ###################################################
            # matched filtering
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].set_title("Mismatch from Default Orbit as a function of time")
            # plot them
            for i, lab in enumerate(["A", "E", "T"]):
                ax[i].set_ylabel("Mismatch " + lab)
                for delta_x, chan in zip(sigma_vec, chans):
                    overlap = np.cumsum(chan[i] * chans_default[i])
                    overlap /= (np.cumsum(chans_default[i]**2) * np.cumsum(chan[i]**2))**0.5
                    overlap = np.abs(1 - overlap)
                    ax[i].semilogy(np.arange(len(chan[0]))[10:] * dt / YRSID_SI, overlap[10:], ':', linewidth=5, label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
            ax[2].set_xlabel("Time [years]")
            ax[2].legend()
            plt.legend()
            plt.savefig(fname[:-3] + "_mismatch.png")
            plt.close("all")