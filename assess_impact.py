
# run with python as
# nohup python assess_impact.py > out.out &
import os, sys

if len(sys.argv) > 2:
    gb_frequency = [float(sys.argv[1])]
else:
    gb_frequency = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # Default frequency of GB binary

# Get CUDA device and frequency of GB binary from command-line arguments
if len(sys.argv) > 1:
    cuda_device = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

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



fpath = "new_orbits.h5"
T = 1.0  # years
plot_orbit_3d(fpath, T)

################################################################
use_gpu = True
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
    return random_vectors_on_sphere(size=size) * rho# * (1 + periodic[:,None])/2
    # return np.ones_like(random_vectors_on_sphere(size=size)) * rho
    
    
def compute_inn_and_den(fft_def, fft_dev, psd_, mask_sum, df):
    """
    Compute the inn and den quantities for FFT-based analysis.

    :param fft_def: FFT of the default signal
    :param fft_dev: FFT of the deviated signal
    :param psd_: Power Spectral Density (PSD) values
    :param mask_sum: Mask for the frequency range of interest
    :param df: Frequency resolution
    :return: Tuple containing inn and den values
    """
    matched_SNR = xp.sum(fft_def[mask_sum].conj() * fft_dev[mask_sum] / psd_).real * df
    loglike_diff = xp.sum(xp.abs(fft_def[mask_sum] - fft_dev[mask_sum])**2 / psd_).real * df
    snr_dev = xp.sum(fft_dev[mask_sum].conj() * fft_dev[mask_sum] / psd_).real * df
    snr_def = xp.sum(fft_def[mask_sum].conj() * fft_def[mask_sum] / psd_).real * df
    mismatch = xp.abs(1 - matched_SNR / xp.sqrt(snr_dev * snr_def))
    print(f"matched_SNR: {matched_SNR}, snr_def: {snr_def}, snr_dev: {snr_dev}, mismatch: {mismatch}")
    return matched_SNR, snr_def, snr_dev, mismatch, loglike_diff

# Create orbit objects
fpath = "new_orbits.h5"
orb_default = ESAOrbits(fpath,use_gpu=use_gpu)
deviation = {which: np.zeros_like(getattr(orb_default, which + "_base")) for which in ["ltt", "x", "n", "v"]}
orb_default.configure(linear_interp_setup=True, deviation=deviation)
# default orbit
gb_lisa_esa = get_response(orb_default)

# define variations
# 10 km, 100 km, 50 km (along-track, cross-track, radial) from message
# worst from fig 24 file:///Users/lorenzo.speri/Downloads/s40295-021-00263-2.pdf
# division by 3 because of the three sigma and multiply by 1e3 to convert to meters
# radial
sig_ref = 1/1000
sigma_radial = sig_ref * 1e3 /3 #50e3
# along-track
sigma_along = sig_ref * 1e3 /3#10e3
# cross-track
sigma_cross =  sig_ref * 1e3 /3#100e3
list_sigma = [sigma_radial, sigma_along, sigma_cross]

change = "x"
sigma_vec = [1, 3, 10, 100]
orbit_list = []

for delta_x in sigma_vec:
    # create the deviation dictionary
    orb_dev = ESAOrbits(fpath,use_gpu=use_gpu)
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
    plt.semilogy(time_vec/86400, np.linalg.norm(deviation_lof,axis=1)/1e3 ,label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
plt.xlim([0.0, 14*4])
plt.ylim([1., 4e4])
plt.xlabel("Time [days]")
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
            ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
                        (arr_def[:, sc, ii]-arr[:, sc, ii])/1e3, 
                        label=f"SC{sc}",color=coord_color[sc][1], alpha=0.3)
            
        ax[ii].axhline(list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].axhline(-list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].set_ylabel(coord_color[ii][0])
        # ax[ii].set_xlim(0, 30)
        

    ax[2].set_xlabel("Time [days]")
plt.xlim([0.0, 30])
plt.savefig("orbits_deviation.png")

# define GB default parameters
A = 1e-18
f = 2.35962078e-3
fdot = 1.5e-17
# import and interpolate psd ../EMRI-FoM/pipeline/TDI2_AE_psd.npy
psd = np.load("../EMRI-FoM/pipeline/TDI2_AE_psd.npy")
# interpolate the psd with a cubic spline
from scipy.interpolate import CubicSpline
psd_interp = CubicSpline(psd[:, 0], psd[:, 1])

# decie how many variations
channel_generator = [get_response(orb_dev) for orb_dev in orbit_list]
# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
# randomly draw the sky coordinates
Ndraws = 10
for f in gb_frequency:
    par_list = np.asarray([draw_parameters(A=A, f=f, fdot=fdot) for i in range(Ndraws)])
    fname = f"results/test_deviation_A{A}_f{f}_fdot{fdot}.h5"
    if os.path.exists(fname):
        print(f"File {fname} already exists. Skipping generation.")
        continue

    print("------------------------------")
    print("Saving to file: ", fname)
        
    rms_dict = {}
    mismatch_dict = {}
    for realization in range(Ndraws):
        
        print(realization, par_list[realization])
        print("Generating channels")
        chans = [channel_generator[i](*par_list[realization]) for i in range(len(channel_generator))]
        chans_default = gb_lisa_esa(*par_list[realization])
        # if use_gpu:
        #     chans = [[el.get() for el in chans[i]] for i in range(len(channel_generator))]
        #     chans_default = [chans_default[i].get() for i in range(len(chans_default))]
        print("Channels generated")

        # Save deviations for each delta_x
        for delta_x, chan in zip(sigma_vec, chans):

            # Compute and save RMS
            res_list = []
            for i, lab in enumerate(["A", "E", "T"]):
                # rms = xp.abs(chan[i] - chans_default[i]) / (2 * xp.mean(chans_default[i]**2))**0.5
                # window = 1000
                # rms = xp.convolve(rms, xp.ones(window) / window, mode='same')
                tukey_window = xp.asarray(tukey(len(chan[i]), alpha=0.5))
                fft_f = xp.fft.rfftfreq(len(chan[i]), dt)
                df = fft_f[1] - fft_f[0]
                # mask_sum = (fft_f > f-1000*df) & (fft_f < f+1000*df)
                mask_sum = (fft_f > 1e-5) & (fft_f < 1.0)
                fft_def = xp.fft.rfft(chans_default[i]*tukey_window,axis=1)*dt
                fft_dev = xp.fft.rfft(chan[i]*tukey_window,axis=1)*dt
                
                psd_ = xp.asarray(psd_interp(fft_f[mask_sum].get()))
                df = fft_f[1] - fft_f[0]

                # Replace the $SELECTION_PLACEHOLDER$ with a call to the function
                matched_SNR, snr_def, snr_dev, mismatch, loglike_diff = compute_inn_and_den(fft_def, fft_dev, psd_, mask_sum, df)
                # plot psd
                # plt.figure()
                # plt.loglog(fft_f[mask_sum].get(), psd_.get(), label="psd", alpha=0.5)
                # # plot difference
                # plt.loglog(fft_f[mask_sum].get(), xp.abs(fft_def[mask_sum] - fft_dev[mask_sum]).get()**2, label="def", alpha=0.5)
                # plt.loglog(fft_f[mask_sum].get(), xp.abs(fft_def[mask_sum]).get()**2, label="def", alpha=0.5)
                # plt.loglog(fft_f[mask_sum].get(), xp.abs(fft_dev[mask_sum]).get()**2, label="dev", alpha=0.5)
                # plt.xlabel("Frequency [Hz]")
                # plt.ylabel("Amplitude")
                # plt.title(f"PSD and FFT for {lab} channel")
                # plt.legend()
                # plt.savefig(f"fft_{lab}.png")
                # rms = xp.abs(1-xp.real(fft_dev[mask_sum].conj() * fft_def[mask_sum]).sum()/den)
                # if i == 0:
                #     plt.figure(); plt.loglog(fft_f[mask_sum].get(), xp.abs(fft_dev[mask_sum]).get(), label="dev", alpha=0.5); plt.loglog(fft_f[mask_sum].get(), xp.abs(fft_def[mask_sum]).get(), label="def", alpha=0.5); plt.savefig("fft_abs.png")
                #     plt.figure(); plt.loglog(fft_f[mask_sum][::10].get(), xp.real(fft_dev[mask_sum].conj()*fft_def[mask_sum])[::10].get(), label="def", alpha=0.5); plt.savefig(f"fft_match.png")
                #     breakpoint()
                # # plt.figure(); plt.plot((chans_default[i] - chan[i]).get()); plt.savefig("fft.png")
                

                # print(delta_x, i, "RMS: ", rms)
                res_ = dict(zip(["mismatch", "matched_SNR", "snr_def", "snr_dev", "loglike_diff"], [mismatch, matched_SNR, snr_def, snr_dev, loglike_diff]))
                res_list.append(res_)

            mismatch_dict[f"mismatch_real{realization}_sigma{delta_x}"] = np.asarray([res["mismatch"] for res in res_list])
        
    with h5py.File(fname, "w") as h5file:
        h5file.create_dataset("sigma_vec", data=sigma_vec)
        h5file.create_dataset("parameters", data=par_list)

        h5file.create_dataset("time", data=np.arange(len(chans_default[0]))[::100] * dt)
        # create delta_x group
        for delta_x in sigma_vec:
            # create rms and mismatch datasets
            rms_to_save = np.asarray([rms_dict[f"rms_real{realization}_sigma{delta_x}"] for realization in range(Ndraws)])
            mism_to_save = np.asarray([mismatch_dict[f"mismatch_real{realization}_sigma{delta_x}"] for realization in range(Ndraws)])
            h5file.create_group(f"sigma_{int(delta_x)}")
            h5file.create_dataset(f"sigma_{int(delta_x)}/rms", data=rms_to_save)
            h5file.create_dataset(f"sigma_{int(delta_x)}/mismatch", data=mism_to_save)
        
        # # Save plots as before
        # if realization <0:
        #     ###################################################
        #     # deviation orbit
        #     fig, ax = plt.subplots(3, 1, sharex=True)
        #     ax[0].set_title("Deviations from Default Orbit")
        #     # plot them
        #     for i, lab in enumerate(["A", "E", "T"]):
        #         ax[i].plot(np.arange(len(chans_default[0])) * dt / YRSID_SI, chans_default[i], 'k')
        #         ax[i].set_ylabel(lab)
        #         for delta_x, chan in zip(sigma_vec, chans):
        #             ax[i].plot(np.arange(len(chan[0])) * dt / YRSID_SI, chan[i], ':', label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
        #     ax[2].set_xlabel("Time [years]")
        #     ax[2].legend()
        #     plt.savefig(fname[:-3] + "_deviation_orbit.png")

        #     ###################################################
        #     fig, ax = plt.subplots(3, 1, sharex=True)
        #     ax[0].set_title("Root Mean Square from Default Orbit")
        #     # plot them
        #     for i, lab in enumerate(["A", "E", "T"]):
        #         ax[i].set_ylabel("Relative Error " + lab)
        #         for delta_x, chan in zip(sigma_vec, chans):
        #             rms = np.abs(chan[i] - chans_default[i]) / (2 * np.mean(chans_default[i]**2))**0.5
        #             window = 1000
        #             rms = np.convolve(rms, np.ones(window) / window, mode='same')
        #             new_time = np.arange(rms.shape[0]) * dt / YRSID_SI
        #             ax[i].semilogy(new_time, rms, ':', label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
        #     ax[2].set_xlabel("Time [years]")
        #     ax[2].legend()
        #     plt.legend()
        #     plt.savefig(fname[:-3] + "_tdi_deviation.png")

        #     ###################################################
        #     # matched filtering
        #     fig, ax = plt.subplots(3, 1, sharex=True)
        #     ax[0].set_title("Mismatch from Default Orbit as a function of time")
        #     # plot them
        #     for i, lab in enumerate(["A", "E", "T"]):
        #         ax[i].set_ylabel("Mismatch " + lab)
        #         for delta_x, chan in zip(sigma_vec, chans):
        #             overlap = np.cumsum(chan[i] * chans_default[i])
        #             overlap /= (np.cumsum(chans_default[i]**2) * np.cumsum(chan[i]**2))**0.5
        #             overlap = np.abs(1 - overlap)
        #             ax[i].semilogy(np.arange(len(chan[0]))[10:] * dt / YRSID_SI, overlap[10:], ':', linewidth=5, label=f"{int(delta_x)}-sigma deviation", alpha=0.5)
        #     ax[2].set_xlabel("Time [years]")
        #     ax[2].legend()
        #     plt.legend()
        #     plt.savefig(fname[:-3] + "_mismatch.png")
        #     plt.close("all")