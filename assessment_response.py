# %% [markdown]
# # Requirement on antenna pattern
# ## SCI-MIS-0250: Knowledge of the antenna pattern in amplitude
# The susceptibility of the observatory to the strain of a GW shall be known to better than 10−4 in amplitude for any polarisation of the GW, at any given time during the mission lifetime, and for any given position of the GW source in the sky.
# ## SCI-MIS-0260: Knowledge of the antenna pattern in phase
# The susceptibility of the observatory to the strain of a GW shall be known to better than 10−2 rad in phase for any polarisation of the GW, at any given time during the mission lifetime, and for any given position of the GW source in the sky.
# ## Rationale: 
# Many science objectives and science investigations depend on the knowledge of the strain of the gravitational wave signal that needs to be calculated from the measured signal with sufficient accuracy. Requiring the which is given by this requirement. In the flow-down, this sets requirements on spacecraft (S/C) position knowledge, knowledge of the laser wavelength, precision of the reconstruction etc.

# %%
# run with python as
# nohup python assess_impact.py > out.out &
import os, sys

# Get CUDA device and frequency of GB binary from command-line arguments
cuda_device = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import sawtooth
from scipy.signal.windows import tukey
import h5py
from myresponse import ResponseWrapper
from astropy import units as un
from lisatools.utils.constants import *
# make nice plots
np.random.seed(2601)

from utils import *

fpath = "new_orbits.h5"
T = 30.0/365  # years
plot_orbit_3d(fpath, T)


# %%
psd = np.load("../EMRI-FoM/pipeline/TDI2_AE_psd.npy")
# interpolate the psd with a cubic spline
from scipy.interpolate import CubicSpline
psd_interp = CubicSpline(psd[:, 0], psd[:, 1])
plt.loglog(psd[:, 0], psd[:, 1], label="Original PSD")


# %%

def get_response(orbit, T=1.0, dt=10., use_gpu=True, t0 = 10000.0):
    gb = GBWave(use_gpu=use_gpu, T=T, dt=dt)
    # default settings
    # order of the langrangian interpolation
    order = 25
    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"
    index_lambda = 6
    index_beta = 7
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
    size = len(time_vec)
    periodic = sawtooth(2 * np.pi * (time_vec-t_initial)/period)
    # res =  random_vectors_on_sphere(size=size)[0] * rho * (1 + periodic[:,None])/2
    periodic = (1-np.cos(2 * np.pi * (time_vec-t_initial)/period)) / 2
    res =  random_vectors_on_sphere(size=size)[0] * rho * periodic[:,None]
    # fixed random
    res = np.ones_like(random_vectors_on_sphere(size=size)) * random_vectors_on_sphere(size=size)[0] * rho
    # random
    res = random_vectors_on_sphere(size=size) * rho
    return res
    
    
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
    d_h = 4 * xp.sum(fft_def[mask_sum].conj() * fft_dev[mask_sum] / psd_).real * df
    d_d = 4 * xp.sum(fft_dev[mask_sum].conj() * fft_dev[mask_sum] / psd_).real * df
    h_h = 4 * xp.sum(fft_def[mask_sum].conj() * fft_def[mask_sum] / psd_).real * df
    A_est = d_h / h_h
    tan_phi_est = 4 * xp.sum(fft_def[mask_sum].conj() * fft_dev[mask_sum] / psd_).imag * df / d_h
    Im = xp.sum(fft_def[mask_sum].conj() * fft_dev[mask_sum] / psd_).imag * df
    Re = d_h 
    tan_phi_est = Im / Re
    phi_est = np.arctan(tan_phi_est)
    # print("estimated A and phi and cos phi", A_est, phi_est, np.cos(phi_est))
    # loglike_diff = 4 * xp.sum(xp.abs(fft_def[mask_sum] - fft_dev[mask_sum])**2 / psd_).real * df
    relative_diff_abs = xp.abs(xp.abs(fft_def[mask_sum]) - xp.abs(fft_dev[mask_sum])) / xp.abs(fft_def[mask_sum])
    diff_angle = xp.abs(xp.angle(fft_def[mask_sum]) - xp.angle(fft_dev[mask_sum]))
    mismatch = xp.abs(1 - d_h / xp.sqrt(h_h * d_d))
    snr = xp.sqrt(h_h)
    # print("mismatch", mismatch, "snr", snr, "loglike_diff")
    return 1-A_est.get(), phi_est.get(), mismatch.get(), snr.get(), relative_diff_abs.mean().get(), diff_angle.mean().get()

def compute_information_matrix(A, delta_phi, rho_squared=1.0):
    """
    Compute the Fisher Information Matrix based on the second derivatives of the log-likelihood.

    Parameters:
        A (float): Amplitude scaling factor.
        delta_phi (float): Phase difference in radians.
        rho_squared (float): The squared signal-to-noise ratio (SNR), ρ².

    Returns:
        np.ndarray: 2x2 Fisher Information Matrix.
    """
    # Compute the second derivatives
    d2L_dA2 = rho_squared  # -∂²_A ℒ
    d2L_dphi2 = rho_squared * A * np.cos(delta_phi)  # -∂²_δφ ℒ
    d2L_dAdphi = -rho_squared * np.sin(delta_phi)  # -∂_A ∂_δφ ℒ

    # Construct the Fisher Information Matrix
    fisher_matrix = np.array([
        [d2L_dA2, d2L_dAdphi],
        [d2L_dAdphi, d2L_dphi2]
    ])

    return fisher_matrix

A = 1.0  # Example amplitude scaling factor
delta_phi = 0.0  # Example phase difference in radians

fisher_matrix = compute_information_matrix(A, delta_phi)
print("Fisher Information Matrix:")
print(fisher_matrix)

from lisaorbits import StaticConstellation

def perturbed_static_orbits(arm_lengths, armlength_error, rotation_error, translation_error, rot_fac = 2.127):
    """ Apply perturbations to the static orbits 
    
    We want to create a situation where the armlengths are known very well, but the
    absolute positions of the spacecraft are not known very well.

    This is achieved by applying a small perturbation to the armlengths
    followed by a rotation and translation to the spacecraft 
    positions (around random axis and directions).
    
    Parameters
    ----------
    arm_lengths : np.array
        Nominal armlengths of the constellation.
    armlength_error : float
        Standard deviation of the perturbation to apply to the armlengths, in meters.
    rotation_error : float
        Standard deviation of the rotation to apply to the spacecraft positions, in equivalent meters of displacement.
    translation_error : float
        Standard deviation of the translation to apply to the spacecraft positions, in meters.

    """
    # Create new orbits with perturbed armlengths
    arm_dev = np.random.normal(0, armlength_error, size=(3,))
    print("arm_dev", arm_dev)
    perturbed_ltt_orbits = StaticConstellation.from_armlengths(arm_lengths[0] + arm_dev[0], 
                                                       arm_lengths[1] + arm_dev[1], 
                                                       arm_lengths[2] + arm_dev[2])
    
    # Apply rotation by an angle phi along a random axis in 3d space

    # Generate a random rotation matrix
    # Random axis of rotation
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize the axis

    # Average distance of the spacecraft from the center of mass
    avg_distance = np.mean(np.linalg.norm(perturbed_ltt_orbits.sc_positions, axis=1))

    # In the small angle approximation, the rotation by an angle phi causes a displacement
    # of the spacecraft positions by a distance d = r * phi. Solving for phi and using d = error_magnitude gives
    # phi = d / r
    # TODO: improve on the math here; not all rotations affect all S/C, so this is
    # off by a factor of 2 or so; for now just fitted by hand
    angle = rot_fac * rotation_error / avg_distance

    # Rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(axis, axis)

    # Apply the rotation to the spacecraft positions
    rotated_positions = np.dot(R, perturbed_ltt_orbits.sc_positions.T).T

    # Apply translation to the spacecraft positions
    translation = np.random.normal(0, translation_error, size=(3,))
    perturbed_positions = rotated_positions + translation
    print("translation", translation)
    print("rotation", angle)
    # Create a new StaticConstellation object with the perturbed positions
    perturbed_orbits = StaticConstellation(perturbed_positions[0], perturbed_positions[1], perturbed_positions[2])
    return perturbed_orbits

def get_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3, dt=10.):
    porbit = perturbed_static_orbits(
        arm_lengths=arm_lengths, 
        armlength_error=armlength_error, 
        rotation_error=rotation_error, 
        translation_error=translation_error
    )
    porbit.write("temp_orbit.h5", dt=dt, size=int(T*YRSID_SI/dt), t0=0.0, mode="w")    
    return ESAOrbits("temp_orbit.h5",use_gpu=True)

def create_orb_dev(delta_x, fpath, use_gpu):
    """
    Create an orb_dev object with deviations based on the given sigma.

    Parameters:
    delta_x (float): Size of the deviation.
    fpath (str): File path for the orbit data.
    use_gpu (bool): Whether to use GPU for computations.

    Returns:
    ESAOrbits: The orb_dev object with configured deviations.
    """
    # create the deviation dictionary
    orb_dev = ESAOrbits(fpath, use_gpu=use_gpu)
    deviation = {which: np.zeros_like(getattr(orb_dev, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    
    # time
    time_vec = orb_dev.t_base
    xbase = orb_dev.x_base
    local_orbital_frame_pos = np.sum(xbase, axis=1) / 3
    
    # loop over spacecraft
    for sc in range(3):
        # deviation in the local orbital frame
        deviation_lof = get_variation(time_vec, t_initial=0, period=14 * 86400, rho=delta_x)
        deviation["x"][:, sc, :] += deviation_lof
        deviation["v"][:, sc, :] += np.gradient(deviation_lof, time_vec, axis=0)
    
    orb_dev.configure(linear_interp_setup=True, deviation=deviation)
    return orb_dev


# %%
# Create orbit objects
# fpath = "new_orbits.h5"
orb_default = get_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0.0, rotation_error=0.0, translation_error=0.0) #ESAOrbits(fpath,use_gpu=use_gpu)
deviation = {which: np.zeros_like(getattr(orb_default, which + "_base")) for which in ["ltt", "x", "n", "v"]}
orb_default.configure(linear_interp_setup=True, deviation=deviation)
# default orbit
gb_lisa_esa = get_response(orb_default)

# define variations
# fig 72 of ESA-LISA-ESOC-MAS-RP-0001 Iss2Rev0 - LISA Consolidated Report on Mission Analysis.pdf
# division by 3 because of the three sigma and multiply by 1e3 to convert to meters
# radial
sig_ref = 1e3
sigma_radial = sig_ref * 1e3 /3 #50e3
# along-track
sigma_along = sig_ref * 1e3 /3#10e3
# cross-track
sigma_cross =  sig_ref * 1e3 /3#100e3
list_sigma = [sigma_radial, sigma_along, sigma_cross]

change = "x"
sigma_vec = np.arange(0, 10)
orbit_list = []

for delta_x in sigma_vec:

    # Replace the placeholder with the function call
    # orb_dev = create_orb_dev(delta_x * list_sigma[0], fpath, use_gpu)
    orb_dev = get_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3)

    orb_dev.configure(linear_interp_setup=True, deviation=deviation)
    orbit_list.append(orb_dev)


# %%
plot_orbit_3d("temp_orbit.h5", T)
plt.savefig("temp_orbit.png")

# %%
coord_color = [(r"$x_{\rm ref} - x$ [km]", "C0"), (r"$y_{\rm ref} - y$ [km]","C1"), (r"$z_{\rm ref}- z$ [km]","C2")]
fig, ax = plt.subplots(3, 1, sharex=True)
for orb_dev in orbit_list:
    arr = getattr(orb_dev, change)
    arr_def = getattr(orb_default, change)
    for ii in range(3):
        for sc in range(3):
            ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
                        (arr_def[:, sc, ii]-arr[:, sc, ii])/1e3, 
                        label=f"SC{sc}",color=coord_color[sc][1], alpha=0.3)
        ax[ii].axhline(0.0, linestyle='--', color='k')
        # ax[ii].axhline(list_sigma[ii]/1e3, linestyle='--', color='k')
        # ax[ii].axhline(-list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].set_ylabel(coord_color[ii][0])
        # ax[ii].set_xlim(0, 30)
        

    ax[2].set_xlabel("Time [days]")
plt.savefig("orbits_deviation.png")

# %%
sc = 0
coord_color = [("$L_{12}$ [s]", "g")]
fig, ax = plt.subplots(3, 1, sharex=True)
for orb_dev in orbit_list[:1]:
    arr = getattr(orb_dev, change)
    arr_def = getattr(orb_default, change)
    ii = 0
    deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1)/C_SI
    ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
                deviation, 
                color=coord_color[sc][1], alpha=0.3, label="default")
    
    ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1)/C_SI
    ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, ref, linestyle='--', color='k',alpha=0.2, label="deviation")
    ax[ii].set_ylabel(coord_color[ii][0])
    # ax[ii].set_ylim([ref[0].min(), ref[1000].max()])
        
for orb_dev in orbit_list[:1]:
    arr = getattr(orb_dev, change)
    arr_def = getattr(orb_default, change)
    ii = 0
    deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1) # /C_SI
    ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1) # /C_SI
    diff = np.abs(ref-deviation)
    ax[1].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, diff, linestyle='--', color='k',alpha=0.2)
    ax[1].set_ylabel("Armlength difference [m]")


for orb_dev in orbit_list[:1]:
    arr = getattr(orb_dev, change)
    arr_def = getattr(orb_default, change)
    ii = 0
    deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1)/C_SI
    ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1)/C_SI
    diff = np.abs(ref-deviation)/ref
    ax[2].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, diff, linestyle='--', color='k',alpha=0.2)
    ax[2].set_ylabel("Relative")

ax[2].set_xlabel("Time [days]")
# plt.xlim([0.0, 7])
ax[0].legend()
plt.savefig("armlength_deviation.png")


# %%
plt.figure()
sc = 0
for sc in range(3):
    for delta_x, orb_dev in zip(sigma_vec[:1], orbit_list[:1]):
        # plot the deviation
        time_vec = orb_dev.t
        # deviation_lof = orb_dev.deviation["x"][:, sc, :]
        deviation_lof = orb_dev.x[:, sc, :] - orb_default.x[:, sc, :]
        # plot deviation
        # print(deviation_lof)
        # print(delta_x, np.diff(np.linalg.norm(deviation_lof,axis=1))/1e3 )

        plt.semilogy(time_vec/86400, np.linalg.norm(deviation_lof,axis=1)/1e3 ,label=f"deviation sc{sc}", alpha=0.5)

plt.axhline(1e3, linestyle='--', color='k', label="3 sigma Reference from ESA")
# plt.xlim([0.0, 7])
# plt.ylim([0.5, 4e4])
plt.xlabel("Time [days]")
plt.legend()
plt.ylabel("Deviation Radius [km]")

# %%
plt.figure()
ff = np.logspace(-4, 0.0, 100)
plt.loglog(ff,psd_interp(ff))

# %%
# define GB default parameters
A = 1e-20
gb_frequency = np.asarray([1e-3])#np.logspace(-4, -0.0, 10)
fdot = 0.0
# sky
import healpy as hp
nside = 12
npix = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
# betas ecliptic latitude https://arxiv.org/pdf/2204.06633
# lambs ecliptic longitude
betas, lambs = np.pi / 2 - thetas, phis

gw_response_map = np.zeros(npix)
for pix in range(npix):
    gw_response_map[pix] = np.random.normal(0, 1)
hp.mollview(gw_response_map)
hp.graticule()


# %%
import time
from tqdm import tqdm
dt = 0.25
T = 1.0/365
Number_of_deviations = 2
for _ in range(Number_of_deviations):
    orb_dev = get_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3)
    orb_dev.configure(linear_interp_setup=True)
    orbit_list.append(orb_dev)

channel_generator = [get_response(orb_dev, T=T, dt=0.25) for orb_dev in orbit_list]
gb_lisa_esa = get_response(orb_default, T=T, dt=0.25)

def process_realization(param, h, chans_default, channel_generator, psd_interp, dt):
    """
    Process a single parameter realization and compute A_est, phi_est, mismatch, snr, and FFTs.

    Parameters:
    param (numpy.ndarray): Array of parameter realizations.
    h (numpy.ndarray): Waveform data.
    chans_default (list): List of default channel data.
    channel_generator (list): List of channel generator functions.
    psd_interp (CubicSpline): Interpolated PSD function.
    dt (float): Time step.
    f (float): Frequency of interest.

    Returns:
    tuple: A_est, phi_est, mismatch, snr, fft_def, fft_dev, fft_f, mask_sum
    """
    # tic = time.time()
    N = chans_default[0].shape[0]
    tukey_window = xp.asarray(tukey(N, alpha=0.01))
    fft_f = xp.fft.rfftfreq(N, dt)
    df = fft_f[1] - fft_f[0]
    delta_w = 10 * df
    mask_sum = (fft_f > param[1] - delta_w) * (fft_f < param[1] + delta_w)
    # print("fmin fmax", fft_f[mask_sum].min(), fft_f[mask_sum].max())
    fft_def = xp.fft.rfft(chans_default * tukey_window, axis=1) * dt
    
    # psd_ = xp.asarray(psd_interp(fft_f[mask_sum].get()))
    psd_ = 1.0
    
    # toc = time.time()
    # print("time to compute FFT", toc-tic, len(channel_generator))
    ffts = xp.asarray([xp.fft.rfft(xp.asarray(channel_generator[i].apply_response(h, param[6], param[7])) * tukey_window, axis=1) * dt for i in range(len(channel_generator))])
    # print("ffts", ffts.shape, fft_def.shape)
    temp_dict = {}
    for tdi in range(fft_def.shape[0]):
        temp_dict[tdi] = np.asarray([compute_inn_and_den(fft_def[tdi], el, psd_, mask_sum, df) for el in ffts[:,tdi]])
        # plt.figure()
        # plt.loglog(fft_f[mask_sum].get(), np.abs(1-ffts[0,tdi][mask_sum].get()/fft_def[tdi][mask_sum].get())**2, label="Dev")
        # plt.axvline(param[1], linestyle='--', color='k', label="Signal")
        # plt.savefig(f"fft_{tdi}.png")
    return temp_dict

param = np.asarray([A, 1e-3, fdot, 0.0, 0.0, 0.0, 0.0, 0.0])
h = gb_lisa_esa.generate_waveform(*param, hp_flag=1.0, hc_flag=0.0)
# test with no deviation
chans_default = xp.asarray(gb_lisa_esa.apply_response(h, param[6], param[7]) )
process_realization(param=param, h=h, chans_default=chans_default, channel_generator=channel_generator[:Number_of_deviations],psd_interp=psd_interp,dt=dt)
# print(process_realization(param=param, h=h, chans_default=chans_default, channel_generator=[gb_lisa_esa],psd_interp=psd_interp,dt=dt))
print("Number of points", chans_default[0].shape[0])

# %%
# randomly draw the sky coordinates
results_dict = {}
for ff in gb_frequency:#tqdm(gb_frequency, desc="Processing frequency"):
    print("frequency", ff)
    # fmax = 5 * ff
    # Ncyc = 1000
    # dt = np.min([10., 1/(2*fmax)])
    # T = Ncyc / (np.pi*2*ff) / YRSID_SI # 1.0/365/24 * 6
    print("T", T, "dt", dt, "days", T*YRSID_SI/86400)
    channel_generator = [get_response(orb_dev, T=T, dt=dt) for orb_dev in orbit_list]
    gb_lisa_esa = get_response(orb_default, T=T, dt=dt)
    # draw random parameters
    # same settings of Olaf
    psi = 0.0 # np.random.uniform(0, 2 * np.pi)
    iota = 0.0 # np.arccos(np.random.uniform(-1, 1))
    phi0 = 0.0 # np.random.uniform(0, 2 * np.pi)
    temp = np.asarray([A, ff, 0.0, iota, phi0, psi, 0.0, 0.0])
    h = gb_lisa_esa.generate_waveform(*temp, hp_flag=1.0, hc_flag=1.0)
    # plt.figure(); 
    # plt.plot(h.get(), label="default"); 
    # plt.plot(chans_default[0].get(), label="chans_default");
    # 
    out_list = []
    for ii in tqdm(range(len(betas)), desc="Processing sky"):#range(len(betas)):#
        temp[6], temp[7] = lambs[ii], betas[ii]
        # test with no deviation
        chans_default = xp.asarray(gb_lisa_esa.apply_response(h, temp[6], temp[7]))
        out = process_realization(param=temp, h=h, chans_default=chans_default, channel_generator=channel_generator[:Number_of_deviations], psd_interp=psd_interp, dt=dt)
        out_list.append(out)
    # save the results
    results_dict[ff] = out_list

# results_dict


# %%
tdi= 0
temp_arr = np.asarray([el[tdi] for el in results_dict[gb_frequency[0]]])
temp_arr.shape

# %%
tdi = 0

amp_ind = 0
phase_ind = 1
mism_ind = 2
snr_ind = 3

indices = {"Amplitude": amp_ind, "Phase": phase_ind, "Mismatch": mism_ind, "SNR": snr_ind, "Relative TDI Amplitude": 4, "TDI Phase": 5}
ylabels = {"Amplitude": r"Relative Amplitude Error", "Phase": r"Phase Error", "Mismatch": "Mismatch", "SNR": "SNR", "Relative TDI Amplitude": r"Relative TDI Amplitude", "TDI Phase": r"TDI Phase"}
requirements = {"Amplitude": 1e-4, "Phase": 1e-2, "Mismatch": None, "SNR": None, "Relative TDI Amplitude": 1e-4, "TDI Phase": 1e-2}
colors = {"Amplitude": "C0", "Phase": "C1", "Mismatch": "C2"}

for key, ind in indices.items():
    plt.figure(figsize=(5, 4))
    for f in gb_frequency:
        temp_arr = np.asarray([el[tdi] for el in results_dict[f]])[:, 0, ind]
        res_list = np.abs(temp_arr.flatten())
        
        plt.plot(f*np.ones_like(res_list), res_list, "o", alpha=0.1, color="C0")
        
        # res_median = np.median(res_list, axis=0)
        # res_max = np.max(res_list, axis=0)
        # res_min = np.min(res_list, axis=0)
        # plt.plot(f, res_median, "P", alpha=0.8, color="C0", label=f"Median {key}" if f == gb_frequency[0] else None)
        # plt.plot(f, res_max, "^", alpha=0.8, color="C2", label=f"Max {key}" if f == gb_frequency[0] else None)
        # plt.plot(f, res_min, "v", alpha=0.8, color="C3", label=f"Min {key}" if f == gb_frequency[0] else None)
    
    if requirements[key]:
        plt.axhline(y=requirements[key], linestyle=":", color="C5", alpha=0.8, label=f"{key} Requirement")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("GW Frequency [Hz]")
    plt.ylabel(ylabels[key])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{key.lower()}_error_plot.png", dpi=300)

# %%
f_plot = gb_frequency[0]
for key, ind in indices.items():
    # plot across the sky
    gw_response_map = np.zeros(npix)
    temp_arr = np.asarray([el[tdi] for el in results_dict[f_plot]])
    for i in range(npix):
        res_list = np.abs(temp_arr[i, 1, ind])
        gw_response_map[i] += res_list
    hp.mollview(gw_response_map, title=f"Frequency {f_plot}", unit=ylabels[key], cmap="viridis")
    hp.graticule()
    plt.savefig(f"{key.lower()}_error_sky.png", dpi=300)
