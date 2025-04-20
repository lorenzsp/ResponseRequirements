import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import sawtooth
from scipy.signal.windows import tukey
from fastlisaresponse import pyResponseTDI, ResponseWrapper
from astropy import units as un
from lisatools.utils.constants import *
# make nice plots
np.random.seed(2601)

from utils import *
import os

fpath = "new_orbits.h5"
T = 1.0#/12  # years
plot_orbit_3d(fpath, T, Nshow=12)

################################################################
use_gpu = False
gb = GBWave(use_gpu=use_gpu )
dt = 5.0

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
    dt = orb_default.dt
    for ii in range(3):
        for sc in range(1):
            ax[ii].plot(np.arange(arr.shape[0]) * dt / 86400, 
                        (arr_def[:, sc, ii]-arr[:, sc, ii])/1e3, 
                        label=f"SC{sc}",color=coord_color[sc][1], alpha=0.3)
            
        ax[ii].axhline(list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].axhline(-list_sigma[ii]/1e3, linestyle='--', color='k')
        ax[ii].set_ylabel(coord_color[ii][0])
        # ax[ii].set_xlim(0, 30)
        

    ax[2].set_xlabel("Time [days]")

# ax[0].set_title(f"SC Coordinates")
plt.xlim([0.0, 30])

plt.savefig("orbits_deviation.png")