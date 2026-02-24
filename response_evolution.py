import h5py
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits import InterpolatedOrbits
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter

from lisaconstants import c
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from segwo_utils import *
from tqdm import trange
np.random.seed(2601)

A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# Choices to run
# Define a frequency axis
# random time array for testing
# array_ltts = np.random.uniform(0, 365*86400, size=1)  # 100 random times over a year

f = np.logspace(-4, 0., 500)
array_ltts = np.linspace(0, 365*86400, 120)  # 100 time points over a year

# evolving's orbits
with h5py.File("processed_trajectories.h5", "r") as dset:
    t_orb_dataset = dset["t_interp"][()]
    x_orb_dataset = dset["spacecraft_positions"][()]
    v_orb_dataset = dset["spacecraft_velocities"][()]
    ltts = dset['owlt_12_23_31_13_32_21'][()]

t_orb = t_orb_dataset
x_orb = np.median(x_orb_dataset, axis=0)  # Use the median over all realizations
v_orb = np.median(v_orb_dataset, axis=0)  # Use the median over all realizations
ltts_median = np.median(ltts, axis=0)  # Use the median over all realizations
print(t_orb.shape, x_orb.shape, v_orb.shape)
distance = np.linalg.norm(x_orb[:,0] - x_orb[:,1],axis=-1)/1e9
print(f"Distance between SC1 and SC2: {distance.mean()} [m], std: {distance.std()} [Mm]")
realizations = x_orb_dataset.shape[0]
print(f"Number of realizations: {realizations}")
orbits = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)
# Compute the positions of the spacecraft and the light travel times at t=0.0
ltts = orbits.compute_ltt(t=array_ltts)
print("Light travel times shape", ltts.shape)
positions = orbits.compute_position(t=array_ltts)
print("Positions shape", positions.shape)

# other realization
x_orb = x_orb_dataset[1]
v_orb = v_orb_dataset[1]
orbits_real = InterpolatedOrbits(t_orb, x_orb, v_orb, interp_order=3)
ltts_real = orbits_real.compute_ltt(t=array_ltts)
positions_real = orbits_real.compute_position(t=array_ltts)

# plt.figure(); plt.hist((ltts_median[:1] - ltts).flatten()); plt.show()
# We use healpy to define a grid of points on the sky
nside = 6

npix = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
# Conversion from colatitude to latitude
betas, lambs = np.pi / 2 - thetas, phis

# Example usage
strain2x = compute_strain2x(f, betas, lambs, ltts, positions, orbits, A, E, T)
strain2x_real = compute_strain2x(f, betas, lambs, ltts_real, positions_real, orbits_real, A, E, T)
to_plot = strain2x# / strain2x_real

channel = 0  # Choose A, E, or T
pol = 0  # Choose polarization
sky_index = 10  # Choose a specific sky location (pixel index)
time_index = 0  # Choose a specific time index
frequency_index = int(len(f)*0.5) # Choose a specific frequency index

# plot of the amplitude and phase as function of time
plt.figure()
plt.subplot(2, 1, 1)
plt.title(f"Response evolution for frequency {f[frequency_index]:.2e} Hz, sky pixel {sky_index}, channel {channel}, pol {pol}")
vec_f = [1e-3, 5e-3, 1e-2, 5e-2]
color_list = plt.cm.viridis(np.linspace(0, 1, len(vec_f)))
for freq_, color in zip(vec_f, color_list):  # Plot a few frequencies for comparison
    strain2x_ = compute_strain2x(np.asarray([freq_]), betas, lambs, ltts, positions, orbits, A, E, T)
    plt.semilogy(array_ltts/86400, np.abs(strain2x_[:, 0, sky_index, channel, pol])/np.abs(strain2x_[0, 0, sky_index, channel, pol]), label=f"Frequency {freq_:.2e} Hz", color=color)
    # plt.semilogy(array_ltts/86400, np.abs(strain2x_[:, 0, sky_index, channel, pol]), label=f"Frequency {freq_:.2e} Hz", color=color)
plt.xlabel("Time [days]")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.subplot(2, 1, 2)
for freq_, color in zip(vec_f, color_list):  # Plot a few frequencies for comparison
    strain2x_ = compute_strain2x(np.asarray([freq_]), betas, lambs, ltts, positions, orbits, A, E, T)
    plt.plot(array_ltts/86400, np.cos(np.angle(strain2x_[:, 0, sky_index, channel, pol])), label=f"Frequency {freq_:.2e} Hz", color=color)
    # plt.axvline(1/freq_ * 86400, color=color, linestyle='--', alpha=0.5)  # Add vertical line at the period of the frequency
plt.xlabel("Time [days]")
plt.ylabel("Cosine Phase")
plt.tight_layout()
plt.savefig('response_evolution.png', dpi=300)
plt.show()

# --- Animation 1: frequency spectrum evolution (for a fixed sky pixel) ---
abs_out = np.abs(to_plot[:, :, sky_index, channel, pol])
fig1, ax1 = plt.subplots()
line, = ax1.loglog(f, np.abs(to_plot[0, :, sky_index, channel, pol]))
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("TDI Response Amplitude")
ax1.grid()

def update_spectrum(i):
    y = abs_out[i]
    line.set_data(f, y)
    ax1.relim()
    ax1.autoscale_view()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax1.set_ylim([1e-3, 20.0])  # Set a fixed lower limit for better visibility
    ax1.set_ylim([abs_out.min(), abs_out.max()])  # Set a fixed lower limit for better visibility
    ax1.set_title(f"Time {array_ltts[i]/86400:.1f} days")
    return (line,)

ani1 = animation.FuncAnimation(fig1, update_spectrum, frames=len(ltts), interval=400, blit=False)
ani1.save('spectrum_evolution.gif', writer=PillowWriter(fps=5))
print('Saved spectrum_evolution.gif')

gw_response_map = np.zeros((len(ltts),npix))
gw_response_map += np.abs(to_plot[:, frequency_index, :, channel, pol])
# --- Animation 2: sky response evolution (for a fixed frequency) ---
fig2 = plt.figure(figsize=(8, 4))
# Draw initial frame and capture the image artist (pass figure number to healpy)
hp.mollview(gw_response_map[0], fig=fig2.number, title=f"Time {array_ltts[0]/86400:.1f} days", cmap='viridis')
hp.graticule()

def update_map(i):
    # Clear the axes and redraw the mollview for the i-th frame
    fig2.clf()
    hp.mollview(gw_response_map[i], fig=fig2.number, title=f"Time {array_ltts[i]/86400:.1f} days", cmap='viridis', min=gw_response_map.min(), max=gw_response_map.max())
    hp.graticule()
    # Try to return the image artist created by healpy so writers have frames
    try:
        imgs = fig2.axes[0].images
        if len(imgs) > 0:
            return (imgs[0],)
    except Exception:
        pass
    return ()

ani2 = animation.FuncAnimation(fig2, update_map, frames=len(ltts), interval=400, blit=False)
ani2.save('sky_evolution.gif', writer=PillowWriter(fps=5))
print('Saved sky_evolution.gif')