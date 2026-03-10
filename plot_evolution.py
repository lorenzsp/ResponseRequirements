"""
Plot script: loads results produced by run_evolution.py and generates all
time-evolution figures and animations.

No heavy computation is performed here.

Usage
-----
    python plot_evolution.py [--data_file segwo_results/evolution_data.h5]
                             [--output_dir .]
"""

import argparse
import os

import h5py
import healpy as hp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

plt.rcParams['text.usetex'] = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SEGWO Evolution — Plot Only")
parser.add_argument('--data_file', type=str,
                    default="segwo_results/evolution_data.h5",
                    help="HDF5 file produced by run_evolution.py.")
parser.add_argument('--output_dir', type=str, default=".",
                    help="Directory where plots and animations are saved.")
args = parser.parse_args()

hdf5_path  = args.data_file
output_dir = args.output_dir.rstrip('/') + '/'
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(hdf5_path):
    raise FileNotFoundError(
        f"{hdf5_path} not found. Run run_evolution.py first."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading data from {hdf5_path} …")
with h5py.File(hdf5_path, "r") as hf:
    f           = hf["frequencies"][()]
    array_ltts  = hf["time_points"][()]
    betas       = hf["betas"][()]
    lambs       = hf["lambs"][()]
    npix        = int(hf.attrs["npix"])
    nside       = int(hf.attrs["nside"])

    strain2x_nominal = (hf["nominal/strain2x_real"][()] +
                        1j * hf["nominal/strain2x_imag"][()])

print(f"  strain2x_nominal shape : {strain2x_nominal.shape}")
print(f"  frequencies            : {len(f)}")
print(f"  time points            : {len(array_ltts)}")

# We plot the nominal (median) orbit response
to_plot = strain2x_nominal

# Fixed indices for single-pixel / channel / pol inspection
channel   = 0
pol       = 0
sky_index = 10

# Extra frequencies used in the time-evolution panel
vec_f       = [1e-3, 5e-3, 1e-2, 5e-2]
color_list  = plt.cm.viridis(np.linspace(0, 1, len(vec_f)))

# ---------------------------------------------------------------------------
# Figure 1: Normalized amplitude and cosine-phase evolution
# ---------------------------------------------------------------------------
print("Generating response_evolution.png …")
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax_amp, ax_phase = axes
ax_amp.set_title(
    f"Response evolution — sky pixel {sky_index}, "
    f"channel {'AET'[channel]}, pol {pol}"
)

for freq_, color in zip(vec_f, color_list):
    fidx = np.argmin(np.abs(f - freq_))        # exact: freq_ is in f by construction
    amp  = np.abs(to_plot[:, fidx, sky_index, channel, pol])
    ax_amp.semilogy(array_ltts / 86400, amp / amp[0],
                    label=f"{freq_:.2e} Hz", color=color)
    ax_phase.plot(array_ltts / 86400,
                  np.cos(np.angle(to_plot[:, fidx, sky_index, channel, pol])),
                  label=f"{freq_:.2e} Hz", color=color)

ax_amp.set_ylabel("Normalized Amplitude")
ax_amp.legend(fontsize=8)
ax_phase.set_xlabel("Time [days]")
ax_phase.set_ylabel("cos(Phase)")
ax_phase.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "response_evolution.png"), dpi=300)
plt.close()
print("  Saved response_evolution.png")

# ---------------------------------------------------------------------------
# Animation 1: Frequency-spectrum evolution (fixed sky pixel)
# ---------------------------------------------------------------------------
print("Generating spectrum_evolution.gif …")
abs_out = np.abs(to_plot[:, :, sky_index, channel, pol])   # (T, F)

fig1, ax1 = plt.subplots()
(line,) = ax1.loglog(f, abs_out[0])
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("TDI Response Amplitude")
ax1.grid()
ax1.set_ylim([abs_out[abs_out > 0].min(), abs_out.max()])


def update_spectrum(i):
    line.set_data(f, abs_out[i])
    ax1.set_title(f"Time {array_ltts[i] / 86400:.1f} days")
    return (line,)


ani1 = animation.FuncAnimation(fig1, update_spectrum,
                                frames=len(array_ltts), interval=400, blit=False)
ani1.save(os.path.join(output_dir, "spectrum_evolution.gif"),
          writer=PillowWriter(fps=5))
plt.close(fig1)
print("  Saved spectrum_evolution.gif")

# ---------------------------------------------------------------------------
# Animation 2: Sky-response map evolution (fixed frequency)
# ---------------------------------------------------------------------------
print("Generating sky_evolution.gif …")
frequency_index = len(f) // 2
gw_response_map = np.abs(to_plot[:, frequency_index, :, channel, pol])  # (T, P)

fig2 = plt.figure(figsize=(8, 4))
hp.mollview(gw_response_map[0], fig=fig2.number,
            title=f"Time {array_ltts[0] / 86400:.1f} days", cmap='viridis')
hp.graticule()


def update_map(i):
    fig2.clf()
    hp.mollview(gw_response_map[i], fig=fig2.number,
                title=f"Time {array_ltts[i] / 86400:.1f} days",
                cmap='viridis',
                min=gw_response_map.min(), max=gw_response_map.max())
    hp.graticule()
    try:
        imgs = fig2.axes[0].images
        if imgs:
            return (imgs[0],)
    except Exception:
        pass
    return ()


ani2 = animation.FuncAnimation(fig2, update_map,
                                frames=len(array_ltts), interval=400, blit=False)
ani2.save(os.path.join(output_dir, "sky_evolution.gif"),
          writer=PillowWriter(fps=5))
plt.close(fig2)
print("  Saved sky_evolution.gif")

print(f"\nAll evolution plots saved to {output_dir}")
