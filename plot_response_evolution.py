"""
Plot the time evolution and frequency spectrum of the nominal response.

Loads results produced by run_evolution.py and generates:
  - response_evolution.png   (normalised amplitude & cosine-phase vs time)
  - frequency_spectrum.png   (TDI response amplitude vs frequency at selected times)

Usage
-----
    python plot_response_evolution.py [--data_file segwo_results/evolution_data.h5]
                                      [--output_dir .]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot nominal response: time evolution & frequency spectrum")
parser.add_argument('--data_file', type=str,
                    default="segwo_results/evolution_data.h5",
                    help="HDF5 file produced by run_evolution.py.")
parser.add_argument('--output_dir', type=str, default=".",
                    help="Directory where plots are saved.")
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

    strain2x_nominal = (hf["nominal/strain2x_real"][()]
                        + 1j * hf["nominal/strain2x_imag"][()])

print(f"  strain2x_nominal shape : {strain2x_nominal.shape}")
print(f"  frequencies            : {len(f)}")
print(f"  time points            : {len(array_ltts)}")

to_plot = strain2x_nominal

# Fixed indices
channel   = 0
pol       = 0
sky_index = 10

vec_f      = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
color_list = plt.cm.viridis(np.linspace(0, 1, len(vec_f)))

# ---------------------------------------------------------------------------
# Figure 1: Normalized amplitude and cosine-phase evolution
# ---------------------------------------------------------------------------
print("Generating response_evolution.png …")
fig, axes = plt.subplots(2, 1, figsize=(3.25*2, 5), sharex=True)

ax_amp, ax_phase = axes

for freq_, color in zip(vec_f, color_list):
    fidx = np.argmin(np.abs(f - freq_))
    amp  = np.abs(to_plot[:, fidx, sky_index, channel, pol])
    ax_amp.semilogy(array_ltts / 86400, amp / amp[0],
                    label=f"$f_0=${freq_:.0e} Hz", color=color)
    ax_phase.plot(array_ltts / 86400,
                  np.cos(np.angle(to_plot[:, fidx, sky_index, channel, pol])),
                  label=f"$f_0=${freq_:.0e} Hz", color=color)

ax_amp.set_ylabel("Normalized Amplitude")
ax_amp.legend(fontsize=8, loc='lower left')
ax_phase.set_xlabel("Time [days]")
ax_phase.set_ylabel("cos(Phase)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "response_evolution.png"), dpi=300)
plt.close()
print("  Saved response_evolution.png")

# ---------------------------------------------------------------------------
# Figure 2: Frequency spectrum at selected times
# ---------------------------------------------------------------------------
print("Generating frequency_spectrum.png …")
plt.figure(figsize=(3.25, 4))

# frequency-domain error (realization 1 vs nominal)
fft_response_nominal = np.fft.fft(strain2x_nominal,axis=0)
f_vec = np.fft.fftfreq(strain2x_nominal.shape[0], d=(array_ltts[1] - array_ltts[0]))
# perform shift to get positive frequencies in the right order
fft_response_nominal = np.fft.fftshift(fft_response_nominal, axes=0)
f_vec = np.fft.fftshift(f_vec)
for freq_, color in zip(vec_f, color_list):
    fidx = np.argmin(np.abs(f - freq_))
    plt.semilogy(f_vec, np.abs(fft_response_nominal[:, fidx, 10, 0 , 0]), label="$f_0=${:.0e} Hz".format(freq_), linestyle="-", color=color)

# plt.legend(bbox_to_anchor=(0.2,1.1))
plt.xlabel("$f_0 - f$ [Hz]")
plt.ylabel("Absolute value of FFT response")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frequency_spectrum.png"), dpi=300)
plt.close()
print("  Saved frequency_spectrum.png")

print(f"\nAll static plots saved to {output_dir}")
