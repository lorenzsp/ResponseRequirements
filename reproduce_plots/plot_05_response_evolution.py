"""
Plot 05 — Response amplitude and phase evolution vs time.

Lines coloured by frequency showing how the normalised amplitude and
cos(phase) of the LISA response evolve over the observation window.

Requires:
  <workspace_root>/data/response_evolution_plot.h5   (already lightweight)

Output:
  reproduce_plots/figures/response_amplitude_phase_evolution.png
"""

from pathlib import Path
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import apply_style, DATA_DIR, OUTPUT_DIR, COL1

apply_style()

# ── Load data ─────────────────────────────────────────────────────────────────
with h5py.File(DATA_DIR / "response_evolution_plot.h5", "r") as ds:
    array_ltts_  = ds["array_ltts"][:]
    f_           = ds["frequencies"][:]
    phase_map_   = ds["phase_map"][:]
    amp_map_     = ds["amp_map"][:]

t_days    = array_ltts_ / 86400.0
freq_vals = np.asarray(f_)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(COL1 * 2, COL1), sharex=True)

if np.all(freq_vals > 0):
    norm      = mcolors.LogNorm(vmin=freq_vals.min(), vmax=freq_vals.max())
    cbar_label = "Frequency [Hz]"
else:
    norm      = mcolors.Normalize(vmin=freq_vals.min(), vmax=freq_vals.max())
    cbar_label = "Frequency"

cmap   = plt.cm.viridis
colors = [cmap(norm(fi)) for fi in freq_vals]

for fi, color in zip(freq_vals, colors):
    i = np.where(freq_vals == fi)[0][0]
    ax_amp.plot(t_days, amp_map_[i] / amp_map_[i][0], color=color, lw=2,
                label=f"{fi:.2} Hz")
    ax_phase.plot(t_days, phase_map_[i], color=color, lw=2)

ax_amp.set_ylabel("Normalized Amplitude")
ax_amp.legend(title="Frequency", bbox_to_anchor=(1.00, 1), loc='upper left',
              title_fontsize=10)

ax_phase.set_ylabel(r"Cos(Response Phase)")
ax_phase.set_xlabel("Time [days]")

plt.tight_layout()
out = OUTPUT_DIR / "response_amplitude_phase_evolution.png"
plt.savefig(out, dpi=300)
print(f"✓ {out.name}")
plt.show()
