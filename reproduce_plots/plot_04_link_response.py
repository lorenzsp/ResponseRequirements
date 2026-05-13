"""
Plot 04 — Link response evolution (amplitude and phase contour maps).

Requires:
  <workspace_root>/data/link_response_maps.h5   (already lightweight)

Output:
  reproduce_plots/figures/link_response_evolution.png
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
with h5py.File(DATA_DIR / "link_response_maps.h5", "r") as ds:
    f_          = ds["frequencies"][()]
    amp_map     = ds["amp_map"][()]
    phase_map   = ds["phase_map"][()]
    array_ltts  = ds["array_ltts"][()]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(COL1, COL1), sharex=True)

positive_vals = amp_map[amp_map > 0]
vmin = positive_vals.min()

cf1 = axes[0].contourf(
    array_ltts / 86400,
    f_,
    amp_map,
    levels=5,
    norm=mcolors.LogNorm(vmin=vmin, vmax=amp_map.max()),
)
fig.colorbar(cf1, ax=axes[0], label='Absolute Value')
axes[0].set_ylabel('Frequency [Hz]')
axes[0].set_yscale('log')

cf2 = axes[1].contourf(array_ltts / 86400, f_, phase_map)
fig.colorbar(cf2, ax=axes[1], label='Phase')
axes[1].set_xlabel('Time [days]')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].set_yscale('log')

plt.tight_layout()
out = OUTPUT_DIR / "link_response_evolution.png"
plt.savefig(out, dpi=300)
print(f"✓ {out.name}")
plt.show()
