# reproduce_plots

Self-contained scripts to reproduce every saved figure in the paper,
each reading only lightweight HDF5 files.

## Directory layout

```
reproduce_plots/
├── config.py                    shared matplotlib style & path constants
├── save_all_data.py             data-extraction step (run once)
├── data/                        lightweight HDF5 files (created by save_all_data.py)
│   ├── orbit_data.h5
│   ├── perturbation_data.h5
│   ├── mismatch_frequency_data.h5
│   ├── amplitude_phase_errors.h5
│   └── gb_segwo_mismatch.h5
├── figures/                     output PDFs/PNGs (created automatically)
├── plot_01_orbit.py             LISA + toy static orbit (3-D)
├── plot_02_static_perturbations.py   static orbit perturbation distributions
├── plot_03_evolving_perturbations.py evolving orbit perturbation distributions
├── plot_04_link_response.py     link response amplitude/phase maps
├── plot_05_response_evolution.py    response amplitude & phase vs time
├── plot_06_amplitude_phase_errors.py amplitude & phase error vs frequency
├── plot_07_mismatch_vs_frequency.py  mismatch vs GW frequency (all cases)
├── plot_08_gb_mismatch.py       GB mismatch comparison
└── plot_09_mcmc_corner.py       MCMC corner overlay (boosted vs non-boosted)
```

## Quick start

**Step 1 — extract lightweight data** (needs the full `segwo_results/` tree):

```bash
cd <workspace_root>
python reproduce_plots/save_all_data.py
```

This writes five compact HDF5 files into `reproduce_plots/data/`.
The following files from `<workspace_root>/data/` are used directly
and need no extraction step:

| File | Used by |
|------|---------|
| `data/link_response_maps.h5` | plot_04 |
| `data/response_evolution_plot.h5` | plot_05 |
| `data/mcmc_chains_processed.h5` | plot_09 |
| `data/gb_mismatch_results_15.0days.h5` | plot_08 |
| `data/gb_bias_results_15.0days.h5` | plot_08 |

**Step 2 — reproduce any individual figure**:

```bash
cd <workspace_root>
python reproduce_plots/plot_07_mismatch_vs_frequency.py
```

All figures are saved to `reproduce_plots/figures/`.

## Figure summary

| Script | Output file | Data source(s) |
|--------|-------------|----------------|
| `plot_01_orbit.py` | `lisa_orbit.png`, `toy_orbit.png` | `data/orbit_data.h5` |
| `plot_02_static_perturbations.py` | `static_orbit_perturbation_distributions.pdf` | `data/perturbation_data.h5` |
| `plot_03_evolving_perturbations.py` | `evolving_orbit_perturbation_distributions.pdf` | `data/perturbation_data.h5` |
| `plot_04_link_response.py` | `link_response_evolution.png` | `../data/link_response_maps.h5` |
| `plot_05_response_evolution.py` | `response_amplitude_phase_evolution.png` | `../data/response_evolution_plot.h5` |
| `plot_06_amplitude_phase_errors.py` | `amplitude_phase_errors.png` | `data/amplitude_phase_errors.h5` |
| `plot_07_mismatch_vs_frequency.py` | `mismatch_vs_frequency.pdf` | `data/mismatch_frequency_data.h5` |
| `plot_08_gb_mismatch.py` | `gb_mismatch_plot.png` | `data/gb_segwo_mismatch.h5` + `../data/gb_*.h5` |
| `plot_09_mcmc_corner.py` | `mcmc_corner_overlay.pdf` | `../data/mcmc_chains_processed.h5` |
