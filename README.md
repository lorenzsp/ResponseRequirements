# ResponseRequirements

## Overview

This repository contains computational tools for quantifying **uncertainties and mismodeling** in the gravitational wave response of the **Laser Interferometer Space Antenna (LISA)**—a space-based gravitational wave observatory planned for the mid-2030s.

### Key Research Questions

1. **Orbit Uncertainties**: How do uncertainties in spacecraft positions and velocities (determined via ground-based radiometric tracking) propagate into the gravitational wave response?

2. **Response Mismodeling**: What is the impact of neglecting spacecraft velocity-dependent terms in the response kernel on parameter estimation and source localization?

### Main Contributions

- Comprehensive waveform **mismatch analysis** across frequency, sky position, and polarization
- Comparison of **static toy constellation** models with realistic **evolving orbits** from mission-representative orbit-determination simulations
- Validation using **galactic binary inference** to quantify parameter bias and localization errors
- High-performance implementations in JAX for fast response calculations

The code leverages ESA's **GODOT** astrodynamics library and **MIDAS** orbit-determination package to generate realistic spacecraft trajectories, then evaluates mismatches between responses computed with nominal and perturbed orbital parameters.

![LISA orbits](report/figures/3d_orbit_around_sun.png)

## Key Findings

- **Orbit uncertainties** produce negligible mismatches across the LISA band, with effects dominated at higher frequencies (∝ f²)
- **Velocity mismodeling** (neglecting v/c terms) has larger impact at low frequencies (∝ f⁻²), reaching ~10⁻³ mismatch at 0.1 mHz
- MCMC analysis shows mild parameter biases (~0.6σ) when using velocity-neglecting templates on high-SNR galactic binary signals

## Installation
Follow these steps to set up the project on your local machine:


1. **Clone the repository**:
    ```bash
    git clone https://github.com/lorenzsp/ResponseRequirements.git
    cd ResponseRequirements
    ```

2. **Install dependencies**:
    Ensure you have conda installed. Then run:
    ```bash
    conda create -n lisa_resp -c conda-forge gcc_linux-64 gxx_linux-64 python=3.12
    conda activate lisa_resp
    pip install lisaorbits lisaconstants pytdi tqdm numpy Cython scipy jupyter ipython h5py matplotlib tqdm
    ```

3. **Install plotting dependencies** (for figure reproduction):
    ```bash
    pip install healpy corner eryn
    ```

## Usage

The main analysis workflow is orchestrated by:

```bash
bash run_segwo_analysis.sh
```

This script runs:
- **Orbit perturbation analysis** (static toy model)
- **Orbit-determination simulation** (realistic evolving orbits)
- **Mismatch computation** across frequency and sky
- **Figure generation** for publication

### Key Scripts

- `gb_bias_analysis.py` — Differential evolution optimization to study parameter bias under model mismatch
- `gb_fit.py` — MCMC inference comparing relativistic vs. non-relativistic response templates
- `test_nonrel_models.py` — Validation of differences between old and new response models
- `PaperPlots.ipynb` — Publication-quality figure generation from full results
- `PreprocessSegwoResults.ipynb` — **Lightweight preprocessing** to extract minimal data from `segwo_results/`
- `PreprocessedPaperPlots.ipynb` — **Fast plot generation** from preprocessed data only

## Reproducing Publication Figures

### Quick Start (Recommended for Distribution)

If you have limited storage or want to share reproducible results without 100+ MB data files:

```bash
# Step 1: Preprocess segwo_results/ → minimal HDF5 files (~15 MB total)
jupyter notebook PreprocessSegwoResults.ipynb
# This extracts frequency-averaged statistics and residual distributions

# Step 2: Generate all publication figures
jupyter notebook PreprocessedPaperPlots.ipynb
# Output: paper_plots/ directory with all plots
```

**Advantages**:
- ✓ Only ~15 MB of data needed (vs 100+ MB for full `segwo_results/`)
- ✓ Figures generate in seconds
- ✓ Ideal for GitHub distribution and reproducibility

### Full Reproducibility (With All Raw Data)

If you have the complete `segwo_results/` directory from the simulation:

```bash
# Generate plots directly from full results
jupyter notebook PaperPlots.ipynb
```

**Note**: The full notebook requires `segwo_results/` (~100+ MB) but provides access to:
- Individual realization data for sensitivity analysis
- Full sky maps and frequency-sky contours
- Raw Monte Carlo samples for custom histograms

### Data Files Required

**For lightweight workflow** (`PreprocessedPaperPlots.ipynb`):
```
data/
  ├── segwo_results_processed.h5      (~10 MB) — Frequency-averaged statistics
  ├── segwo_sky_maps.h5              (~5 MB)  — HEALPix sky maps
  ├── processed_trajectories.h5              — Orbit plots (already in repo)
  ├── link_response_maps.h5                  — Link response evolution
  ├── response_evolution_plot.h5             — Response amplitude/phase
  ├── gb_mismatch_results_15.0days.h5       — Galactic binary mismatches
  ├── gb_bias_results_15.0days.h5           — Parameter biases
  └── mcmc_chains_processed.h5              — MCMC posteriors (optional)
```

**For full workflow** (`PaperPlots.ipynb`):
```
segwo_results/
  ├── static/
  │   ├── arm1_rot0.0_trans0.0_boost0.0/
  │   ├── arm1_rot0.0_trans0.0_boost1.0/
  │   ├── arm0.0_rot50000.0_trans0.0_boost0.0/
  │   ├── arm0.0_rot50000.0_trans0.0_boost1.0/
  │   ├── arm0.0_rot0.0_trans50000.0_boost0.0/
  │   └── arm0.0_rot0.0_trans50000.0_boost1.0/
  ├── 15.0days_evolving_boost0.0/
  └── 15.0days_evolving_boost1.0/
```

### Generated Figures

Both workflows produce the same publication-ready plots in `paper_plots/`:

1. `perturbation_distributions.pdf` — Histograms of orbital residuals (LTT, position, angle)
2. `amplitude_phase_errors.png` — Strain error evolution with frequency
3. `mismatch_vs_frequency.pdf` — Mismatch across the LISA band
4. `mismatch_sky_*.pdf` — HEALPix Mollweide projections (optional)
5. `response_amplitude_phase_evolution.png` — Link response time evolution
6. `link_response_evolution.png` — Response contours
7. `gb_mismatch_plot.png` — Galactic binary mismatch comparisons
8. `gb_parameter_biases.png` — Parameter systematic errors
9. `mcmc_corner_overlay.pdf` — MCMC posterior comparison (optional)

### Storage Considerations

| Workflow | Data Size | Generation Time | Best For |
|----------|-----------|-----------------|----------|
| **Lightweight** (`PreprocessedPaperPlots.ipynb`) | ~15 MB | ~10 sec | Reproducibility, distribution, CI/CD |
| **Full** (`PaperPlots.ipynb`) | ~100+ MB | ~30 sec | Raw data access, sensitivity analysis |

**Option 1**: Commit lightweight data to GitHub (recommended)
```bash
git add data/segwo_results_processed.h5 data/segwo_sky_maps.h5
git lfs track "*.h5"  # If using Git LFS
git commit -m "Add preprocessed SEGWO results for figure reproduction"
```

**Option 2**: Host full data on Zenodo/OSF
```bash
# Upload segwo_results/ to persistent repository
# Reference in paper and README with DOI
```

**Option 3**: Generate on-demand
```bash
# Include segwo_results/ generation instructions in CI/CD pipeline
# Keep only metadata/plots in repository
```

### Troubleshooting Figure Reproduction

**"FileNotFoundError: Missing segwo_results/..."**
- You don't have the full simulation data. Use `PreprocessedPaperPlots.ipynb` instead (lightweight)
- Or run `PreprocessSegwoResults.ipynb` first if you have `segwo_results/`

**"KeyError: 'segwo_results_processed.h5'"**
- Run `PreprocessSegwoResults.ipynb` to generate it from `segwo_results/`
- Make sure you're in the correct directory

**"ImportError: lisaorbits, lisaconstants, etc."**
- Install missing dependencies:
```bash
pip install lisaorbits lisaconstants pytdi healpy corner eryn
```

**Fast iteration for paper edits**:
```bash
# Modify plotting code in PreprocessedPaperPlots.ipynb
# Figures regenerate in seconds (no preprocessing needed)
```




## Citation

If you use this code or data in your research, please cite the following paper:

**BibTeX entry:**
```bibtex
@article{Speri2026,
  author = {Speri, Lorenzo and Hartwig, Olaf and Martens, Waldemar and Jennrich, Oliver and Joffre, Eric and Armano, Michele and Hewitson, Martin and L\"utzgendorf, Nora},
  title = {Uncertainties and mismodeling of the gravitational wave response of the {L}aser {I}nterferometer {S}pace {A}ntenna},
  journal = {},
  year = {2026},
  note = {In preparation},
  url = {https://github.com/lorenzsp/ResponseRequirements}
}
```

## Contact
For questions or feedback, please contact [Lorenzo Speri](https://github.com/lorenzsp).