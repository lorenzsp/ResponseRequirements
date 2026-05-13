#!/usr/bin/env bash
# Run the full reproduce_plots pipeline from the workspace root.
# Usage:  bash reproduce_plots/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT"

# echo "============================================================"
# echo " Step 1 — Extract lightweight data"
# echo "============================================================"
# python reproduce_plots/save_all_data.py

echo ""
echo "============================================================"
echo " Step 2 — Reproduce all plots"
echo "============================================================"

PLOTS=(
    reproduce_plots/plot_01_orbit.py
    reproduce_plots/plot_02_static_perturbations.py
    reproduce_plots/plot_03_evolving_perturbations.py
    reproduce_plots/plot_04_link_response.py
    reproduce_plots/plot_05_response_evolution.py
    reproduce_plots/plot_06_amplitude_phase_errors.py
    reproduce_plots/plot_07_mismatch_vs_frequency.py
    reproduce_plots/plot_08_gb_mismatch.py
    reproduce_plots/plot_09_mcmc_corner.py
)

for script in "${PLOTS[@]}"; do
    echo ""
    echo "── $(basename "$script") ──"
    python "$script"
done

echo ""
echo "============================================================"
echo " All figures saved to reproduce_plots/figures/"
echo "============================================================"
