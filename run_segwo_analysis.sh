#!/usr/bin/env bash
# Full pipeline: compute → save HDF5 → plot figures.
# Each "run" step is independent; re-run only the "plot" step to regenerate figures.

set -e

# ---------------------------------------------------------------------------
# COMPUTE
# ---------------------------------------------------------------------------
echo "=== [1/6] Computing: static orbits ==="
python run_analysis.py --run_flag static

echo ""
echo "=== [2/6] Computing: evolving orbits at t=15 days ==="
python run_analysis.py --run_flag evolving --time_eval 15

echo ""
echo "=== [3/6] Computing: evolving orbits at t=0 days ==="
python run_analysis.py --run_flag evolving --time_eval 0

# echo ""
# echo "=== [4/6] Computing: response evolution (nominal + realization 1) ==="
# python run_evolution.py

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/6] Plotting: static ==="
python plot_analysis.py --results_dir segwo_results/static/

echo ""
echo "=== [5b] Plotting: evolving at t=15 days ==="
python plot_analysis.py --results_dir "segwo_results/15.0daysevolving/"

echo ""
echo "=== [5c] Plotting: evolving at t=0 days ==="
python plot_analysis.py --results_dir "segwo_results/0.0daysevolving/"

# echo ""
# echo "=== [6/6] Plotting: response evolution ==="
# python plot_evolution.py --data_file segwo_results/evolution_data.h5 --output_dir .

echo ""
echo "All done."