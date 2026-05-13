#!/usr/bin/env bash
# Full pipeline: compute → save HDF5 → plot figures.
# Each "run" step is independent; re-run only the "plot" step to regenerate figures.

set -e

# ---------------------------------------------------------------------------
# COMPUTE
# ---------------------------------------------------------------------------
echo "=== [1/6] Computing: static orbits ==="
python run_analysis.py --run_flag static --boost_flag 0
python run_analysis.py --run_flag static --boost_flag 1

echo ""
echo "=== [2/6] Computing: evolving orbits at t=15 days ==="
python run_analysis.py --run_flag evolving --time_eval 15 --boost_flag 0
python run_analysis.py --run_flag evolving --time_eval 15 --boost_flag 1

echo ""
echo "=== [3/6] Computing: evolving orbits at t=0 days ==="
python run_analysis.py --run_flag evolving --time_eval 0 --boost_flag 0
python run_analysis.py --run_flag evolving --time_eval 0 --boost_flag 1

echo "All done."