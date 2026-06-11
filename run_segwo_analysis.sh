#!/usr/bin/env bash
# Full pipeline: compute → save HDF5 → plot figures.
# Each "run" step is independent, so you can run them separately if you want to save time.

# ---------------------------------------------------------------------------
# COMPUTE
# ---------------------------------------------------------------------------
# echo "=== [1/6] Computing: static orbits ==="
# python run_analysis.py --run_flag static --boost_flag 0
# python run_analysis.py --run_flag static --boost_flag 1

echo ""
echo "=== [2/6] Computing: evolving orbits at t=30 days ==="
python run_analysis.py --run_flag evolving --time_eval 30 --boost_flag 0
python run_analysis.py --run_flag evolving --time_eval 30 --boost_flag 1

# echo ""
# echo "=== [3/6] Computing: evolving orbits at t=0 days ==="
# python run_analysis.py --run_flag evolving --time_eval 0 --boost_flag 0
# python run_analysis.py --run_flag evolving --time_eval 0 --boost_flag 1

# python run_analysis.py --run_flag evolving --time_eval 365 --boost_flag 1

echo "All done."