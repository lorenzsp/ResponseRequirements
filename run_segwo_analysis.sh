echo "Running segwo analysis for static and evolving orbits with different time evaluations..."
python segwo_analysis.py --run_flag static
echo "\n"
echo "Running segwo analysis for evolving orbits with time evaluation of 15 days..."
python segwo_analysis.py --run_flag evolving --time_eval 15
echo "\n"
echo "Running segwo analysis for evolving orbits with time evaluation of 0 days..."
python segwo_analysis.py --run_flag evolving --time_eval 0