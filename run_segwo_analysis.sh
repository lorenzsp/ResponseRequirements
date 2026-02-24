echo "Running segwo analysis for static and evolving orbits with different time evaluations..."
python segwo_analysis.py --run_flag static
echo "\n"
echo "Running segwo analysis for evolving orbits with time evaluation of 60 days..."
python segwo_analysis.py --run_flag evolving --time_eval 45
echo "\n"
echo "Running segwo analysis for evolving orbits with time evaluation of 30 days..."
python segwo_analysis.py --run_flag evolving --time_eval 30
# python segwo_analysis.py --run_flag periodic_dev