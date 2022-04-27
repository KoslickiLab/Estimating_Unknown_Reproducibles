## THIS SHELL SCRIPT IS WRITTEN FOR GROUPING ALL NECESSARY PYTHON SCRIPTS TO RUN THE PROJECT STEP BY STEP

## set working directory
work_folder=$(pwd)

## Run experiment
python ${work_folder}/scripts/run_tests.py --N -1 --seed 123 --T 100 --cpu 20 --savepath ${work_folder}/processed_files/processed_Nall_seed123.pkl
