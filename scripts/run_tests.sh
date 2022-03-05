## THIS SHELL SCRIPT IS WRITTEN FOR GROUPING ALL NECESSARY PYTHON SCRIPTS TO RUN THE PROJECT STEP BY STEP

## set working directory
work_folder=$(pwd)

## Run experiment
for count in 500 1000 2000
do
    python run_tests.py --N $count --seed 12345 --T 100
done
