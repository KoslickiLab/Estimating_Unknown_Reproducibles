## THIS SHELL SCRIPT IS WRITTEN FOR GROUPING ALL NECESSARY PYTHON SCRIPTS TO RUN THE PROJECT STEP BY STEP

## set working directory
work_folder=$(pwd)

## complie the cython extension scripts
# cd ${work_folder}/scripts
# bash ./compile.sh
# cd ${work_folder}

# ## create a text file with names of all FNA files from a directory for easy use with MinHash
# python ./scripts/write_genome_file.py --genome_dir ${work_folder}/files/organism_files --out_file ${work_folder}/files/genome_file.txt

# ## Run MinHash to create a count estimator database (.h5) file.
python ./scripts/makeDNAdatabase.py --in_file ${work_folder}/files/genome_file.txt --out_file ${work_folder}/files/database_test/cdb_n1000_k31/cdb.h5 --threads 8 --num_hashes 1000 --k_size 31

## Run experiment
# python mut_test_script.py -a 0.3 0.2 0.5 -r1 0.01 0.02 0.07 -seed 10