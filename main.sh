## THIS SHELL SCRIPT IS WRITTEN FOR GROUPING ALL NECESSARY PYTHON SCRIPTS TO RUN THE PROJECT STEP BY STEP

## set working directory
work_folder=$(pwd)

## create a text file with names of all FNA files from a directory for easy use with MinHash
python ./scripts/write_genome_file.py --genome_dir ${work_folder}/files/organism_files_10kplus --out_file ${work_folder}/files/genome_file_10kplus.txt --suffix "fna.gz"

## Run MinHash to create a count estimator database (.h5) file.
/usr/bin/time -v python ./scripts/makeDNAdatabase.py --in_file ${work_folder}/files/genome_file_10kplus.txt --out_file ${work_folder}/files/database_test/cdb_n1000_k31/cdb.h5 --threads 128 --num_hashes 1000 --k_size 31

## Run experiment
echo 'N    Time Difference' >> comparison_op.txt
for count in 10 50 100 500 1000 5000 10000
do
    START=$(date +%s)
    python ./scripts/mut_test_script.py --set_auto --N $count --db_file ${work_folder}/files/database_test/cdb_n1000_k31/cdb.h5 --seed 10 --unif_param 0.01 0.07 --output_dir ${work_folder}/results
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo $count'    '$DIFF >> comparison_op.txt
done
