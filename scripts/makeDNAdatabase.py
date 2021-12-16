import os
import sys
import MinHash as MH
import multiprocessing
from itertools import *
import argparse
import khmer
from khmer.khmer_args import optimal_size
import logging

# This function will make a single min hash sketch upon being given a file name, sketch size, prime, and k-mer size
def make_minhash(genome, max_h, prime, ksize):
	MHS = MH.CountEstimator(n=max_h, max_prime=prime, ksize=ksize, save_kmers='y', input_file_name=genome)
	# Just use HLL to estimate the number of kmers, no need to get exact count
	hll = khmer.HLLCounter(0.01, ksize)
	hll.consume_seqfile(genome)
	MHS._true_num_kmers = hll.estimate_cardinality()
	MHS.input_file_name = genome
	return MHS


# Unwrap for Python2.7
def make_minhash_star(arg):
	return make_minhash(*arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script creates training/reference sketches for each FASTA/Q file listed in the input file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prime', help='Prime (for modding hashes)', default=9999999999971)
    parser.add_argument('--threads', type=int, help="Number of threads to use", default=-1)
    parser.add_argument('--num_hashes', type=int, help="Number of hashes to use.", default=500)
    parser.add_argument('--k_size', type=int, help="K-mer size", default=21)
    parser.add_argument('--intersect_nodegraph', action="store_true", help="Optional flag to export Nodegraph file (bloom filter) containing all k-mers in the training database. Saved in same location as out_file. This is to be used with QueryDNADatabase.py")
    parser.add_argument('--in_file', help="Input file: file containing (absolute) file names of training genomes.")
    parser.add_argument('--out_file', help='Output training database/reference file (in HDF5 format)')
    args = parser.parse_args()

	## set up basic log information
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.threads == -1:
        logging.info(f"The parameter 'threads' is set to the maximum available threads {multiprocessing.cpu_count()} in this machine")
        num_threads = multiprocessing.cpu_count()
    elif 0 < args.threads > multiprocessing.cpu_count():
        logging.info(f"The parameter 'threads' is set larger than the maximum number of threads in this machine. So, set it to its maximum available threads {multiprocessing.cpu_count()}")
        num_threads = multiprocessing.cpu_count()
    else:
        logging.info(f"The parameter 'threads' is set to {args.threads}")
        num_threads = args.threads

    if args.k_size > 31:
        logging.error("Unfortunately, ksize must be size 32 or smaller (due to khmer contraints). Please reduce the ksize or use MakeStreamingDNADatabase.py instead.")
        exit(0)

    input_file_names = os.path.abspath(args.in_file)
    if not os.path.exists(input_file_names):
        logging.error(f"Input file {input_file_names} does not exist.")
        exit(0)

    ## check output path
    out_file = os.path.abspath(args.out_file)
    if not os.path.exists(os.path.dirname(out_file)):
        logging.warning(f"Output directory {os.path.dirname(out_file)} does not exist. We're going to create such directory")
        os.makedirs(os.path.dirname(out_file))

    if args.intersect_nodegraph is True:
        intersect_nodegraph_file = os.path.splitext(out_file)[0] + ".intersect.Nodegraph"
    else:
        intersect_nodegraph_file = None

    file_names = list()
    fid = open(input_file_names, 'r')
    with open(input_file_names, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            if not os.path.exists(line):
                logging.error(f"Training genome {line} does not exist.")
                exit()
            file_names.append(line)

	# Open the pool and make the sketches
    with multiprocessing.Pool(processes=num_threads) as pool:
        genome_sketches = pool.map(make_minhash_star, zip(file_names, repeat(args.num_hashes), repeat(args.prime), repeat(args.k_size)))


	# Export all the sketches
    MH.export_multiple_to_single_hdf5(genome_sketches, out_file)

	# If requested, save all the k-mers into a big Nodegraph (unfortunately, need to pass through the data again since we
	# a-priori don't know how big of a table we need to make
    if intersect_nodegraph_file is not None:
        total_num_kmers = 0
        for sketch in genome_sketches:
            total_num_kmers += sketch._true_num_kmers
        res = optimal_size(total_num_kmers, fp_rate=0.001)
        intersect_nodegraph = khmer.Nodegraph(args.k_size, res.htable_size, res.num_htables)
        for file_name in file_names:
            intersect_nodegraph.consume_seqfile(file_name)
        intersect_nodegraph.save(intersect_nodegraph_file)
