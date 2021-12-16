import os
import numpy as np
import make_data
import freq_est_from_mut as ferm
import argparse
import logging

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script runs random tests on a given organism dictionary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--abundance_list', help='A list of Relative abundance of organisms in simulated sample', default=[0.05], type = float, nargs = '*')
	parser.add_argument('--mut_rate_list', help='A list of mutation rate of organisms in simulated sample', default=[0.05], type = float, nargs = '*')
	parser.add_argument('--db_file', default = "files/databases/cdb_n1000_k31/cdb.h5", help='file with count estimators')
	parser.add_argument('--weight', help='False negative discount weight', default=None, type = float)
	parser.add_argument('--seed', help='Random seed', default=None, type = int)
	args = parser.parse_args()

	## set up basic log information
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	## set up random seed
	if args.seed is not None:
		np.random.seed(args.seed)

	## basic checking
	N = len(args.abundance_list)
	if len(args.mut_rate_list) != N:
		logging.error("r1 and abundance must be same length.")
		exit(0)

	## calculate original data
	orig_A_matrix = make_data.get_original_data(db_file=args.db_file, N=N, filepath = os.path.dirname(args.db_file))

	## calculate mutated data
	mut_organisms = make_data.get_mutated_data(orig_A_matrix, args.abundance_list, args.mut_rate_list)

	FE = ferm.frequency_estimator(mut_organisms, w=args.weight)
	print("Recovered frequencies at " + str(0.05) + " mutation threshold:")
	print(np.round(FE.freq_est,4))