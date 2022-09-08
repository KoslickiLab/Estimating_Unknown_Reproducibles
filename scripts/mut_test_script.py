import os
import numpy as np
import make_data
import freq_est_from_mut as ferm
import est_eval as ee
import argparse
import logging
import pandas as pd
import MinHash as mh

#######################
#DEPRECATED: SW TO FIX#
#######################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script runs random tests on a given organism dictionary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--abundance_list', help='A list of Relative abundance of organisms in simulated sample', default=[0.05], type = float, nargs = '+')
	parser.add_argument('--mut_rate_list', help='A list of mutation rate of organisms in simulated sample', default=[0.05], type = float, nargs = '+')
	parser.add_argument('--db_file', default = "files/databases/cdb_n1000_k31/cdb.h5", help='file with count estimators')
	parser.add_argument('--set_auto', action='store_true', help='Automatically set abundance list and mutation rate list')
	parser.add_argument('--N', help='If set --set_auto, then this parameter is used to generate # of abundance and mutation rate (default: 100, set -1 to use all genomes)', default=100, type = int)
	parser.add_argument('--unif_param', help='If set --set_auto, then this parameter is used as the parameters of uniform distribution to generate mutation rate', default=[0.01, 0.07], type = float, nargs = '+')
	parser.add_argument('--weight', help='False negative discount weight', default=None, type = float)
	parser.add_argument('--seed', help='Random seed', default=None, type = int)
	parser.add_argument('--output_dir', help='Output directory', default='../results', type = str)
	parser.add_argument('--savepath', help='File location to save processed dictionary', type = str)
	parser.add_argument('--loadpath', help='File location to load processed dictionary from', type = str)
	args = parser.parse_args()

	## set up basic log information
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
	logging.info('Program starts running...')

	## set up random seed
	if args.seed is not None:
		np.random.seed(args.seed)

	## get metadata from database
	args.metadata = mh.get_info_from_single_hdf5(args.db_file)

	## basic checking
	if args.set_auto is True:
		logging.info('Abundance list (follows dirichlet distribution) and mutation rate list (follows uniform distribution) are automatically set!')
		if args.N <= 0 and args.N != -1:
			logging.error('N is smaller or equal to 0. Please set it larger than 0 or -1 to use all genomes')
			exit(0)
		else:
			if args.N == -1:
				logging.info(f"The parameter 'N' is set to -1. So use all {len(args.metadata.file_names)} genomes.")
				N = len(args.metadata.file_names)
			elif args.N > len(args.metadata.file_names):
				logging.warning(f"The parameter 'N' is set larger than the total numbers of genomes {len(args.metadata.file_names)}. So set it to {len(args.metadata.file_names)}.")
				N = len(args.metadata.file_names)
			else:
				N = args.N
			args.abundance_list = list(np.random.dirichlet(np.ones(N),size=1).reshape(-1))
			logging.info('Abundance list computed:')
			logging.info(np.round(args.abundance_list))
			if len(args.unif_param) != 2:
				logging.error("The parameter 'unif_param' has the number of values which is not equal to 2")
				exit(0)
			else:
				args.mut_rate_list = list(np.random.uniform(args.unif_param[0],args.unif_param[1],N))
				logging.info('Mutation rates computed:')
				logging.info(args.mut_rate_list)
	else:
		N = len(args.abundance_list)
		if args.N > len(args.metadata.file_names):
			logging.warning(f"The parameter 'N' is set larger than the total numbers of genomes {len(args.metadata.file_names)}. So set it to {len(args.metadata.file_names)}.")
			N = len(args.metadata.file_names)
		if len(args.mut_rate_list) != N:
			logging.error("the mutation rate list and the abundance list must have the same length.")
			exit(0)

	if not os.path.isdir(args.output_dir):
		logging.warning(f"Output directory {args.output_dir} doesn't exit. It will be automatically generated.")
		os.makedirs(args.output_dir)

	## calculate original data
	logging.info("calculate original data.")
	orig_A_matrix = make_data.get_original_data(db_file = args.db_file, file_names = args.metadata.file_names, N = N, savepath = args.output_dir)

	## calculate mutated data
	logging.info("calculate mutated data.")
	mut_organisms = make_data.get_mutated_data(orig_A_matrix, args.abundance_list, args.mut_rate_list)

	logging.info("estimate frequency.")
	FE = ferm.frequency_estimator(mut_organisms, w=args.weight)
	if N <= 100:
		logging.info("True frequencies:")
		logging.info(np.round(args.abundance_list,4))
		logging.info("True mutation rates:")
		logging.info(np.round(args.mut_rate_list,4))
		logging.info("Recovered frequencies at " + str(0.05) + " mutation threshold:")
		logging.info(np.round(FE.freq_est,4))
	logging.info("Evaluating estimate:")
	EE = ee.est_evaluator(args.abundance_list,args.mut_rate_list,0.05,FE.freq_est)
	fp, fn = EE.classification_err()
	max_pos_rt = EE.max_nonzero_rate()
	logging.info("False Pos: " + str(fp) + ", False Neg: " + str(fn) + ", Max mut rate: " +str(np.round(max_pos_rt,3)))

	## save results into file
	res = pd.DataFrame(zip(orig_A_matrix.fasta_files, args.abundance_list, args.mut_rate_list, np.round(FE.freq_est,4)))
	res.columns = ['organism_files', 'abundance', 'mutation_rate', 'estimated_freq']
	res.to_csv(os.path.join(args.output_dir,"final_results.txt"), sep='\t', index=None)
	logging.info('Program is finished!!!')
