import os
import numpy as np
import make_data
import freq_est_from_mut as ferm
import est_eval as ee
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import MinHash as mh
import warnings
import lp_solver
import time
import random
from math import ceil
from scipy.stats import binom
import csv
from datetime import datetime
import pathlib
warnings.filterwarnings("ignore")


#Parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script runs random tests on a given organism dictionary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', default = 51, help='kmer size')
    parser.add_argument('--db_file', default = None, help='file with count estimators')
    parser.add_argument('--n', help='Number of kmers from sketch per organism', default=1000, type = int)
    parser.add_argument('--N', help='Number of organisms in dictionary', default=100, type = int)
    parser.add_argument('--unif_param', help='Bounds for random mutation rate', default=[0.01, 0.09], type = float, nargs = '+')
    parser.add_argument('--weight', help='False negative discount weight', default=None, type = float)
    parser.add_argument('--mut_thresh', help='mutation threshold', default=0.05, type = float)
    parser.add_argument('--alpha', help='Proportion of dictionary in sample', default=0.05, type = float)
    parser.add_argument('--seed', help='Random seed', default=None, type = int)
    parser.add_argument('--output_dir', help='Output directory', default='../results', type = str)
    parser.add_argument('--T', help='Number of tests', default=1, type = int)
    args = parser.parse_args()

    k = args.k
    n = args.n
    N = args.N
    if args.db_file is None:
        db_file = '../files/database_test/cdb_n'+str(n)+'_k'+str(k)+'/cdb.h5'
    unif_param = args.unif_param
    weight = args.weight
    mut_thresh = args.mut_thresh
    alpha = args.alpha
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    output_dir = args.output_dir
    T = args.T
    
    #filename for results
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    result_path = '../results/test_results_%(ts)s' % {'ts': date_time}
    abs_path = os.path.abspath(result_path)
    #create path if doesn't exist
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    abs_filepath = result_path + '/results.csv'
    
    #todo: log arguments
    arg_filepath = result_path + '/args.csv'
    with open(arg_filepath, 'w') as args_f:
        writer = csv.writer(args_f)
        headers = ['k','n','N','db_file','unif_param','weight','mut_thresh','alpha','seed','output_dir','T']
        row = [k,n,N,db_file,unif_param,weight,mut_thresh,alpha,seed,output_dir,T]
        writer.writerow(headers)
        writer.writerow(row)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    logging.info('Program started. Recording results in folder %(abs_filepath)s.' % {'abs_filepath': abs_filepath})

    with open(abs_filepath, 'w') as f:
        writer = csv.writer(f)
        row = ['False positives', 'False negatives', 'Max positive rate', 'Min zero rate', 'Absolute error', 'True unknown percent', 'Est unknown pct', 'True unknown minus est unknown']
        writer.writerow(row)

        logging.info('Beginning test suite with random seed %(seed)d.' % {"seed": seed})
        metadata = mh.get_info_from_single_hdf5(db_file)

        logging.info('Compiling original dictionary.')
        orig_data = make_data.get_original_data(db_file = db_file, file_names = metadata.file_names, \
            N=N, filepath = output_dir)
        logging.info('Preprocessing dictionary.')
        proc_data = make_data.processed_data(orig_data, mut_thresh = mut_thresh)

        logging.info('Beginning test loop.')
        for t in range(T):
            s = ceil(alpha*proc_data.N)
            if t == 0:
                logging.info('Abundance is automatically set according to dirichlet distribution.')
                logging.info('Mutation rates automatically set according to uniform distribution between %(lower)2f and %(upper)2f.' %{'lower': unif_param[0], 'upper': unif_param[1]})
            support_abundance = list(np.random.dirichlet(np.ones(s),size=1).reshape(-1))
            support_mut = list(np.random.uniform(unif_param[0],unif_param[1],s))
            support = random.sample(list(range(proc_data.N)),s)
            proc_abundance = [0]*proc_data.N
            proc_mut_rate_list = [0]*proc_data.N
            for i in range(s):
                proc_abundance[support[i]] = support_abundance[i]
                proc_mut_rate_list[support[i]] = support_mut[i]

            logging.info('Generating mutated data for iteration %(t)d.' % {'t': t+1})
            mut_organisms = make_data.get_mutated_data(proc_data, proc_abundance, proc_mut_rate_list)

            logging.info('Estimating Frequencies for iteration %(t)d.' % {'t': t+1})
            FE = ferm.frequency_estimator(mut_organisms, w=weight)

            logging.info('Evaluating performance for iteration %(t)d.' % {'t': t+1})

            EE = ee.est_evaluator(proc_abundance,proc_mut_rate_list,mut_thresh,FE.freq_est)
            fp, fn, rfp, rfn = EE.classification_err()
            num_fp = np.sum(fp)
            num_fn = np.sum(fn)
            max_pos_rt = EE.max_nonzero_rate()
            min_zero_rt = EE.min_zero_rate()

            logging.info('Saving results for iteration.')
            row = [num_fp, num_fn, max_pos_rt, min_zero_rt, EE.abs_err(), round(EE.unknown_pct(),6), EE.unknown_pct_est(), EE.unknown_pct()-EE.unknown_pct_est()]
            writer.writerow(row)

            logging.info('Test iteration %(t)d complete.' % {'t': t+1})

        logging.info('Testing complete.')