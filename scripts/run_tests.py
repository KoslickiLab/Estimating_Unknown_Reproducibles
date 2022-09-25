import os
import sys
sys.path.append('/home/cqm5886/work/Estimating_Unknown/test/scripts')
import numpy as np
import make_data
import freq_est_from_mut as ferm
import est_eval as ee
import run_single_test as rst
import argparse
import logging
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
import pickle
import pathlib
import zipfile
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
    parser.add_argument('--cpu', help='Number of cpu', default=50, type = int)
    parser.add_argument('--nosparse', help='Set to use dense matrices instead of sparse', action='store_false')
    parser.add_argument('--save_results', help='Results will be saved if set.', action='store_true')
    parser.add_argument('--save_mutations', help='Mutated strings will be saved if set.', action='store_true')    
    parser.add_argument('--savepath', help='File location to save processed dictionary', type = str)
    parser.add_argument('--loadpath', help='File location to load processed dictionary from', type = str)
    args = parser.parse_args()
    
    k = args.k
    n = args.n
    N = args.N
    
    if args.db_file is None:
        db_file = '../files/database_test/cdb_n'+str(n)+'_k'+str(k)+'/cdb.h5'
    else:
        db_file = args.db_file
    unif_param = args.unif_param
    weight = args.weight
    mut_thresh = args.mut_thresh
    alpha = args.alpha
    seed = args.seed
    sparse = args.nosparse
    save_results = args.save_results
    save_mut = args.save_mutations
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    output_dir = args.output_dir
    savepath = args.savepath
    loadpath = args.loadpath
    T = args.T
    cpu = args.cpu
    
    #filename for results
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    result_path = '../results/test_results_%(ts)s' % {'ts': date_time}
    mut_path = result_path + '/mutations'
    abs_path = os.path.abspath(result_path)
    #create path if doesn't exist
    if save_results:
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
        abs_filepath = result_path + '/results.csv'
        arg_filepath = result_path + '/args.csv'
        uncorr_idx_filepath = result_path + '/uncorr_idx.csv'
        proc_abund_filepath = result_path + '/processed_abundance.csv'
    if save_mut:
        pathlib.Path(mut_path).mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    if save_results:
        logging.info('Program started. Recording results in folder %(abs_filepath)s.' % {'abs_filepath': abs_filepath})
    else:
        logging.info('Program started. Parameter --save_results set not set; results WILL NOT be recorded.')

    logging.info('Beginning test suite with random seed %(seed)d.' % {"seed": seed})
    metadata = mh.get_info_from_single_hdf5(db_file)
    
    if N == -1:
        logging.info(f"The parameter 'N' is set to -1. Program will use all {len(metadata.file_names)} genomes.")
        N = len(metadata.file_names)

    if (savepath is not None) and (loadpath is not None):
        logging.info('Both savepath and loadpath provided. Dictionary will be loaded from loadpath but no new dictionary will be saved.')
    
    if loadpath is not None:
        logging.info(f'Loading processed data from path {os.path.abspath(loadpath)}')
        with open(loadpath, 'rb') as infile:
            proc_data = pickle.load(infile)
    else:
        logging.info('Loadpath not provided; compiling original dictionary from scratch.')
        orig_data = make_data.get_original_data(db_file = db_file, file_names = metadata.file_names, \
        N=N, sparse_flag = sparse)
        logging.info('Dictionary generated with type %(dict_type)s' % {"dict_type": type(orig_data.dictionary)})
        logging.info('Preprocessing dictionary.')
        proc_data = make_data.processed_data(orig_data, mut_thresh = mut_thresh, savepath = savepath)
        if savepath is not None:
            logging.info(f'Processed data saved at path {os.path.abspath(savepath)}')
    if save_results:
        with open(uncorr_idx_filepath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(proc_data.uncorr_indices)
    
    s = ceil(alpha*proc_data.N)
    if save_results:
        with open(arg_filepath, 'w') as args_f:
            writer = csv.writer(args_f)
            headers = ['k','n','s','raw_N','proc_N','db_file','unif_param','weight',
                       'mut_thresh','alpha','seed','output_dir','T']
            row = [k,n,s, N,proc_data.N, db_file,unif_param,weight,mut_thresh,alpha,seed,output_dir,T]
            writer.writerow(headers)
            writer.writerow(row)

    logging.info('Beginning test loop.')
    logging.info('Abundance is automatically set according to dirichlet distribution.')
    logging.info('Mutation rates automatically set according to uniform distribution between %(lower)2f and %(upper)2f.' %{'lower': unif_param[0], 'upper': unif_param[1]})
    if save_results:
        with open(abs_filepath, 'w') as f:
            writer = csv.writer(f)
            row = ['Iteration','False positives', 'False negatives', 'Max positive rate', 'Min zero rate', 'Absolute error', 'True unknown percent', 'Est unknown pct', 'True unknown minus est unknown','simulation_time','algo_time']
            writer.writerow(row)
    performance = np.zeros((T,6))
    
    tests = []
    for t in range(T):
        if save_mut:
            curr_mut_path = mut_path + '/iter_%(t)d' % {'t': t+1}
            pathlib.Path(curr_mut_path).mkdir(parents=True, exist_ok=True)
        else:
            curr_mut_path = None
        
        curr_test = rst.test_instance(
            s,
            unif_param,
            proc_data,
            mut_thresh = mut_thresh,
            weight = weight,
            seed = seed,
            cpu = cpu,
            writepath = curr_mut_path,
            run_now = True,
        )
        
        if save_mut:
            support_file = curr_mut_path + '/supp.csv'
            with open(support_file, 'w') as f:
                writer = csv.writer(f)
                headers = ['raw_idx','abundance','mut_rate','known_flag','organism']
                writer.writerow(headers)
                for i in range(len(curr_test.support)):
                    row = [curr_test.support_raw[i],
                           curr_test.support_abundance[i], 
                           curr_test.support_mut[i],
                           curr_test.support_known_flag[i],
                           curr_test.support_orgs[i],
                          ]
                    writer.writerow(row)

        performance[t,:] = [
            curr_test.num_fp,
            curr_test.num_fn,
            curr_test.max_pos_rt,
            curr_test.min_zero_rt,
            curr_test.abs_err,
            curr_test.unknown_pct_diff,
        ]

        logging.info('Saving results for iteration.')

        if save_results:
            row = [
                t+1,
                curr_test.num_fp,
                curr_test.num_fn,
                curr_test.max_pos_rt,
                curr_test.min_zero_rt,
                curr_test.abs_err,
                round(curr_test.unknown_pct,6),
                curr_test.unknown_pct_est,
                curr_test.unknown_pct_diff,
                curr_test.sim_time,
                curr_test.algo_time,
            ]
            with open(abs_filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            with open(proc_abund_filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(curr_test.proc_abundance)
                
        logging.info('Test iteration %(t)d complete.' % {'t': t+1})
        
    avg_perf = np.mean(performance, axis = 0)
    logging.info('Testing complete.')
    logging.info('Performance:\nAvg false positives: %2f\nAvg false negatives: %2f\nAvg Abs Error: %2f\nAvg Est Error: %2f'%(avg_perf[0],avg_perf[1],avg_perf[4],avg_perf[5]))
        