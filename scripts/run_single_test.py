import os
import sys
sys.path.append('/home/cqm5886/work/Estimating_Unknown/test/scripts')
import numpy as np
import make_data
import freq_est_from_mut as ferm
import est_eval as ee
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
warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser(description="This script runs random tests on a given organism dictionary.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--k', default = 51, help='kmer size')
# parser.add_argument('--db_file', default = None, help='file with count estimators')
# parser.add_argument('--n', help='Number of kmers from sketch per organism', default=1000, type = int)
# parser.add_argument('--N', help='Number of organisms in dictionary', default=100, type = int)
# parser.add_argument('--unif_param', help='Bounds for random mutation rate', default=[0.01, 0.09], type = float, nargs = '+')
# parser.add_argument('--weight', help='False negative discount weight', default=None, type = float)
# parser.add_argument('--mut_thresh', help='mutation threshold', default=0.05, type = float)
# parser.add_argument('--alpha', help='Proportion of dictionary in sample', default=0.05, type = float)
# parser.add_argument('--seed', help='Random seed', default=None, type = int)
# parser.add_argument('--output_dir', help='Output directory', default='../results', type = str)
# parser.add_argument('--T', help='Number of tests', default=1, type = int)
# parser.add_argument('--cpu', help='Number of cpu', default=50, type = int)
# parser.add_argument('--nosparse', help='Set to use dense matrices instead of sparse', action='store_false')
# parser.add_argument('--save_results', help='Results will be saved if set.', action='store_true')
# parser.add_argument('--savepath', help='File location to save processed dictionary', type = str)
# parser.add_argument('--loadpath', help='File location to load processed dictionary from', type = str)

class test_instance():
    def __init__(
        self,
        s,
        unif_param,
        proc_data,
        mut_thresh = 0.05,
        weight = None,
        seed = None,
        cpu = 1,
        run_now = True,
    ):
        self.s = s
        self.unif_param = unif_param
        self.proc_data = proc_data
        self.N_proc = self.proc_data.N
        self.mut_thresh = mut_thresh
        self.weight = weight
        self.seed = seed
        if run_now:
            self.run_test(cpu = cpu)
        
    def run_test(self, cpu = 1):
        sim_start = time.time()

        self.support = random.sample(list(range(self.N_proc)),self.s)
        self.support_abundance = list(np.random.dirichlet(np.ones(self.s),size=1).reshape(-1))
        self.support_mut = list(np.random.uniform(self.unif_param[0],self.unif_param[1],self.s))
        self.proc_abundance = np.zeros(self.N_proc)
        self.proc_abundance[self.support] = self.support_abundance

        self.proc_mut_rate_list = np.zeros(self.N_proc)
        self.proc_mut_rate_list[self.support] = self.support_mut

        self.mut_organisms = make_data.get_mutated_data(self.proc_data, self.proc_abundance,
                                                        self.proc_mut_rate_list, seed = self.seed, use_cpu = cpu)

        sim_end = time.time()
        self.sim_time = sim_end - sim_start

        self.FE = ferm.frequency_estimator(self.mut_organisms, w=self.weight)
        
        self.EE = ee.est_evaluator(
            self.proc_abundance,
            self.proc_mut_rate_list,
            self.mut_thresh,
            self.FE.freq_est
        )

        algo_end = time.time()
        self.algo_time = algo_end - sim_end
        
        self.fp, self.fn, rfp, rfn = self.EE.classification_err()
        self.num_fp = np.sum(fp) 
        self.num_fn = np.sum(fn)
        self.max_pos_rt = self.EE.max_nonzero_rate()
        self.min_zero_rt = self.EE.min_zero_rate()
        self.abs_err = self.EE.abs_err(),
        self.unknown_pct_diff = self.EE.unknown_pct()-self.EE.unknown_pct_est()

