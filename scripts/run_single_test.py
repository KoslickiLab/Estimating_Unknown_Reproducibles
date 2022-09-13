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
import pdb


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
        writepath = None,
        run_now = True,
    ):
        self.s = s
        self.unif_param = unif_param
        self.proc_data = proc_data
        self.N_proc = self.proc_data.N
        self.mut_thresh = mut_thresh
        self.weight = weight
        self.seed = seed
        self.writepath = writepath
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
                                                        self.proc_mut_rate_list, seed = self.seed, use_cpu = cpu, writepath = self.writepath)

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
        self.num_fp = np.sum(self.fp) 
        self.num_fn = np.sum(self.fn)
        self.max_pos_rt = self.EE.max_nonzero_rate()
        self.min_zero_rt = self.EE.min_zero_rate()
        self.abs_err = self.EE.abs_err()
        self.unknown_pct = self.EE.unknown_pct()
        self.unknown_pct_est = self.EE.unknown_pct_est()
        self.unknown_pct_diff = self.EE.unknown_pct()-self.EE.unknown_pct_est()

