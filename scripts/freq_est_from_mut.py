import lp_solver as lps
import numpy as np

#inputs: all_mutations_from_fasta object (simulated data)
#outputs: frequency estimator
class frequency_estimator():
	def __init__(self, mut_organisms, mut_thresh = 0.05, w = None):
		self.mut_organisms = mut_organisms
		self.k = self.mut_organisms.orig_A_matrix.k
		self.num_hashes = self.mut_organisms.orig_A_matrix.num_hashes
		self.mut_thresh = mut_thresh
		self.w = w if w is not None else self.estimate_w()
		self.LPS = lps.lp_solver(self.mut_organisms.orig_A_matrix.dictionary, self.mut_organisms.mut_kmer_ct, self.w, self.mut_organisms.N, run_now = True)
		self.count_est = self.LPS.x_opt
		self.freq_est = self.LPS.x_opt/self.mut_organisms.total_kmers
		self.est_unk_pct = np.sum(self.freq_est)

	def estimate_w(self,est_n_orgs = 1000, p_val = 0.99, n_tests = 10000):
		prob = (1-self.mut_thresh)**self.k
		b = []
		for i in range(n_tests):
			b.append(min(np.random.binomial(self.num_hashes, prob, (est_n_orgs, 1))))
		min_est = np.quantile(b,p_val)
		w = min_est/(self.num_hashes-min_est)
		return w
