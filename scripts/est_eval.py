import numpy as np

class est_evaluator:
    def __init__(self, abundance_list, mut_rate_list, mut_thresh, freq_est):
        self.true_abundance = np.array(abundance_list)
        self.true_mut_rate = np.array(mut_rate_list)
        self.mut_thresh = mut_thresh
        self.freq_est = np.array(freq_est)
        
    def max_nonzero_rate(self):
        nonzero_idx = np.nonzero(self.freq_est)
        if np.size(nonzero_idx) == 0:
            return 0
        return(np.max(self.true_mut_rate[nonzero_idx]))
    
    def min_zero_rate(self):
        est_zero_idx = np.nonzero(self.freq_est == 0)
        support_idx = np.nonzero(self.true_abundance > 0)
        est_zero_supp_idx = np.intersect1d(est_zero_idx,support_idx)
        if np.size(est_zero_supp_idx) == 0:
            return 1
        return(np.min(self.true_mut_rate[est_zero_supp_idx]))
    
    def abs_err(self):
        return np.sum(np.abs(self.freq_est - self.true_abundance*(self.true_mut_rate <= self.mut_thresh)))

    def classification_err(self):
        mut_positive = (self.true_mut_rate <= self.mut_thresh).astype(int)
        supp_positive = (self.true_abundance > 0).astype(int)
        true_positive = (mut_positive*supp_positive)
        est_positive = (self.freq_est > 0).astype(int)
        diff = true_positive - est_positive
        false_pos = (diff < 0)
        false_neg = (diff > 0)
        rel_false_pos = (self.true_mut_rate - self.mut_thresh)*false_pos
        rel_false_neg = (self.true_mut_rate - self.mut_thresh)*false_neg
        return false_pos, false_neg, rel_false_pos, rel_false_neg
    
    def unknown_pct(self):
        return 1-np.sum(self.true_abundance*(self.true_mut_rate <= self.mut_thresh))
    
    def unknown_pct_est(self):
        return 1-np.sum(self.freq_est)

    
    
    
    
        