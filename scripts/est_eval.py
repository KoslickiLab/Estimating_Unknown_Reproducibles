import numpy as np

class est_evaluator:
    def __init__(self, abundance_list, mut_rate_list, mut_thresh, freq_est):
        self.true_abundance = np.array(abundance_list)
        self.true_mut_rate = np.array(mut_rate_list)
        self.mut_thresh = mut_thresh
        self.freq_est = np.array(freq_est)
        
    def max_nonzero_rate(self):
        nonzero_idx = np.nonzero(self.freq_est)
        return(np.max(self.true_mut_rate[nonzero_idx]))
    
    def min_zero_rate(self):
        zero_idx = np.nonzero(self.freq_est == 0)
        return(np.min(self.true_mut_rate[zero_idx]))
    
    def abs_err(self):
        return np.sum(np.abs(self.freq_est - self.true_abundance*(self.true_mut_rate <= self.mut_thresh)))

    def classification_err(self):
        true_class = (self.true_mut_rate <= self.mut_thresh).astype(int)
        est_class = (self.freq_est > 0).astype(int)
        diff = true_class - est_class
        false_pos = (diff < 0)
        false_neg = (diff > 0)
        rel_false_pos = (self.true_mut_rate - self.mut_thresh)*false_pos
        rel_false_neg = (self.true_mut_rate - self.mut_thresh)*false_neg
        return false_pos, false_neg, rel_false_pos, rel_false_neg

    
    
    
    
        