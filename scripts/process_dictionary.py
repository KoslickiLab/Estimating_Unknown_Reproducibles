import numpy as np
from sklearn.preprocessing import normalize


def process_dictionary(A_raw, corr_thresh, rel_max_thresh):
    uncorr_idx = get_uncorr_idx(A_raw, corr_thresh)
    A_uncorr = A_raw[:,uncorr_idx]
    A_proc = flatten_dictionary(A_uncorr, rel_max_thresh)
    return A_proc, uncorr_idx

def get_uncorr_idx(A, corr_thresh):
    A_norm = normalize(A, norm = 'l2', axis = 0)
    corrs = A_norm.transpose() * A_norm
    uncorr_idx = [0]
    N = A_norm.shape[1]
    for i in range(1,N):
        corr_flag = False
        for j in uncorr_idx:
            if corrs[i,j] > corr_thresh:
                corr_flag = True
                break
        if not(corr_flag):
            uncorr_idx.append(i)
    return np.array(uncorr_idx).astype(int)

def flatten_dictionary(A, rel_max_thresh):
    A_flat = A.copy()
    N = A.shape[1]
    print(N)
    for i in range(N):
        col = A[:,i]
        col_plus = col[col.nonzero()]
        m = col_plus.min()
        removed = 0
        for j in col.nonzero()[0]:
            if col[j,0]/m > rel_max_thresh:
                A_flat[j,i] = rel_max_thresh*m
                removed += A[j,i] - rel_max_thresh
        total_i = 1/m
        total_i_adj = 1/m - removed
        A_flat[:,i] = A_flat[:,i]*total_i/total_i_adj
    return A_flat