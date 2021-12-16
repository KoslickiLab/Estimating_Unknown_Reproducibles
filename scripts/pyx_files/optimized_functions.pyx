
cpdef list get_kmer_count(long k, dict kmer_to_idx, str seq):
	cdef long L = len(seq);
	cdef long n_unique_kmers = len(kmer_to_idx)
	cdef list count = [0] * n_unique_kmers
	cdef long i
	cdef str kmer
	for i in range(L-k+1):
		kmer = seq[i:(i+k)]
		if kmer in kmer_to_idx:
			count[kmer_to_idx[kmer]] += 1
	return count