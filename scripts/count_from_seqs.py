import os
import numpy as np
from Bio import SeqIO
import MinHash as mh
import re

def get_kmer_count(k, kmer_to_idx, seq):
	L = len(seq);
	n_unique_kmers = len(kmer_to_idx)
	count = np.zeros(n_unique_kmers)
	for i in range(L-k+1):
		kmer = seq[i:(i+k)]
		if kmer in kmer_to_idx:
			count[kmer_to_idx[kmer]] += 1
	return count

def count_from_seqs(k, kmer_to_idx, seqs):
	n_unique_kmers = len(kmer_to_idx)
	n_kmers_in_seqs = 0
	count = np.zeros(n_unique_kmers)
	for seq in seqs:
		n_kmers_in_seqs += (len(seq) - k + 1)
		count += get_kmer_count(k, kmer_to_idx, seq)
	count = count/n_kmers_in_seqs
	return count, n_kmers_in_seqs