from Bio import SeqIO
import re
import numpy as np
from functools import reduce
import operator
import exec_ext_files.optimized_functions as op_funcs

def extract_ATCG_seq_from_genome(bio_seqio_iter):
	notACTG = re.compile('[^ACTG]')
	seq_split_ACTG = notACTG.split(bio_seqio_iter.seq.upper().encode().decode())
	return list(filter(None, seq_split_ACTG))

def fasta_to_ATCG_seq_list(fasta_file):
	fasta_seqs = SeqIO.parse(open(fasta_file),'fasta')
	seqs = reduce(operator.concat, map(extract_ATCG_seq_from_genome,fasta_seqs))
	return seqs

def count_from_seqs(k, kmer_to_idx, seqs):
	n_unique_kmers = len(kmer_to_idx)
	n_kmers_in_seqs = 0
	count = np.zeros(n_unique_kmers)
	for seq in seqs:
		n_kmers_in_seqs += (len(seq) - k + 1)
		count += np.array(op_funcs.get_kmer_count(k, kmer_to_idx, seq))
	count = count/n_kmers_in_seqs
	return count, n_kmers_in_seqs