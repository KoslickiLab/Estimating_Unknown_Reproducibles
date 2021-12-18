from Bio import SeqIO
import re
import numpy as np
from functools import reduce
import operator
import gzip
import pathlib
# import exec_ext_files.optimized_functions as op_funcs

def extract_ATCG_seq_from_genome(bio_seqio_iter):
	notACTG = re.compile('[^ACTG]')
	seq_split_ACTG = notACTG.split(bio_seqio_iter.seq.upper().encode().decode())
	return list(filter(None, seq_split_ACTG))

def fasta_to_ATCG_seq_list(fasta_file):
	if '.gz' in pathlib.Path(fasta_file).suffixes:
		with gzip.open(fasta_file,'rt') as infile:
			fasta_seqs = SeqIO.parse(infile,'fasta')
			seqs = reduce(operator.concat, map(extract_ATCG_seq_from_genome,fasta_seqs))
	else:
		with open(fasta_file,'rt') as infile:
			fasta_seqs = SeqIO.parse(infile,'fasta')
			seqs = reduce(operator.concat, map(extract_ATCG_seq_from_genome,fasta_seqs))
	return seqs

def get_kmer_count(k, kmer_to_idx, seq):
	L = len(seq);
	n_unique_kmers = len(kmer_to_idx)
	## use a triky skill to count total kmers (store the total kmers of this sequence in the last position of the count array)
	n_kmers_in_seq = L-k+1
	count = np.zeros(n_unique_kmers+1)
	count[-1] = n_kmers_in_seq
	for i in range(n_kmers_in_seq):
		kmer = seq[i:(i+k)]
		if kmer in kmer_to_idx:
			count[kmer_to_idx[kmer]] += 1
	return count

def count_from_seqs(k, kmer_to_idx, seqs):
	n_unique_kmers = len(kmer_to_idx)
	temp_count = np.zeros(n_unique_kmers+1)
	for seq in seqs:
		temp_count += get_kmer_count(k, kmer_to_idx, seq)
	n_kmers_in_seqs = temp_count[-1]
	count = temp_count[:-1]
	count = count/n_kmers_in_seqs
	return [count, n_kmers_in_seqs]
