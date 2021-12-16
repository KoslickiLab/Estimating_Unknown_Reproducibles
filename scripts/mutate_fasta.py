import os
import numpy as np
from Bio import SeqIO
import glob
import warnings
import MinHash as mh
import itertools
from pathlib import Path
import re

class mutated_fasta():
	def __init__(self, M,N,k,rel_abundance,r1,org_fna_files,kmer_to_idx,total_kmers = pow(10,9), rnd = True):
		self.mut_ct = np.zeros((M,1))
		self.rel_abundance
		self.total_kmers = total_kmers

		#counts for each mutated organism
		self.indiv_ct = []
		self.raw_seqs = []
		self.mutated_seqs = []

		for file_idx, fna_file in enumerate(org_fna_files[0:len(r1)]):
		    #extract genome sequence from fna file
		    curr_raw_seqs = fasta_to_str_seq(fna_file)
		    self.raw_seqs.append(curr_raw_seqs)
		    if r1[file_idx] > 0:
			    curr_mutated_seqs = mutate_fasta_seqs(curr_raw_seqs, r1[file_idx])
			self.mutated_seqs.append(curr_mutated_seqs)
		    self.indiv_ct.append(count_from_seqs(k,kmer_to_idx,curr_seqs))
		    self.mut_ct = self.mut_ct + self.total_kmers*rel_abundance[file_idx]*indiv_ct[file_idx];
		if rnd:
			mut_ct = np.round(mut_ct)

			self.mut_ct = mut_ct
			self.indiv_ct = indiv_ct
			self.mut_seqs = mut_seqs

	def get_mutated_seq(seq, r1):
		L = len(seq)
		mut_seq_list = list(seq)
		mut_flag = np.random.binomial(1,r1,L)
		mut_indices = np.nonzero(mut_flag)
		mut_indices = mut_indices[0]
		nucleotides = {'A','C','G','T'}
		for mut_idx in mut_indices:
			possible_muts = list(nucleotides.difference({seq[mut_idx]}))
			mut_seq_list[mut_idx] = np.random.choice(possible_muts)
		mut_seq = ""
		mut_seq = mut_seq.join(mut_seq_list)
		return mut_seq

	def mutate_fasta_seqs(seqs, r1):
		mut_seqs = []
		for seq in seqs:
			seq = get_mutated_seq(seq,r1)
			mut_seqs.append(seq)
		return mut_seqs

def mutated_data(M,N,k,rel_abundance,r1,org_fna_files,kmer_to_idx, total_kmers = pow(10,9), rnd = True):
	mut_ct = np.zeros((M,1))

	#counts for each mutated organism
	indiv_ct = []
	read_seqs = []

	#get mutated vectors
	for file_idx, fna_file in enumerate(org_fna_files[0:len(r1)]):
	    curr_seqs = fasta_to_str_seq(fna_file)
	    if r1[file_idx] > 0:
		    curr_seqs = mutate_fasta_seqs(curr_seqs, r1[file_idx])
	    read_seqs.append(curr_seqs)
	    indiv_ct.append(count_from_seqs(k,kmer_to_idx,curr_seqs))
	    mut_ct = mut_ct + total_kmers*rel_abundance[file_idx]*indiv_ct[file_idx];
	if rnd:
		mut_ct = np.round(mut_ct)

	return mut_data(mut_ct, indiv_ct, read_seqs)