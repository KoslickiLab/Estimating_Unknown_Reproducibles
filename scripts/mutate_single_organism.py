import numpy as np
from Bio import SeqIO
import utils

#class for mutating storing mutated and unmutated data on single organism
#input: single FNA file
#output: mutated FNA file and metadata
class get_mutated_seq_and_kmers():
	""" This class is to randomly mutate the sequence and generate the corresponding muated kmer count list for a single organism based on the given mutated rate"""
	def __init__(self, fasta_file, kmer_to_idx, mut_rate=0):
		self.fasta_file = fasta_file
		self.mut_rate = mut_rate
		self.k = len(list(kmer_to_idx.keys())[0])
		self.raw_seqs = utils.fasta_to_ATCG_seq_list(fasta_file)
		if mut_rate > 0:
			self.mutated_seqs = self.mutate_fasta_seqs()
		else:
			self.mutated_seqs = self.raw_seqs
		self.mut_kmer_ct, self.n_kmers_in_seqs = utils.count_from_seqs(self.k,kmer_to_idx,self.mutated_seqs)

	def mutate_fasta_seqs(self):
		return list(map(self.get_mutated_seq, self.raw_seqs))

	def get_mutated_seq(self, seq):
		L = len(seq)
		mut_seq_list = list(seq)
		mut_flag = np.random.binomial(1,self.mut_rate,L)
		mut_indices = np.nonzero(mut_flag)[0]
		nucleotides = {'A','C','G','T'}
		for mut_idx in mut_indices:
			possible_muts = list(nucleotides.difference({seq[mut_idx]}))
			mut_seq_list[mut_idx] = np.random.choice(possible_muts)
		mut_seq = "".join(mut_seq_list)
		return mut_seq