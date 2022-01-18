import os
import numpy as np
import MinHash as mh
import pickle
import mutate_single_organism as mso
import utils
from multiprocessing import Pool, cpu_count

class get_original_data:
	'''
	This class is to take in count estimator(genome sketch) file generated from CMash and creates dictionary matrix and associated metadata
	'''
	def __init__(self, db_file, file_names, N, filename = None, filepath = None):
		self.file_names = file_names
		self.dict_files_from_db(db_file, N=N)
		if filename is not None:
			self.filename = filename
		else:
			self.filename = self.dict_filename(filepath,N=N)
		# if self.filename is not None:
			# with open(self.filename, 'wb') as f:
			# 	pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

	def dict_files_from_db(self,db_file, N):
		genome_files = self.file_names[:N]
		genome_data = mh.import_multiple_from_single_hdf5(db_file, genome_files)
		idx_to_kmer, kmer_to_idx = self.kmer_union_from_count_estimators(genome_data)
		self.k = len(idx_to_kmer[0])
		self.num_hashes = len(genome_data[0]._kmers)
		self.fasta_files = genome_files
		self.idx_to_kmer = idx_to_kmer
		self.kmer_to_idx = kmer_to_idx
		dict_matrix = self.matrix_from_fasta_files()
		self.dictionary = dict_matrix

	def matrix_from_fasta_files(self):
		N = len(self.fasta_files)
		max_available_cpu = int(cpu_count()*(2/3))
		if N < max_available_cpu:
			n_processes = N
		else:
			n_processes = max_available_cpu
		params = zip(self.fasta_files, [self.k]*N, [self.kmer_to_idx]*N)
		with Pool(processes=n_processes) as excutator:
			res = excutator.map(self.get_count_from_single_organism, params)
		freq_matrix_A = np.vstack(res).T
		return freq_matrix_A

	@staticmethod
	def get_count_from_single_organism(this_param):
		fasta_file, k, kmer_to_idx = this_param
		curr_seqs = utils.fasta_to_ATCG_seq_list(fasta_file)
		this_normalized_count, _ = utils.count_from_seqs(k, kmer_to_idx, curr_seqs)
		return this_normalized_count

	@staticmethod
	def kmer_union_from_count_estimators(count_estimators):
		idx_to_kmer = list({kmer for ce in count_estimators for kmer in ce._kmers})
		kmer_to_idx = {kmer:idx for idx, kmer in enumerate(idx_to_kmer)}
		return [idx_to_kmer, kmer_to_idx]

	def dict_filename(self,filepath, N=None):
		if os.path.isfile(filepath):
			dict_dir = os.path.dirname(filepath)
		elif os.path.isdir(filepath):
			dict_dir = filepath
		else:
			raise ValueError('Argument is not a file or directory.')
		add = '_N'+str(N) if N is not None else ''
		dict_files = os.path.join(dict_dir, 'A_matrix'+add+'.pkl')
		return dict_files


#inputs: org_dict object, sample abundance (list), sample mutation rates (list)
#output: all_mutations_from_dict object, contains multiple mutated samples and aggregated kmer frequency vector
class get_mutated_data():
	'''
	This class is to randomly mutate the original genomes and generate aggregated kmer frequency vector
	'''
	def __init__(self, orig_A_matrix, abundance_list, mut_rate_list, total_kmers = None, rnd = True):
		self.orig_A_matrix = orig_A_matrix
		self.fasta_files = orig_A_matrix.fasta_files
		self.kmer_to_idx = orig_A_matrix.kmer_to_idx
		self.mut_kmer_ct = np.zeros(len(self.kmer_to_idx))
		self.abundance_list = abundance_list
		self.mut_rate_list = mut_rate_list
		self.rnd = rnd
		self.N = len(self.fasta_files)
		if total_kmers is None:
			#scale total kmers for number of organisms--needs fine-tuning        
			self.total_kmers = self.N*pow(10,8)
		else:
			self.total_kmers = total_kmers
		self.mut_orgs = []

		self.get_all_mutated_organism()

	def get_all_mutated_organism(self):
		max_available_cpu = int(cpu_count()*(2/3))
		if self.N < max_available_cpu:
			n_processes = self.N
		else:
			n_processes = max_available_cpu
		params = zip(self.fasta_files, [self.total_kmers]*self.N, self.mut_rate_list, self.abundance_list, [self.kmer_to_idx]*self.N)
		with Pool(processes=n_processes) as excutator:
			res = list(excutator.map(self.get_single_mutated_organism, params))
		self.mut_orgs += [curr_mut_org for (curr_mut_org, _) in res]
		self.mut_kmer_ct += np.sum(np.vstack([mut_kmer_ct for (_, mut_kmer_ct) in res]), axis=0)

        #this line is a problem
		if self.rnd:
			self.mut_kmer_ct = np.round(self.mut_kmer_ct)

	@staticmethod
	def get_single_mutated_organism(this_param):
		fasta_file, total_kmers, mut_rate, rel_abundance, kmer_to_idx = this_param
		curr_mut_org = mso.get_mutated_seq_and_kmers(fasta_file, kmer_to_idx, mut_rate)
		mut_kmer_ct = total_kmers*rel_abundance*curr_mut_org.mut_kmer_ct
		return [curr_mut_org, mut_kmer_ct]