import os
import numpy as np
import MinHash as mh
import pickle
import mutate_single_organism as mso
import utils
import scipy.sparse.linalg as spla
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
from multiprocessing import Pool, cpu_count

class get_original_data:
	'''
	This class is to take in count estimator(genome sketch) file generated from CMash and creates dictionary matrix and associated metadata
	'''
	def __init__(self, db_file, file_names, N, sparse_flag = True, savepath = None):
		self.file_names = file_names
		self.N = N
		self.sparse = sparse_flag
		self.dict_files_from_db(db_file)

	def dict_files_from_db(self,db_file):
		genome_files = self.file_names[:self.N]
		genome_data = mh.import_multiple_from_single_hdf5(db_file, genome_files)
		idx_to_kmer, kmer_to_idx = self.kmer_union_from_count_estimators(genome_data)
		self.k = len(idx_to_kmer[0])
		self.num_hashes = len(genome_data[0]._kmers)
		self.fasta_files = genome_files
		self.idx_to_kmer = idx_to_kmer
		self.kmer_to_idx = kmer_to_idx
		if self.sparse:
			self.dictionary = self.sparse_matrix_from_fasta_files()
		else:
			self.dictionary = self.matrix_from_fasta_files()
		#save dictionary/reload if already computed

	@staticmethod
	def kmer_union_from_count_estimators(count_estimators):
		idx_to_kmer = list({kmer for ce in count_estimators for kmer in ce._kmers})
		kmer_to_idx = {kmer:idx for idx, kmer in enumerate(idx_to_kmer)}
		return [idx_to_kmer, kmer_to_idx]

	#Dense matrix generation
	def matrix_from_fasta_files(self):
		max_available_cpu = int(cpu_count()*(2/3))
		if self.N < max_available_cpu:
			n_processes = self.N
		else:
			n_processes = max_available_cpu
		params = zip(self.fasta_files, [self.k]*self.N, [self.kmer_to_idx]*self.N)
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

	#Sparse matrix generation
	def sparse_matrix_from_fasta_files(self):
		max_available_cpu = int(cpu_count()*(2/3))
		if self.N < max_available_cpu:
			n_processes = self.N
		else:
			n_processes = max_available_cpu
		params = zip(self.fasta_files, [self.k]*self.N, [self.kmer_to_idx]*self.N)
		with Pool(processes=n_processes) as executor:
			res = executor.map(self.get_sparse_count_from_single_organism, params)
		counts = []
		row_idx = []
		col_idx = []
		for i in range(self.N):
			L = len(res[i][0])
			counts = counts + res[i][0]
			row_idx = row_idx + res[i][1]
			col_idx = col_idx + ([i]*L)
		freq_matrix = csc_matrix((counts, (row_idx,col_idx)))
		return freq_matrix
	
	@staticmethod
	def get_sparse_count_from_single_organism(this_param):
		fasta_file, k, kmer_to_idx = this_param
		curr_seqs = utils.fasta_to_ATCG_seq_list(fasta_file)
		this_normalized_count, indices = utils.sparse_count_from_seqs(k, kmer_to_idx, curr_seqs)
		return this_normalized_count, indices
    

class processed_data:
	'''
	This class takes a get_original_data object and pre-processes the dictionary in two ways.
	1. Correlated columns are removed
	2. Large entries in columns violating 0/1 assumption are reduced to a maximum value.
	Inputs: get_original_data object, correlation threshold, relative maximum threshold
	Outputs: processed_data object: similar fields to original_data but with modified dictionary, N, and filenames
	'''
	def __init__(self, orig_data, corr_thresh = None, mut_thresh = None, savepath = None, rel_max_thresh = 5):
		self.orig_data = orig_data
		self.sparse = orig_data.sparse
		self.N_orig = self.orig_data.N
		self.k = self.orig_data.k
		self.num_hashes = self.orig_data.num_hashes
		if corr_thresh is None and mut_thresh is None:
			raise ValueError("Either corr_thresh or mut_thresh must be provided.")
		if corr_thresh is not None and mut_thresh is not None:
			raise ValueWarning("Both corr_thresh and mut_thresh provided; corr_thresh will be used.")
			self.corr_thresh = corr_thresh
		elif corr_thresh is not None:
			self.corr_thresh = corr_thresh
		else:
			non_mut_prob = (1-mut_thresh)**self.k
			self.corr_thresh = 2*non_mut_prob            
		self.rel_max_thresh = rel_max_thresh
		if self.sparse:
			self.uncorr_indices = self.sparse_uncorr_idx()
		else:
			self.uncorr_indices = self.uncorr_idx()
		self.N = len(self.uncorr_indices)
		self.idx_to_kmer = self.orig_data.idx_to_kmer
		self.kmer_to_idx = self.orig_data.kmer_to_idx
		self.fasta_files = [self.orig_data.fasta_files[i] for i in self.uncorr_indices]
		#self.dictionary = self.orig_data.dictionary[:,self.uncorr_indices]
		if self.sparse:
			self.dictionary = self.sparse_flatten_dictionary()
		else:
			self.dictionary = self.flatten_dictionary()
		if savepath is not None:
			with open(savepath, 'wb') as outfile:
				pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
                
	def uncorr_idx(self):
		orig_dict = self.orig_data.dictionary
		norm_dict = orig_dict/np.linalg.norm(orig_dict, axis = 0)
		corrs = np.dot(np.transpose(norm_dict),norm_dict)
		uncorr_idx = [0]
		for i in range(1,self.N_orig):
			corr_flag = False
			for j in range(0,i):
				if corrs[i,j] > self.corr_thresh:
					corr_flag = True
					break
			if not(corr_flag):
				uncorr_idx.append(i)
		return np.array(uncorr_idx).astype(int)

	def sparse_uncorr_idx(self):
		orig_dict = self.orig_data.dictionary
		norm_dict = normalize(orig_dict, norm = 'l2', axis = 0)
		corrs = norm_dict.transpose() * norm_dict
		uncorr_idx = [0]
		for i in range(1,self.N_orig):
			corr_flag = False
			for j in range(0,i):
				if corrs[i,j] > self.corr_thresh:
					corr_flag = True
					break
			if not(corr_flag):
				uncorr_idx.append(i)
		return np.array(uncorr_idx).astype(int)
    
	#reduces the peaks of dictionary columns
	#assumes min(dict[:,i]) corresponds to count of exactly one kmer
	def flatten_dictionary(self):
		base_dict = self.orig_data.dictionary[:,self.uncorr_indices]
		flat_dict = np.copy(base_dict)
		for i in range(self.N):
			col = base_dict[:,i]
			m = np.min(col[col>0])
			removed = 0
			#todo: more scientific choice of threshold
			for j in range(len(self.kmer_to_idx)):
				if col[j]/m > self.rel_max_thresh:
					flat_dict[j,i] = self.rel_max_thresh*m
					removed += base_dict[j,i] - self.rel_max_thresh
			total_i = 1/m
			total_i_adj = 1/m - removed
			flat_dict[:,i] = flat_dict[:,i]*total_i/total_i_adj
		return flat_dict

	def sparse_flatten_dictionary(self):
		base_dict = self.orig_data.dictionary[:,self.uncorr_indices]
		flat_dict = base_dict.copy()
		for i in range(self.N):
			col = base_dict[:,i]
			col_plus = col[col.nonzero()]
			m = col_plus.min()
			removed = 0
			for j in col.nonzero()[0]:
				if col[j,0]/m > self.rel_max_thresh:
					flat_dict[j,i] = self.rel_max_thresh*m
					removed += base_dict[j,i] - self.rel_max_thresh
			total_i = 1/m
			total_i_adj = 1/m - removed
			flat_dict[:,i] = flat_dict[:,i]*total_i/total_i_adj
		return flat_dict
    
	def process_abund_and_mut(self, abundance_list, mut_rate_list):
		proc_abundance = [abundance_list[i] for i in self.uncorr_indices]
		proc_abundance = proc_abundance/np.sum(proc_abundance)
		proc_mut_rate_list = [mut_rate_list[i] for i in self.uncorr_indices]
		return proc_abundance, proc_mut_rate_list
                    
                
#inputs: proc_data object, sample abundance (list), sample mutation rates (list)
#output: get_mutated_data object, contains multiple mutated samples and aggregated kmer frequency vector
class get_mutated_data():
	'''
	This class is to randomly mutate the original genomes and generate aggregated kmer frequency vector.
	Supports both sparse and dense matrix formats for proc_data.
	'''
	
	def __init__(self, proc_data, abundance_list, mut_rate_list, total_kmers = None, rnd = True):
		self.orig_A_matrix = proc_data
		self.fasta_files = proc_data.fasta_files
		self.kmer_to_idx = proc_data.kmer_to_idx
		self.mut_kmer_ct = np.zeros(len(self.kmer_to_idx))
		self.abundance_list = abundance_list
		self.mut_rate_list = mut_rate_list
		self.rnd = rnd
		self.N = len(self.fasta_files)
		if total_kmers is None:
			#scale total kmers for number of organisms--needs fine-tuning        
			self.total_kmers = self.N*pow(10,10)
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
		#TEMPORARY for debugging
		try:
			with Pool(processes=n_processes) as excutator:
				res = list(excutator.map(self.get_single_mutated_organism, params))
		except Exception as err:
			print("Error: {0}".format(err))
			raise RuntimeError('Debugging exception found.')
		self.mut_orgs += [curr_mut_org for (curr_mut_org, _) in res]
		self.mut_kmer_ct += np.sum(np.vstack([mut_kmer_ct for (_, mut_kmer_ct) in res]), axis=0)
		if self.rnd:
			self.mut_kmer_ct = np.round(self.mut_kmer_ct)

	@staticmethod
	def get_single_mutated_organism(this_param):
		try:
			fasta_file, total_kmers, mut_rate, rel_abundance, kmer_to_idx = this_param
			curr_mut_org = mso.get_mutated_seq_and_kmers(fasta_file, kmer_to_idx, mut_rate)
			mut_kmer_ct = total_kmers*rel_abundance*curr_mut_org.mut_kmer_ct
			return [curr_mut_org, mut_kmer_ct]
		#TEMPORARY for debugging
		except Exception as err:
			print("Error: {0}".format(err))
			raise RuntimeError('Debugging exception found.')