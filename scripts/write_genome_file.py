import os
import argparse
import logging
from glob import glob

def write_genome_file(filepath, genome_dir, N = None):
	fna_files = glob(os.path.join(genome_dir,'*.fna'))
	if N is not None:
		fna_files = fna_files[:N]
	with open(filepath,'w+') as fid:
		for item in fna_files:
			fid.write(item + "\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--genome_dir', type=str, help='The path of directory with all FNA files', default='./files/organism_files')
	parser.add_argument('--out_file', type=str, help='The path of directory with all FNA files', default='./files/genome_file.txt')
	args = parser.parse_args()

	## set up basic log information
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	## set up input and output files/paths
	genome_dirname = os.path.abspath(args.genome_dir)
	out_file = os.path.abspath(args.out_file)
	if not os.path.isdir(genome_dirname):
		logging.error(f"Genome directory {genome_dirname} does not exist.")
		exit(0)

	## write a genome file for downstream analysis
	logging.info(f"Writing a genome file to {out_file}.")
	write_genome_file(out_file, genome_dirname)

