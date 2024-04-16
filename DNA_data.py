import pandas as pd
from Bio import SeqIO
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import os
import glob
import torch

# now we want some example corpus
# you can download a fasta genome if you have not already

# first thing is top read in the genome file, break it into chromosomes and remove all N.
# we will not use unknown regions in this first example


def process_fasta_file(file_path):
    # Parse the FASTA file
    for record in SeqIO.parse(file_path, "fasta"):
        # Get the chromosome name and sequence
        chromosome_name = record.id
        sequence = str(record.seq)

        # Remove all "N" characters from the sequence
        sequence = sequence.replace("N", "").upper()

        # Write the sequence to a new file
        with open(f"{chromosome_name}.fasta", "w") as output_file:
            # we don't want the chromosome name in the sequence
            #output_file.write(f">{chromosome_name}\n")
            output_file.write(sequence)

# Usage:
# process_fasta_file("hg38.fa") # validated these fasta are reasonable
# raise Error
# let's train a byte pair encoder on chromosome 1
            

# rewrite the BPE for your transcripts
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(dir_path, vocab_size=30000):
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Get list of paths to text files
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]

    # Train the tokenizer
    tokenizer.train(files=file_paths, 
                    vocab_size=vocab_size, 
                    #min_frequency=2, 
                    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"], 
                    show_progress=True)

    return tokenizer

def train_bpe_tokenizer(dir_path, vocab_size=30000, chunk_size=10000):
    # Initialize a tokenizer with Byte-Level BPE
    tokenizer = Tokenizer(models.BPE())

    # Use byte-level processing as pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # Get list of paths to text files
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]


    # Initialize a trainer with byte-level and custom parameters
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, 
                                  special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"], 
                                  show_progress=True)

    # Read and chunk the file
    with open(file_path, 'r') as file:
        lines = []
        chunk = file.read(chunk_size)
        while chunk:
            lines.append(chunk)
            chunk = file.read(chunk_size)
    print(len(lines))
    # Train the tokenizer on the chunks
    tokenizer.train_from_iterator(lines, trainer=trainer)

    return tokenizer

# chunk up the original big fasta before dataloader ingest
def chunk_file(input_file, output_dir, chunk_size=8388608):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            output_file = os.path.join(output_dir, f'chunk_{i}.txt')
            with open(output_file, 'w') as out_f:
                out_f.write(chunk)
            i += 1


# dataset-- let's pre split up the fasta into chunks of context_len size for quick I/O
# before being used for training
class DNA_dataset(torch.utils.data.Dataset):
    def __init__(self, seq_dir):
        super().__init__()
        # get an array of filenames from seq_dir
        self.dna_files = sorted(glob.glob(seq_dir+"/*.txt"))

    def __getitem__(self, index):
        # open the txt file and read in as string
        with open(self.dna_files[index], 'r') as f:
            content = f.read()
        return content

    def __len__(self):
        return len(self.dna_files)
    
# Usage:
    
chunk_file('human_genome_contigs/chr1.fasta', 'chr1_chunks',)#chunk_size=2048)
# 112540 chunks @ 2048
            
tokenizer = train_tokenizer("transcript_strs")
tokenizer.save("transcript_tokenizer.json")
# Usage:
#tokenizer = train_bpe_tokenizer("human_genome_contigs/chr1.fasta")
# save it 
#tokenizer.save("chr1_tokenizer.json")
# reload it
# tokenizer = Tokenizer.from_file("chr1_tokenizer.json")
# #print(tokenizer.get_vocab())
# # Encode the sequence
# sequence = "ATGCTGCTGGGGGATCGCTGCACGTACTCGACCGGGGGCTTTACGAAAAAAAAGATCGGCTTTTTTTTTTAATGCGTCCCCATATA"
# encoded_sequence = tokenizer.encode(sequence)