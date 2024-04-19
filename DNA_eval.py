


# read in the clinvar vcf
# transform to csv with 
# #CHROM, POS, REF, ALT, 
# then parse the info field
# ALLELEID=2193183;CLNDISDB=MeSH:D030342,MedGen:C0950123;CLNDN=Inborn_genetic_diseases;CLNHGVS=NC_000001.11:g.69134A>G;CLNREVSTAT=criteria_provided,_single_submitter;CLNSIG=Likely_benign;CLNVC=single_nucleotide_variant;CLNVCSO=SO:0001483;GENEINFO=OR4F5:79501;MC=SO:0001583|missense_variant;ORIGIN=1
# get CLNSIG, CLNVC, GENEINFO, MC, ORIGIN


# s = "ALLELEID=2193183;CLNDISDB=MeSH:D030342,MedGen:C0950123;CLNDN=Inborn_genetic_diseases;CLNHGVS=NC_000001.11:g.69134A>G;CLNREVSTAT=criteria_provided,_single_submitter;CLNSIG=Likely_benign;
# CLNVC=single_nucleotide_variant;CLNVCSO=SO:0001483;GENEINFO=OR4F5:79501;MC=SO:0001583|missense_variant;ORIGIN=1"

# clnsig = re.search('CLNSIG=(.*?);', s).group(1)
# clnvc = re.search('CLNVC=(.*?);', s).group(1)
# geneinfo = re.search('GENEINFO=(.*?);', s).group(1)
# mc = re.search('MC=(.*?);', s).group(1)
# origin = re.search('ORIGIN=(.*?)(;|$)', s).group(1)

# print(f'CLNSIG: {clnsig}')
# print(f'CLNVC: {clnvc}')
# print(f'GENEINFO: {geneinfo}')
# print(f'MC: {mc}')
# print(f'ORIGIN: {origin}')

import pandas as pd
import re
import os
import shutil
import random
import glob

# split up the original transcript_strs into train and validation
# rename transcript_strs to train_samples
# os.rename('transcript_strs', 'train_samples')
# # mkdir val_samples
# os.makedirs("val_samples", exist_ok=True)

# # mv 1000 samples to validation set, val_samples
# # Get a list of files in the source directory
# files = os.listdir('train_samples')
# # Randomly select 1000 files
# random_files = random.sample(files, 1000)

# # Move the selected files
# for file in random_files:
#     shutil.move(os.path.join('train_samples', file), "val_samples")

#raise Error
# now, split up the clinvar vcf into coding and noncoding mutations and smaller subcategories

# save as separate csv files



# subset based on if they appear in my bed files
# how do I make this fast?
# df = pd.read_csv("example_transcript.csv")

# # Select the columns for the BED file
# bed_df = df[['chrom', 'start', 'end']]  # Replace 'chrom', 'start', 'end' with your actual column names

# # Write the BED file
# bed_df.to_csv('file.bed', sep='\t', header=False, index=False)

# Read the VCF file into a dataframe
# Assuming the VCF file has been preprocessed to have 'chrom', 'pos' columns
# Count the number of header lines
num_header_lines = sum(1 for line in open('clinvar.vcf') if line.startswith('##'))

dtypes = {
    '#CHROM': 'str',
    'POS': 'int64',
    # Add more columns as needed
}

# Read the VCF file, skipping the meta-information lines
variants_df = pd.read_csv('clinvar.vcf', sep='\t', skiprows=num_header_lines, header=0,
                          usecols=['#CHROM', 'POS'],  engine='c', dtype=dtypes)

print("read in clinvar")
# # Convert the DataFrame to a Parquet file
# variants_df.to_parquet('clinvar.parquet')

# raise Error
# Read the Parquet file into a DataFrame
# variants_df = pd.read_parquet('clinvar.parquet')

# all_transcripts = glob.glob("human_transcripts/*.csv")

# put them all together in one file

# transcript_bed = []

# for transcript_file in all_transcripts:
#     # Read the BED file into a dataframe
#     transcripts_df = pd.read_csv(transcript_file) #'file.bed', sep='\t', usecols=['chrom', 'start', 'end'])
#     transcript_bed.append(transcripts_df)

# transcripts_df = pd.concat(transcript_bed)
# transcripts_df.to_csv("all_transcripts.bed", sep='\t', header=False, index=False)

transcripts_df = pd.read_csv("all_transcripts.bed", sep='\t', header=False, index=False)


# Define a function to check if a variant falls within any transcript
def is_in_transcript(variant):
    chrom, pos = variant
    return any((transcripts_df['chrom'] == chrom) & (transcripts_df['start'] <= pos) & (transcripts_df['end'] >= pos))

# Apply the function to each variant
variants_df['in_transcript'] = variants_df.apply(is_in_transcript, axis=1)

print(variants_df['in_transcript'].sum())

# Check if any variants fall within transcripts
if variants_df['in_transcript'].any():
    print("Some variants fall within the genomic ranges in the transcripts.")
else:
    print("No variants fall within the genomic ranges in the transcripts.")
raise Error

# Count the number of header lines
num_header_lines = sum(1 for line in open('../clinvar.vcf') if line.startswith('##'))

# Read the VCF file, skipping the meta-information lines
df = pd.read_csv('../clinvar.vcf', sep='\t', skiprows=num_header_lines, header=0)

df['clnsig'] = df['INFO'].apply(lambda x: re.search('CLNSIG=(.*?);', x).group(1) if re.search('CLNSIG=(.*?);', x) else None)
df['clnvc'] = df['INFO'].apply(lambda x: re.search('CLNVC=(.*?);', x).group(1) if re.search('CLNVC=(.*?);', x) else None)
df['geneinfo'] = df['INFO'].apply(lambda x: re.search('GENEINFO=(.*?);', x).group(1) if re.search('GENEINFO=(.*?);', x) else None)
df['mc'] = df['INFO'].apply(lambda x: re.search('MC=(.*?);', x).group(1) if re.search('MC=(.*?);', x) else None)
print(df['mc'] )
df['origin'] = df['INFO'].apply(lambda x: re.search('ORIGIN=(.*?)(;|$)', x).group(1) if re.search('ORIGIN=(.*?)(;|$)', x) else None)
# split mc on the "|"
df['seq_ontology_id'] = df['mc'].apply(lambda x: x.split('|')[0] if x else None)
df['molecular_consequence'] = df['mc'].apply(lambda x: x.split('|')[1].split(',')[0] if x else None)
print(df['molecular_consequence'])
df = df.reindex(columns=['#CHROM', 'POS', 'REF', 'ALT', 'clnsig', 'clnvc', 'geneinfo', 'seq_ontology_id', 'molecular_consequence', 'origin'])



# check if any variant falls within any of the genomic ranges in the transcripts

benign = [
    "Benign", 'Benign/Likely_benign', "Likely_benign"
]

pathogenic = [
    "Pathogenic", 'Pathogenic/Likely_pathogenic', "Likely_pathogenic"
]

regex = '|'.join(benign)
# Use the regular expression with str.contains
benign_df = df[df['clnsig'].str.contains(regex, na=False)]
#print(len(benign_df.index)) # 1,114,427
regex = '|'.join(pathogenic)
pathogenic_df = df[df['clnsig'].str.contains(regex, na=False)]
#print(len(pathogenic_df.index)) # 263711

df_ls = [benign_df, pathogenic_df]
name_ls = ["benign", "pathogenic"]

coding_ls = [
    "missense_variant", "synonymous_variant", "inframe_deletion", "nonsense",
    "inframe_insertion", "inframe_indel", "stop_lost", "genic_downstream_transcript_variant", 
    "genic_upstream_transcript_variant", "initiator_codon_variant"
    ]

for i in range(len(df_ls)):
    sub_df = df_ls[i]
    # Create a regular expression that matches any of the entries in coding_ls
    regex = '|'.join(coding_ls)

    # Use the regular expression with str.contains
    coding_df = sub_df[sub_df['molecular_consequence'].str.contains(regex, na=False)]
    # noncoding is everything else
    noncoding_df = sub_df[~sub_df['molecular_consequence'].str.contains(regex, na=False)]
    # Save the coding and noncoding DataFrames to CSV files
    coding_df.to_csv(f'clinvar_coding_{name_ls[i]}.csv', index=False)
    noncoding_df.to_csv(f'clinvar_noncoding_{name_ls[i]}.csv', index=False)

# because we cannot predict snps for the most part, 
# we have to align these variants to the genome for proper context
    
# example transcript file for now.


# at the end of the day, we want (DNA, benign) pairs. The most obvious issue with BPE is the alignment issue
# so given SNP, align to tokenized transcripts, for now. 
    
# then get the logits from the model and score the variant.
    
# then we define some simple cutoff line to do the classification.

# test against NT, DNA BERT
    
# so let's see how many variants fit nicely in out existing genome chunks


#print(noncoding_df.molecular_consequence.unique())

# ['missense_variant' 'synonymous_variant' 'splice_donor_variant'
#  'intron_variant' 'inframe_deletion' 'nonsense' 'frameshift_variant'
#  'inframe_insertion' 'splice_acceptor_variant' 'no_sequence_alteration'
#  None '5_prime_UTR_variant' '3_prime_UTR_variant'
#  'initiator_codon_variant' 'inframe_indel' 'non-coding_transcript_variant'
#  'stop_lost' 'genic_downstream_transcript_variant'
#  'genic_upstream_transcript_variant']







#raise Error


# subsets = {chrom: df[df['#CHROM'] == chrom] for chrom in df['#CHROM'].unique()}
# for chrom, subset in subsets.items():
#     subset.to_csv(f'clinvar_{chrom}.csv', index=False) 

# need to know if genic or not

# then I want it chunked by chromosome. easier to keep track of

# have the tokenized dataset ready by the end of the week. Take it from Divneet. DO NOT WAIT
# once we have the tokenized dataset, you need to map this onto the tokens.
    

# need to check for the presence of Transposable Elements and Microsatellites in the tokenizer

# how do I even compare the finemapping stuff?

# for now, avoid atacQTLS. Wait to ask pooja for a good dataset.

