


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

raise Error
# now, split up the clinvar vcf into coding and noncoding mutations and smaller subcategories

# save as separate csv files



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
print(df.molecular_consequence.unique())
# Separate the DataFrame into coding and noncoding mutations
# split based on your bed files!

# does it occur in any exon? --> coding

# does it occur in any known cCRE -> known noncoding

# literally anything else --> unknown noncoding

# noncoding is everything else?
coding_df = df[df['molecular_consequence'].str.contains('missense_variant', na=False)]
# noncoding includes at least intron_variant, splice_donor_variant
noncoding_df = df[~df['molecular_consequence'].str.contains('missense_variant', na=False)]
#print(noncoding_df.molecular_consequence.unique())

# ['missense_variant' 'synonymous_variant' 'splice_donor_variant'
#  'intron_variant' 'inframe_deletion' 'nonsense' 'frameshift_variant'
#  'inframe_insertion' 'splice_acceptor_variant' 'no_sequence_alteration'
#  None '5_prime_UTR_variant' '3_prime_UTR_variant'
#  'initiator_codon_variant' 'inframe_indel' 'non-coding_transcript_variant'
#  'stop_lost' 'genic_downstream_transcript_variant'
#  'genic_upstream_transcript_variant']






# Save the coding and noncoding DataFrames to CSV files
coding_df.to_csv('clinvar_coding.csv', index=False)
noncoding_df.to_csv('clinvar_noncoding.csv', index=False)
raise Error


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

