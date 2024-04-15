# create Human Genome dataset strings

import pandas as pd
import pyBigWig
import warnings
from pyfaidx import Fasta
import glob
import multiprocessing as mp

def read_acceptable_contigs(filename, acceptable_contigs):
    df = pd.read_csv(filename) #"ENCODEV45_basic.csv")

    # elim all special contigs
    df = df[df.chrom.isin(acceptable_contigs)]
    return df 

def explode_contigs(df):
    # create a new dataframe called exons, where each exon for each gene has it's own row with start and end sites
    exons = df.copy()

    # clean the trailing , first
    exons['exonStarts'] = exons['exonStarts'].str[:-1]
    exons['exonEnds'] = exons['exonEnds'].str[:-1]

    # Split the 'exonStarts' and 'exonEnds' columns into lists of strings
    exons['exonStarts'] = exons['exonStarts'].str.split(',')
    exons['exonEnds'] = exons['exonEnds'].str.split(',')

    # Use the `explode` function to create a new row for each element in these lists
    exons_exploded = exons.explode(['exonStarts', 'exonEnds'])

    # Create a new DataFrame that contains the exploded 'exonStarts' and 'exonEnds' columns
    clean_exons = exons_exploded[['name', 'name2', 'chrom', 'strand', 'exonStarts', 'exonEnds']]
    clean_exons = clean_exons.rename(columns={'exonStarts': 'start', 'exonEnds': 'end'})
    clean_exons["start"] = clean_exons["start"].astype(int)
    clean_exons["end"] = clean_exons["end"].astype(int)
    return clean_exons


def process_contig(contig, bb):
    element_list = bb.entries(contig, 0, -1)
    print(contig, len(element_list))
    df_list = []
    for entry in element_list:
        pe_chrom = pd.DataFrame()
        pe_chrom["chrom"] = [contig]
        pe_chrom["start"] = [int(entry[0])]
        pe_chrom["end"] = [int(entry[1])]
        element_str = entry[2].split("\t")
        pe_chrom["strand"] = [element_str[2]]
        pe_chrom["ENCODE Accession"] = [element_str[0]]
        pe_chrom["ENCODE classification"] = [element_str[6]]
        pe_chrom["UCSC label"] = [element_str[-3]]
        df_list.append(pe_chrom)
    chrom_df = pd.concat(df_list)
    return chrom_df

def get_promoters_and_enhancers(encodeCcreCombined, acceptable_contigs):
    # now get all the enhancers (ENCODE cCREs, VISTA, Zoonomia cCREs), align them with nearest gene 
    bb = pyBigWig.open(encodeCcreCombined) 

    all_promoters_and_enhancers = []

    #promoters_and_enhancers["chrom"] = 
        # Create a pool of workers
    with mp.Pool(mp.cpu_count()) as pool:
        all_promoters_and_enhancers = pool.starmap(process_contig, [(contig, bb) for contig in acceptable_contigs])


    # for contig in acceptable_contigs:
    #     element_list = bb.entries(contig, 0, -1)
    #     print(contig, len(element_list))
    #     df_list = []
    #     for entry in element_list:
    #         pe_chrom = pd.DataFrame()
    #         pe_chrom["chrom"] = [contig]
    #         pe_chrom["start"] = [int(entry[0])]
    #         pe_chrom["end"] = [int(entry[1])]
    #         element_str = entry[2].split("\t")
    #         pe_chrom["strand"] = [element_str[2]]
    #         pe_chrom["ENCODE Accession"] = [element_str[0]]
    #         pe_chrom["ENCODE classification"] = [element_str[6]]
    #         pe_chrom["UCSC label"] = [element_str[-3]]
    #         df_list.append(pe_chrom)
    #     chrom_df = pd.concat(df_list)
    #     all_promoters_and_enhancers.append(chrom_df)

    all_promoters_and_enhancers = pd.concat(all_promoters_and_enhancers)
    return all_promoters_and_enhancers

def find_closest(row, df1):
    df1['diff'] = abs(df1['txStart'] - row.end)
    closest_entry = df1.nsmallest(1, 'diff').name.values[0]
    return closest_entry

def associate_enhancers(df, all_promoters_and_enhancers, acceptable_contigs):
    # now we need to associate enhancers to TSS
    # Initialize an empty DataFrame to store the results
    merged = pd.DataFrame()

    chrom_pe =[]

    for chrom in acceptable_contigs:
        df1 = df[df.chrom == chrom].sort_values(['chrom', 'txStart'])#[['chrom', 'txStart']]
        df2 = all_promoters_and_enhancers[all_promoters_and_enhancers.chrom == chrom].sort_values(['chrom', 'end'])#[['chrom', 'end']]
        #closest_ls = []
        df2['closest_TSS'] = df2.apply(lambda row: find_closest(row, df1), axis=1)
        # closest_ls = df2.apply(find_closest, axis=1)#.tolist()
        # df2["closest_TSS"] = closest_ls
        chrom_pe.append(df2)
    return chrom_pe


def make_bed_csvs(df, chrom_pe, acceptable_contigs, clean_exons):
    transcript_lens = []
    warnings.filterwarnings('ignore')

    for chrom in acceptable_contigs:
        result_subset = df[df.chrom == chrom]
        chrom_pe_subset = chrom_pe[acceptable_contigs.index(chrom)]

        # Group by 'closest_TSS' to avoid looping through unique entries
        grouped = chrom_pe_subset.groupby('closest_TSS')
        print(chrom, grouped.size())

        for entry, group in grouped:
            corresponding_row = result_subset[result_subset['name'] == entry].iloc[0]
            name = corresponding_row["name"] + "_" + corresponding_row["name2"]

            exon_subset = clean_exons[clean_exons.name == entry].copy()
            exon_subset["ENCODE classification"] = "exon"

            transcript_pe_subset = group[["chrom", "start", "end", "ENCODE classification"]]
            transcript_exon_subset = exon_subset[["chrom", "start", "end", "ENCODE classification"]]

            transcript_bed = pd.concat([transcript_pe_subset, transcript_exon_subset])
            transcript_bed["transcript_and_name"] = name

            transcript_lens.append((transcript_bed.end - transcript_bed.start).sum())

            transcript_bed.to_csv(f"human_transcripts/{name}.csv")

            
def extract_sequences(fasta_file, bed_file):
    sequences = []
    fasta = Fasta(fasta_file)
    with open(bed_file, 'r') as bed:
        next(bed) # skip the first line
        
        for line in bed:
            chrom, start, end = line.split(",")[1:4]
            start, end = int(start), int(end)
            sequence = fasta[chrom][start:end].seq.upper()
            sequences.append(sequence)

    whole_example = " ".join(sequences)
    # add start token
    whole_example = "<BOS>" + whole_example
    # add end token
    whole_example = whole_example + "<EOS>"
    return whole_example

def make_transcript_strings(dirname):        
    all_human_transcripts = glob.glob(dirname)
    fasta_file = 'hg38.fa'

    for trans in all_human_transcripts:
        file_name = "transcript_strs/" + trans.split("/")[-1].split(".")[0] + ".txt"
        my_string = extract_sequences(fasta_file, trans)
        # Open the file in write mode and write the string to it
        with open(file_name, 'w') as file:
            file.write(my_string)
            
def main():
    acceptable_contigs = ['chr1',
                      'chr10',
                      'chr11',
                      'chr12',
                      'chr13',
                      'chr14',
                      'chr15',
                      'chr16',
                      'chr17',
                      'chr18',
                      'chr19',
                      'chr2',
                      'chr20',
                      'chr21',
                      'chr22',
                      'chr3',
                      'chr4',
                      'chr5',
                      'chr6',
                      'chr7',
                      'chr8',
                      'chr9',
                      'chrX',
                      'chrY'
                     ]
    print(1)
    df = read_acceptable_contigs("ENCODEV45_basic.csv", acceptable_contigs)
    print(df)
    clean_exons = explode_contigs(df)
    print(clean_exons)
    all_promoters_and_enhancers = get_promoters_and_enhancers("encodeCcreCombined.bb", acceptable_contigs)
    print(all_promoters_and_enhancers)
    chrom_pe = associate_enhancers(df, all_promoters_and_enhancers, acceptable_contigs)
    print(chrom_pe)
    make_bed_csvs(df, chrom_pe, acceptable_contigs, clean_exons)
    # don't forget to make the transcript_strs dir from command line
    make_transcript_strings("human_transcripts/*")
    


if __name__ == "__main__":
    main()





    
    


