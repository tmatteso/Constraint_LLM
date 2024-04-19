import multiprocessing as mp
import pandas as pd 
import glob

# read in the clinvar file
def read_clinvar(filename):
    num_header_lines = sum(1 for line in open('clinvar.vcf') if line.startswith('##'))

    dtypes = {
        '#CHROM': 'str',
        'POS': 'int64',
        # Add more columns as needed
    }

    # # Read the VCF file, skipping the meta-information lines
    variants_df = pd.read_csv(filename, sep='\t', skiprows=num_header_lines, header=0,
                            usecols=['#CHROM', 'POS'],  engine='c', dtype=dtypes)

    name_ls = []
    for name, group in variants_df.groupby('#CHROM'):
        group.to_csv(f'clinvar_{name}.csv')
        name_ls.append(f'clinvar_{name}.csv')
    return name_ls


# read in the transcripts
def tx_bed_to_chrom_bed(transcript_dir, read_in_full, split_by_chrom=False):
    all_transcripts = glob.glob(transcript_dir)

    # put them all together in one file
    if read_in_full:
        transcript_bed = []

        for transcript_file in all_transcripts:
            # Read the BED file into a dataframe
            transcripts_df = pd.read_csv(transcript_file) #'file.bed', sep='\t', usecols=['chrom', 'start', 'end'])
            # if len(transcripts_df.chrom.unique()) > 1:
            #     print(transcript_file)
            transcript_bed.append(transcripts_df)

        transcripts_df = pd.concat(transcript_bed)
        transcripts_df.to_csv("all_transcripts.bed", sep='\t', header=False, index=False)

    elif not read_in_full:
        transcripts_df = pd.read_csv("all_transcripts.bed", sep='\t', 
                            names=["chrom","start","end",
                                    "ENCODE classification","transcript_and_name"])

    elif not split_by_chrom:
        return transcripts_df

    # split into separate dfs based on chrom and save as separate bed files
    elif split_by_chrom:
        for name, group in transcripts_df.groupby('chrom'):
            print(f'all_transcripts_{name}.bed')
            group.to_csv(f'all_transcripts_{name}.bed', sep='\t', header=False, index=False)
        return transcripts_df


# Define a function to process a single transcript
def process_transcript(chrom_df, name_ls):
    # find the corresponding clinvar chrom chunk
    chrom = chrom_df.split('_')[-1].split('.')[0]
    clinvar_name = [name for name in name_ls if chrom in name][0]
    variants_df = pd.read_csv(clinvar_name, usecols=['#CHROM', 'POS'])

    # Read the BED file into a dataframe
    transcripts_df = pd.read_csv(chrom_df, sep='\t', names=["chrom","start","end",
                                     "ENCODE classification","transcript_and_name"])

    print(chrom_df, len(transcripts_df.index), clinvar_name, len(variants_df.index))
    # Apply the function to each variant
    variants_df['in_transcript'] = variants_df.apply(is_in_transcript, axis=1, args=(transcripts_df,))
    print(chrom_df, variants_df['in_transcript'].sum())
    return variants_df[variants_df['in_transcript'] == True]

# Define a function to check if a variant falls within any transcript
def is_in_transcript(variant, transcripts_df):
    chrom, pos = variant
    return any((transcripts_df['chrom'] == chrom) & (transcripts_df['start'] <= pos) & (transcripts_df['end'] >= pos))

# subset clinvar based on variants that appear in the transcript bed files
def subset_clinvar(name_ls):
    # Apply the function to each variant
    all_transcripts = glob.glob("all_transcripts_*.bed")

    # Create a pool of processes
    pool = mp.Pool(mp.cpu_count())

    # Process each transcript in the pool of processes
    results = pool.starmap(process_transcript, [(transcript, name_ls) for transcript in all_transcripts])

    # Close the pool
    pool.close()
    pool.join()

    print(results)
    results = pd.concat(results)
    results.to_csv("clinvar_subset.csv")

# redo DNA_eval in a non dumb way
def main():
    filename = 'clinvar.vcf'
    name_ls = read_clinvar(filename)
    transcript_dir = "human_transcripts/*.csv"
    read_in_full = False
    #transcripts_df = tx_bed_to_chrom_bed(transcript_dir, read_in_full)
    subset_clinvar(name_ls)

    pass

if __name__ == "__main__":
    main()