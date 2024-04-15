import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# get human genome
url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
filename = 'hg38.fa.gz'
download_file(url, filename)

# get cCREs
url = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb"
filename = "encodeCcreCombined.bb"
download_file(url, filename)

# get TSS from an scp (small enough)
# from my local machine: scp ENCODEV45_basic.csv IP:/dir/Constraint_LLM
