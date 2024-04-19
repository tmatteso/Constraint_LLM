import requests
import urllib.request
# def download_file(url, filename):
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     with open(filename, 'wb') as file:
#         for chunk in response.iter_content(chunk_size=8192):
#             file.write(chunk)

def download_file(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

# get human genome
#url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
#filename = 'hg38.fa.gz'
#download_file(url, filename)

# get cCREs
#url = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb"
#filename = "encodeCcreCombined.bb"
#download_file(url, filename)

# get clinvar for testing
url = "ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
filename = "clinvar.vcf.gz"
download_file(url, filename)

# get TSS from an scp (small enough)
# from my local machine: scp ENCODEV45_basic.csv IP:/dir/Constraint_LLM
