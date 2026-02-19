set -e

# Download the documents
mkdir -p data/BioCXML_PubTator
python -u src/download_PubTator3.py data/docids.tsv data/BioCXML_PubTator

# Make sure we got all of the documents
python -u src/get_path_filenames.py data/BioCXML_PubTator data/BioCXML_filenames.txt False
python -u src/get_file_docids_fast.py data/docids.tsv data/BioCXML_filenames.txt data/BioCXML_file_pmids.tsv 10 1000

# Remove annotations
mkdir -p data/BioCXML
python -u src/strip_annotations.py data/BioCXML_PubTator data/BioCXML

# Download the metadata
cat data/docids.tsv | cut -sf 1 | sort > data/pmids.txt
python -u src/get_pubmed_metadata.py data/pmids.txt data/meta.jsonl.gz 
