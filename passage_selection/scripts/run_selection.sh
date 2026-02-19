set -e

cat /dev/null > data/empty.txt

PREVIOUSLY_SELECTED_PASSAGEIDS="data/empty.txt"
UNAVAILABLE_PMIDS="data/empty.txt"
SELECTION_COUNT=3000
BATCH_SIZE=18000

# Full data is a little over 1,515,000 passages
# Should run three iterations:
# Iteration 1: 85 batches
# Iteration 2: 15 batches
# Iteration 3: 3 batches
# Iteration 4: 1 batch
# Time: about 5 days
# Realistically, you may want to separate the data into subsets

python -u src/select_passages.py data/config.json data/filtered_passages.jsonl.gz data/adjusted_measurements.json ${SELECTION_COUNT} ${BATCH_SIZE} selected_passageids.txt 2>&1 | tee select_passages.log 
