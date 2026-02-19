set -e

# Extract cell type names from CL <-- This has been done previously
#cat /dev/null > data/empty.obo
#python -u src/extract_CL_PCL.py data/cl_v2025-01-08.obo data/empty.obo data/cl.tsv
#grep -P "\tCL:" data/cl.tsv | sed "s/\t/\tcell_phenotype\t/g" | sort | uniq > data/cell_types.tsv

# Run CL dictionary method
mkdir -p data/BioCXML_DICT
echo "{}" > data/empty.json
python -u src/recognize_dict_pool.py data/cell_types.tsv data/BioCXML data/abbreviations.tsv data/empty.json 10 data/BioCXML_DICT

# Combine annotations 
mkdir -p data/BioCXML_ALL
python -u src/combine_annotations.py data/docids.tsv data/BioCXML AnatEM data/BioCXML_AnatEM BioID data/BioCXML_BioID CRAFT data/BioCXML_CRAFT CL_DICT data/BioCXML_DICT JNLPBA data/BioCXML_JNLPBA PubTator data/BioCXML_PubTator data/BioCXML_ALL 2>&1 | tee logs/combine_annotations.log

# Clean up annotations
mkdir -p data/BioCXML_Filtered
python -u src/update_annotations.py data/BioCXML_ALL data/patch.tsv data/BioCXML_Filtered

# Extract passages
python -u src/extract_passages.py data/config.json data/meta.jsonl.gz data/BioCXML_Filtered data/annotated_passages.jsonl.gz
python -u src/calculate_measurements.py data/config.json summary_calculator data/annotated_passages.jsonl.gz data/annotated_measurements.json

# Add MTIX MeSH predictions
python -u src/add_MTIX_predictions.py data/2024-12-31_PMIDs_for_MeSH_MTIX_data_predictions.json.gz data/annotated_measurements.json data/annotated_passages.jsonl.gz data/mtix_passages.jsonl.gz 2>&1 | tee logs/add_MTIX_predictions.log
python -u src/calculate_measurements.py data/config.json summary_calculator data/mtix_passages.jsonl.gz data/mtix_measurements.json

# Transform data, round 1
python -u src/transform_data.py data/config.json data_transform1 data/annotated_passages.jsonl.gz data/mtix_measurements.json data/transformed_passages1.jsonl.gz | tee logs/transform_data1.log
python -u src/calculate_measurements.py data/config.json summary_calculator data/transformed_passages1.jsonl.gz data/transformed_measurements1.json

# Transform data, round 2
python -u src/transform_data.py data/config.json data_transform2 data/transformed_passages1.jsonl.gz data/transformed_measurements1.json data/transformed_passages2.jsonl.gz | tee logs/transform_data2.log
python -u src/calculate_measurements.py data/config.json summary_calculator data/transformed_passages2.jsonl.gz data/transformed_measurements2.json

# Filter data (e.g., REF , large passages)
python -u src/filter_data.py data/config.json data/transformed_passages2.jsonl.gz data/transformed_measurements2.json data/filtered_passages.jsonl.gz | tee logs/filter_data.log
python -u src/calculate_measurements.py data/config.json summary_calculator data/filtered_passages.jsonl.gz data/filtered_measurements.json

# Adjust measurements
python -u src/adjust_measurements.py data/config.json data/filtered_measurements.json data/adjusted_measurements.json 2>&1 | tee logs/adjust_measurements.log 
