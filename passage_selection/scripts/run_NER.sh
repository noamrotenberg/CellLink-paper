set -e

CUDA="${1}"

# Convert format to HFJSON for annotation
mkdir -p data/HFJSON_input
python -u src/convert_bioc_to_hfjson.py data/BioCXML data/HFJSON_input 10

# Run NER models
MODEL_PATH_AnatEM="***ADD***"
MODEL_PATH_BioID="***ADD***"
MODEL_PATH_CRAFT="***ADD***"
MODEL_PATH_JNLPBA="***ADD***"

mkdir -p data/HFJSON_output_AnatEM
rm -rf data/HFJSON_output_AnatEM/*
mkdir -p data/HFJSON_output_BioID
rm -rf data/HFJSON_output_BioID/*
mkdir -p data/HFJSON_output_CRAFT
rm -rf data/HFJSON_output_CRAFT/*
mkdir -p data/HFJSON_output_JNLPBA
rm -rf data/HFJSON_output_JNLPBA/*

./scripts/predict.sh data/HFJSON_input 512 data/HFJSON_corpora/AnatEM_train.json data/HFJSON_corpora/AnatEM_dev.json ${MODEL_PATH_AnatEM} ${CUDA} data/HFJSON_output_AnatEM
./scripts/predict.sh data/HFJSON_input 512 data/HFJSON_corpora/BioID_train.json data/HFJSON_corpora/BioID_dev.json ${MODEL_PATH_BioID} ${CUDA} data/HFJSON_output_BioID
./scripts/predict.sh data/HFJSON_input 512 data/HFJSON_corpora/CRAFT_train.json data/HFJSON_corpora/CRAFT_dev.json ${MODEL_PATH_CRAFT} ${CUDA} data/HFJSON_output_CRAFT
./scripts/predict.sh data/HFJSON_input 512 data/HFJSON_corpora/JNLPBA_train.json data/HFJSON_corpora/JNLPBA_train.json ${MODEL_PATH_JNLPBA} ${CUDA} data/HFJSON_output_JNLPBA

# Convert the NER data back to BioCXML
./scripts/finalize_dir.sh data/BioCXML data/HFJSON_input data/HFJSON_output_AnatEM data/BioCXML_AnatEM
./scripts/finalize_dir.sh data/BioCXML data/HFJSON_input data/HFJSON_output_BioID data/BioCXML_BioID
./scripts/finalize_dir.sh data/BioCXML data/HFJSON_input data/HFJSON_output_CRAFT data/BioCXML_CRAFT
./scripts/finalize_dir.sh data/BioCXML data/HFJSON_input data/HFJSON_output_JNLPBA data/BioCXML_JNLPBA

