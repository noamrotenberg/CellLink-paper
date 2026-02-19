set -e

# Set up inputs
INPUT_PATH=$(realpath ${1})
MAX_LEN="${2}"
TRAIN_JSON_FILENAME=$(realpath ${3})
DEV_JSON_FILENAME=$(realpath ${4})
MODEL_PATH=$(realpath ${5})
CUDA="${6}"
OUTPUT_PATH=$(realpath ${7})

# Log inputs
echo "INPUT_PATH=${INPUT_PATH}"
echo "MAX_LEN=${MAX_LEN}"
echo "TRAIN_JSON_FILENAME=${TRAIN_JSON_FILENAME}"
echo "DEV_JSON_FILENAME=${DEV_JSON_FILENAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA=${CUDA}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

# Prepare
export CUDA_VISIBLE_DEVICES="${CUDA}"
mkdir -p "${OUTPUT_PATH}"
rm -f "${OUTPUT_PATH}/*"

date

# Process files
for input_filename in "${INPUT_PATH}"/*.json; do
	#echo ${input_filename}
	name=$(basename ${input_filename})
	output_filename="${OUTPUT_PATH}/${name}"
	echo "Processing ${input_filename} to ${output_filename}"
	python -u src/run_ner.py \
		--model_name_or_path "$MODEL_PATH" \
		--task_name ner \
		--max_seq_length "$MAX_LEN" \
		--train_file "$TRAIN_JSON_FILENAME" \
		--validation_file "$DEV_JSON_FILENAME" \
		--test_file "$input_filename" \
		--output_dir "$OUTPUT_PATH" \
		--do_predict
	mv ${OUTPUT_PATH}/predictions.json ${output_filename}
done

date
echo "Done."
