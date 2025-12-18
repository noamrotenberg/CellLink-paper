#!/bin/bash

# source ../../bert_env/bin/activate
source ~/bert_env/bin/activate

SCRIPT_SRC_PATH="src"
OUTPUT_DIR="../../model_outputs/NER-exp1_corpusComparisons"
MODEL_PATHS="models"

get_data_src_path() {
    # input to function $1 is CORPUS_NAME
    CORPUS_NAME="$1"
    if [[ "$CORPUS_NAME" == "NLM_CellLink" ]]; then
        DATA_SRC_PATH="../../NLM_CellLink_data/phenoAndHeteroMerged"
    else
        DATA_SRC_PATH="filtered_corpora/${CORPUS_NAME}"
    fi
    echo $DATA_SRC_PATH
}

# filter other corpora to better align annotation guidelines/styles

for CORPUS_NAME in AnatEM BioID CRAFT JNLPBA
do
    python $SCRIPT_SRC_PATH/filter_annotations.py ./corpora/$CORPUS_NAME ./corpora/filter_${CORPUS_NAME}.tsv $(get_data_src_path $CORPUS_NAME)
done

# convert all files to huggingface json format

for CORPUS_NAME in NLM_CellLink AnatEM BioID CRAFT JNLPBA
do
    DATA_SRC_PATH=$(get_data_src_path $CORPUS_NAME)
    
    TRAIN_XML_PATH="${DATA_SRC_PATH}/train.xml"
    DEV_XML_PATH="${DATA_SRC_PATH}/val.xml"
    TEST_XML_PATH="${DATA_SRC_PATH}/test.xml"
    
    TRAIN_JSON="${DATA_SRC_PATH}/train.hf.json"
    DEV_JSON="${DATA_SRC_PATH}/val.hf.json"
    TEST_JSON="${DATA_SRC_PATH}/test.hf.json"
    
    
    # CPU
    # Convert XML to JSON <- do this step once for each file 
    python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $TRAIN_XML_PATH  $TRAIN_JSON
    # python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $DEV_XML_PATH    $DEV_JSON
    python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $TEST_XML_PATH   $TEST_JSON
done

# train a model on each corpus
for CORPUS_NAME in NLM_CellLink AnatEM BioID CRAFT JNLPBA
do
    DATA_SRC_PATH=$(get_data_src_path $CORPUS_NAME)
    
    TRAIN_XML_PATH="${DATA_SRC_PATH}/train.xml"
    TRAIN_JSON="${DATA_SRC_PATH}/train.hf.json"
    # DEV_JSON="${DATA_SRC_PATH}/val.hf.json"
    DEV_JSON=$TRAIN_JSON # some corpora don't have a 3 splits
    
    
    #GPU
    # Train the model on the JSON files
    CUDA=0 # GPU number
    MODEL="${MODEL_PATHS}/${CORPUS_NAME}" # name of directory to output model
    mkdir -p "$MODEL"
    
    # use nvidia-smi or gpustat to find a GPU with memory free
    export CUDA_VISIBLE_DEVICES="${CUDA}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    
    python -u "${SCRIPT_SRC_PATH}/run_ner.py" \
      --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
      --task_name ner \
      --train_file "$TRAIN_JSON" \
      --validation_file "$DEV_JSON" \
      --output_dir "$MODEL" \
      --num_train_epochs=20 \
      --save_steps=10000 \
      --save_total_limit=3 \
      --evaluation_strategy=epoch \
      --learning_rate=3e-05 \
      --per_device_train_batch_size=16 \
      --do_train \
      --do_eval
    
    # for some reason, --validation_file and --do_eval are required
    
    # apply model to test set of each corpus
    for CORPUS_NAME2 in NLM_CellLink AnatEM BioID CRAFT JNLPBA
    do
        DATA_SRC_PATH2=$(get_data_src_path $CORPUS_NAME2)
        TEST_XML_PATH="${DATA_SRC_PATH2}/test.xml"
        TEST_JSON="${DATA_SRC_PATH2}/test.hf.json"
        
        python -u "${SCRIPT_SRC_PATH}/run_ner.py" \
          --model_name_or_path "$MODEL" \
          --task_name ner \
          --train_file "$TRAIN_JSON" \
          --validation_file $DEV_JSON \
          --test_file "$TEST_JSON" \
          --output_dir "$OUTPUT_DIR/${CORPUS_NAME}-${CORPUS_NAME2}" \
          --do_predict
        
        # for some reason, --validation_file is required
        
        
        OUTPUT_JSON_FILE="${OUTPUT_DIR}/${CORPUS_NAME}-${CORPUS_NAME2}/predictions.json"
        OUTPUT_XML_FILE="${OUTPUT_DIR}/${CORPUS_NAME}_model_on_${CORPUS_NAME2}_test_pred.xml"
        
        # Now convert the JSON output into tagged BioC XML
        python -u "${SCRIPT_SRC_PATH}/convert_hfjson_to_bioc.py" $TEST_XML_PATH $TEST_JSON $OUTPUT_JSON_FILE $OUTPUT_XML_FILE
    done
done

# perform eval
for CORPUS_NAME in NLM_CellLink AnatEM BioID CRAFT JNLPBA
do
    for CORPUS_NAME2 in NLM_CellLink AnatEM BioID CRAFT JNLPBA
    do
        OUTPUT_XML_FILE="${OUTPUT_DIR}/${CORPUS_NAME}_model_on_${CORPUS_NAME2}_test_pred.xml"
        DATA_SRC_PATH2=$(get_data_src_path $CORPUS_NAME2)
        TEST_XML_PATH="${DATA_SRC_PATH2}/test.xml"
        echo "eval of $CORPUS_NAME model on $CORPUS_NAME2"
        python ../general_scripts/evaluate.py --reference_path $TEST_XML_PATH --prediction_path $OUTPUT_XML_FILE --evaluation_type span --evaluation_method strict --annotation_type None --logging_level critical
    done
done
