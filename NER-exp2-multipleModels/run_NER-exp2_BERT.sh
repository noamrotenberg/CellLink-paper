#!/bin/bash


# source ../../bert_env/bin/activate
source ~/bert_env/bin/activate

SCRIPT_SRC_PATH="src"
FINAL_OUTPUT_PATH=../../model_outputs
DATA_SRC_PATH="../../NLM_CellLink_data"

for ENTITY_TYPE in cell_phenotype cell_hetero cell_desc
do

    
    TRAIN_XML_PATH="${DATA_SRC_PATH}/splits_by_entity_type/train_${ENTITY_TYPE}_only.xml"
    DEV_XML_PATH="${DATA_SRC_PATH}/splits_by_entity_type/val_${ENTITY_TYPE}_only.xml"
    TEST_XML_PATH="${DATA_SRC_PATH}/splits_by_entity_type/test_${ENTITY_TYPE}_only.xml"
    
    TRAIN_JSON="${DATA_SRC_PATH}/BiomedBERT_hfjson_format/train_${ENTITY_TYPE}_only.json"
    DEV_JSON="${DATA_SRC_PATH}/BiomedBERT_hfjson_format/val_${ENTITY_TYPE}_only.json"
    TEST_JSON="${DATA_SRC_PATH}/BiomedBERT_hfjson_format/test_${ENTITY_TYPE}_only.json"
    
    OUTPUT_DIR="./${ENTITY_TYPE}/output"
    
    
    # CPU
    # Convert XML to JSON <- do this step once for each file 
    python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $TRAIN_XML_PATH  $TRAIN_JSON
    python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $DEV_XML_PATH    $DEV_JSON
    python -u "${SCRIPT_SRC_PATH}/convert_bioc_to_hfjson.py" $TEST_XML_PATH   $TEST_JSON
    
    #GPU
    # Train the model on the JSON files
    
    CUDA=5 # GPU number
    MODEL="./$ENTITY_TYPE/model" # name of directory to output model
    
    mkdir -p "$MODEL"
    mkdir -p "$OUTPUT_DIR"
    
    # use nvidia-smi or gpustat to find a GPU with memory free
    export CUDA_VISIBLE_DEVICES="${CUDA}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    
    python -u "${SCRIPT_SRC_PATH}/run_ner.py" \
      --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
      --task_name ner \
      --train_file "$TRAIN_JSON" \
      --validation_file "$DEV_JSON" \
      --output_dir "$MODEL" \
      --num_train_epochs=20.0 \
      --save_steps=10000 \
      --save_total_limit=3 \
      --evaluation_strategy=epoch \
      --learning_rate=3e-05 \
      --per_device_train_batch_size=16 \
      --do_train \
      --do_eval
    
    # I think I can combine train and test, but maybe safer not to
    # change test_file if desired
    python -u "${SCRIPT_SRC_PATH}/run_ner.py" \
      --model_name_or_path "$MODEL" \
      --task_name ner \
      --train_file "$TRAIN_JSON" \
      --validation_file "$DEV_JSON" \
      --test_file "$TEST_JSON" \
      --output_dir "$OUTPUT_DIR" \
      --do_predict
    
    
    OUTPUT_JSON_FILE="${OUTPUT_DIR}/predictions.json"
    
    # Now convert the JSON output into tagged BioC XML
    python -u "${SCRIPT_SRC_PATH}/convert_hfjson_to_bioc.py" $TEST_XML_PATH $TEST_JSON $OUTPUT_JSON_FILE $FINAL_OUTPUT_PATH/NER_BERT_${ENTITY_TYPE}_output.xml
done

python ../general_scripts/merge_BioCXML_annotations.py $FINAL_OUTPUT_PATH/NER_BERT_cell_phenotype_output.xml $FINAL_OUTPUT_PATH/NER_BERT_cell_hetero_output.xml $FINAL_OUTPUT_PATH/NER_BERT_cell_desc_output.xml $FINAL_OUTPUT_PATH/NER_BERT_combined_output.xml

MERGED="cell_phenotype cell_hetero cell_desc merged"
for ENTITY_TYPE in cell_phenotype cell_hetero cell_desc None "$MERGED"
do
    for EVALUATION_METHOD in strict approx
    do
        python ../general_scripts/evaluate.py --reference_path $DATA_SRC_PATH/test.xml --prediction_path $FINAL_OUTPUT_PATH/NER_BERT_combined_output.xml --evaluation_type span --evaluation_method $EVALUATION_METHOD --annotation_type "$ENTITY_TYPE" --logging_level critical
    done
done
