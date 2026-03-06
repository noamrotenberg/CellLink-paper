#!/bin/bash

CELLLINK_PATH="../../NLM_CellLink_data"
source ../../../agent_env/bin/activate


# run GPT 0-shot inference:
export endpoint= ### insert endpoint url
export api_key= ### insert Azure API key
ZEROSHOT_OUTPUT_PATH=../../model_outputs/NER-exp2_zeroshot_gpt-5_2.xml
python src/OpenAI_zeroshot_inference.py $CELLLINK_PATH/test.xml $ZEROSHOT_OUTPUT_PATH ../../model_outputs/NER-exp2_zeroshot_gpt-5_2_cache.json gpt-5.2

MERGED="cell_phenotype cell_hetero cell_vague merged"
for ANNOTATION_TYPE in cell_phenotype cell_hetero cell_vague None "$MERGED"
do
    for EVALUATION_METHOD in strict approx
    do
        python ../general_scripts/evaluate.py --reference_path $CELLLINK_PATH/test.xml --prediction_path $ZEROSHOT_OUTPUT_PATH --evaluation_type span --evaluation_method $EVALUATION_METHOD --annotation_type "$ANNOTATION_TYPE"
    done
done