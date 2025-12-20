#!/bin/bash

CELLLINK_PATH="../../NLM_CellLink_data"

# before this script, run the following:
# python ..\general_scripts\split_BioCXML_files_by_entity_type.py $CELLLINK_PATH $CELLLINK_PATH/splits_by_entity_type
# python src/BioCXML_to_LLMjson.py $CELLLINK_PATH/splits_by_entity_type $CELLLINK_PATH/LLM_json_format/


# run GPT 0-shot inference:
source ../../agent_env/bin/activate
export endpoint= ### insert endpoint url
export api_key= ### insert Azure API key
ZEROSHOT_OUTPUT_PATH=../../model_outputs/NER-exp2_zeroshot_gpt4_1.xml
python src/OpenAI_zeroshot_inference.py $CELLLINK_PATH/test.xml $ZEROSHOT_OUTPUT_PATH ../../model_outputs/NER-exp2_zeroshot_cache.json gpt-4.1

MERGED="cell_phenotype cell_hetero cell_desc merged"
for ANNOTATION_TYPE in cell_phenotype cell_hetero cell_desc None "$MERGED"
do
    for EVALUATION_METHOD in strict approx
    do
        python ../general_scripts/evaluate.py --reference_path $CELLLINK_PATH/test.xml --prediction_path $ZEROSHOT_OUTPUT_PATH --evaluation_type span --evaluation_method $EVALUATION_METHOD --annotation_type "$ANNOTATION_TYPE"
    done
done

# LLAMA fine-tuning and inference: ### TO DELETE
python src/LLAMA_finetuning.py $CELLLINK_PATH/LLM_json_format cell_hetero cell_hetero_only/LLAMA /data/rotenbergnh/llama_trials/meta-llama
python src/LLAMA_inference.py  $CELLLINK_PATH/LLM_json_format cell_hetero cell_hetero_only/LLAMA /data/rotenbergnh/llama_trials/meta-llama
# convert back to BioC-XML
# change to TEST
python src/LLM_output_processing.py cell_hetero_only/LLAMA/output-step376.json $CELLLINK_PATH/val.xml cell_hetero ../../model_outputs/LLAMA_cell_hetero_step376.xml

# from laptop: python \\hpcdrive.nih.gov\data\corpus_paper\NER-exp2-multipleModels\src\LLM_output_processing.py \\hpcdrive.nih.gov\data\corpus_paper\NER-exp2-multipleModels\cell_hetero_only\LLAMA\output-step376.json \\hpcdrive.nih.gov\data\NLM_CellLink_data\val.xml cell_hetero \\hpcdrive.nih.gov\data\model_outputs\LLAMA_cell_hetero_step376.xml 
# from laptop: python \\hpcdrive.nih.gov\data\corpus_paper\NER-exp2-multipleModels\src\LLM_output_processing.py \\hpcdrive.nih.gov\data\corpus_paper\NER-exp2-multipleModels\cell_hetero_only\LLAMA\output-step376.json \\hpcdrive.nih.gov\data\NLM_CellLink_data\val.xml cell_hetero \\hpcdrive.nih.gov\data\model_outputs\LLAMA_cell_hetero_step376.xml


