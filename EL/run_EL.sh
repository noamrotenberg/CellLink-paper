#!/bin/bash

source ../../../agent_env/bin/activate

CELLLINK_PATH="../../NLM_CellLink_data"

# embeddings model endpoint and api-key
export endpoint_embeddings=####
export api_key_embeddings= ####

# agent / gpt-5.2 model endpoint and api-key
export endpoint_agent= ####
export api_key_agent= ####

uv run normalize.py $CELLLINK_PATH/test.xml ../../model_outputs/EL_test_output.xml cell_types.tsv abbreviations.tsv ../../model_outputs/OpenAI_embeddings_cache.json