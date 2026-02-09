# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 20:12:53 2025

@author: rotenbergnh
"""

import os
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import pandas as pd
import torch
import json
import sys
import numpy as np


if len(sys.argv) != 5:
    raise Exception("Usage: python LLAMA_finetuning.py <LLM_json_data_dirpath> <entity_type> <model_dirpath> <model_cache_dir>")
    
LLM_json_data_dirpath = sys.argv[1]
entity_type = sys.argv[2]
model_dirpath = sys.argv[3]
model_cache_dir = sys.argv[4]

# LLM_json_data_dirpath = "../../NLM_CellLink_data/LLM_json_format"
# entity_type = "cell_hetero"
# model_dirpath = "cell_hetero_only/LLAMA"
# model_cache_dir = '/data/rotenbergnh/llama_trials/meta-llama' # necessary to prevent using up storage in /home/rotenbergnh


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EPOCHS = 3


if entity_type == "cell_hetero":
    entity_name = "heterogeneous cell populations"
elif entity_type == "cell_phenotype":
    entity_name = "cell phenotypes (specific cell types and their states)"
elif entity_type == "cell_vague":
    entity_name = "vague cell populations"

# test_path = os.path.join(LLM_json_data_dirpath, f"{dataset}_{entity_type}_only_LLM.json")
##############***** change to test set
test_path = os.path.join(LLM_json_data_dirpath, f"val_{entity_type}_only_LLM.json")

df_test  = pd.read_json(test_path)


#### Delete
# get tokenizer
model_checkpoint_path = os.path.join(model_dirpath, f'checkpoint-{376}')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, cache_dir=model_cache_dir)
PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
# tokenizer.padding_side = "right" # right was working better

def format_test_example(row: dict):
    system_instructions = "You are an expert biomedical researcher who always answers questions accurately."
    prompt_instructions = f"Extract {entity_name} from the biomedical text that follows; " + \
                          "Include genes and all other modifiers in the name.\n"
    
    prompt = prompt_instructions + "Here is the instance to process:\n" + row['passage']
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
        # {"role": "assistant", "content": json.dumps(row['output'], indent=3)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# format the training examples into a new text column


df_test['text'] = df_test.apply(format_test_example, axis=1)

for checkpoint_num in ['376', '564']:
    # load model
    model_checkpoint_path = os.path.join(model_dirpath, f'checkpoint-{checkpoint_num}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, cache_dir=model_cache_dir)
    
    PAD_TOKEN = "<|pad|>"

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    # tokenizer.padding_side = "right" # right was working better
    
    # here we are loading the raw model, if you can't load it on your GPU, you can just change device_map to cpu
    # we won't need gpu here anyway
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='auto',
        cache_dir=model_cache_dir
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "mergedModel-step" + checkpoint_num))
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(base_model, model_checkpoint_path) #, cache_dir=model_cache_dir)
    model = model.merge_and_unload()
    # new_model.save_pretrained(os.path.join(OUTPUT_DIR, "mergedModel-step" + checkpoint_num))
    
    
    
    ### inference
    
    # MODEL_NAME = "/data/rotenbergnh/llama_finetuning/experiment0/checkpoint-75"
    # MODEL_NAME = os.path.join(OUTPUT_DIR, "mergedModel-step" + checkpoint_num)
    # output_filepath = '/data/ScheuermannGroup/noam/output.json'
    output_filepath = os.path.join(model_dirpath, f"output-step{checkpoint_num}.json")
    # processed_dataset_path = "/data/rotenbergnh/llama_finetuning/2024-10-30_finetuning_LLAMA/2024-12-19_processed_dataset/"
    # df_test = pd.read_json(f"{processed_dataset_path}test.json", lines=True)
    # df_val = pd.read_json(os.path.join(base_data_path, "val_processed_LLAMA.jsonl"), orient='records', lines=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load trained model
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME, 
    #     quantization_config=quantization_config,
    #     device_map="auto",
    #     cache_dir=model_cache_dir
    # )
    model.eval()
    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        return_full_text=False
    )
    
    def is_valid_json_list(string):
        try:
            my_list = json.loads(string)
            assert(type(my_list) == list)
            assert(all(type(item) == str for item in my_list))
            return True
        except:
            return False
    
    def extract_response(output):
        return output[0]['generated_text'][11:]
    
    questions = df_test['text'].tolist()
    outputs = pipe(questions, batch_size=32, do_sample = False) ## no sampling --> temp = 0
    
    recursive_depths = []
    
    # for question, output in zip(questions, outputs):
    def process_output(output, question, pipe, recursive_depth = 0):
        output = extract_response(output)
        if is_valid_json_list(output):
            recursive_depths.append(recursive_depth)
            return json.loads(output)
        elif recursive_depth < 4:
            new_output = pipe([question], do_sample=True, temperature=0.7, top_p=0.9)[0]
            return process_output(new_output, question, pipe, recursive_depth + 1)
        else:
            # try 5 times (including original output) max
            recursive_depths.append(-1)
            return []
    
    processed_outputs = [process_output(output, question, pipe) for question, output in zip(questions, outputs)]
    
    with open(output_filepath, 'w') as writefp:
        json.dump(processed_outputs, writefp, indent=3)
    
    print("Wrote", output_filepath, f"(checkpoint {checkpoint_num})")
    recursive_depths = np.asarray(recursive_depths)
    print(np.sum(recursive_depths != 0), f"({int(round(np.mean(recursive_depths != 0)*100))}%) outputs not properly formatted on first try")
    print(np.sum(recursive_depths == -1), f"({int(round(np.mean(recursive_depths == -1)*100))}%) outputs forced to empty list after 5 chances.")
    