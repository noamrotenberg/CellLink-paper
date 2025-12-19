# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:45:13 2024

@author: rotenbergnh
"""


"""
sinteractive --cpus-per-task=8 --mem=20g --gres=gpu:a100 # or v100x
source /data/rotenbergnh/llama_trials/llama_test/bin/activate
"""

# script developed with help from this tutorial: https://kickitlikeshika.github.io/2024/07/24/how-to-fine-tune-llama-3-models-with-LoRA.html

import random
import numpy as np
import torch
import pandas as pd
import json
import datasets
import os
import sys


if len(sys.argv) != 5:
    raise Exception("Usage: python LLAMA_finetuning.py <LLM_json_data_dirpath> <entity_type> <model_dirpath> <model_cache_dir>")
    
LLM_json_data_dirpath = sys.argv[1]
entity_type = sys.argv[2]
model_dirpath = sys.argv[3]
model_cache_dir = sys.argv[4]
# model_cache_dir = '/data/rotenbergnh/llama_trials/meta-llama' # necessary to prevent using up storage in /home/rotenbergnh


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EPOCHS = 3


if entity_type == "cell_hetero":
    entity_name = "heterogeneous cell populations"
    max_seq_length = 700
elif entity_type == "cell_phenotype":
    entity_name = "cell phenotypes (specific cell types and their states)"
    max_seq_length = 900
elif entity_type == "cell_desc":
    entity_name = "vague cell populations"
    max_seq_length = 700


# base_data_path = f"/data/rotenbergnh/llama_finetuning/2025-06-20_LLAMA_NER_for_paper/2025-06-18_Robert_splitv2_{entity_type.replace('_','-')}_only/"
# train_path, val_path, test_path = [dataset.join(data_path_split) for dataset in ["train", "devel", "test"]]
train_path, val_path = [os.path.join(LLM_json_data_dirpath, f"{dataset}_{entity_type}_only_LLM.json") for dataset in ["train", "val"]]
# processed_dataset_path = "/data/rotenbergnh/llama_finetuning/2024-10-30_finetuning_LLAMA/2024-12-19_processed_dataset/"
# OUTPUT_DIR = os.path.join(base_data_path, "model")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

"""
2. Load and Quantize Model

The 8B model is still quite big to fit on average Colab GPUs (e.g T4), so It’s 
recommended to quantize the model to a lower precision rate before starting training.

Here’s how we can load and quantize the model using BitsAndBytes to 8-bit

Note: This will reduce GPU utilization from 18GB to approximately 6GB.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=model_cache_dir) #, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config,
    device_map="auto",
    cache_dir=model_cache_dir
)

"""
3. Add Padding Token

Llama 3 tokenizers do not have a padding token by default, so, to train the model 
in batches, we will need to configure this ourselves, and it has also proven to 
show better results even when training with a batch size of one sample.
"""

PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
# tokenizer.padding_side = "right" # right was working better

# we added a new padding token to the tokenizer, we have to extend the embeddings
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(tokenizer.pad_token, tokenizer.pad_token_id)
# output: ('<|pad|>', 128256)


# 4. Format Training Examples

df_train = pd.read_json(train_path)
df_val   = pd.read_json(val_path)

def format_example(row: dict):
    system_instructions = "You are an expert biomedical researcher who always answers questions accurately."
    prompt_instructions = f"Extract {entity_name} from the biomedical text that follows; " + \
                          "Include genes and all other modifiers in the name.\n"
    
    prompt = prompt_instructions + "Here is the instance to process:\n" + row['passage']
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(row['output'], indent=3)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# format the training examples into a new text column


for set_name, df in [("train", df_train), ("val", df_val)]:
    df['text'] = df.apply(format_example, axis=1)
    # df.to_json(os.path.join(base_data_path, set_name + "_processed_LLAMA.jsonl"), orient='records', lines=True)


# dataset = datasets.load_dataset(
#     "json",
#     data_files={set_name: f"{processed_dataset_path}{set_name}.json" for set_name in ('train', 'val', 'test')}
# )
dataset = {set_name: datasets.Dataset.from_pandas(df) for set_name, df in [("train", df_train), ("val", df_val)]}




from trl import DataCollatorForCompletionOnlyLM

# in order to only evaluate the generation of the model, we shouldn't consider the text that were already inputed, we will use the end header id token to get the generated text only, and mask everything else
response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# 6) LORA configurations

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

# this is recommended by original lora paper: using lora, we should target the linear layers only
lora_config = LoraConfig(
    r=8,  # rank for matrix decomposition; could be lower than 32 (original)
    lora_alpha=16, # could be lower
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(model.print_trainable_parameters())


# 7) training configurations
from trl import SFTConfig, SFTTrainer


sft_config = SFTConfig(
    output_dir=model_dirpath,
    dataset_text_field='text',  # this is the name of the key of the input
    max_seq_length=max_seq_length, # could be lower, but this worked best # used to be 4096
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,  # training batch size
    per_device_eval_batch_size=16,  # eval batch size
    gradient_accumulation_steps=2,  # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps = 4 * 2 = 8 steps
    optim="paged_adamw_8bit",  # paged adamw
    eval_strategy='steps',
    eval_steps=0.2,
    save_strategy='epoch',
    logging_steps=5, # save train metrics every 5 steps
    learning_rate=1e-5, # used to be 1e-4
    fp16=True,  # also try bf16=True
    warmup_ratio=0.2 / NUM_EPOCHS,  # used to be 0.1
    save_total_limit=3,
    lr_scheduler_type="cosine",  # scheduler
    save_safetensors=True,  # saving to safetensors
    dataset_kwargs={
        "add_special_tokens": False,  # we template with special tokens already
        "append_concat_token": False,  # no need to add additional sep token
    },
    seed=0
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    # tokenizer=tokenizer,
    data_collator=collator,
)

### train
trainer.train()


# then, perform inference
    