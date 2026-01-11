# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:37:20 2025

@author: rotenbergnh
"""

import openai
import os
import pydantic
from pydantic import BaseModel
from typing import List, Literal
import bioc
import LLM_output_processing
import sys
import json



class TextAnnotation(pydantic.BaseModel):
    """
    A text annotation is a span of text and entity type.
    Use `text` for the source text and `entity_type` for the entity type.
    """
    text: str
    entity_type: Literal["cell_phenotype", "cell_hetero", "cell_desc"]

class TextAnnotationResult(BaseModel):
    annotations: List[TextAnnotation]


def query_LLM(model_name, passage_text):
    system_instructions = """
    Extract all cell populations from the text. Return them as a list of annotations.
    There are 3 types of cell populations to extract: specific cell_phenotype (cell types
    and their states, such as "hepatocytes", "microglia", "activated fibroblast"),
    cell_hetero (heterogeneous cell populations, such as "kidney cells", "secreting cells",
    "cancer cells"), and cell_desc (vague cell population descriptions, such as 
    "Ly6a-expressing cells", "microglia-like cells", "CAR T cells", "neural subsets").
    """
    
    prompt = f"Here is the text to annotate:\n{passage_text}"
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
    ]
    
    
    response = client.chat.completions.parse(
                    model=model_name, messages=messages, temperature=0, max_completion_tokens=1000,
                    # response_format= {"type": "json_object"},
                    response_format=TextAnnotationResult
    )
    # json.loads(response.dict()['choices'][0]['message']['content'])['annotations'] # returns a dict
    return response.choices[0].message.parsed.annotations


if __name__ == '__main__':
    if len(sys.argv) not in [4, 5]:
        raise Exception("Usage: python BioCXML_to_LLMjson.py <test_xml_path> <output_xml_path> <cache_path> <model_name>")
    
    test_xml_path = sys.argv[1]
    output_xml_path = sys.argv[2]
    cache_path = sys.argv[3]
    model_name = sys.argv[4]
    

    # load test document
    
    with open(test_xml_path, 'r', encoding='utf-8') as readfp:
        bioc_collection = bioc.load(readfp)
    
    for doc in bioc_collection.documents:
        for p in doc.passages:
            p.clear_annotations()
    
    
    client = openai.AzureOpenAI(
      azure_endpoint = os.environ['endpoint'],
      api_key = os.environ['api_key'],
      api_version = "2024-08-01-preview"  
    )
    print("Created AzureOpenAI client.")
    
    if os.path.isfile(cache_path):
        with open(cache_path) as readfp:
            cache = json.load(readfp)
        print("Loaded cache.")
    else:
        cache = dict()
    
    num_documents = len(bioc_collection.documents)
    for i, doc in enumerate(bioc_collection.documents):
        for p in doc.passages:
            if p.text not in cache:
                predicted_annotations = query_LLM(model_name, p.text)
                # predicted_annotations = [TextAnnotation(text='chondrocytes', entity_type='cell_phenotype'), TextAnnotation(text='chondroprogenitors', entity_type='cell_phenotype'), TextAnnotation(text='MSCs', entity_type='cell_phenotype'), TextAnnotation(text='proliferating MSCs', entity_type='cell_phenotype')]
                ann_texts = [ann.text for ann in predicted_annotations]
                entity_types = [ann.entity_type for ann in predicted_annotations]
                cache[p.text] = {"ann_texts": ann_texts, "entity_types": entity_types}
                print(i, '/', num_documents)
                
                if (i % 5 == 0) or (i == num_documents - 1):
                    # save cache
                    with open(cache_path, 'w') as writefp:
                        json.dump(cache, writefp)
            
    for i, doc in enumerate(bioc_collection.documents):
        for p in doc.passages:
            ann_texts = cache[p.text]["ann_texts"]
            entity_types = cache[p.text]["entity_types"]
            p.annotations = LLM_output_processing.predictions_to_biocAnnotations(p, ann_texts, entity_types)
        
    # save file
    with open(output_xml_path, 'w', encoding='utf-8') as writefp:
        bioc.dump(bioc_collection, writefp)

