# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:58:39 2025

@author: rotenbergnh
"""

import os
import sys
import bioc
import copy

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Usage: python BioCXML_to_LLMjson.py <input_directory> <output_directory>")
    # process the "train", "val", and "test" files in the input_directory to output_directory
    
    input_dirpath = sys.argv[1]
    output_dirpath = sys.argv[2]
    
    for split in ["train", "val", "test"]:
        filepath = os.path.join(input_dirpath, split + '.xml')
        with open(filepath, 'r', encoding='utf-8') as readfp:
            input_collection = bioc.load(readfp)
        
        for entity_type in ["cell_phenotype", "cell_hetero", "cell_desc", "pheno_hetero_merged"]:
            output_collection = copy.deepcopy(input_collection)
            for doc in output_collection.documents:
                for p in doc.passages:
                    p.annotations = [ann for ann in p.annotations if ann.infons['type'] == entity_type]
            
            if entity_type == "pheno_hetero_merged":
                output_filepath = os.path.join(output_dirpath, f"{split}_{entity_type}.xml")
            else:
                output_filepath = os.path.join(output_dirpath, f"{split}_{entity_type}_only.xml")
            
            with open(output_filepath, 'w', encoding='utf-8') as writefp:
                bioc.dump(output_collection, writefp)