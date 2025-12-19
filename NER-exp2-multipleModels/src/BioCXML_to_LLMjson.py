# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:35:15 2024

@author: rotenbergnh
"""


import os
import bioc
import json
import sys


def convert_file(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as readfile:
        collection = bioc.load(readfile)
    
    dataset = []
    for doc in collection.documents:
        for passage in doc.passages:
            sorted_passage_annotations = list(sorted(passage.annotations, key=lambda ann: ann.locations[0].offset))
            dataset.append({"passage": passage.text, "output": [ann.text for ann in sorted_passage_annotations]})
    
    with open(output_filepath, 'w', encoding='utf-8') as writefile:
        json.dump(dataset, writefile, indent=3)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Usage: python BioCXML_to_LLMjson.py <input_directory> <output_directory>")
    # accept an input directory with files and an output directory with files
    # this script doesn't convert files within other directories in the input directory
    
    input_dirpath = sys.argv[1]
    output_dirpath = sys.argv[2]
    
    
    for filename in os.listdir(input_dirpath):
        if filename.endswith('.xml'):
            output_filename = filename.rsplit('.', maxsplit=1)[0] + "_LLM.json"
            convert_file(os.path.join(input_dirpath, filename),
                         os.path.join(output_dirpath, output_filename))