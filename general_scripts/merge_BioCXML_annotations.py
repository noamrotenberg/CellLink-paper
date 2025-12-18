# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 20:53:16 2025

@author: rotenbergnh
"""

import os
import sys
import bioc
import copy

# use this script to merge annotations across 
# BioC-XML collections that have the same exact passages
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Usage: python mergeBioCXML_annotations.py <input_filepaths> ... <output_filepath>")
    
    input_filepaths = sys.argv[1:-1]
    output_filepath = sys.argv[-1]
    
    with open(input_filepaths[0], 'r', encoding='utf-8') as readfp:
        output_collection = bioc.load(readfp)
    
    for input_filepath in input_filepaths[1:]:
        with open(input_filepath, 'r', encoding='utf-8') as readfp:
            collection_i = bioc.load(readfp)
        
        for doc1, doc2 in zip(output_collection.documents, collection_i.documents):
            for p1, p2 in zip(doc1.passages, doc2.passages):
                assert(p1.text == p2.text)
                p1.annotations += p2.annotations
                # do not filter out any equivalent annotations
    
    with open(output_filepath, 'w', encoding='utf-8') as writefp:
        bioc.dump(output_collection, writefp)