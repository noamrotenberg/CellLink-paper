# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 05:57:21 2025

@author: rotenbergnh
"""

import json
import bioc
import re
import string
import sys
import copy


def overlap(ann1, ann2):
    ann1_offset = ann1.locations[0].offset
    ann1_end = ann1.locations[0].end
    ann2_offset = ann2.locations[0].offset
    ann2_end = ann2.locations[0].end
    if (ann1_offset <= ann2_offset) and (ann2_offset <= ann1_end):
        return True
    elif (ann2_offset <= ann1_offset) and (ann1_offset <= ann2_end):
        return True
    else:
        return False


def iterate_over_passages(bioc_collection):
    for doc in bioc_collection.documents:
        for p in doc.passages:
            yield p


def find_phrase(text, offset, phrase):
    # ChatGPT offered this complicated regex: ** need to check
    # escape any regex‐metacharacters in your phrase
    esc = re.escape(phrase)
    
    # build a character‐class for _any_ punctuation:
    punct = re.escape(string.punctuation)   # e.g. !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    
    # lookaround that asserts: left‐side is start, whitespace, or punctuation
    left  = r'(?:(?<=^)|(?<=\s)|(?<=[' + punct + r']))'
    # lookaround that asserts: right‐side is end,   whitespace, or punctuation
    right = r'(?:(?=$)|(?=\s)|(?=['  + punct + r']))'
    
    pattern = left + esc + right
    regex   = re.compile(pattern)
    
    # return all (non‐overlapping) match‐objects
    return list(regex.finditer(text, offset))



# I started making the above into a function... but it's not ready yet -Noam
def predictions_to_biocAnnotations(passage, ann_texts, entity_types):
    bioc_annotations = []
    for ann_text, entity_type in zip(ann_texts, entity_types):
        matches = find_phrase(passage.text, 0, ann_text)
        if len(matches) == 0:
            # try case-insensitive match:
            no_case_matches = find_phrase(passage.text.lower(), 0, ann_text.lower())
            if len(no_case_matches) > 0:
                matches = no_case_matches
                
        if len(matches) > 0:
            for match in matches:
                new_ann = bioc.BioCAnnotation()
                new_ann.text = passage.text[match.start():match.end()]
                new_ann.add_location(bioc.BioCLocation(match.start() + passage.offset, match.end() - match.start() + passage.offset))
                new_ann.infons['type'] = entity_type
                bioc_annotations.append(new_ann)
    
    # review all annotations and remove overlap
    sorted_annotations = sorted(bioc_annotations, key=lambda ann: len(ann.text), reverse=True)
    # print(len(bioc_annotations))
    output_annotations = []
    for ann in sorted_annotations:
        # other_annotations = copy.deepcopy(bioc_annotations)
        # other_annotations.remove(ann)
        if not any([overlap(ann, ann_i) for ann_i in output_annotations if ann_i != ann]):
            output_annotations.append(ann)
    
    return output_annotations

if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception("Usage: python LLM_output_processing.py <LLM_json_output_path> <original_BioCXML_path> <entity_type> <output_xml_path>")
    # process the "train", "val", and "test" files in the input_directory to output_directory
    
    LLM_json_output_path = sys.argv[1]
    original_BioCXML_path = sys.argv[2]
    entity_type = sys.argv[3]
    output_xml_path = sys.argv[4]
    
    with open(LLM_json_output_path) as readfp:
        llama_json = json.load(readfp)
    
    
    with open(original_BioCXML_path, 'r', encoding='utf-8') as readfp:
        bioc_collection = bioc.load(readfp)
    
    # clear annotations
    for p in iterate_over_passages(bioc_collection):
        p.annotations = []

    # add llama annotations to bioc collection
    for p, annotations_i in zip(iterate_over_passages(bioc_collection), llama_json):
        p.annotations = predictions_to_biocAnnotations(p, annotations_i, [entity_type]*len(annotations_i))
    
    # save file
    with open(output_xml_path, 'w', encoding='utf-8') as writefp:
        bioc.dump(bioc_collection, writefp)

