"""
Unfortunately, this script won't be runnable directly, because we do not plan to release
annotator-specific data, which is necessary in order to calculate inter-annotator agreement.
This script is very similar (but not identical) to evaluate.py for LLM evaluation.
Please feel free to contact us if you have any questions.
"""


# -*- coding: utf-8 -*-
import hashlib
import argparse
import codecs
import collections
import datetime
import logging
import os
import xml.etree.ElementTree as ElementTree
import gzip
import numpy as np
import pandas as pd
import bioc
import copy


root = "MESH:ROOT"
log_format = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"

# Returns precision, recall & f-score for the specified reference and prediction files

log = logging.getLogger(__name__)

evaluation_config = collections.namedtuple("evaluation_config", ("annotation_type", "evaluation_type"))
evaluation_count = collections.namedtuple("evaluation_count", ("tp", "fp", "fn"))
evaluation_result = collections.namedtuple("evaluation_result", ("precision", "recall", "f_score"))
span_annotation = collections.namedtuple("span_annotation", ("passage_id", "type", "locations", "text"))
identifier_annotation = collections.namedtuple("identifier_annotation", ("passage_id", "type", "identifier"))
span_identifier_annotation = collections.namedtuple("span_identifier_annotation", ("passage_id", "type", "locations", "text", "identifier"))
annotation_location = collections.namedtuple("annotation_location", ("offset", "length"))

def read_doc_ids(filename):
    doc_ids = set()
    if filename.endswith(".gz"):
        file = gzip.open(filename, 'rt', encoding="utf-8") 
    else:
        file = codecs.open(filename, 'r', encoding="utf-8") 
    for line in file:
        line = line.strip()
        if len(line) > 0:
            doc_ids.add(line) 
    file.close()
    return doc_ids

def get_annotations_from_XML(input_collection, input_filename, eval_config):
    unique_annotators = {element.text for element in input_collection.findall(".//infon[@key='annotator']")}
    annotation_set_dict = {annotator:set() for annotator in unique_annotators}
    passage_text_dict = collections.defaultdict(dict)
    for document in input_collection.findall(".//document"):
        pseudodoc_id = document.find(".//id").text
        for passage in document.findall(".//passage"):
    # for passage in input_collection.findall(".//passage"):
            passage_id = passage.findtext("infon[@key='passage_id']")
            passage_offset = int(passage.find(".//offset").text)
            if passage.find(".//text") is None:
                continue
            passage_text = passage.find(".//text").text
            passage_end = passage_offset + len(passage_text)
            passage_text_dict[pseudodoc_id][passage_offset] = passage_text
            if passage.find(".//annotation") is not None:
                PMID = passage.find(".//infon[@key='article-id_pmid']").text
            for annotation in passage.findall(".//annotation"):
                annotation_id = annotation.attrib["id"]
                annotators = annotation.findall(".//infon[@key='annotator']")
                if len(annotators) > 1: raise Exception("Multiple annotators found for 1 annotation") # is this even possible? if not, delete
                if len(annotators) == 0:
                    continue ### *** address pre-annotation
                annotator = annotators[0].text #*** we will need to deal with pre-annotations - they won't be assigned to a specific annotator I think, but their approval is annotator-specific
                
                if eval_config.annotation_type == 'pool':
                    type = 'pool'
                else:
                    type = annotation.find(".//infon[@key='type']").text
                
                
                if not eval_config.annotation_type is None and type != eval_config.annotation_type:
                    continue
                if eval_config.evaluation_type == "span":
                    locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
                    if sum(location.length for location in locations) == 0:
                        log.warning("Ignoring zero-length annotation: pseudodoc ID = {}, annotation ID = {}".format(pseudodoc_id, annotation_id))
                        continue
                    if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                        log.warning("Ignoring annotation with span outside of passage: pseudodoc ID = {}, annotation ID = {}".format(pseudodoc_id, annotation_id))
                        continue
                    locations.sort()
                    annotation_text = annotation.find(".//text").text
                    location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                    annotation = span_annotation(passage_id, type, tuple(locations), annotation_text)
                    if annotation_text != location_text:
                        log.error("Annotation text {} does not match text at location(s) {}: pseudodoc ID = {}, annotation ID = {}".format(annotation_text, location_text, pseudodoc_id, annotation_id))
                    annotation_set_dict[annotator].add(annotation)
                elif eval_config.evaluation_type == "identifier":
                    identifier_node = annotation.find(".//infon[@key='identifier']")
                    if identifier_node is None or identifier_node.text is None:
                        annotation = identifier_annotation(passage_id, type, "placeholder")
                        #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                        annotation_set_dict[annotator].add(annotation)
                    else:
                        for identifier in identifier_node.text.split(","):
                            if "CL:" not in identifier:
                                identifier = "placeholder"
                            annotation = identifier_annotation(passage_id, type, identifier)
                            #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                            annotation_set_dict[annotator].add(annotation)
                elif eval_config.evaluation_type == "span_identifier":
                    locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
                    if sum(location.length for location in locations) == 0:
                        log.warning("Ignoring zero-length annotation: pseudodoc ID = {}, annotation ID = {}".format(pseudodoc_id, annotation_id))
                        continue
                    if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                        log.warning("Ignoring annotation with span outside of passage: pseudodoc ID = {}, annotation ID = {}".format(pseudodoc_id, annotation_id))
                        continue
                    locations.sort()
                    annotation_text = annotation.find(".//text").text
                    identifier_node = annotation.find(".//infon[@key='identifier']")
                    location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                    if annotation_text != location_text:
                        log.error("Annotation text {} does not match text at location(s) {}: pseudodoc ID = {}, annotation ID = {}".format(annotation_text, location_text, pseudodoc_id, annotation_id))
                    if identifier_node is None or identifier_node.text is None:
                        # log.warning("Ignoring annotation {} with no identifier: pseudodoc ID = {}, annotation ID = {}".format(annotation_text, pseudodoc_id, annotation_id))
                        # continue
                        annotation = span_identifier_annotation(passage_id, "", tuple(locations), annotation_text, "placeholder")
                        #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                        annotation_set_dict[annotator].add(annotation)
                    else:
                        for identifier in identifier_node.text.split(","):
                            if "CL:" not in identifier:
                                identifier = "placeholder"
                            annotation = span_identifier_annotation(passage_id, type, tuple(locations), annotation_text, identifier)
                            #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                            annotation_set_dict[annotator].add(annotation)
    # print("file {} returning {} annotations".format(input_filename, len(annotation_set)))
    return annotation_set_dict, passage_text_dict


def get_annotations_from_TSV(input_filename, eval_config):
    raise ValueError("Not implemented")

def get_annotations_from_file(input_filename, eval_config):
    try:
        if input_filename.endswith(".xml"):
            log.info("Reading XML file {}".format(input_filename))
            parser = ElementTree.XMLParser(encoding="utf-8")
            input_collection = ElementTree.parse(input_filename, parser=parser).getroot()
            return get_annotations_from_XML(input_collection, input_filename, eval_config)
        if input_filename.endswith(".tsv"):
            log.info("Reading TSV file {}".format(input_filename))
            return get_annotations_from_TSV(input_filename, eval_config)
        log.info("Ignoring file {}".format(input_filename))
        return set(), dict()
    except Exception as e:
        raise RuntimeError("Error while processing file {}".format(input_filename)) from e
    

def get_annotations_from_path(input_path, eval_config):
    annotation_set_dict = dict()
    # annotation_set = set()
    passage_text_dict = collections.defaultdict(set)
    if os.path.isdir(input_path):
        log.info("Processing directory {}".format(input_path))
        dir = os.listdir(input_path)
        for item in dir:
            input_filename = input_path + "/" + item
            if os.path.isfile(input_filename):
                new_annotation_set_dict, passage_text_dict2 = get_annotations_from_file(input_filename, eval_config)
                for annotator, new_annotations in new_annotation_set_dict.items():
                    if annotator in annotation_set_dict:
                        annotation_set_dict[annotator].update(new_annotations)
                    else:
                        annotation_set_dict[annotator] = new_annotations
                # annotation_set.update(annotation_set2)
                passage_text_dict.update(passage_text_dict2)
    elif os.path.isfile(input_path):
        annotation_set_dict, passage_text_dict = get_annotations_from_file(input_path, eval_config)
        # annotation_set2, passage_text_dict2 = get_annotations_from_file(input_path, eval_config)
        # annotation_set.update(annotation_set2)
        # passage_text_dict.update(passage_text_dict2)
    else:  
        raise RuntimeError("Path is not a directory or normal file: {}".format(input_path))
    return annotation_set_dict, passage_text_dict

def calculate_evaluation_count(reference_annotations, predicted_annotations):
    reference_annotations = set(reference_annotations)
    predicted_annotations = set(predicted_annotations)
    annotations = set()
    annotations.update(reference_annotations)
    annotations.update(predicted_annotations)
    annotations = list(annotations)
    try:
        annotations.sort()
    except:
        print(annotations)
        raise Exception()
    results = collections.Counter()
    for a in annotations:
        r = a in reference_annotations
        p = a in predicted_annotations
        results[(r, p)] += 1
        log.debug("annotation = {} in reference = {} in predicted = {}".format(str(a), r, p))
    log.debug("Raw results = {}".format(str(results)))
    return evaluation_count(results[(True, True)], results[(False, True)], results[(True, False)])

def calculate_evaluation_result(eval_count):
    if eval_count.tp == 0:
        return evaluation_result(0.0, 0.0, 0.0)
    p = eval_count.tp / (eval_count.tp + eval_count.fp)
    r = eval_count.tp / (eval_count.tp + eval_count.fn)
    f = 2.0 * p * r / (p + r)
    return evaluation_result(p, r, f)

def do_strict_eval(reference_annotations, predicted_annotations):
    eval_count = calculate_evaluation_count(reference_annotations, predicted_annotations)
    log.info("TP = {0}, FP = {1}, FN = {2}".format(eval_count.tp, eval_count.fp, eval_count.fn))
    eval_result = calculate_evaluation_result(eval_count)
    return eval_result

def get_locations(annotations):
    locations = collections.defaultdict(list)
    for annotation in annotations:
        locations[annotation.passage_id].append({(annotation.type, offset, offset + length) for offset, length in annotation.locations})
    return locations



def do_approx_span_eval(reference_annotations, predicted_annotations, pool=False):
    tp1, fn = 0, 0
    predicted_locations = get_locations(predicted_annotations,)
    for annotation in reference_annotations:
        predicted_locations2 = predicted_locations[annotation.passage_id]
        found = False
        for location in annotation.locations:
            if pool:
                found |= any([location.offset < end2 and start2 < location.offset + location.length for type, start2, end2 in predicted_locations2])
            else:
                found |= any([(location.offset < end2 and start2 < location.offset + location.length) and annotation.type == type \
                              for type, start2, end2 in predicted_locations2])
        if found:
            tp1 += 1
        else:
            fn += 1
    log.info("REFERENCE: TP = {0}, FN = {1}".format(tp1, fn))
        
    tp2, fp = 0, 0
    reference_locations = get_locations(reference_annotations)
    for annotation in predicted_annotations:
        reference_locations2 = reference_locations[annotation.passage_id]
        found = False
        for location in annotation.locations:
            if pool:
                found |= any([location.offset < end2 and start2 < location.offset + location.length for type, start2, end2 in reference_locations2])
            else:
                found |= any([(location.offset < end2 and start2 < location.offset + location.length) and annotation.type == type for type, start2, end2 in reference_locations2])
        if found:
            tp2 += 1
        else:
            fp += 1
    log.info("PREDICTED: TP = {0}, FP = {1}".format(tp2, fp))

    if tp1 + tp2 == 0:
        return evaluation_result(0.0, 0.0, 0.0)
    p = tp2 / (tp2 + fp)
    r = tp1 / (tp1 + fn)
    f = 2.0 * p * r / (p + r)
    return evaluation_result(p, r, f)


def do_approx_macro_avg_span_eval(annotator1_annotations, annotator2_annotations):
    # this function assumes that there aren't any duplicate overlaps
    # (i.e., ref annotations don't overlap with themselves and pred annotations don't overlap with themselves)
    
    overlap = 0.5*(approx_macro_avg_span_helper(annotator1_annotations, annotator2_annotations) + \
                   approx_macro_avg_span_helper(annotator2_annotations, annotator1_annotations))

    return evaluation_result(0,0,overlap)


def approx_macro_avg_span_helper(reference_annotations, predicted_annotations):
    if len(reference_annotations) == 0:
        return 0
    overlap1 = 0
    for entity_type in set({ann.type for ann in reference_annotations}):
        if entity_type is None: raise Exception("not implemented")
        ref_annotations_i = [ann for ann in reference_annotations if ann.type == entity_type]
        predicted_locations = get_locations([ann for ann in predicted_annotations if ann.type == entity_type],)
        for ref_annotation in ref_annotations_i:
            predicted_locations2 = predicted_locations[ref_annotation.passage_id]
            for ref_location in ref_annotation.locations:
                ref_start, ref_end = ref_location.offset, ref_location.offset + ref_location.length
                for pred_start, pred_end in predicted_locations2:
                    if ref_start < pred_end and pred_start < ref_end:
                        text_overlap = ref_annotation.text[max(ref_start, pred_start) - ref_start : 
                                                           min(ref_end, pred_end) - ref_start]
                        overlap1 += len(text_overlap.strip().split()) / len(ref_annotation.text.strip().split())

    overlap1 /= len(reference_annotations)
    # print(overlap_list)
    return overlap1


def get_docid2identifiers(annotations):
    docid2identifiers = collections.defaultdict(set)
    for docid, type, identifier in annotations:
        if annotation_type == type:
            docid2identifiers[docid].add(identifier)
    return docid2identifiers

def do_approx_identifier_eval(lca_hierarchy, reference_annotations, predicted_annotations):
    reference_docid2identifiers = get_docid2identifiers(reference_annotations)
    predicted_docid2identifiers = get_docid2identifiers(predicted_annotations)
    docids = set(reference_docid2identifiers.keys())
    docids.update(predicted_docid2identifiers.keys())
    
    precision = list()
    recall = list()
    f_score = list()
    for docid in docids:
        log.info("Evaluating document {}".format(docid))
        reference_identifiers = reference_docid2identifiers[docid]
        predicted_identifiers = predicted_docid2identifiers[docid]
        reference_augmented, predicted_augmented = lca_hierarchy.get_augmented_sets(reference_identifiers, predicted_identifiers)
        log.info("{}: len(reference_identifiers) = {} len(reference_augmented) = {}".format(docid, len(reference_identifiers), len(reference_augmented)))
        log.info("{}: len(predicted_identifiers) = {} len(predicted_augmented) = {}".format(docid, len(predicted_identifiers), len(predicted_augmented)))
        eval_count = calculate_evaluation_count(reference_augmented, predicted_augmented)
        log.info("{}: TP = {}, FP = {}, FN = {}".format(docid, eval_count.tp, eval_count.fp, eval_count.fn))
        eval_result = calculate_evaluation_result(eval_count)
        log.info("{}: P = {:.4f}, R = {:.4f}, F = {:.4f}".format(docid, eval_result.precision, eval_result.recall, eval_result.f_score))
        precision.append(eval_result.precision)
        recall.append(eval_result.recall)
        f_score.append(eval_result.f_score)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)
    avg_f_score = sum(f_score) / len(f_score)
    return evaluation_result(avg_precision, avg_recall, avg_f_score)


def filter_annotations(annotations, docid_set):
    annotations2 = set()
    for annotation in annotations:
        if annotation[0] in docid_set:
            annotations2.add(annotation)
    return annotations2

def filter_passages(passage_text_dict, docid_set):
    passage_text_dict2 = dict()
    for document_id, document_dict in passage_text_dict.items():
        if document_id in docid_set:
            passage_text_dict2[document_id] = document_dict
    return passage_text_dict2

def get_md5sum_from_path(input_path):
    file_hash = hashlib.md5()
    if os.path.isdir(input_path):
        log.info("Processing directory {}".format(input_path))
        dir = os.listdir(input_path)
        dir.sort()
        for item in dir:
            input_filename = input_path + "/" + item
            if not os.path.isfile(input_filename):
                continue
            with open(input_filename, "rb") as f:
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
    elif os.path.isfile(input_path):
        with open(input_path, "rb") as f:
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
    else:  
        raise RuntimeError("Path is not a directory or normal file: {}".format(input_path))
    return file_hash.hexdigest()

def run_docids(docids_file, reference_annotations, reference_passages):
    print("docids_file = {}".format(docids_file))
    docid_set = read_doc_ids(docids_file)
    log.info("Read {} document IDs from file {}".format(len(docid_set), docids_file))
    reference_annotations = filter_annotations(reference_annotations, docid_set)
    reference_passages = filter_passages(reference_passages, docid_set)
    # predicted_annotations = filter_annotations(predicted_annotations, docid_set)
    # predicted_passages = filter_passages(predicted_passages, docid_set)
    return reference_annotations, reference_passages

def num_entities(path):
    with open(path, 'r', encoding='utf-8') as readfp:
        collection = bioc.load(readfp)
    num_phenotype = 0
    num_hetero = 0
    num_desc = 0
    num_IDs = 0
    
    for doc in collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                if ann.infons['type'] == "cell_phenotype" or ann.infons['type'] == "cell_type":
                    num_phenotype += 1
                elif ann.infons['type'] == "cell_hetero":
                    num_hetero += 1
                elif ann.infons['type'] == "cell_desc":
                    num_desc += 1
                
                if ann.infons.get('identifier', "") is not None and \
                    len(ann.infons.get('identifier', '')) > 0:
                    num_IDs += 1
    # return str((num_phenotype, num_hetero, num_desc, num_IDs))
    return (num_phenotype, num_hetero, num_desc, num_IDs)


def run_a_metric(round1_filename, evaluation=('strict', 'span_identifier', None), parents_filename = None):
    # must be done 1 file at a time
    
    evaluation_method, evaluation_type, annotation_type = evaluation
    
    eval_config = evaluation_config(annotation_type, evaluation_type)
    reference_annotations, reference_passages = get_annotations_from_path(round1_filename, eval_config)
    unique_annotators = list(reference_annotations.keys())
    for false_annotator in ['NR', 'AAA@gmail.com,BBB@gmail.com', 'CCC@gmail.com,DDD@gmail.com']: # redacted false annotators (fixing rare TeamTat glitches)
        if false_annotator in unique_annotators:
            unique_annotators.remove(false_annotator)
    print(round1_filename, unique_annotators)
    assert(len(unique_annotators) == 2), "num annotators should be 2"

    # results_df = pd.DataFrame(index=[e[0] + ',' + e[1] + ',' + str(e[2]) for e in evaluations])
    # annotator_combinations = [annotator1 + " & " + annotator2 for i, annotator1 in enumerate(unique_annotators[:-1]) for annotator2 in unique_annotators[i+1:]]
    # results_df = pd.DataFrame(index = annotator_combinations)
    
    # results_df["total num pheno, hetero, desc; IDs"] = num_entities(round1_filename)
    

        
    annotator1, annotator2 = unique_annotators
    eval_result = run_a_metric_main(reference_annotations[annotator1], reference_annotations[annotator2], 
                                    evaluation_method, evaluation_type, annotation_type, parents_filename)

    print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))
    # print("EVAL_RESULTS\t{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}".format(args.reference_path, get_md5sum_from_path(args.reference_path), 
    #                                                                         evaluation_type, evaluation_method, annotation_type, 
    #                                                                         eval_result.precision, eval_result.recall, eval_result.f_score))
    
    if type(eval_result) == float or type(eval_result) == int:
        return eval_result
    else:
        return eval_result.f_score

def run_a_metric_main(annotations1, annotations2, evaluation_method, evaluation_type, annotation_type, parents_filename):
    
    if evaluation_method == "strict":
        eval_result = do_strict_eval(annotations1, annotations2)
    elif evaluation_method == "approx" and evaluation_type == "span":
        eval_result = do_approx_span_eval(annotations1, annotations2, annotation_type=='pool')
    # we did not use the below methods but it may be insteresting for others
    # elif evaluation_method == "approx_macro_avg" and evaluation_type == "span":
    #     if annotation_type == "pool":
    #         raise Exception("Does not make sense for this metric because annotations can overlap.")
    #     eval_result = do_approx_macro_avg_span_eval(annotations1, annotations2)
    # elif evaluation_method == "approx" and evaluation_type == "identifier":
    #     if parents_filename is None:
    #         raise RuntimeError("Approximate identifier evaluation requires a parents filename")
    #     lca_hierarchy = lca.lca_hierarchy(root)
    #     lca_hierarchy.load_parents(parents_filename)
    #     eval_result = do_approx_identifier_eval(lca_hierarchy, annotations1, annotations2) # note: I think this ignores annotation_type
    elif evaluation_method == "approx" and evaluation_type == "span_identifier":
        raise ValueError("Not implemented")
    else:
        raise ValueError("Unknown evaluation method: {}".format(evaluation_method))
    
    return eval_result


if __name__ == "__main__":
    
    start = datetime.datetime.now()
    args = argparse.ArgumentParser().parse_args()
    args.parents_filename = None
    args.logging_level = "INFO"
    args.docids_file = False # pseudodoc IDs
        
        
    logging.basicConfig(level=args.logging_level.upper(), format=log_format)
    
    if log.isEnabledFor(logging.DEBUG):
        for arg, value in sorted(vars(args).items()):
            log.info("Argument {0}: {1}".format(arg, value))


    ref_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2025-04-22_corpus_paper_prep\IAA_subgroups\total_results_by_annotator_pair_after_training"
    
    # multi_ref_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2025-04-22_corpus_paper_prep\IAA_subgroups\total_results_by_MeSH_by_pair_after_training"
    multi_results_dict = {}
    # for dirname in os.listdir(multi_ref_path): # if not multi_ref, then replace this line with: for dirname in [ref_path]
    multi_ref_path = ref_path
    for dirname in [""]:
        ref_path = os.path.join(multi_ref_path, dirname)
        results_dict = {}
        
        
        for file in os.listdir(ref_path):
            if not (file.endswith('.xml') or file.endswith('.json')):
                continue
            args.reference_path = os.path.join(ref_path, file)
        
            # this following line is just to get a list of the annotators...
            reference_annotations, reference_passages = get_annotations_from_path(args.reference_path, evaluation_config(None, 'span'))
            unique_annotators = list(reference_annotations.keys())
            for false_annotator in ['NR', 'AAA@gmail.com,BBB@gmail.com', 'CCC@gmail.com,DDD@gmail.com']: # redacted false annotators (fixing rare TeamTat glitches)
                if false_annotator in unique_annotators:
                    unique_annotators.remove(false_annotator)
            
            evaluations = [('strict', 'span_identifier', None)] # for IAA by passage type or by MeSH cluster, use only this line
            # evaluations = [('strict', 'span_identifier', 'cell_phenotype'), ('strict', 'span_identifier', 'cell_hetero'), ('strict', 'span_identifier', None), ('strict', 'span_identifier', 'pool'),
            #                 ('strict', 'span', 'cell_phenotype'), ('strict', 'span', 'cell_hetero'), ('strict', 'span', 'cell_desc'), ('strict', 'span', None), ('strict', 'span', 'pool'),
            #         ('approx', 'span', 'cell_phenotype'), ('approx', 'span', 'cell_hetero'), ('approx', 'span', 'cell_desc'), ('approx', 'span', None), ('approx', 'span', 'pool'),
            #         ('strict', 'identifier', 'cell_phenotype'), ('strict', 'identifier', 'cell_hetero'), ('strict', 'identifier', None), ('strict', 'identifier', 'pool')]
            results_df = pd.DataFrame(index=[e[0] + ',' + e[1] + ',' + str(e[2]) for e in evaluations])
            annotator_combinations = [annotator1 + " & " + annotator2 for i, annotator1 in enumerate(unique_annotators[:-1]) for annotator2 in unique_annotators[i+1:]]
            results_df = pd.DataFrame(index = annotator_combinations)
            
            results_df["num phenotype"], results_df["num hetero"], results_df["num desc"], results_df["num IDs"] = num_entities(args.reference_path)
            
            for evaluation_method, evaluation_type, annotation_type in evaluations:
                results_i = list()
                if evaluation_type == "span_identifier":
                    print()
                for annotator_combination in annotator_combinations:
                    annotator1, annotator2 = annotator_combination.split(' & ')
                    print("comparing", annotator1, "and", annotator2)
                
                    print("\n\n\n\n", evaluation_method, evaluation_type, annotation_type)
                    eval_config = evaluation_config(annotation_type, evaluation_type)
                    reference_annotations, reference_passages = get_annotations_from_path(args.reference_path, eval_config)
                    if args.docids_file:
                        reference_annotations, reference_passages = run_docids(args.docids_file, reference_annotations, reference_passages)
            
                    annotations1 = reference_annotations[annotator1]
                    annotations2 = reference_annotations[annotator2]
                    eval_result = run_a_metric_main(annotations1, annotations2, evaluation_method, evaluation_type, 
                                      annotation_type, args.parents_filename)
                    
                    
                    print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))
                    # print("EVAL_RESULTS\t{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}".format(args.reference_path, get_md5sum_from_path(args.reference_path), 
                    #                                                                         evaluation_type, evaluation_method, annotation_type, 
                    #                                                                         eval_result.precision, eval_result.recall, eval_result.f_score))
                    
                    if type(eval_result) == float or type(eval_result) == int:
                        results_i.append(eval_result)
                    else:
                        results_i.append(eval_result.f_score)
                results_df[evaluation_method + ',' + evaluation_type + ',' + str(annotation_type)] = results_i
            print("Elapsed time: {}".format(datetime.datetime.now() - start))
            results_dict[file] = results_df
        
        results = pd.concat(results_dict)
        
        average_results_df = copy.deepcopy(results_df)
        average_results_dict = {}
        for col_name in results:
            if ("cell_phenotype" in col_name) or ("cell_hetero" in col_name) or ("cell_desc" in col_name):
                weights = results["num " + col_name.split('_')[-1]]
                average_results_dict[col_name] = "{:.3f}".format(np.average(results[col_name], weights=weights)) + 'w'
            elif ("pool" in col_name) or ("None" in col_name):
                weights = results["num phenotype"] + results["num hetero"] + results["num desc"]
                average_results_dict[col_name] = "{:.3f}".format(np.average(results[col_name], weights=weights)) + 'w'
            else: # unweighted:
                average_results_dict[col_name] = np.average(results[col_name])
        
        for key in average_results_dict.keys():
            if 'w' not in str(average_results_dict[key]):
                average_results_dict[key] = "{:.3f}".format(average_results_dict[key]) + 'u'
        
        
        
        results.loc[('', 'un/weighted average (u,w)'),:] = average_results_dict
        multi_results_dict[dirname] = results
    collapsed_results = {key: val.loc["","strict,span_identifier,None"].loc["un/weighted average (u,w)"] for key, val in multi_results_dict.items()}