import argparse
import collections
import datetime
import logging
import os
import sys
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ElementTree


# partially-edited script from Dr. Leaman is found in "2024 06 27 from Dr Leaman BC7T2-evaluation_v3.zip"

root = "MESH:ROOT"
log_format = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"

# Returns precision, recall & f-score for the specified reference and prediction files

log = logging.getLogger(__name__)

evaluation_config = collections.namedtuple("evaluation_config", ("annotation_types", "evaluation_type"))
evaluation_count = collections.namedtuple("evaluation_count", ("tp", "fp", "fn"))
evaluation_result = collections.namedtuple("evaluation_result", ("precision", "recall", "f_score"))
span_annotation = collections.namedtuple("span_annotation", ("passage_id", "type", "locations", "text"))
identifier_annotation = collections.namedtuple("identifier_annotation", ("passage_id", "type", "identifier"))
annotation_location = collections.namedtuple("annotation_location", ("offset", "length"))

def get_annotations_from_XML(input_collection, input_filename, eval_config, NER_blacklist):
    annotation_set = set()
    passage_text_dict = collections.defaultdict(dict)
    # for document in input_collection.findall(".//document"):
    #     document_id = document.find(".//id").text
    #     for passage_idx, passage in enumerate(document.findall(".//passage")):
    for passage in input_collection.findall(".//passage"):
            passage_id = passage.find(".//passage_id").text
            passage_offset = int(passage.find(".//offset").text)
            if passage.find(".//text") is None:
                continue
            passage_text = passage.find(".//text").text
            passage_end = passage_offset + len(passage_text)
            passage_text_dict[passage_id][passage_offset] = passage_text
            for annotation in passage.findall(".//annotation"):
                annotation_id = annotation.attrib["id"]
                type = annotation.find(".//infon[@key='type']").text
                if (not eval_config.annotation_types is None) and not (type in eval_config.annotation_types): # I changed "and" to "or"... #########
                    continue
                if eval_config.evaluation_type == "span":
                    locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
                    if sum(location.length for location in locations) == 0:
                        log.warning("Ignoring zero-length annotation: document ID = {}, annotation ID = {}".format(passage_id, annotation_id))
                        continue
                    if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                        log.warning("Ignoring annotation with span outside of passage: document ID = {}, annotation ID = {}".format(passage_id, annotation_id))
                        continue
                    locations.sort()
                    annotation_text = annotation.find(".//text").text
                    location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                    annotation = span_annotation(passage_id, type, tuple(locations), annotation_text)
                    if annotation_text != location_text:
                        log.error("Annotation text {} does not match text at location(s) {}: document ID = {}, annotation ID = {}".format(annotation_text, location_text, passage_id, annotation_id))
                    if not (annotation_text.lower() in NER_blacklist):
                        annotation_set.add(annotation)
                if eval_config.evaluation_type == "identifier":
                    identifier_node = annotation.find(".//infon[@key='identifier']")
                    if identifier_node is None:
                        continue
#                     print(identifier_node.text)
                    if identifier_node.text is None:
#                         print("**an identifier is None!!**")
                        identifier_node.text = "None"
                    elif not ("CL:" in identifier_node.text):
                        raise Exception("non-CL ID:" + identifier_node.text)
                    for identifier in identifier_node.text.split(","):
                        # annotation = identifier_annotation(passage_id, type, identifier)
                        annotation = identifier_annotation(str(passage_id), type, identifier)
                        print("using passage idx!")
                        #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                        annotation_set.add(annotation)
    return annotation_set, passage_text_dict

def get_annotations_from_JSON(input_collection, input_filename, eval_config):
    annotation_set = set()
    passage_text_dict = collections.defaultdict(dict)
    for document in input_collection["documents"]:
        # document_id = document["id"]
        for passage in document["passages"]:
            passage_id = passage["passage_id"]
            passage_offset = passage["offset"]
            passage_text = passage.get("text")
            passage_end = passage_offset + len(passage_text)
            if passage_text is None:
                continue
            passage_text_dict[passage_id][passage_offset] = passage_text
            for annotation in passage["annotations"]:
                annotation_id = annotation["id"]
                type = annotation["infons"]["type"]
                #if not eval_config.annotation_types is None and not (type in eval_config.annotation_types):
                #    continue
                if eval_config.evaluation_type == "span":
                    locations = [annotation_location(location["offset"], location["length"]) for location in annotation["locations"]]
                    if sum(location.length for location in locations) == 0:
                        log.warning("Ignoring zero-length annotation: document ID = {}, annotation ID = {}".format(passage_id, annotation_id))
                        continue
                    if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                        log.warning("Ignoring annotation with span outside of passage: passage ID = {}, annotation ID = {}".format(passage_id, annotation_id))
                        continue
                    locations.sort()
                    annotation_text = annotation["text"]
                    location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                    annotation = span_annotation(document["id"], type, tuple(locations), annotation_text)
                    #log.debug("BioCJSON file {} span annotation {}".format(input_filename, str(annotation)))
                    if annotation_text != location_text:
                        log.error("Annotation text {} does not match text at location(s) {}: passage ID = {}, annotation ID = {}".format(annotation_text, location_text, passage_id, annotation_id))
                    annotation_set.add(annotation)
                if eval_config.evaluation_type == "identifier":
                    for identifier in annotation["infons"]["identifier"].split(","):
                        annotation = identifier_annotation(document["id"], type, identifier)
                        #log.debug("BioCJSON file {} identifier annotation {}".format(input_filename, str(annotation)))
                        annotation_set.add(annotation)
    return annotation_set, passage_text_dict
            
def get_annotations_from_file(input_filename, eval_config, NER_blacklist = []):
    try:
        if input_filename.endswith(".xml"):
            log.info("Reading XML file {}".format(input_filename))
            parser = ElementTree.XMLParser(encoding="utf-8")
            input_collection = ElementTree.parse(input_filename, parser=parser).getroot()
            return get_annotations_from_XML(input_collection, input_filename, eval_config, NER_blacklist)
        if input_filename.endswith(".json"):
            raise Exception("did not update with: NER_blacklist")
            log.info("Reading JSON file {}".format(input_filename))
            #with codecs.open(input_filename, 'r', encoding="utf8") as input:
            #    input_collection = json.load(input)
            #return get_annotations_from_JSON(input_collection, input_filename, eval_config)
        log.info("Ignoring file {}".format(input_filename))
        return set(), dict()
    except Exception as e:
        raise RuntimeError("Error while processing file {}".format(input_filename)) from e

def get_annotations_from_path(input_path, eval_config, NER_blacklist=[]):
    annotation_set = set()
    passage_text_dict = collections.defaultdict(set)
    if os.path.isdir(input_path):
        log.info("Processing directory {}".format(input_path))
        dir = os.listdir(input_path)
        for item in dir:
            input_filename = input_path + "/" + item
            if os.path.isfile(input_filename):
                annotation_set2, passage_text_dict2 = get_annotations_from_file(input_filename, eval_config, NER_blacklist)
                annotation_set.update(annotation_set2)
                passage_text_dict.update(passage_text_dict2)
    elif os.path.isfile(input_path):
        annotation_set2, passage_text_dict2 = get_annotations_from_file(input_path, eval_config, NER_blacklist)
        annotation_set.update(annotation_set2)
        passage_text_dict.update(passage_text_dict2)
    else:  
        raise RuntimeError("Path is not a directory or normal file: {}".format(input_path))
    return annotation_set, passage_text_dict

def calculate_evaluation_count(reference_annotations, predicted_annotations):
    log2 = list()
    reference_annotations = set(reference_annotations)
    predicted_annotations = set(predicted_annotations)
    annotations = set()
    annotations.update(reference_annotations)
    annotations.update(predicted_annotations)
    annotations = list(annotations)
    annotations.sort()
    results = collections.Counter()
    for a in annotations:
        r = a in reference_annotations
        p = a in predicted_annotations
        results[(r, p)] += 1
        log.debug("annotation = {} in reference = {} in predicted = {}".format(str(a), r, p))
        log2.append("annotation = {} in reference = {} in predicted = {}".format(str(a), r, p))
    log.debug("Raw results = {}".format(str(results)))
    return evaluation_count(results[(True, True)], results[(False, True)], results[(True, False)]), log2

def calculate_evaluation_result(eval_count):
    if eval_count.tp == 0:
        return evaluation_result(0.0, 0.0, 0.0)
    p = eval_count.tp / (eval_count.tp + eval_count.fp)
    r = eval_count.tp / (eval_count.tp + eval_count.fn)
    f = 2.0 * p * r / (p + r)
    return evaluation_result(p, r, f)

def do_strict_eval(reference_annotations, predicted_annotations):
    eval_count, _ = calculate_evaluation_count(reference_annotations, predicted_annotations)
    log.info("TP = {0}, FP = {1}, FN = {2}".format(eval_count.tp, eval_count.fp, eval_count.fn))
    eval_result = calculate_evaluation_result(eval_count)
    return eval_result


def get_locations(annotations):
    locations = collections.defaultdict(list)
    for annotation in annotations:
        locations[annotation.passage_id].append({(annotation.type, offset, offset + length) for offset, length in annotation.locations})
    return locations

def do_approx_span_eval_AnyMatching(reference_annotations, predicted_annotations, pool=False):
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

# def overlaps(location, start2, end2):
#     return (location.offset < end2) and (start2 < location.offset + location.length)

# def overlaps(loc1, loc2):
#     loc1_end = loc1.offset + loc1.length
#     loc2_end = loc2.offset + loc2.length
#     return (loc1.offset < loc2_end) and (loc2.offset < loc1_end)

def find_approx_match(ref_annotation, predicted_annotations, pool):
    # given a list of predicted annotations, return the index of the 
    for ref_loc in ref_annotation.locations:
        for i, pred_annotation in enumerate(predicted_annotations):
            if (ref_annotation.passage_id == pred_annotation.passage_id) and \
              (pool or ref_annotation.type == pred_annotation.type):
                for pred_loc in pred_annotation.locations:
                    if overlaps(ref_loc, pred_loc):
                        return i
    return -1

def do_approx_span_eval_GREEDY(reference_annotations, predicted_annotations, pool=False):
    # this one is greedy and the results will depend on the order of the annotations
    tp, fn, fp = 0, 0, 0
    
    unused_predicted_annotations = copy.deepcopy(predicted_annotations)
    
    for ref_annotation in reference_annotations:
        found_ind = find_approx_match(ref_annotation, unused_predicted_annotations, pool)
        if found_ind == -1:
            fn += 1
        else:
            unused_predicted_annotations.pop(found_ind)

    
    # remaining predicted spans are false positives
    fp = len(unused_predicted_annotations)
        
    log.info("TP = {0}, FP = {1}, FN = {2}".format(tp, fp, fn))

    if tp == 0:
        return evaluation_result(0.0, 0.0, 0.0)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)
    return evaluation_result(p, r, f)


def overlaps(start1, end1, start2, end2):
    return (start1 < end2) and (start2 < end1)

def do_approx_span_eval(reference_annotations, predicted_annotations, pool=False):
    tp, fn, fp = 0, 0, 0

    ref_locs = get_locations(reference_annotations)
    pred_locs = get_locations(predicted_annotations)
    all_docids = set(ref_locs.keys()).union(pred_locs.keys())

    for doc_id in all_docids:
        refs = ref_locs.get(doc_id, [])
        preds = pred_locs.get(doc_id, [])

        if not refs:
            fp += len(preds)
            continue
        if not preds:
            fn += len(refs)
            continue

        # Cost matrix: 0 = valid match, 1 = invalid
        cost = np.ones((len(refs), len(preds)))

        for i, (r_type, r_start, r_end) in enumerate(refs):
            for j, (p_type, p_start, p_end) in enumerate(preds):
                if overlaps(r_start, r_end, p_start, p_end) and (pool or (r_type == p_type)):
                    cost[i, j] = 0

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = sum(cost[r, c] == 0 for r, c in zip(row_ind, col_ind))

        tp += matches
        fn += len(refs) - matches
        fp += len(preds) - matches

    if tp == 0:
        return evaluation_result(0.0, 0.0, 0.0)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)
    return evaluation_result(p, r, f)



def get_docid2identifiers(annotations):
    docid2identifiers = collections.defaultdict(set)
    for docid, type, identifier in annotations:
        if type in annotation_types:
            docid2identifiers[docid].add(identifier)
    return docid2identifiers

def verify_document_sets(reference_passages, predicted_passages):
    verification_errors = list()
    # Verify that reference path and prediction path contain the same set of documents
    reference_docids = set(reference_passages.keys())
    predicted_docids = set(predicted_passages.keys())
    if len(reference_docids - predicted_docids) > 0:
        verification_errors.append("Prediction path is missing documents {}".format(", ".join(reference_docids - predicted_docids)))
    if len(predicted_docids - reference_docids) > 0:
        verification_errors.append("Prediction path contains extra documents {}".format(", ".join(predicted_docids - reference_docids)))
    # Verify that the reference and predicted files are the same
    docids = reference_docids.intersection(predicted_docids)
    for passage_id in docids:
        reference_passage_offsets = set(reference_passages[passage_id].keys())
        predicted_passage_offsets = set(predicted_passages[passage_id].keys())
        if len(reference_passage_offsets) != len(predicted_passage_offsets):
            verification_errors.append("Number of passages does not match for document {0}, {1} != {2}".format(passage_id, len(reference_passage_offsets), len(predicted_passage_offsets)))
        elif reference_passage_offsets != predicted_passage_offsets:
            verification_errors.append("Passage offsets do not match for document {}".format(passage_id))
        else:
            for offset in reference_passage_offsets:
                if reference_passages[passage_id][offset] != predicted_passages[passage_id][offset]:
                    verification_errors.append("Passage text does not match for document {0}, offset {1}".format(passage_id, offset))
    return verification_errors

def log_entity_types(ref_annotations, prediction_annotations):
    ref_types = [ann.type for ann in ref_annotations]
    log.info("ref annotations type counts: {}".format({item:ref_types.count(item) for item in ref_types}))
    prediction_types = [ann.type for ann in prediction_annotations]
    log.info("prediction annotations type counts: {}".format({item:prediction_types.count(item) for item in prediction_types}))

def filter_entity(annotations, entity_to_keep): # entity_to_keep can be string or dict
    if not (entity_to_keep is None):
        return set(filter(lambda ann: ann[1] in entity_to_keep, annotations))

if __name__ == "__main__":
    
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Evaluation script for NLM CellLink")
    parser.add_argument("--reference_path", "-r", type=str, required=True, help="path to directory or file containing the reference annotations, i.e. the annotations considered correct")
    parser.add_argument("--prediction_path", "-p", type=str, required=True, help="path to directory or file containing the predicted annotations, i.e. the annotations being evaluated")
    parser.add_argument("--evaluation_type", "-t", choices = {"span", "identifier"}, required=True, help="The type of evaluation to perform")
    parser.add_argument("--evaluation_method", "-m", choices = {"strict", "approx"}, required=True, help="Whether to perform a strict or approximate evaluation")
    parser.add_argument("--annotation_type", "-a", type=str, required=True, help="The annotation type to consider, all others are ignored. 'None' considers all types, but it still must match")
    parser.add_argument("--logging_level", "-l", type=str, default="INFO", help="The logging level, options are {critical, error, warning, info, debug}")
    parser.add_argument("--no_document_verification", dest='verify_documents', action='store_const', const=False, default=True, help='Do not verify that reference and predicted document sets match')
    
    args = parser.parse_args()
    evaluation_type = args.evaluation_type
    evaluation_method = args.evaluation_method
    annotation_types = args.annotation_type.split() if not args.annotation_type.lower() == "none" else None
    logging.basicConfig(level=args.logging_level.upper(), format=log_format)
    
    if log.isEnabledFor(logging.DEBUG):
        for arg, value in sorted(vars(args).items()):
            log.info("Argument {0}: {1}".format(arg, value))

    eval_config = evaluation_config(annotation_types, evaluation_type)
    reference_annotations, reference_passages = get_annotations_from_path(args.reference_path, eval_config)
    predicted_annotations, predicted_passages = get_annotations_from_path(args.prediction_path, eval_config)
    print(len(reference_annotations), len(predicted_annotations))
    print("annotation_types:", annotation_types)
    print((annotation_types is not None) and ("merged" in annotation_types))
    if (annotation_types is not None) and ("merged" in annotation_types):
        reference_annotations = [ann._replace(type="merged") for ann in reference_annotations]
        predicted_annotations = [ann._replace(type="merged") for ann in predicted_annotations]
    else:
        filter_entity(reference_annotations, annotation_types)
        filter_entity(predicted_annotations, annotation_types)
    log_entity_types(reference_annotations, predicted_annotations)
    
    if args.verify_documents:
        verification_errors = verify_document_sets(reference_passages, predicted_passages)
        for verification_error in verification_errors:
            log.error(verification_error)
        if len(verification_errors) > 0:
            sys.exit(1)

    if evaluation_method == "strict":
        eval_result = do_strict_eval(reference_annotations, predicted_annotations)
    elif evaluation_method == "approx" and evaluation_type == "span":
        eval_result = do_approx_span_eval(reference_annotations, predicted_annotations)
    else:
        raise ValueError("Unknown evaluation method: {}".format(evaluation_method))
    print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))
    log.info("Elapsed time: {}".format(datetime.datetime.now() - start))