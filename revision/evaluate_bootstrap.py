import argparse
import collections
import datetime
import json
import logging
import os
import sys
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ElementTree
import re


root = "MESH:ROOT"
log_format = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"

# Returns precision, recall & f-score for the specified reference and prediction files

log = logging.getLogger(__name__)

evaluation_config = collections.namedtuple("evaluation_config", ("annotation_types", "evaluation_type"))
evaluation_count = collections.namedtuple("evaluation_count", ("tp", "fp", "fn"))
evaluation_result = collections.namedtuple("evaluation_result", ("precision", "recall", "f_score"))
bootstrap_evaluation_result = collections.namedtuple(
    "bootstrap_evaluation_result",
    ("point_estimate", "sample_counts", "sample_results", "confidence_intervals"),
)
span_annotation = collections.namedtuple("span_annotation", ("passage_id", "type", "locations", "text"))
identifier_annotation = collections.namedtuple("identifier_annotation", ("passage_id", "type", "identifier"))
span_identifier_annotation = collections.namedtuple("span_identifier_annotation", ("passage_id", "type", "locations", "text", "identifier"))
annotation_location = collections.namedtuple("annotation_location", ("offset", "length"))

def type_matches(entity_type, annotation_types):
    if annotation_types is None:
        return entity_type, True
    if "merged" in annotation_types:
        return "merged", True
    if entity_type in annotation_types:
        return entity_type, True
    return entity_type, False

def get_annotations_from_XML(input_collection, input_filename, eval_config):
    annotation_set = set()
    passage_text_dict = collections.defaultdict(dict)
    for document in input_collection.findall(".//document"):
        document_id = document.find(".//id").text
        collate_key = document_id
        #print(f"collate_key = {collate_key}")
        for passage_idx, passage in enumerate(document.findall(".//passage")):
        #for passage in input_collection.findall(".//passage"):
                #passage_id = passage.findtext("infon[@key='passage_id']")
                passage_offset = int(passage.find(".//offset").text)
                if passage.find(".//text") is None:
                    continue
                passage_text = passage.find(".//text").text
                passage_end = passage_offset + len(passage_text)
                passage_text_dict[collate_key][passage_offset] = passage_text
                # the passage_offset key is unnecessary but not harmful, so we keep for backwards compatibility
                for annotation in passage.findall(".//annotation"):
                    annotation_id = annotation.attrib["id"]
                    entity_type, entity_match = type_matches(annotation.find(".//infon[@key='type']").text, eval_config.annotation_types)
                    if not entity_match:
                        continue
                    if eval_config.evaluation_type == "span":
                        locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
                        if sum(location.length for location in locations) == 0:
                            log.warning("Ignoring zero-length annotation: passage ID = {}, annotation ID = {}".format(collate_key, annotation_id))
                            continue
                        if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                            log.warning("Ignoring annotation with span outside of passage: passage ID = {}, annotation ID = {}".format(collate_key, annotation_id))
                            continue
                        locations.sort()
                        annotation_text = annotation.find(".//text").text
                        location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                        annotation = span_annotation(collate_key, entity_type, tuple(locations), annotation_text)
                        if annotation_text != location_text:
                            log.error("Annotation text {} does not match text at location(s) {}: passage ID = {}, annotation ID = {}".format(annotation_text, location_text, collate_key, annotation_id))
                        annotation_set.add(annotation)
                    if eval_config.evaluation_type == "identifier":
                        identifier_node = annotation.find(".//infon[@key='identifier']")
                        ## Robert, can you check how I deal with the identifier field for 
                        # identifier_annotation and span_identifier_annotation?
                        # for identifier-only, I split the identifiers. keeping the skos term.
                        # for identifier_span, I leave the identifier as is
                        if (identifier_node is None) or (identifier_node.text is None) or \
                            (identifier_node.text.lower() == "none") or (len(identifier_node.text) == 0):
                            continue
                        for identifier in re.split(',|;', identifier_node.text):
                            annotation = identifier_annotation(str(collate_key), entity_type, identifier)
                            #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                            annotation_set.add(annotation)
                    elif eval_config.evaluation_type == "span_identifier":
                        locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
                        if sum(location.length for location in locations) == 0:
                            log.warning("Ignoring zero-length annotation: passage ID = {}, annotation ID = {}".format(collate_key, annotation_id))
                            continue
                        if any((location.offset < passage_offset or location.offset + location.length > passage_end) for location in locations):
                            log.warning("Ignoring annotation with span outside of passage: passage ID = {}, annotation ID = {}".format(collate_key, annotation_id))
                            continue
                        locations.sort()
                        annotation_text = annotation.find(".//text").text
                        identifier_node = annotation.find(".//infon[@key='identifier']")
                        location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
                        if annotation_text != location_text:
                            log.error("Annotation text {} does not match text at location(s) {}: passage ID = {}, annotation ID = {}".format(annotation_text, location_text, collate_key, annotation_id))
                        if (identifier_node is None) or (identifier_node.text is None) or \
                            (identifier_node.text.lower() == "none") or (len(identifier_node.text) == 0):
                            identifier = ""
                        else:
                            identifier = identifier_node.text
                        
                        annotation = span_identifier_annotation(collate_key, entity_type, tuple(locations), annotation_text, identifier)
                        #log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
                        annotation_set.add(annotation)
    return annotation_set, passage_text_dict

            
def get_annotations_from_file(input_filename, eval_config):
    try:
        if input_filename.endswith(".xml"):
            log.info("Reading XML file {}".format(input_filename))
            parser = ElementTree.XMLParser(encoding="utf-8")
            input_collection = ElementTree.parse(input_filename, parser=parser).getroot()
            return get_annotations_from_XML(input_collection, input_filename, eval_config)
        log.info("Ignoring file {}".format(input_filename))
        return set(), dict()
    except Exception as e:
        raise RuntimeError("Error while processing file {}".format(input_filename)) from e

def get_annotations_from_path(input_path, eval_config):
    annotation_set = set()
    passage_text_dict = collections.defaultdict(set)
    if os.path.isdir(input_path):
        log.info("Processing directory {}".format(input_path))
        dir = os.listdir(input_path)
        for item in dir:
            input_filename = input_path + "/" + item
            if os.path.isfile(input_filename):
                annotation_set2, passage_text_dict2 = get_annotations_from_file(input_filename, eval_config)
                annotation_set.update(annotation_set2)
                passage_text_dict.update(passage_text_dict2)
    elif os.path.isfile(input_path):
        annotation_set2, passage_text_dict2 = get_annotations_from_file(input_path, eval_config)
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


def calculate_evaluation_results(eval_counts):
    """Vectorized precision, recall, and F-score for an (n, 3) count array."""
    eval_counts = np.asarray(eval_counts)
    if eval_counts.ndim != 2 or eval_counts.shape[1] != 3:
        raise ValueError("eval_counts must have shape (n, 3) for TP, FP, and FN")

    tp = eval_counts[:, 0].astype(float)
    fp = eval_counts[:, 1].astype(float)
    fn = eval_counts[:, 2].astype(float)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return np.column_stack((precision, recall, f_score))

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
    
    # breakpoint()
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

        for i, ref in enumerate(refs):
            for ref_loc in ref:
                for j, pred in enumerate(preds):
                    for pred_loc in pred:
                        r_type, r_start, r_end = ref_loc
                        p_type, p_start, p_end = pred_loc
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


def calculate_approx_document_count(reference_locations, predicted_locations, pool=False):
    """Calculate approximate span counts for one document."""
    if not reference_locations:
        return evaluation_count(0, len(predicted_locations), 0)
    if not predicted_locations:
        return evaluation_count(0, 0, len(reference_locations))

    # Cost matrix: 0 = valid match, 1 = invalid. The assignment therefore
    # maximizes the number of non-conflicting overlap matches.
    cost = np.ones((len(reference_locations), len(predicted_locations)))
    for i, reference_annotation in enumerate(reference_locations):
        for reference_location in reference_annotation:
            for j, predicted_annotation in enumerate(predicted_locations):
                for predicted_location in predicted_annotation:
                    reference_type, reference_start, reference_end = reference_location
                    predicted_type, predicted_start, predicted_end = predicted_location
                    if overlaps(reference_start, reference_end, predicted_start, predicted_end) and (
                        pool or reference_type == predicted_type
                    ):
                        cost[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = sum(cost[row, column] == 0 for row, column in zip(row_ind, col_ind))
    return evaluation_count(
        matches,
        len(predicted_locations) - matches,
        len(reference_locations) - matches,
    )


def calculate_document_evaluation_counts(
    reference_annotations,
    predicted_annotations,
    document_ids,
    evaluation_type,
    evaluation_method,
):
    """Reduce each document to its additive TP, FP, and FN contribution."""
    if evaluation_method == "strict":
        references_by_document = collections.defaultdict(set)
        predictions_by_document = collections.defaultdict(set)
        for annotation in reference_annotations:
            references_by_document[annotation.passage_id].add(annotation)
        for annotation in predicted_annotations:
            predictions_by_document[annotation.passage_id].add(annotation)

        document_counts = {}
        for document_id in document_ids:
            references = references_by_document.get(document_id, set())
            predictions = predictions_by_document.get(document_id, set())
            document_counts[document_id] = evaluation_count(
                len(references & predictions),
                len(predictions - references),
                len(references - predictions),
            )
        return document_counts

    if evaluation_method == "approx" and evaluation_type == "span":
        references_by_document = get_locations(reference_annotations)
        predictions_by_document = get_locations(predicted_annotations)
        return {
            document_id: calculate_approx_document_count(
                references_by_document.get(document_id, []),
                predictions_by_document.get(document_id, []),
            )
            for document_id in document_ids
        }

    if evaluation_method == "approx" and evaluation_type == "identifier":
        raise ValueError("Bootstrap sampling does not support identifier/approx evaluation")
    raise ValueError(
        "Unknown evaluation configuration: {}/{}".format(evaluation_type, evaluation_method)
    )


def do_bootstrap_eval(
    reference_annotations,
    predicted_annotations,
    document_ids,
    evaluation_type,
    evaluation_method,
    n_samples,
    random_seed=None,
):
    """Perform document-level bootstrap sampling without reevaluating replicates.

    Annotation matching is performed once per document. All bootstrap samples
    are then drawn together as document multiplicities, and their TP/FP/FN
    totals are calculated by matrix multiplication.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero for bootstrap sampling")
    if evaluation_type == "identifier" and evaluation_method == "approx":
        raise ValueError("Bootstrap sampling does not support identifier/approx evaluation")

    document_ids = list(document_ids)
    if not document_ids:
        raise ValueError("Cannot bootstrap an evaluation containing no documents")

    counts_by_document = calculate_document_evaluation_counts(
        reference_annotations,
        predicted_annotations,
        document_ids,
        evaluation_type,
        evaluation_method,
    )
    #log.debug(f"TRACE counts_by_document = {counts_by_document}")
    document_counts = np.asarray(
        [counts_by_document[document_id] for document_id in document_ids],
        dtype=np.int64,
    )
    #log.info(f"document_counts.shape = {document_counts.shape} document_counts = {document_counts}")

    point_count_values = document_counts.sum(axis=0)
    point_count = evaluation_count(*point_count_values.tolist())
    point_estimate = calculate_evaluation_result(point_count)
    log.info(
        "TP = {0}, FP = {1}, FN = {2}".format(
            point_count.tp, point_count.fp, point_count.fn
        )
    )

    rng = np.random.default_rng(random_seed)
    document_probabilities = np.full(len(document_ids), 1.0 / len(document_ids))
    # Each row contains the number of times every document occurs in one
    # bootstrap sample. Every sample therefore contains len(document_ids)
    # document draws with replacement.
    sample_document_counts = rng.multinomial(
        len(document_ids), document_probabilities, size=n_samples
    ).astype(np.int32, copy=False)
    #log.debug(f"TRACE sample_document_counts.shape = {sample_document_counts.shape} sample_document_counts = {sample_document_counts}")

    # Columns are TP, FP, and FN. This evaluates all samples together rather
    # than looping over bootstrap replicates.
    sample_counts = sample_document_counts @ document_counts
    #log.debug(f"TRACE sample_counts.shape = {sample_counts.shape} sample_counts = {sample_counts}")
    sample_results = calculate_evaluation_results(sample_counts)
    #log.debug(f"TRACE sample_results.shape = {sample_results.shape} sample_results = {sample_results}")
    confidence_intervals = np.percentile(sample_results, (2.5, 97.5), axis=0).T
    #log.debug(f"TRACE confidence_intervals.shape = {confidence_intervals.shape} confidence_intervals = {confidence_intervals}")

    return bootstrap_evaluation_result(
        point_estimate,
        sample_counts,
        sample_results,
        confidence_intervals,
    )


def write_sample_metrics(sample_results, output_path):
    """Write one JSON record per bootstrap sample."""
    sample_metrics = [
        {
            "precision": float(precision),
            "recall": float(recall),
            "f_score": float(f_score),
        }
        for precision, recall, f_score in sample_results
    ]
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(sample_metrics, output_file, indent=2)
        output_file.write("\n")



# def get_docid2identifiers(annotations):
#     docid2identifiers = collections.defaultdict(set)
#     for docid, type, identifier in annotations:
#         if type in annotation_types:
#             docid2identifiers[docid].add(identifier)
#     return docid2identifiers

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

def filter_passages(annotations, passages, reference_passages):
    filtered_passages = copy.deepcopy(passages)
    for passage_id in passages.keys():
        if passage_id not in reference_passages:
            del filtered_passages[passage_id]
    
    filtered_annotations = copy.deepcopy([ann for ann in annotations if ann.passage_id in filtered_passages])
    return filtered_annotations, filtered_passages



def main(reference_path, prediction_path, evaluation_type, evaluation_method, annotation_type,
         logging_level, verify_documents, skip_extra_pred_passages, n_samples=0,
         random_seed=None, sample_metrics_path=None):
    
    start = datetime.datetime.now()
    input_annotation_type = annotation_type
    del annotation_type
    annotation_types = input_annotation_type.split() if not input_annotation_type.lower() == "none" else None
    logging.basicConfig(level=logging_level.upper(), format=log_format)
    
    if log.isEnabledFor(logging.DEBUG):
        for arg, value in sorted(locals().items()):
            log.info("Argument {0}: {1}".format(arg, value))

    eval_config = evaluation_config(annotation_types, evaluation_type)
    reference_annotations, reference_passages = get_annotations_from_path(reference_path, eval_config)
    predicted_annotations, predicted_passages = get_annotations_from_path(prediction_path, eval_config)
    log.info(f"Extracted {len(reference_annotations)} reference annotations from {len(reference_passages)} passages.")
    log.info(f"Extracted {len(predicted_annotations)} predicted annotations from {len(predicted_passages)} passages.")
    if None in reference_passages:
        raise Exception("At least one reference document could not find an ID")
    if None in predicted_passages:
        raise Exception("At least one predicted document could not find an ID")
    log.info(f"Annotation types extracted: {annotation_types}")
    
    if skip_extra_pred_passages:
        predicted_annotations, predicted_passages = filter_passages(predicted_annotations, predicted_passages, reference_passages)
    
    if (annotation_types is not None) and ("merged" in annotation_types):
        reference_annotations = [ann._replace(type="merged") for ann in reference_annotations]
        predicted_annotations = [ann._replace(type="merged") for ann in predicted_annotations]
    log_entity_types(reference_annotations, predicted_annotations)
    
    if verify_documents:
        verification_errors = verify_document_sets(reference_passages, predicted_passages)
        for verification_error in verification_errors:
            log.error(verification_error)
        if len(verification_errors) > 0:
            raise Exception("Input and reference documents did not match.")

    if n_samples < 0:
        raise ValueError("n_samples cannot be negative")
    if sample_metrics_path is not None and n_samples == 0:
        raise ValueError("sample_metrics_path requires n_samples greater than zero")

    if n_samples > 0:
        # passage_id is the document identifier used throughout this evaluator;
        # one document may contain multiple passage offsets.
        document_ids = sorted(set(reference_passages).union(predicted_passages))
        if None in document_ids:
            raise Exception("At least one document could not find an ID")
       
        eval_result = do_bootstrap_eval(
            reference_annotations,
            predicted_annotations,
            document_ids,
            evaluation_type,
            evaluation_method,
            n_samples,
            random_seed,
        )
        if sample_metrics_path is not None:
            write_sample_metrics(eval_result.sample_results, sample_metrics_path)
    elif evaluation_method == "strict":
        eval_result = do_strict_eval(reference_annotations, predicted_annotations)
    elif evaluation_method == "approx" and evaluation_type == "span":
        eval_result = do_approx_span_eval(reference_annotations, predicted_annotations)
    else:
        raise ValueError("Unknown evaluation method: {}".format(evaluation_method))
    log.info("Elapsed time: {}".format(datetime.datetime.now() - start))
    return eval_result


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluation script for NLM CellLink")
    parser.add_argument("--reference_path", "-r", type=str, required=True, help="path to directory or file containing the reference annotations, i.e. the annotations considered correct")
    parser.add_argument("--prediction_path", "-p", type=str, required=True, help="path to directory or file containing the predicted annotations, i.e. the annotations being evaluated")
    parser.add_argument("--evaluation_type", "-t", choices = {"span", "identifier"}, required=True, help="The type of evaluation to perform")
    parser.add_argument("--evaluation_method", "-m", choices = {"strict", "approx"}, required=True, help="Whether to perform a strict or approximate evaluation")
    parser.add_argument("--annotation_type", "-a", type=str, required=True, help="The annotation type to consider, all others are ignored. 'None' considers all types, but it still must match. 'merged' considers all types as a single type")
    parser.add_argument("--logging_level", "-l", type=str, default="INFO", help="The logging level, options are {critical, error, warning, info, debug}")
    parser.add_argument("--no_document_verification", dest='verify_documents', action='store_const', const=False, default=True, help='Do not verify that reference and predicted document sets match')
    parser.add_argument("--skip_extra_pred_passages", action='store_true', help="If there are passages only found in the prediction input, then skip them.")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of document-level bootstrap samples; 0 disables bootstrap sampling")
    parser.add_argument("--random_seed", type=int, default=None, help="Optional random seed for reproducible bootstrap samples")
    parser.add_argument(
        "--sample_metrics_path", "--sample_metrics_file",
        dest="sample_metrics_path",
        type=str,
        default=None,
        help="Optional JSON file for the precision, recall, and F-score of each bootstrap sample",
    )
    
    args = parser.parse_args()
    eval_result = main(**vars(args))
    if isinstance(eval_result, bootstrap_evaluation_result):
        point_estimate = eval_result.point_estimate
        print("P = {0:.3f}, R = {1:.3f}, F = {2:.3f}".format(
            point_estimate.precision, point_estimate.recall, point_estimate.f_score
        ))
        means = eval_result.sample_results.mean(axis=0)
        print("Document-level bootstrap (n = {}):".format(args.n_samples))
        for metric_name, mean, confidence_interval in zip(
            ("P", "R", "F"), means, eval_result.confidence_intervals
        ):
            print("{} mean = {:.4f}, 95% CI = [{:.4f}, {:.4f}]".format(
                metric_name, mean, confidence_interval[0], confidence_interval[1]
            ))
    else:
        print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))
