import math
import sys
import gzip
import json
import random
from collections import deque

import scoring
import PerformanceProfiler

def read_ids(filename):
    ids = set()
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            ids.add(line)
    return ids

def load_dimension_scorers(dimension2data_type, dimension2scoring_config, measurements, sample_size):
    # Get list of annotations
    prior_scorer = scoring.annotation_prior_scorer()
    dimension_scorer_list = list()
    for index, (dimension, values) in enumerate(measurements.items()):
        if not dimension in dimension2scoring_config:
            continue
        # Get the scorer
        scorer_type, scorer_function, scoring_weight, params = dimension2scoring_config[dimension]
        if scorer_type == "prior":
            scorer_name_fields = scorer_function.split(".")
            scorer_class = getattr(sys.modules[scorer_name_fields[0]], scorer_name_fields[1])
            # Add the annotation to the scorer
            data_type = dimension2data_type[dimension]
            if data_type == "singleton" or data_type == "count_dict":
                per_doc_mean, item_values = values
                sample_total = per_doc_mean * sample_size
                for item_name, value in item_values.items():
                    scorer_instance = scorer_class(value, sample_total, scoring_weight)
                    prior_scorer.add_annotation((dimension, item_name), scorer_instance)
            elif data_type == "number":
                mean, stdev = values
                scorer_instance = scorer_class(mean, stdev, scoring_weight)
                prior_scorer.add_annotation((dimension, None), scorer_instance)
        elif scorer_type == "selection":
            data_type = dimension2data_type[dimension]
            if data_type == "singleton" or data_type == "count_dict":
                scorer_name_fields = scorer_function.split(".")
                scorer_class = getattr(sys.modules[scorer_name_fields[0]], scorer_name_fields[1])
                per_doc_mean, item_values = values
                scorer_instance = scorer_class(dimension, per_doc_mean, *params)
                dimension_scorer_list.append((scoring_weight, scorer_instance))
                for item_name, value in item_values.items():
                    scorer_instance.add_annotation((dimension, item_name), value)
            elif data_type == "number":
                raise ValueError("Not implemented")
        else:
            raise ValueError()
        
    print("Total number of prior annotations: {}".format(len(prior_scorer.annotation2index)))
    for weight, dimension_scorer in dimension_scorer_list:
        print("Total number of annotations for dimension {}: {}".format(dimension_scorer.dimension, len(dimension_scorer.annotation2index)))

    print("Initializing scoring")
    for weight, dimension_scorer in dimension_scorer_list:
        dimension_scorer.finalize()

    return prior_scorer, dimension_scorer_list

def create_scorer(dimension2data_type, prior_scorer, dimension_scorer_list, passages, visualize = False):
    PerformanceProfiler.start("create_scorer()")
    # Get annotation index/counts for each passage
    print("Converting passage annotations to indexes")
    scorer = scoring.normalized_passage_scorer(prior_scorer, dimension_scorer_list)
    for passage in passages:
        passageid = passage["passage_id"]
        passage_data = passage["data"]
        annotation_counts = dict()
        for dimension, values in passage_data.items():
            data_type = dimension2data_type.get(dimension)
            if data_type is None:
                continue
            elif data_type == "singleton":
                annotation_counts[(dimension, values)] = 1
            elif data_type == "count_dict":
                for value, count in values.items():
                    annotation_counts[(dimension, value)] = count
            elif data_type == "number":
                annotation_counts[(dimension, None)] = values
        scorer.add_passage(passageid, annotation_counts)
    print("Initial number of passages in queue is {}".format(len(scorer.passageid2info)))
    
    print("Finalizing indexes")
    scorer.finalize()
    PerformanceProfiler.end("create_scorer()")
    if visualize:
        PerformanceProfiler.visualize()
    return scorer

def select_passage_ids(passageid2pmid, scorer, sample_size, visualize = False):
    PerformanceProfiler.start("select_passage_ids()")
    pmid2passageids = dict()
    for passageid, pmid in passageid2pmid.items():
        if not pmid in pmid2passageids:
            pmid2passageids[pmid] = set()
        pmid2passageids[pmid].add(passageid)

    print("Selecting")
    selected_passageids = list()
    while len(selected_passageids) < sample_size and len(scorer.passageid2info) > 0:
        iteration = len(selected_passageids)
        print("Iteration {}: starting".format(iteration))
        PerformanceProfiler.start("select_passage_ids()@iteration")
        # Get best passage
        score, passageid, dimension_summary, item_analysis = scorer.select_lowest()
        if passageid is None or dimension_summary is None or item_analysis is None:
            print(f"WARN: scorer returned None for passageid: score = {score} passageid = {passageid} dimension_summary = {dimension_summary} item_analysis = {item_analysis}")
            break
        print("Iteration {}: accepting {}, {}".format(iteration, passageid, score))
        scorer.select_passage(passageid)
        pmid = passageid2pmid[passageid]
        for passageid2 in pmid2passageids[pmid]:
            scorer.drop_passage(passageid2)
        for dimension_name, dimension_weight, dimension_score in dimension_summary:
            print("\tdim = {} weight = {} score = {}".format(dimension_name, dimension_weight, dimension_score))
        item_analysis_list = [(score, count, annotation) for annotation, (count, score) in item_analysis.items()]
        item_analysis_list.sort()
        print("\tScores:")
        for score, count, (dimension, annotation) in item_analysis_list:
            print("\t\t{} @ {} = {}".format(annotation, count, score))
        selected_passageids.append(passageid)
        print("Iteration {}: complete".format(iteration))
        PerformanceProfiler.end("select_passage_ids()@iteration")
        if visualize:
            PerformanceProfiler.visualize()
    PerformanceProfiler.end("select_passage_ids()")
    if visualize:
        PerformanceProfiler.visualize()
    return selected_passageids

class FunnelSelector:

    def __init__(self, batch_size, output_size, selector):
        self.batch_size = batch_size
        self.output_size = output_size
        self.selector = selector

    def batch_select(self, items, visualize = False):
        PerformanceProfiler.start("FunnelSelector.batch_select()")
        batch_count = math.ceil(len(items) / self.batch_size)
        actual_batch_size = math.ceil(len(items) / batch_count)
        print(f"FunnelSelector.batch_select(): running {batch_count} batches of size {actual_batch_size}; len(items) = {len(items)}")
        unprocessed_items = deque(items)
        selected_items = list()
        batch_number = 0
        while len(unprocessed_items) > 0:
            PerformanceProfiler.start("FunnelSelector.batch_select()@batch")
            batch_number += 1
            batch = list()
            while len(batch) < actual_batch_size and len(unprocessed_items) > 0:
                batch.append(unprocessed_items.popleft())
            print(f"FunnelSelector.batch_select(): batch {batch_number} of {batch_count}: len(batch) = {len(batch)}")
            batch_selected_items = self.selector(batch, self.output_size, visualize and batch_number == 1) 
            print(f"FunnelSelector.batch_select(): batch {batch_number} of {batch_count}: len(batch_selected_items) = {len(batch_selected_items)}")
            selected_items.extend(batch_selected_items)
            PerformanceProfiler.end("FunnelSelector.batch_select()@batch")
        print(f"FunnelSelector.batch_select(): len(selected_items) = {len(selected_items)}")
        PerformanceProfiler.end("FunnelSelector.batch_select()")
        if visualize:
            PerformanceProfiler.visualize()
        return selected_items

    def iterative_batch_select(self, items):
        iteration = 0
        current_items = list(items)
        while len(current_items) > self.output_size:
            PerformanceProfiler.start("FunnelSelector.iterative_batch_select()@iterate")
            iteration += 1
            print(f"FunnelSelector.iterative_batch_select(): iteration={iteration}, len(current_items)={len(current_items)}")
            random.shuffle(current_items)
            next_items = self.batch_select(current_items, iteration <= 1)
            if len(next_items) >= len(current_items):
                raise RuntimeError("Funnel did not shrink the item set")
            current_items = next_items
            PerformanceProfiler.end("FunnelSelector.iterative_batch_select()@iterate")
            PerformanceProfiler.visualize()
        return current_items

def main():
    config_filename = sys.argv[1]
    passages_filename = sys.argv[2]
    measurements_input_filename = sys.argv[3]
    output_size = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    selected_passageids_filename = sys.argv[6]
    if output_size >= batch_size:
        raise ValueError("output_size must be smaller than batch size")

    # Load config
    with open(config_filename) as config_file:
        config = json.load(config_file)
    print("Loaded {} dimensions".format(len(config)))
    dimension2data_type = {dimension_name: dimension_config["data_type"] for dimension_name, dimension_config in config.items()}
    print("dimension2data_type = {}".format(dimension2data_type))
    
    print("Preparing scorers from config")
    dimension2scoring_config = dict()
    for dimension_name, dimension_config in config.items():
        if not "scorer" in dimension_config:
            continue
        scorer_type, scorer_function, scoring_weight, params = dimension_config["scorer"]
        if scorer_type == "prior":
            dimension2scoring_config[dimension_name] = (scorer_type, scorer_function, scoring_weight, params)
        elif scorer_type == "selection":
            dimension2scoring_config[dimension_name] = (scorer_type, scorer_function, scoring_weight, params)
        else:
            raise ValueError()

    print("Loading measurements")
    open_func = gzip.open if measurements_input_filename.endswith(".gz") else open
    with open_func(measurements_input_filename, "rt") as file:
        measurements = json.load(file)
    print("Number of measurements is {}".format(len(measurements)))
    passage_count = measurements["PASSAGE_COUNT"]
    print("Number of passages is {}".format(passage_count))

    print("Preparing scorers from measurements")
    prior_scorer, dimension_scorer_list = load_dimension_scorers(dimension2data_type, dimension2scoring_config, measurements, output_size)

    print("Loading passages")
    passages = list()
    open_func = gzip.open if passages_filename.endswith(".gz") else open
    with open_func(passages_filename, "rt") as passage_input_file:
        for line_index, line in enumerate(passage_input_file):
            if (line_index + 1) % 5000 == 0:
                print("Loading passage {} of {}".format(line_index + 1, passage_count))
            passage = json.loads(line)
            passages.append(passage)
    print("Number of passages is {} ".format(len(passages)))

    def select_passages_inner(passages_inner, sample_size_inner, visualize = False):
        PerformanceProfiler.start("select_passages_inner()")
        if len(passages_inner) <= sample_size_inner:
            PerformanceProfiler.end("select_passages_inner()")
            return passages_inner
        scorer = create_scorer(dimension2data_type, prior_scorer, dimension_scorer_list, passages_inner, visualize)
        passageid2pmid_inner = {passage["passage_id"]: passage["pmid"] for passage in passages_inner}
        selected_passageids = set(select_passage_ids(passageid2pmid_inner, scorer, sample_size_inner, visualize))
        selected_passages = [passage for passage in passages_inner if passage["passage_id"] in selected_passageids]
        PerformanceProfiler.end("select_passages_inner()")
        if visualize:
            PerformanceProfiler.visualize()
        return selected_passages
    
    funnel_selector = FunnelSelector(batch_size, output_size, select_passages_inner)
    selected_passages = funnel_selector.iterative_batch_select(passages)

    PerformanceProfiler.visualize()

    print("Writing selected passageids")
    with open(selected_passageids_filename, "w") as output_file:
        for passage in selected_passages:
            output_file.write(passage["passage_id"] + "\n")

    print("Done.")

if __name__ == '__main__':
    main()