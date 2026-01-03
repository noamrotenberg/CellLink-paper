# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:41:13 2025

@author: rotenbergnh
"""

import os
import bioc
import pandas as pd
import collections
import numpy as np
import re
import matplotlib_venn
import matplotlib.pyplot as plt
import json
import io


# unfortunately, many of the files here are not available to the public because 
# they are data directly produced by the annotators and/or include test set passages

output_dir = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2025-04-22_corpus_paper_prep/"

# change path for IAA script: ## ** DELETE
eval_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-16_post-annotation_pipeline_for_pseudodocs/"
os.chdir(eval_path)
import evaluation
# import IAA_evaluation

# these paths refer to files produced by the annotators (not available to the public)
post_round3_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files\Results\PostRound3Processing"
post_round1_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files\Results\PostRound1Processing"

merged_output_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files/total_results_addl_metadata.xml"

merged_collection_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2025-04-22_corpus_paper_prep\train-test_split\merged.xml"

train_dev_test_split_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2025-04-22_corpus_paper_prep\train-test_split"


PASSAGE_JSONL_PATH = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files\CellLink_metadata.jsonl"
with open(PASSAGE_JSONL_PATH, 'r') as readfp:
    PASSAGE_METADATA = [json.loads(line) for line in readfp]

# ensure only 1 entry per passage
assert (collections.Counter([val['passage_id'] for val in PASSAGE_METADATA]).most_common(1)[0][1] == 1)
PASSAGE_METADATA_DICT = {val['passage_id']: val for val in PASSAGE_METADATA}

CL_ID_to_name_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files\CellOntology_id-name-definition-2025-02-13.csv"
ID_TO_MENTION_DICT = json.loads(pd.read_csv(CL_ID_to_name_path, index_col=1).dropna().T.to_json())
ID_TO_MENTION_DICT = {key: val['cl_name'] for key, val in ID_TO_MENTION_DICT.items()}
ID_TO_MENTION_DICT["None"] = "N/A"


def merge_collections(input_path, output_filename=None, add_missing_infons=True):
    output_collection = bioc.BioCCollection()
    
    for path, dn, filenames in os.walk(input_path):
        for file in filenames:
            filepath = os.path.join(path, file)
            if filepath[-4:] == '.xml':
                with open(filepath, 'r', encoding='utf-8') as readfp:
                    collection_i = bioc.load(readfp)
                for doc in collection_i.documents:
                    
                    if add_missing_infons:
                        # add any missing infons
                        for p in doc.passages:
                            # add passage_id:
                            if 'article-id_pmid' in p.infons and 'passage_idx' in p.infons and \
                                not 'passage_id' in p.infons:
                                raise Exception()
                                # p.infons['passage_id'] = p.infons['article-id_pmid'] + "_" + p.infons['passage_idx']
                            p.infons['set_name'] = doc.id
                            p.infons['set_num'] = int(re.search('[0-9]+', doc.id.split('set')[1]).group())
                            p.infons['date-folderName'] = "2025-" + path.split("2025-")[-1][:5] # extract from directory name
                    
                    output_collection.add_document(doc)
                    
                    
    if output_filename is not None:
        with open(output_filename, 'w', encoding='utf-8') as writefp:
            bioc.dump(output_collection, writefp)
    return output_collection

# these collections are made using the files produced by the annotators (not available to the public)
round3_collection = merge_collections(post_round3_path , merged_output_path)
round1_collection = merge_collections(post_round1_path)



def split_collection_by_set_num(input_collection, set_num):
    # create 2 collections, where set number < set_num and set number >= set_num
    collection1 = bioc.BioCCollection()
    collection2 = bioc.BioCCollection()
    
    for doc in input_collection.documents:
        set_num_i = int(re.search('[0-9]+', doc.id.split('set')[1]).group())
        if set_num_i < set_num:
            collection1.add_document(doc)
        else:
            collection2.add_document(doc)
    
    return collection1, collection2
    

def get_passages_and_annotations(collection):
    annotatable_passages = list()
    for doc in collection.documents:
        for p in doc.passages:
            p.infons['set_ID'] = doc.id
        annotatable_passages += [p for p in doc.passages if (len(p.text) > 0) and (p.infons.get('annotatable', True) != "no")]
            
    
    for p in annotatable_passages:
        # remove any unmerged annotations (i.e., same offset, text, ID, entity type)
        annotations_without_repeats = []
        annotations_p_uniqueness_set = set()
        for ann in p.annotations:
            uniqueness_set_i = (tuple(ann.locations), ann.text, ann.infons['identifier'], ann.infons['type'])
            if uniqueness_set_i not in annotations_p_uniqueness_set:
                annotations_p_uniqueness_set.add(uniqueness_set_i)
                annotations_without_repeats.append(ann)
            else:
                raise Exception("Found unmerged annotations")
                print("Found (and filtering) a duplicate annotation")
                print(p.infons['passage_id'], ';', ann.text)
        p.annotations = annotations_without_repeats
        
        # deal with None annotations:
        for ann in p.annotations:
            if (ann.infons['identifier'] is None) or (ann.infons['identifier'].lower() == "none") or (ann.infons['identifier'] == ""):
                ann.infons['identifier'] = "None"
                
    
    all_annotations = []
    for p in annotatable_passages:
        all_annotations += p.annotations
    cell_pheno_annotations = [ann for ann in all_annotations if ann.infons['type'] == "cell_phenotype"] # in ['cell_type', 'cell_phenotype', 'cell_pheno']]
    cell_hetero_annotations = [ann for ann in all_annotations if ann.infons['type'] == 'cell_hetero']
    cell_desc_annotations = [ann for ann in all_annotations if ann.infons['type'] in ['cell_desc', 'cell_vague']]
    assert(len(all_annotations) == len(cell_pheno_annotations) + len(cell_hetero_annotations) + len(cell_desc_annotations))
    annotation_groups = [all_annotations, cell_pheno_annotations, cell_hetero_annotations, cell_desc_annotations]
    return annotatable_passages, annotation_groups

def ID_is_None(identifier):
    # remove skos labels
    identifier = identifier.replace("(skos:related)", "")
    identifier = identifier.replace("(skos:exact)", "")
    all_sub_IDs = re.split(',|;', identifier)
    None_IDs = ["none", '-', ""]
    if all([sub_ID.lower() in None_IDs for sub_ID in all_sub_IDs]):
        return True
    else:
        return False


# def ID_is_exact(identifier):
#     if "(skos:related)" in identifier:
#         return False
#     elif "-" in identifier:
#         return False
#     elif "(skos:exact)" in identifier:
#         return True
#     else:
#         return False


def ID_is_exact(identifier):
    print(identifier, end='\t', file=writefp)
    if "(skos:related)" in identifier:
        print(False, file=writefp)
        return False
    elif "-" in identifier:
        print(False, file=writefp)
        return False
    elif "(skos:exact)" in identifier:
        print(True, file=writefp)
        return True
    else:
        print(False, file=writefp)
        return False

def ID_is_exact(identifier):
    if "(skos:related)" in identifier:
        return False
    elif "-" in identifier:
        return False
    elif "(skos:exact)" in identifier:
        return True
    elif "None" == identifier:
        return False
    else:
        raise Exception("Identifier qualifiers not formatted as expected.")

def nonexact_IDs(identifier):
    answer = []
    for ID_i in re.split(',|;', identifier):
        if "skos:exact" in ID_i:
            answer.append(ID_i)
    return answer


def get_all_IDs(annotation_list, include_None=False):
    # include_None means "None" is counted as 1 ID
    all_IDs = []
    for ann in annotation_list:
        identifier = ann.infons['identifier']
        # if (identifier is None) or (identifier.lower() == 'none') or (len(identifier) == 0):=
        # remove skos labels
        identifier = identifier.replace("(skos:related)", "")
        identifier = identifier.replace("(skos:exact)", "")
        all_IDs += re.split(',|;', identifier)
    
    for i, ID in enumerate(all_IDs):
        if ID_is_None(ID):
            all_IDs[i] = "None"
    if not include_None:
        all_IDs = [ID for ID in all_IDs if ID != "None"]
    
    return all_IDs


def get_unique_IDs(annotation_list):
    return set(get_all_IDs(annotation_list))


def count_unique_IDs(annotation_list):
    return len(get_unique_IDs(annotation_list))


def get_set_names(passages, PMID):
    # get the names of the sets where PMID is (given all passages)
    return [p.infons['set_ID'] for p in passages if p.infons.get('article-id_pmid', '') == PMID]


def pretty_fraction_string(numerator, denominator, num_sig_figs):
    return f"{numerator}/{denominator} ({numerator/denominator:.{num_sig_figs}f})"


def get_all_tokens(list_of_strings):
    all_tokens = []
    for s in list_of_strings:
        # this is our tokenization:
        all_tokens += re.findall(r'\b\w+\b', s.replace("_", " ").lower())
    return all_tokens


def general_stats(collection, writefp = None):

    annotatable_passages, annotation_groups = get_passages_and_annotations(collection)
    (all_annotations, cell_pheno_annotations, cell_hetero_annotations, cell_desc_annotations) = annotation_groups
    
    num_passages = len(annotatable_passages)

    unique_PMIDs = set([p.infons['article-id_pmid'] for p in annotatable_passages])
    
    print("number of passages annotated:", num_passages, file=writefp)
    print("number of PMIDs used:", len(unique_PMIDs), file=writefp)
    duplicate_PMID_counts = [tup for tup in collections.Counter([p.infons['article-id_pmid'] for p in annotatable_passages]).most_common() if tup[1] > 1]
    print(f"Multiple ({len(duplicate_PMID_counts)}) PMIDs with >1 passage:", duplicate_PMID_counts, file=writefp)
    print("approx number of sentences:", sum([p.text.count('.') if '.' in p.text else 1 for p in annotatable_passages]), file=writefp)
    
    passage_lengths = [len(p.text) for p in annotatable_passages]
    print(f"passage length: {np.mean(passage_lengths):.1f} (mean), {np.median(passage_lengths)} (median),",
          f"{np.min(passage_lengths)} (min), {np.max(passage_lengths)} (max)", file=writefp)
    annotations_per_passage = [len(p.annotations) for p in annotatable_passages]
    print(f"number of annotations per passage: {np.mean(annotations_per_passage):.1f} (mean), {np.median(annotations_per_passage)} (median),",
          f"{np.min(annotations_per_passage)} (min), {np.max(annotations_per_passage)} (max)", file=writefp)
    
    passage_years = [PASSAGE_METADATA_DICT.get(p.infons['passage_id'], {'data':{'PUBLICATION_YEAR':"missing"}})['data']['PUBLICATION_YEAR'] for p in annotatable_passages]
    passage_years_frequency = collections.Counter(passage_years)
    print("passage years frequency:", sorted(passage_years_frequency.most_common()), file=writefp)
    
    df = pd.DataFrame(index=["Total", "cell_pheno", "cell_hetero", "cell_desc"])
    
    df['num_annotations'] = [
        len(all_annotations),
        len(cell_pheno_annotations),
        len(cell_hetero_annotations),
        len(cell_desc_annotations)
        ]
    
    all_corpus_tokens= get_all_tokens([p.text for p in annotatable_passages])
    print(f"Total number of corpus tokens: {len(all_corpus_tokens)}, {len(set(all_corpus_tokens))} of which are unique.", file=writefp)
    
    all_annotation_tokens= get_all_tokens([ann.text for ann in all_annotations])
    print(f"Total number of annotation tokens: {len(all_annotation_tokens)}, {len(set(all_annotation_tokens))} of which are unique.", file=writefp)
    
    df['% of annotations'] = round(df['num_annotations'] *100 / df['num_annotations']['Total'], 1)
    df['num annotations per 100 tokens'] = df['num_annotations'] *100 / len(all_corpus_tokens)

    df['num unique mentions'] = [len(set([ann.text for ann in annotations])) for annotations in annotation_groups]
    df['num unique non-exactMatch mentions'] = [len(set([ann.text for ann in annotations if not ID_is_exact(ann.infons['identifier'])])) for annotations in annotation_groups]
    
    df['num unique IDs'] = [count_unique_IDs(annotations) for annotations in annotation_groups]
    
    annotation_lengths = []
    for annotations in annotation_groups:
        ann_text_lengths = [len(ann.text) for ann in annotations]
        annotation_lengths.append(f"{np.mean(ann_text_lengths):.1f} +/- {np.std(ann_text_lengths):.1f}, {np.median(ann_text_lengths)}, " + \
                                  f"({np.min(ann_text_lengths)}, {np.max(ann_text_lengths)})")
    df['Length in char (mean +/- std, median, (min, max))'] = annotation_lengths
    
    df['% of mentions w/o exactMatch'] = [len([ann for ann in annotations if not ID_is_exact(ann.infons['identifier'])])*100//len(annotations) for annotations in annotation_groups]

    # calculate % of cell ellipses
    cell_endings = ["neuron", "cyte", "blast", "clast", "phage", "cell", "precursor", 
                    "neutrophil", "platelet", "glia", "troph", "sperm", "fiber", "phore", "tube"]
    df['approx % of mentions are cell ellipses'] = [len([1 for ann in annotations if \
                                                         not any([cell_ending in ann.text for cell_ending in cell_endings]) and \
                                                             ann.text[:-1].isupper()])*100//len(annotations) for annotations in annotation_groups]

    print(df.to_string(), file=writefp)
    
    print("\n\ntop 100 most frequent mentions:", collections.Counter([ann.text for ann in all_annotations]).most_common(100), file=writefp)
    
    print("\n\ntop 100 most frequent IDs:", collections.Counter([(ID, ID_TO_MENTION_DICT.get(ID, "[no match]")) for ID in get_all_IDs(all_annotations, include_None=True)]).most_common(100),
          file=writefp)
    
    print("\nlist of novel/non-reported (no ID) cell phenotypes:", set([ann.text for ann in cell_pheno_annotations if ID_is_None(ann.infons['identifier'])]), file=writefp)
    print("\nlist of novel/non-reported (no ID) cell_hetero:", set([ann.text for ann in cell_hetero_annotations if ID_is_None(ann.infons['identifier'])]), file=writefp)
    
    # passage stats:
    
    # journals
    # MeSH categories
    # % human, % mouse venn diagram
    # % cell lines
    # passage types
    
    passage_ids = [p.infons['passage_id'] for p in annotatable_passages]
    missing_passage_ids = [p_id for p_id in passage_ids if p_id not in PASSAGE_METADATA_DICT]
    if len(missing_passage_ids) > 0:
        print('\n' + str(len(missing_passage_ids)), "missing passage IDs:", missing_passage_ids, file=writefp)
        raise Exception("missing passage IDs")
    
    # journals
    journal_frequency = collections.Counter([PASSAGE_METADATA_DICT[p_id]['data']['JOURNAL_NAME'] for p_id in passage_ids])
    print('\n' + str(len(journal_frequency)), "unique journals:", journal_frequency.most_common(), file=writefp)
    
    # passage types
    passage_types_frequency = collections.Counter([p.infons.get('section_type', 'None') + '/' + p.infons.get('type', "None") for p in annotatable_passages])
    print('\n' + str(len(passage_types_frequency)), "passage types:", passage_types_frequency.most_common(), file=writefp)
    
    passages_by_type = {"Title": [], "Section title": [], "Abstract": [], "Intro": [], "Results": [], 
                        "Discussion & Conclusion": [], "Table": [],
                        "Table and figure captions": [], "Other [abbreviations, appendix]": []}
    
    for p in annotatable_passages:
        passage_type1 = p.infons.get('section_type', 'None')
        passage_type2 = p.infons.get('type', "None")
        section_with_type = passage_type1 + '/' + passage_type2
        
        if section_with_type in ["TITLE/front", "None/title"]:
            passages_by_type["Title"].append(p)
        elif "title" in section_with_type.lower():
            passages_by_type["Section title"].append(p)
        elif "abstract" in section_with_type.lower():
            passages_by_type["Abstract"].append(p)
        elif passage_type1 == "INTRO":
            passages_by_type["Intro"].append(p)
        elif "results" in section_with_type.lower():
            passages_by_type["Results"].append(p)
        elif passage_type1 in ["DISCUSS", "CONCL"]:
            passages_by_type["Discussion & Conclusion"].append(p)
        elif section_with_type == "TABLE/table":
            passages_by_type["Table"].append(p)
        elif ("table" in section_with_type) or ("fig" in section_with_type):
            passages_by_type["Table and figure captions"].append(p)
        else:
            passages_by_type["Other [abbreviations, appendix]"].append(p)
    
    # we did not use the commented statistics below, but they could still be interesting
    # average annotations by passage type:
    # averages_by_passagetype = {"cell_phenotype":[], "cell_hetero":[], "cell_desc":[]}
    # num_annotations_by_passage_type = pd.DataFrame(index=["cell_phenotype", "cp_rate", "cp_rate2", "cell_hetero",
    #                                                       "ch_rate", "ch_rate2", "cell_desc", "cd_rate", "cd_rate2"])
    # for passage_type, passages_i in passages_by_type.items():
    #     for pt in ["cell_phenotype", "cell_hetero", "cell_desc"]:
    #         averages_by_passagetype[pt].append([])
    #     annotation_counts = {"cell_phenotype":0, "cell_hetero":0, "cell_desc":0}
    #     annotation_averages = {"cell_phenotype":0, "cell_hetero":0, "cell_desc":0}
    #     num_char = 0
    #     for p in passages_i:
    #         annotation_counts_i = {"cell_phenotype":0, "cell_hetero":0, "cell_desc":0}
    #         num_char += len(p.text)
    #         for ann in p.annotations:
    #             annotation_counts[ann.infons['type']] += 1
    #             annotation_counts_i[ann.infons['type']] += 1
    #         for key in annotation_averages.keys():
    #             annotation_averages[key] += annotation_counts_i[key] / len(p.text)
    #             averages_by_passagetype[key][-1].append(annotation_counts_i[key] / len(p.text))
            
    #     num_annotations_by_passage_type[passage_type] = [annotation_counts["cell_phenotype"],
    #                 annotation_counts["cell_phenotype"]*100/num_char, annotation_averages["cell_phenotype"]*100/len(passages_i),
    #                 annotation_counts["cell_hetero"], annotation_counts["cell_hetero"]*100/num_char, annotation_averages["cell_hetero"]*100/len(passages_i),
    #                 annotation_counts["cell_desc"], annotation_counts["cell_desc"]*100/num_char, annotation_averages["cell_desc"]*100/len(passages_i)]
        
    # print(num_annotations_by_passage_type)
    
    num_annotations_by_passage_type = pd.DataFrame(index=["cell_phenotype", "cp-%", "cell_hetero",
                                                          "ch-%", "cell_desc", "cd-%"])
    for passage_type, passages_i in passages_by_type.items():
        annotation_counts = {"cell_phenotype":0, "cell_hetero":0, "cell_desc":0, "total":0}
        for p in passages_i:
            for ann in p.annotations:
                annotation_counts[ann.infons['type']] += 1
                annotation_counts["total"] += 1
            
        num_annotations_by_passage_type[passage_type] = [annotation_counts["cell_phenotype"],
                    annotation_counts["cell_phenotype"]*100/annotation_counts["total"],
                    annotation_counts["cell_hetero"], annotation_counts["cell_hetero"]*100/annotation_counts["total"], 
                    annotation_counts["cell_desc"], annotation_counts["cell_desc"]*100/annotation_counts["total"]]
        
    print('\n' + num_annotations_by_passage_type.T.to_string(), file=writefp)
    
    # annotatable passages with metadata in PASSAGE_METADATA_DICT:
    
    human_MH_passages = [p.infons['passage_id'] for p in annotatable_passages \
                         if 'Human' in PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['MESH_CLUSTERS']]
    mouse_MH_passages = [p.infons['passage_id'] for p in annotatable_passages \
                         if 'Mice' in PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['MESH_CLUSTERS']]
    print(f"\nHuman & mouse: fraction of passages containing the human MeSH tag: {pretty_fraction_string(len(human_MH_passages),len(annotatable_passages),4)};",
          f"mouse MeSH group: {pretty_fraction_string(len(mouse_MH_passages),len(annotatable_passages),4)}, and both mouse and human:",
          f"{pretty_fraction_string(len(set(human_MH_passages).intersection(mouse_MH_passages)),len(annotatable_passages),4)}", file=writefp)
    
    # % cell lines:
    cell_line_MH_passages = [p.infons['passage_id'] for p in annotatable_passages \
                         if 'Cell lines' in PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['MESH_CLUSTERS']]
    print("\ncell line: Fraction of passages containing a cell line MeSH group:", 
          pretty_fraction_string(len(cell_line_MH_passages), len(annotatable_passages), 4), file=writefp)
    
    # MeSH categories:
    MeSH_category_acumulator = []
    for p in annotatable_passages:
        MeSH_category_acumulator += list(PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['MESH_CLUSTERS'])
    MeSH_category_frequencies = collections.Counter(MeSH_category_acumulator)
    print(f"\nMost common MeSH clusters across all ({len(annotatable_passages)}) passages:", 
          [(tup[0], pretty_fraction_string(tup[1],len(annotatable_passages),2)) for tup in MeSH_category_frequencies.most_common()], file=writefp)
    if len(MeSH_category_frequencies) == 21:
        print("All 21 anatomy & disease MeSH categories are in the corpus.", file=writefp)
    else:
        print(f"There are {len(MeSH_category_frequencies)}/21 MeSH categories in the corpus.", file=writefp)
    
    return df
    

def total_stats(collection, writefp = None):
    
    df = general_stats(collection, writefp)
    
    annotatable_passages, annotation_groups = get_passages_and_annotations(collection)
    (all_annotations, cell_pheno_annotations, cell_hetero_annotations, cell_desc_annotations) = annotation_groups
    
    # plot cumulative # of new IDs & mentions over time
    dates = set([p.infons['date-folderName'] for p in annotatable_passages])
    unique_mentions_by_date = {date: set([ann.text for p in annotatable_passages for ann in p.annotations \
                                          if p.infons['date-folderName'] == date]) for date in dates}
    unique_IDs_by_date = {date: get_unique_IDs([ann for p in annotatable_passages for ann in p.annotations \
                                          if p.infons['date-folderName'] == date]) for date in dates}
    
    dates = list(dates)
    dates.sort()
    fig = plt.figure(figsize=(18, 6))
    for i, (dict_i, name) in enumerate([(unique_mentions_by_date, "mentions"), (unique_IDs_by_date, "IDs")]):
        plt.subplot(1,2,i+1)
        cumulative_count = []
        cumulative_set = set()
        for date in dates: # dates are sorted
            cumulative_set.update(dict_i[date])
            cumulative_count.append(len(cumulative_set))
        
        x_axis = np.asarray([0] + [date[6:] for date in dates])
        # x_axis[::2] = "" # skip every other tick
        plt.plot(x_axis, [0] + cumulative_count, marker='o')
        plt.xticks(x_axis[::2], size=10)
        plt.title(f"cumulative number of unique {name}")
    
    # show every other tick
    # plt.setp(plot[0].axes.get_xticklabels(), visible=False)
    # plt.setp(plot[0].axes.get_xticklabels()[::2], visible=True)
    plt.savefig(os.path.join(output_dir, "cumulative_diversity.jpg"))
    plt.show()
    
    return df

    
    
def differential_analysis(collection1, collection2, name1, name2):
    annotatable_passages1, annotation_groups1 = get_passages_and_annotations(collection1)
    (all_annotations1, cell_pheno_annotations1, cell_hetero_annotations1, cell_desc_annotations1) = annotation_groups1
    
    annotatable_passages2, annotation_groups2 = get_passages_and_annotations(collection2)
    (all_annotations2, cell_pheno_annotations2, cell_hetero_annotations2, cell_desc_annotations2) = annotation_groups2
    
    df = pd.DataFrame(index=["1 only", "1 & 2", "2 only"])
    
    # unique mentions (all)
    unique_mentions1 = set([ann.text for ann in all_annotations1])
    unique_mentions2 = set([ann.text for ann in all_annotations2])
    df["num unique mentions"] = [
        len(unique_mentions1 - unique_mentions2),
        len(unique_mentions1.intersection(unique_mentions2)),
        len(unique_mentions2 - unique_mentions1),
        ]
    
    
    # unique IDs (all)
    unique_IDs1 = get_unique_IDs(all_annotations1)
    unique_IDs2 = get_unique_IDs(all_annotations2)
    df["num unique IDs"] = [
        len(unique_IDs1 - unique_IDs2),
        len(unique_IDs1.intersection(unique_IDs2)),
        len(unique_IDs2 - unique_IDs1),
        ]
    
    
    fig = plt.figure(figsize=(9, 4))
    for i, col in enumerate(df):
        plt.subplot(1,2,i+1)
        matplotlib_venn.venn2(subsets=(df[col][0], df[col][2], df[col][1]), set_labels=(name1, name2))
        plt.title(col)
    plt.savefig(os.path.join(output_dir, "differential_uniqueness.jpg"), )
    plt.show()
    
    return df


def tri_differential_analysis(bioc_collections, names):
    assert((len(bioc_collections) == 3) and (len(names) == 3))
    
    unique_mentions_per_name = []
    unique_IDs_per_name = []
    unique_journals_by_name = []
    unique_PMIDs_by_name = []
    for collection, name in zip(bioc_collections, names):
        annotatable_passages, annotation_groups = get_passages_and_annotations(collection)
        (all_annotations, cell_pheno_annotations, cell_hetero_annotations, cell_desc_annotations) = annotation_groups
        
        unique_mentions_per_name.append(set([ann.text for ann in all_annotations]))
        unique_IDs_per_name.append(set(get_all_IDs(all_annotations, include_None=True)))
        
        unique_journals_by_name.append(set([PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['JOURNAL_NAME'] for p in annotatable_passages \
                                            if p.infons['passage_id'] in PASSAGE_METADATA_DICT]))
        unique_PMIDs_by_name.append(set([p.infons['article-id_pmid'] for p in annotatable_passages]))
    
    
    plt.figure(figsize=(6, 9))
    
    assert(('train' in names[0].lower()) and ('devel' in names[1].lower()) and ('test' in names[2].lower()))
    names = ['train', 'val', 'test']
    
    plt.subplot(2,2,1)
    matplotlib_venn.venn3(unique_mentions_per_name, names)
    plt.title("unique mentions")
    
    plt.subplot(2,2,2)
    matplotlib_venn.venn3(unique_IDs_per_name, names)
    plt.title("unique IDs")
    
    plt.subplot(2,2,3)
    matplotlib_venn.venn3(unique_journals_by_name, names)
    plt.title("unique journals")
    
    plt.subplot(2,2,4)
    matplotlib_venn.venn3(unique_PMIDs_by_name, names)
    plt.title("unique pmids")
    
    plt.show()
    


PASSAGE_CATEGORIES_DICT = {
    "intro_paragraph": ['INTRO/paragraph'],
    "results_paragraph": ['RESULTS/paragraph'],
    "fig_or_table_caption": ['FIG/fig_caption', 'TABLE/table_caption'],
    "discussion_paragraph": ['DISCUSS/paragraph'],
    "abstract": ['ABSTRACT/abstract', 'None/abstract'], 
    "abbreviations": ['ABBR/paragraph', 'ABBR/footnote'],
    "title_and_subsection_titles": ['RESULTS/title_2', 'TITLE/front', 'None/title', 'FIG/fig_title_caption',
              'INTRO/title_1', 'INTRO/title_3', 'DISCUSS/title_2', 'SUPPL/title_1',
              'RESULTS/title_3', 'ABSTRACT/abstract_title_1'],
    "table": ['TABLE/table'],
    "other": ['TABLE/table_footnote', 'CONCL/paragraph', 'TABLE/table_foot',
              'SUPPL/paragraph', 'SUPPL/footnote', 'INTRO/title_2', '<RARE>']
}
PASSAGE_TYPES_DICT = {val: key for key, val_list in PASSAGE_CATEGORIES_DICT.items() for val in val_list}

def save_IAA_by_passage_type(collection, save_path):
    # collection = round1_collection
    
    collections_by_annotators = dict()
    # assumption: all annotators in a document have annotated the entire document (which is true for our case)
    for doc in collection.documents:
        for p in doc.passages:
            p.annotations = [ann for ann in p.annotations if ann.infons.get('annotator') not in ["NR", "noam"]] # remove Noam
        annotators_i = frozenset([ann.infons['annotator'] for p in doc.passages for ann in p.annotations if 'annotator' in ann.infons])
        if annotators_i not in collections_by_annotators:
            collections_by_annotators[annotators_i] = bioc.BioCCollection()
        
        collections_by_annotators[annotators_i].add_document(doc)
    
    for annotator_set, collection_i in collections_by_annotators.items():
        passages, _ = get_passages_and_annotations(collection_i)
        
        count_passage_types = True
        if count_passage_types:
            passage_types = []
            for p in passages:
                if p.infons.get('passage_id') in PASSAGE_METADATA_DICT:
                    passage_types.append(PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['PASSAGE_TYPE'])
            print(collections.Counter(passage_types))
        
        for category, types in PASSAGE_CATEGORIES_DICT.items():
            output_doc_i = bioc.BioCDocument()
            output_doc_i.id = f"all {category} passages"
            output_doc_i.passages = [p for p in passages if (p.infons['passage_id'] in PASSAGE_METADATA_DICT) and
                                      (PASSAGE_METADATA_DICT[p.infons['passage_id']]['data']['PASSAGE_TYPE'] in types)]
            output_collection_i = bioc.BioCCollection()
            output_collection_i.add_document(output_doc_i)
            with open(os.path.join(save_path, category + '_' + '-'.join(annotator_set) + '.xml'), 'w', encoding='utf-8') as writefp:
                bioc.dump(output_collection_i, writefp)
    
# save_IAA_by_passage_type(round1_collection, os.path.join(tmp_dirpath, '2025-01-20_IAA_by_passage_type'))

def plot_IAA(filepath):
    # plot IAA over time
    # assumption: folder names start with the date in YYYY-MM-DD format
    times = []
    strict_span_identifier_None = []
    strict_span_identifier_cellPhenotype = []
    for path, dirnames, filenames in os.walk(filepath):
        print(filenames)
        print(dirnames)
        for dirname in dirnames:
            if dirname == '2025-01-09_set11-16_round1 individual- cell type training project 4_v2':
                continue
            times.append(dirname[6:10])
            # print([evaluation.run_a_metric(os.path.join(filepath, dirname, filename)) \
            #         for filename in os.listdir(os.path.join(path,dirname))])
            strict_span_identifier_None.append(np.mean([evaluation.run_a_metric(os.path.join(filepath, dirname, filename), 
                                                         evaluation=('strict', 'span_identifier', None)) \
                                 for filename in os.listdir(os.path.join(path,dirname))]))
            strict_span_identifier_cellPhenotype.append(np.mean([evaluation.run_a_metric(os.path.join(filepath, dirname, filename), 
                                                         evaluation=('strict', 'span_identifier', "cell_phenotype")) \
                                 for filename in os.listdir(os.path.join(path,dirname))]))
            # for filename in os.listdir(os.path.join(path, dirname)):
            #     # print(os.path.isfile(os.path.join(filepath, dirname, filename)))
            #     print(evaluation.run_a_metric(os.path.join(filepath, dirname, filename)))
    plt.plot(times, strict_span_identifier_None, marker='o')
    plt.plot(times, strict_span_identifier_cellPhenotype, marker='o')
    plt.legend(["all", "cell_pheno"])
    plt.ylim([0.5, 1])
    plt.title("strict span_identifier IAA over time")
    plt.xlabel("date")
    plt.xticks(times[::2], size=8)
    # plt.savefig(os.path.join(output_dir, f"IAA_over_time.jpg"))
    plt.show()

# plot_IAA(post_round1_path)

# with open(os.path.join(output_dir, "total_stats.txt"), 'w') as writefp:
#     df1 = total_stats(round3_collection, writefp)

# or, without writing:
# df1 = total_stats(round3_collection)
# df1_T = df1.T

with open(merged_collection_path, 'r', encoding='utf-8') as readfp:
    merged_collection = bioc.load(readfp)

with open("tmp.txt", 'w') as writefp:
    df1 = general_stats(merged_collection, writefp)
df1_T = df1.T


def train_val_test_stats(tdt_split_path, writefp=None, ds_names = ['train', 'dev', 'test']):
    datasets = []
    
    for ds_name in ds_names:
        with open(os.path.join(tdt_split_path, ds_name + '.xml'), 'r', encoding='utf-8') as readfp:
            datasets.append(bioc.load(readfp))
    
    # Capture general_stats output for each collection into a list
    outputs = []
    for collection_i in datasets:
        buf = io.StringIO()
        general_stats(collection_i, writefp=buf)
        output = buf.getvalue().strip()
        stats = output.split('\n\n')
        outputs.append([x.strip() for x in stats])

    # Interleave stats
    for i in range(len(outputs[0])):
        print(f"--train\n{outputs[0][i]}\n--val\n{outputs[1][i]}\n--test\n{outputs[2][i]}\n\n", file=writefp)
    
    tri_differential_analysis(datasets, ds_names)


# train_val_test_stats(train_dev_test_split_path, None) ### uncomment
# with open(os.path.join(train_dev_test_split_path, "Noam_stats.txt"), 'w') as writefp:
#     train_val_test_stats(train_dev_test_split_path, writefp)

# old_collection, recent_collection = split_collection_by_set_num(round3_collection, set_split_num)
# df2 = differential_analysis(old_collection, recent_collection, f"sets 11-{set_split_num-1}", f"sets {set_split_num}-{latest_set_num}")

annotatable_passages, annotation_groups = get_passages_and_annotations(round3_collection)
(all_annotations, cell_pheno_annotations, cell_hetero_annotations, cell_desc_annotations) = annotation_groups

# additional stats that we did not use but might be interesting:
# passage_IDs = [p.infons['passage_id'] for p in annotatable_passages]
# all_raw_annotations = [PASSAGE_METADATA_DICT[ID]['data'].get('RAW_ANNOTATIONS') for ID in passage_IDs]
# pubtator_annotations = [ann for ann_list in all_raw_annotations if ann_list is not None for ann in ann_list if ann[2] == "Gene"]

# num_passages_with_pubtator_gene = 0
# for p in annotatable_passages:
#     ID = p.infons['passage_id']
#     if ID in PASSAGE_METADATA_DICT:
#         annotations_i = PASSAGE_METADATA_DICT[ID]['data'].get("RAW_ANNOTATIONS")
#         if annotations_i is not None:
#             for ann in annotations_i:
#                 if ann[2]=="Gene":
#                     all_cell_annotations = 
#             if any([ann[2]=="Gene" for ann in annotations_i]):
#                 num_passages_with_pubtator_gene += 1
# print(num_passages_with_pubtator_gene / len(passage_IDs))
