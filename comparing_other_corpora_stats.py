# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:11:12 2025

@author: rotenbergnh
"""

import bioc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re
import os
from nltk.tokenize import sent_tokenize


our_data_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files/total_results.xml"
AnatEM_path = "NER-exp1-corpusComparisons/filtered_corpora/AnatEM"
BioID_path = "NER-exp1-corpusComparisons/filtered_corpora/BioID"
CRAFT_path = "NER-exp1-corpusComparisons/filtered_corpora/CRAFT"
JNLPBA_path = "NER-exp1-corpusComparisons/filtered_corpora/JNLPBA"


sourceData_NLP_path = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-10-23_SourceData-NLP_dataset\2024-10-23_cell_types_only\all.xml"


datasets = [("NLM CellLink", our_data_path), ("AnatEM", AnatEM_path), ("BioID", BioID_path), ("CRAFT", CRAFT_path), ("JNLPBA", JNLPBA_path), ("sourceData_NLP", sourceData_NLP_path)]
ds_names = [ds[0] for ds in datasets]

def get_unique_IDs(annotation_list):
    unique_identifiers = set()
    for ann in annotation_list:
        identifier = ann.infons.get('identifier')
        if identifier in [None, "None"]:
            continue
            # unique_identifiers.add(None)
        else:
            # assume text
            # remove skos labels
            identifier = identifier.replace("(skos:related)", "")
            identifier = identifier.replace("(skos:exact)", "")
            unique_identifiers.update(re.split(',|;', identifier))
    return unique_identifiers

def count_unique_IDs(annotation_list):
    return len(get_unique_IDs(annotation_list))


def get_passages_from_path(path):
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as readfp:
            bioc_collection_i = bioc.load(readfp)
        return [p for doc in bioc_collection_i.documents for p in doc.passages]
    elif os.path.isdir(path):
        passages = []
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath_i = os.path.join(dirpath, filename)
                if filepath_i.endswith('.xml'):
                    with open(filepath_i, 'r', encoding='utf-8') as readfp:
                        bioc_collection_i = bioc.load(readfp)
                        passages += [p for doc in bioc_collection_i.documents for p in doc.passages]
    return passages


passage_groups = dict()
annotation_groups = dict()
annotation_texts = dict()
annotation_counts_decrFreq = dict()
for name, path_i in datasets:
    print("loading", name)
    passages_i = get_passages_from_path(path_i)
    passages_i = [p for p in passages_i if p.infons.get("annotatable") not in ["False", "no"]]
    passage_groups[name] = passages_i
    annotation_groups[name] = [ann for p in passage_groups[name] for ann in p.annotations]
    annotation_texts[name] = [ann.text for ann in annotation_groups[name]]
    annotation_counts_decrFreq[name] = [x[1] for x in collections.Counter(annotation_texts[name]).most_common()]

print("finished loading")

# add 2 other categories:
for new_ds_name in ["all others", "all other manually annotated corpora"]:
    ds_names += [new_ds_name]
    if new_ds_name == "all other manually annotated corpora":
        passage_groups[new_ds_name] = sum([passage_groups[name] for name in ["AnatEM", "BioID", "CRAFT", "JNLPBA"]], [])
    else:
        passage_groups["all others"] = sum([passage_groups[name] for name in ds_names[1:-1]], [])
    annotation_groups[new_ds_name] = sum([p.annotations for p in passage_groups[new_ds_name]], [])
    annotation_texts[new_ds_name] = [ann.text for ann in annotation_groups[new_ds_name]]
    annotation_counts_decrFreq[new_ds_name] = [x[1] for x in collections.Counter(annotation_texts[new_ds_name]).most_common()]


def get_all_tokens(list_of_strings):
    all_tokens = []
    for s in list_of_strings:
        all_tokens += re.findall(r'\b\w+\b', s.replace("_", " ").lower())
    return all_tokens


# total stats
stats = dict()
for name in ds_names:
    
    stats[name] = dict()

    # unique_PMIDs = set([p.infons['article-id_pmid'] for p in passages])
    
    stats[name]["num passages"] = len(passage_groups[name])
    stats[name]["num passages w/ >0 annotations"] = len([1 for p in passage_groups[name] if len(p.annotations) > 0])
    stats[name]["num sentences"] = sum([len(sent_tokenize(p.text)) for p in passage_groups[name]])
    
    # print("number of PMIDs used:", len(unique_PMIDs), file=writefp)
    # duplicate_PMID_counts = [tup for tup in collections.Counter([p.infons['article-id_pmid'] for p in annotatable_passages]).most_common() if tup[1] > 1]
    # print(f"Multiple ({len(duplicate_PMID_counts)}) passages found from the same PMID:", duplicate_PMID_counts, file=writefp)
    
    stats[name]["num tokens"] = len(get_all_tokens([p.text for p in passage_groups[name]]))
    
    stats[name]["num annotations"] = len(annotation_groups[name])
    
    stats[name]["num annotations per 100 tokens"] = stats[name]["num annotations"] * 100 / stats[name]["num tokens"]
    
    anns_per_passage = [len(p.annotations) for p in passage_groups[name]]
    stats[name]["num annotations/passage (mean, median, min, max)"] = \
        f"{np.mean(anns_per_passage):.1f}, {np.median(anns_per_passage)}, {np.min(anns_per_passage)}, {np.max(anns_per_passage)}"
    
    
    # average annotation length
    ann_lengths = [len(ann.text) for ann in annotation_groups[name]]
    stats[name]["ann length (mean +/- std, median, min, max)"] = \
        f"{np.mean(ann_lengths):.1f} +/- {np.std(ann_lengths):.1f}, {np.median(ann_lengths)}, {np.min(ann_lengths)}, {np.max(ann_lengths)}"


    stats[name]['num unique mentions'] = len(set([ann.text for ann in annotation_groups[name]]))
    stats[name]['unique/total mentions'] = stats[name]['num unique mentions']/stats[name]["num annotations"]
    
    stats[name]["num unique tokens in annotations"] = len(set(get_all_tokens([ann.text for ann in annotation_groups[name]])))
    
    unique_IDs = get_unique_IDs(annotation_groups[name])
    if all([ID in [None, "None"] for ID in unique_IDs]):
        stats[name]['num unique ID'] = 0
        stats[name]['num unique no-ID & related (novel/unreported) mentions'] = 0
    else:
        stats[name]['num unique ID'] = len(unique_IDs)
        if "all other" in name:
            stats[name]['num unique no-ID & related (novel/unreported) mentions'] = "XX-manual"
        # elif (None in unique_IDs) or ("None" in unique_IDs):
        else:
            stats[name]['num unique no-ID & related (novel/unreported) mentions'] = len(set([ann.text for ann in annotation_groups[name] if (ann.infons.get('identifier') in [None, "None"]) \
                                                                                             or ("rel" in ann.infons.get("identifier","")) or (',' in ann.infons.get("identifier", ""))]))
        # else:
        #     stats[name]['num unique no-ID & related (novel/unreported) mentions'] = 0
                


df = pd.DataFrame(stats)

print("stats done")

print(df.to_string())

# plots

# plt.figure()
# for name in ds_names:
#     plt.plot(np.linspace(0, 1, num=len(annotation_groups[name])), 1 / np.repeat(annotation_counts_decrFreq[name],annotation_counts_decrFreq[name])[::-1])
# plt.legend(ds_names)
# plt.title("Distribution of mention frequencies")
# plt.xlabel("Frequency ranking (per-mention)")
# plt.ylabel("1 / frequency")
# plt.show()

# # plt.figure()
# # for name in ds_names:
# #     plt.plot(np.linspace(0, 1, num=len(annotation_groups[name])), 1 / np.repeat(annotation_counts_decrFreq[name],annotation_counts_decrFreq[name]))
# # plt.legend(ds_names)
# # plt.title("Reciprocal mentions frequency (arranged in decreasing frequency)")
# # plt.xlabel("scaled frequency index (per-mention)")
# # plt.ylabel("1 / frequency")
# # # plt.annotate("This plot says ~25% of anns\n appear once,\n ~8% twice, etc", (0, .25))
# # plt.show()

# cutoff = 8
# plt.figure()
# for name in ds_names:
#     plt.plot( np.arange(cutoff)+1, [np.sum(np.asarray(annotation_counts_decrFreq[name]) <= i) / len(annotation_counts_decrFreq[name]) for i in range(len(annotation_counts_decrFreq[name]))][1:cutoff+1] )
# plt.legend(ds_names)
# plt.title("Cumulative count of unique mentions*")
# plt.xlabel("Frequency")
# plt.ylabel("Fraction of unique mentions with count <= x")
# plt.ylim(0, 1)
# plt.show()
# # * cut-off at {cutoff}


# plots that are interesting but we didn't keep:
"""
plt.figure()
for name in ds_names:
    cumulative_counts_i = np.cumsum(annotation_counts_decrFreq[name])
    plt.plot(np.linspace(0, 1, len(annotation_counts_decrFreq[name])), cumulative_counts_i / len(annotation_groups[name]))
plt.legend(ds_names)
plt.title("unique mentions Q-Q plot (arranged in decreasing frequency)")
plt.xlabel("fraction of unique mentions")
plt.ylabel("cumulative fraction unique mentions (quartile)")
plt.show()


for name in ds_names[:-1]:
    plt.hist(annotation_counts_decrFreq[name], alpha=0.5)
plt.legend(ds_names[:-1])
plt.title("unique mentions histogram")
plt.xlabel("frequency of a mention")
plt.ylabel("# of mentions with this frequency")
plt.show()


for name in ds_names[:-1]:
    plt.hist(np.log(annotation_counts_decrFreq[name]), alpha=0.5)
plt.legend(ds_names[:-1])
plt.title("log of unique mentions histogram")
plt.xlabel("log frequency of a mention")
plt.ylabel("# of mentions with this log frequency")
plt.show()

for name in ds_names[:-1]:
    plt.hist(np.log(annotation_counts_decrFreq[name]), alpha=0.5, log=True)
plt.legend(ds_names[:-1])
plt.title("log-log of unique mentions histogram")
plt.xlabel("log frequency of a mention")
plt.ylabel("log # of mentions with this log frequency")
plt.show()

for name in ds_names[:-1]:
    plt.hist(annotation_counts_decrFreq[name], alpha=0.5, log=True)
plt.legend(ds_names[:-1])
plt.title("unique mentions log histogram")
plt.ylabel("log # of mentions with this frequency")
plt.show()

for name in ds_names:
    plt.plot(annotation_counts_decrFreq[name])
plt.legend(ds_names)
plt.title("unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("frequency index (per-unique mention)")
plt.ylabel("frequency of the xth most frequent mentions")
plt.show()

for name in ds_names:
    plt.plot(np.repeat(annotation_counts_decrFreq[name], annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("frequency index (per-mention)")
plt.ylabel("frequency")
plt.show()

for name in ds_names:
    plt.plot(np.log(annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("unique mentions log frequency (arranged in decreasing frequency)")
plt.xlabel("frequency index (per-unique mention)")
plt.ylabel("log frequency of the xth most frequent mentions")
plt.show()

for name in ds_names:
    plt.plot(np.log(np.arange(len(annotation_counts_decrFreq[name]))), np.log(annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("log-log unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("log of frequency index (per-unique mention)")
plt.ylabel("log frequency of the xth most frequent mentions")
plt.show()

for name in ds_names:
    plt.plot(np.log(np.arange(len(annotation_texts[name]))), np.log(np.repeat(annotation_counts_decrFreq[name],annotation_counts_decrFreq[name])))
plt.legend(ds_names)
plt.title("log-log mentions frequency (arranged in decreasing frequency)")
plt.xlabel("log of frequency index (per-mention)")
plt.ylabel("log frequency")
plt.show()

for name in ds_names:
    plt.plot((np.arange(len(annotation_counts_decrFreq[name]))), 1 / np.asarray(annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("reciprocal unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("frequency index (per-unique mention)")
plt.ylabel("1 / frequency of the xth most frequent mentions")
plt.show()

for name in ds_names:
    plt.plot(np.linspace(0,1, len(annotation_counts_decrFreq[name])), 1 / np.asarray(annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("reciprocal unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("scaled frequency index (per-unique mention)")
plt.ylabel("1 / frequency of the xth most frequent mentions")
plt.annotate("This plot says ~75% of unique mentions\n appear once (6000/8000)", (0.5, .2))
plt.show()

for name in ds_names:
    plt.plot(np.linspace(0, 1, num=len(annotation_groups[name])), 1 / np.repeat(annotation_counts_decrFreq[name],annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("reciprocal mentions frequency (arranged in decreasing frequency)")
plt.xlabel("scaled frequency index (per-mention)")
plt.ylabel("1 / frequency")
plt.annotate("This plot says ~25% of anns\n appear once,\n ~8% twice, etc", (0, .25))
plt.show()

for name in ds_names:
    plt.plot(np.linspace(0,1, len(annotation_counts_decrFreq[name])), np.log(annotation_counts_decrFreq[name]))
plt.legend(ds_names)
plt.title("log unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("scaled frequency index (per unique mention)")
plt.ylabel("log frequency of the xth most frequent mentions")
plt.show()

for name in ds_names:
    plt.plot(np.log(np.arange(len(annotation_counts_decrFreq[name])) / len(annotation_counts_decrFreq[name])), np.log(np.asarray(annotation_counts_decrFreq[name])))
plt.legend(ds_names)
plt.title("log-log unique mentions frequency (arranged in decreasing frequency)")
plt.xlabel("log of scaled frequency index (per-unique mention)")
plt.ylabel("log frequency")
plt.show()

for name in ds_names:
    plt.plot( [np.sum(np.asarray(annotation_counts_decrFreq[name]) <= i) / len(annotation_counts_decrFreq[name]) for i in range(len(annotation_counts_decrFreq[name]))] )
plt.legend(ds_names)
plt.title("unique mentions cumulative count frequencies")
plt.xlabel("unique mentions count")
plt.ylabel("fraction of unique mentions that have count <= x")
plt.show()

"""


print("done")