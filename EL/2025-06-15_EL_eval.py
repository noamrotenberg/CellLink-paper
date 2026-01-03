# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:46:43 2025

@author: rotenbergnh

"""

import bioc
import re
import json

input_path = "../../model_outputs/EL_test_output.xml"

model_names = ["SapBERT", "MedCPT-Query", "OpenAI-txt-emb-3-L", "GPT-4.1_Agent"]

CL_names_filename = r"C:\Users\rotenbergnh\OneDrive - National Institutes of Health\cell type NLP extraction\2024-12-30_annotation_files\validation\cl.json"
with open(CL_names_filename, "r") as file:
    cell_types_dict = json.load(file)


with open(input_path, 'r', encoding='utf-8') as readfp:
    input_collection = bioc.load(readfp)


def exactIDsOnly_iterator(bioc_collection):
    # iterate only over annotations that have a single identifier
    # skip coordination ellipses; skip related annotations iff more than 1 ID
    for doc in bioc_collection.documents:
        for p in doc.passages:
            for ann in p.annotations:
                identifier_i = ann.infons['identifier']
                if (len(identifier_i) != 0) and ("none" not in identifier_i.lower()) and \
                    (';' not in identifier_i) and (',' not in identifier_i) and ("related" not in identifier_i):
                    identifier_i = identifier_i.replace("(skos:exact)", "")
                    if identifier_i not in cell_types_dict:
                        print(ann.infons['identifier'], identifier_i)
                    yield (p, ann, (identifier_i, ))

# we didn't use this but it could be interesting:
# def singleIDsOnly_iterator(bioc_collection):
#     # iterate only over annotations that have a single identifier
#     # skip coordination ellipses; skip related annotations iff more than 1 ID
#     for doc in bioc_collection.documents:
#         for p in doc.passages:
#             for ann in p.annotations:
#                 identifier_i = ann.infons['identifier']
#                 if (len(identifier_i) != 0) and ("none" not in identifier_i.lower()) and \
#                     (';' not in identifier_i) and (',' not in identifier_i):
#                     identifier_i = identifier_i.replace("(skos:exact)", "").replace("(skos:related)", "")
#                     if identifier_i not in cell_types_dict:
#                         print(ann.infons['identifier'], identifier_i)
#                     yield (p, ann, (identifier_i, ))

def allLabels_iterator(bioc_collection):
    # extract all IDs from all cell_pheno and cell_hetero annotations (including if empty)
    for doc in bioc_collection.documents:
        for p in doc.passages:
            for ann in p.annotations:
                if ann.infons['type'] in ['cell_phenotype', 'cell_hetero']:
                    identifier_i = ann.infons['identifier']
                    identifier_i = identifier_i.replace("(skos:related)", "")
                    identifier_i = identifier_i.replace("(skos:exact)", "")
                    identifier_i = identifier_i.replace("None", "")
                    
                    all_IDs = re.split(',|;', identifier_i)
                    all_IDs = list(filter(lambda x: x not in ['-', ''], all_IDs))
                    
                    if any([ID_i not in cell_types_dict for ID_i in all_IDs]):
                        print(ann.infons['identifier'], all_IDs)
                    
                    yield (p, ann, all_IDs)
                    

for entity_types in [["cell_phenotype"], ["cell_hetero"], ["cell_phenotype", "cell_hetero"]]: # 
    print(entity_types)
    for iterator_name, iterator in [("exactIDsOnly_iterator", exactIDsOnly_iterator), 
                                    # ("singleIDsOnly_iterator", singleIDsOnly_iterator), 
                                    ("allLabels_iterator", allLabels_iterator)]:
        print(iterator_name)
        for model_name in model_names:
            ref_tuples = set()
            top1_pred_tuples = set()
            top5_pred_tuples = set()
            top10_pred_tuples = set()
            for passage, ann, ref_IDs_i in iterator(input_collection):
                if ann.infons['type'] in entity_types:
                    if "none" in ann.infons['identifier'].lower():
                        pass
                    ref_tuples.update([(passage.infons['passage_id'], ann.infons['type'], ID_j) for ID_j in ref_IDs_i])
                    top10_tuples_i = [(passage.infons['passage_id'], ann.infons['type'], ann.infons[f"{model_name}_id_{i}"]) for i in range(10)]
                    # filter out any empty IDs
                    top10_tuples_i = [tup for tup in top10_tuples_i if tup[2].lower() not in ["", "-", "none"]]
                    top1_pred_tuples.add(top10_tuples_i[0])
                    top5_pred_tuples.update(top10_tuples_i[:5])
                    top10_pred_tuples.update(top10_tuples_i)
            
            
            
            for k, set_i in [("1", top1_pred_tuples), ("5", top5_pred_tuples), ("10", top10_pred_tuples)]:
                precision_numerator = len(ref_tuples.intersection(set_i))
                precision_denominator = len(set_i)
                precision = precision_numerator/precision_denominator
                recall_numerator = len(ref_tuples.intersection(set_i))
                recall_denominator = len(ref_tuples)
                recall = recall_numerator/recall_denominator
                F1 = 2*precision*recall / (precision + recall)
                print(f"{model_name} top-{k} results: precision {precision:.3f} ({precision_numerator}/{precision_denominator}),",
                      f"recall {recall:.3f} ({recall_numerator}/{recall_denominator}), F1 {F1:.3f}")
        print()



# analyze SapBERT confidence
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

def simpleLabels_NoCoordEllipses_iterator(bioc_collection):
    # iterate only over all cell_pheno and cell_hetero, except for coordination ellipeses
    for doc in bioc_collection.documents:
        for p in doc.passages:
            for ann in p.annotations:
                if ann.infons['type'] in ['cell_phenotype', 'cell_hetero']:
                    identifier_i = ann.infons['identifier']
                    if ';' not in identifier_i:
                        if "(skos:related)" in identifier_i:
                            linkage_type = "related"
                            identifier_i = identifier_i.replace("(skos:related)", "")
                        elif "(skos:exact)" in identifier_i:
                            linkage_type = "exact"
                            identifier_i = identifier_i.replace("(skos:exact)", "")
                        elif identifier_i == "None":
                            linkage_type = "none"
                            identifier_i = ""
                        else:
                            raise Exception(identifier_i)
                        
                        all_IDs = re.split(',', identifier_i)
                        all_IDs = list(filter(lambda x: x not in ['-', ''], all_IDs))
                        
                        if any([ID_i not in cell_types_dict for ID_i in all_IDs]):
                            print(ann.infons['identifier'], all_IDs)
                        
                        SapBERT_id1 = ann.infons['SapBERT_id_0']
                        SapBERT_top1_correct = SapBERT_id1 in all_IDs
                        SapBERT_confidence = float(ann.infons['SapBERT_identifier_score_0'])
                        
                        entity_type = ann.infons['type']
                        
                        yield (ann, entity_type, all_IDs, linkage_type, SapBERT_top1_correct, SapBERT_confidence)


confidence_df = pd.DataFrame(simpleLabels_NoCoordEllipses_iterator(input_collection), 
                             columns=["Annotation", "entity_type", "identifiers", "linkage_type", "SapBERT_correct", "SapBERT_confidence"])
confidence_df['identifier'] = [ann.infons['identifier'] for ann in confidence_df["Annotation"]]
confidence_df['in_CL'] = confidence_df['linkage_type'] == 'exact'
confidence_df["annotation_text"] = [ann.text for ann in confidence_df["Annotation"]]

linkage_types = ["exact", "related", "none"]

# histogram of SapBERT confidence for exact, related, and None types
for linkage_type in linkage_types:
    plt.hist(confidence_df[confidence_df["linkage_type"] == linkage_type]["SapBERT_confidence"], 
             bins= np.linspace(0, 1, 21), 
             edgecolor='black', alpha=0.5, label=linkage_type)
plt.legend()
plt.title("SapBERT confidence by linkage type")
plt.xlabel("SapBERT cosine similarity")
plt.ylabel("number of mentions")
plt.show()
plt.figure()


if True:
    # plots for "ID in CL AND SapBERT prediction correct" vs. "ID not in CL OR SapBERT prediction wrong"
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(confidence_df["in_CL"] & confidence_df["SapBERT_correct"], confidence_df["SapBERT_confidence"])
    
    # Create the ROC curve plot
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity)')
    
    # thresholds[thresholds > 0.5][0]
    
    auc = sklearn.metrics.auc(fpr, tpr)
    # plt.text(0.6, 0.2, f'AUC = {auc:.2f}', fontsize=12)
    # plt.title("ROC curve of 'in ontology AND SAPBERT prediction correct' vs.\n 'not in ontology OR SAPBERT prediction wrong'")
    plt.title("ROC curve of SapBERT successful matching based on confidence")
    plt.show()
    plt.figure()
    
    negatives = confidence_df[~confidence_df["in_CL"] | ~confidence_df["SapBERT_correct"]].sort_values("SapBERT_confidence", ascending=False)
    print("\n\nhighest confidence terms not in CL [~false positives]:", negatives.head(10)['annotation_text'], sep='\n')
    print("\nlowest confidence terms not in CL [~true negatives]:", negatives.tail(10)['annotation_text'], sep='\n')
    
    positives = confidence_df[confidence_df["in_CL"] & confidence_df["SapBERT_correct"]].sort_values("SapBERT_confidence")
    print("\n\nlowest confidence terms in CL [~false negatives]:", positives.head(10)['annotation_text'], sep='\n')
    print("\nhighest confidence terms in CL [~true positives]:", positives.tail(10)['annotation_text'], sep='\n')
    
    plt.hist(positives["SapBERT_confidence"], bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.5)
    plt.hist(negatives["SapBERT_confidence"], bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.5)
    plt.legend(["entity in ontology AND SAPBERT prediction correct", "entity not in ontology OR SAPBERT prediction wrong"])
    plt.title("SAPBERT success in EL")
    plt.ylabel("number of unique mentions")
    plt.xlabel("SAPBERT confidence/cosine similarity")
    plt.show()


if True:
    for entity_types in [["cell_phenotype"], ["cell_hetero"], ["cell_phenotype", "cell_hetero"]]:
        confidence_df_i = confidence_df[[x in entity_types for x in confidence_df["entity_type"]]]
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(confidence_df_i["linkage_type"] == "exact", confidence_df_i["SapBERT_confidence"])
        auc = sklearn.metrics.auc(fpr, tpr)
        print(entity_types, "exact vs related & no ID AUROC:", auc)
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(confidence_df_i["linkage_type"] != "none", confidence_df_i["SapBERT_confidence"])
        auc = sklearn.metrics.auc(fpr, tpr)
        print(entity_types, "exact & related vs. no ID AUROC:", auc)
        
        


