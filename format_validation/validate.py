import datetime
import os
import sys
import json
import collections
import string
import re

from bioc import biocxml

import BioCXMLUtils


def validate_identifier_list(annotation, identifier_list, cell_types_dict):
    errors = []
    if identifier_list == "None":
        return errors
    ellipsis_parts = identifier_list.split(";")
    for ellipsis_part in identifier_list.split(";"):
        if ellipsis_part == "-" and len(ellipsis_parts) > 1:
            # "-" is allowed as an identifier only for coordination ellipses
            continue
        end_paren_index = ellipsis_part.find(")")
        accession_index = end_paren_index + 1 if end_paren_index >= 0 else -1
        qualifier = ellipsis_part[:accession_index] if accession_index >= 0 else None
        if not qualifier in {"(skos:exact)", "(skos:related)"}:
            errors.append(("ERROR", "Identifier part should begin with (skos:exact) or (skos:related)", "Identifier part should begin with (skos:exact) or (skos:related): \"{}\"".format(ellipsis_part)))
        lookup_ellipsis_part = ellipsis_part[accession_index:] if accession_index >= 0 else ellipsis_part
        lookup_identifiers = lookup_ellipsis_part.split(",")
        if len(lookup_identifiers) > 1 and qualifier != "(skos:related)":
            errors.append(("ERROR", "Comma-separated identifiers are only allowed with \"(skos:related)\"", "Comma-separated identifiers are only allowed with \"(skos:related)\": \"{}\"".format(ellipsis_part)))
        for lookup_identifier in lookup_identifiers:
            if not lookup_identifier in cell_types_dict:
                # I prepended a 'c' to some annotations as a flag to double check them
                if lookup_identifier.startswith('c'):
                    if lookup_identifier[1:] in cell_types_dict:
                        errors.append(("ERROR", "'c' check flag", f"'c' check flag found ({lookup_identifier} = {cell_types_dict[lookup_identifier[1:]]['name']})"))
                    else:
                        errors.append(("ERROR", "'c' check flag", "'c' check flag found"))
                else:
                    errors.append(("ERROR", "Identifier not found in cell types dictionary", "Identifier not found in cell types dictionary: \"{}\"".format(lookup_identifier)))
    
    return errors

def process_file(input_filename, standardizer, cell_types_dict):
    if not input_filename.endswith(".xml"):
        return [("WARN", input_filename, None, "File does not end in \".xml\"", "File \"{}\" ignored: does not end in \".xml\"".format(input_filename))]
    errors = list()
    with open(input_filename, "r", encoding='utf-8') as fp:
        collection = biocxml.load(fp)
    for document in collection.documents:
        for passage_index, passage in enumerate(document.passages):
            passage_id = passage.infons["passage_id"] if "passage_id" in passage.infons else "#{}".format(passage_index)
            
            if not "annotatable" in passage.infons:
                errors.append(("ERROR", input_filename, passage_id, "Passage does not contain \"annotatable\" infon",  "Passage does not contain \"annotatable\" infon"))
                continue
            elif not passage.infons["annotatable"] in {"no", "yes"}:
                errors.append(("ERROR", input_filename, passage_id, "Passage \"annotatable\" infon value is unknown",  "Passage \"annotatable\" infon value is unknown: {}".format(passage.infons["annotatable"])))
                continue
            if passage.infons["annotatable"] == "no":
                continue
            if not "passage_id" in passage.infons:
                errors.append(("ERROR", input_filename, passage_id, "Passage does not contain \"passage_id\" infon",  "Passage does not contain \"passage_id\" infon"))
                continue
            pmid = passage.infons.get("article-id_pmid")
            if pmid is None:
                errors.append(("ERROR", input_filename, passage_id, "Passage does not contain \"article-id_pmid\" infon",  "Passage does not contain \"article-id_pmid\" infon"))
                continue
            elif "passage_id" in passage.infons and not passage.infons["passage_id"].startswith(pmid + "_"):
                # Check that the passage_id starts iwth the pmid
                errors.append(("ERROR", input_filename, passage_id, "Passage ID does not start with PMID",  "Passage ID (\"{}\") does not start with PMID (\"{}\")".format(passage.infons["passage_id"], pmid)))
                continue
            pmc = passage.infons.get("article-id_pmc")
            pmid2, pmc2 = standardizer.standardize(None, pmid, pmc)
            if pmid != pmid2 or (not pmc is None and pmc != pmc2 and ("PMC" + pmc) != pmc2):
                errors.append(("ERROR", input_filename, passage_id, "Passage identifiers do not match docids data", "Passage identifiers do not match docids data: ({}, {}) != ({}, {})".format(pmid, pmc, pmid2, pmc2)))
                continue
            
            # check for duplicate annotations
            annotations_p_uniqueness_set = set()
            for ann in passage.annotations:
                uniqueness_set_i = (tuple(ann.locations), ann.text, ann.infons['identifier'], ann.infons['type'])
                if uniqueness_set_i not in annotations_p_uniqueness_set:
                    annotations_p_uniqueness_set.add(uniqueness_set_i)
                else:
                    errors.append(("ERROR", input_filename, passage_id, "Duplicate annotation", "Annotation #{} text ({}) has a duplicate in the same place".format(ann.id, ann.text)))
            
            for i, annotation in enumerate(passage.annotations):
                # if annotation.infons['type'] == 'exact_cell_desc':
                #     annotation.infons['type'] = "cell_desc"
                if not "type" in annotation.infons:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation does not contain \"type\" infon",  "Annotation does not contain \"type\" infon"))
                elif not annotation.infons["type"] in {"cell_phenotype", "cell_hetero", "cell_desc"}:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation \"type\" infon value is unknown",  "Annotation \"type\" infon value is unknown: {}".format(annotation.infons["type"])))
                if not "identifier" in annotation.infons:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation does not contain \"identifier\" infon",  "Annotation does not contain \"identifier\" infon"))
                else:
                    identifier_errors = validate_identifier_list(annotation, annotation.infons["identifier"], cell_types_dict)
                    for error_type, general_error, specific_error in identifier_errors:
                        errors.append(("ERROR", input_filename, passage_id, general_error, "Annotation #{} ({}): {}".format(annotation.id, annotation.text, specific_error)))
                if "type" in annotation.infons and "identifier" in annotation.infons:
                    # Check that desc does not have an identifier
                    if annotation.infons["type"] == "cell_desc" and annotation.infons["identifier"] != "None":
                        errors.append(("ERROR", input_filename, passage_id, "Annotation with type \"cell_desc\" lists an identifier", "Annotation #{} ({}) with type \"cell_desc\" lists an identifier: {}".format(annotation.id, annotation.text, annotation.infons["identifier"])))
                if len(annotation.locations) != 1:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation does not have exactly 1 location", "Annotation #{} ({}) should have exactly 1 location: {}".format(annotation.id, annotation.text, len(annotation.locations))))
                if len(annotation.locations) == 0:
                    continue
                # Validate the location is within the passage
                if annotation.locations[0].offset < passage.offset or annotation.locations[0].offset + annotation.locations[0].length > passage.offset + len(passage.text):
                    errors.append(("ERROR", input_filename, passage_id, "Annotation span is outside of passage", "Annotation #{} ({}) span is outside of passage".format(annotation.id, annotation.text)))
                    continue
                # Validate the text matches the span
                start = annotation.locations[0].offset - passage.offset
                end = start + annotation.locations[0].length
                if passage.text[start:end] != annotation.text:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation text does not match passage", "Annotation #{} text does not match: \"{}\" != \"{}\"".format(annotation.id, passage.text[start:end], annotation.text)))
                    continue
                
                # # check for accidental span endpoint issues (warning, not error)
                # # check if first/last char of an ann is space/punctuation:
                # flagged_characters = string.whitespace + string.punctuation.replace("+","").replace("-","")
                # if (annotation.text[0] in flagged_characters) or (annotation.text[-1] in flagged_characters):
                #     errors.append(("WARNING", input_filename, passage_id, "(warn) Punctuation or whitespace at endpoint(s) of annotation text", "(warn) Annotation #{} text ({}) may contain unnecessary punctuation or whitespace".format(annotation.id, annotation.text)))
                
                # # check if first/last word of ann is cut off
                # passage_text_around_ann = passage.text[max(0, annotation.locations[0].offset - passage.offset - 1) :
                #                                        min(len(passage.text), annotation.locations[0].end - passage.offset + 1)]
                # split_passage = re.split(r'[^\w]+', passage_text_around_ann)
                # split_annotation = re.split(r'[^\w]+', annotation.text)
                # all values in split_annotation should be in split_passage iff the word is separated by punctuation or whitespace
                # if (split_annotation[0] not in split_passage) or (split_annotation[-1] not in split_passage):
                #     errors.append(("WARNING", input_filename, passage_id, "(warn) Annotation may be cut off.", "(warn) Annotation #{} text ({}) may be cut off".format(annotation.id, annotation.text)))
                
                # # check for "primary" in the annotation text (warning, not error)
                # if "primary" in annotation.text.lower():
                #     errors.append(("WARNING", input_filename, passage_id, "(warn) Annotation text contains the word 'primary'", "(warn) Annotation #{} text contains the word 'primary'".format(annotation.id)))
                
                
                
                # check for overlapping annotations
                for ann2 in passage.annotations[i+1:]:
                    # skip if the location, ID, type are exactly the same
                    if not ((annotation.locations[0].offset == ann2.locations[0].offset) and (annotation.text == ann2.text) \
                            and (annotation.infons['identifier'] == ann2.infons['identifier']) and (annotation.infons['type'] == ann2.infons['type'])):
                        # XNOR (exclusive not OR): cell_desc can't overlap with cell_desc; not cell_desc can't overlap with not cell_desc
                        # (not cell_desc == cell_phenotype or cell_hetero)
                        if not ((annotation.infons['type'] == 'cell_desc') ^ (ann2.infons['type'] == 'cell_desc')):
                            ann1_offset = annotation.locations[0].offset
                            ann2_offset = ann2.locations[0].offset
                            if ((ann1_offset >= ann2_offset) and (ann1_offset <= ann2.locations[0].end)) or \
                               ((ann2_offset >= ann1_offset) and (ann2_offset <= annotation.locations[0].end)):
                                   errors.append(("ERROR", input_filename, passage_id, "Overlapping annotations", "Annotation #{} ({}) overlaps with Annotation #{} ({})".format(annotation.id, annotation.text, ann2.id, ann2.text)))
                        else:
                            # overlapping cell_desc & something else. make sure the span isn't exactly the same
                            if annotation.text == ann2.text:
                                errors.append(("WARNING", input_filename, passage_id, "identical annotations", "cell_desc annotation overlaps exactly with another annotation: ('{}' and '{}')".format(annotation.text, ann2.text)))
                
                # 4/23/25 flags:
                # for flag in lowercase_TC_flags:
                #     if any(flag == word for word in re.split(r'\W', annotation.text.lower())):
                #         errors.append(("WARNING", input_filename, passage_id, "hetero T cell flag", "Annotation #{} ({}) might need to be cell_hetero.".format(annotation.id, annotation.text)))
                        
                if "ipsc" in annotation.text.lower():
                    errors.append(("WARNING", input_filename, passage_id, "iPSC flag", "Annotation #{} ({}) contains iPSC.".format(annotation.id, annotation.text)))
                
                # if the annotation text is "[something short] immune [something short]" and is cell_pheno or has an ID:
                if ("immune" in annotation.text.lower()) and (len(annotation.text.lower()) < 15) \
                    and ((annotation.infons['type'] == "cell_phenotype") or annotation.infons['identifier'] not in [None, "None"]):
                        errors.append(("WARNING", input_filename, passage_id, "possible immune cell annotation incorrect", "Annotation #{} ({}) might need to be cell_hetero and without ID (immune cells)".format(annotation.id, annotation.text)))
                
                # check for '#':
                if '#' in annotation.text:
                    errors.append(("WARNING", input_filename, passage_id, "possible cluster #", "Annotation #{} ({}) might need arbitrary numbers removed.".format(annotation.id, annotation.text)))
                    
                # check for mentions that end in ')':
                # if annotation.text[-1] == ')':  
                #     errors.append(("WARNING", input_filename, passage_id, "annotation ends with ')'", "Annotation #{} ({}) might need equivalent parenthetical removed from span".format(annotation.id, annotation.text)))
                    
                # if (("stroma" in annotation.text) or ("CL:0000499" in annotation.infons['identifier'])) and (annotation.infons['type'] == "cell_phenotype"):
                #     errors.append(("WARNING", input_filename, passage_id, "stromal annotation", "Annotation #{}, ({}) might need to be cell_hetero (stromal)".format(annotation.id, annotation.text)))
                    
                # 6/26/25 flag:
                # if annotation.text.lower() in cell_hetero_flags:
                #     errors.append(("WARNING", input_filename, passage_id, "cell_hetero to pheno flag", "Annotation #{}, ({}) might need to be cell_pheno.".format(annotation.id, annotation.text)))
                
            # # identify nearby words (+/- 10 char) of interest that aren't annotated
            # for flag in lowercase_nearby_flags:
            #     num_char_elbow_room = 10 + len(flag)
            #     for match in re.finditer(flag, passage.text):
            #         curflag_start_ind = match.start()
            #         curflag_end_ind = match.end()
            #         nearby_anns = []
            #         curflag_within_ann = False
            #         # loop over all annotations to find annotations that contain or are nearby the flag:
            #         for annotation in passage.annotations:
            #             annotation_begin_idx = annotation.locations[0].offset - passage.offset
            #             annotation_end_idx   = annotation.locations[0].end    - passage.offset
                        
            #             if (annotation_begin_idx <= curflag_start_ind) and (curflag_end_ind <= annotation_end_idx):
            #                 curflag_within_ann = True
            #             elif (annotation_begin_idx - num_char_elbow_room <= curflag_start_ind) and \
            #                 (curflag_end_ind - num_char_elbow_room <= annotation_end_idx):
            #                 nearby_anns.append(annotation)
                   
            #         if not curflag_within_ann:
            #             for annotation in nearby_anns:
            #                 errors.append(("WARNING", input_filename, passage_id, "nearby_flag is nearby annotation", "Annotation #{} ({}) is nearby the flag word {}".format(annotation.id, annotation.text, flag)))
                    
            #     if flag in passage.text[max(0, annotation_begin_idx - num_char_elbow_room) : annotation_begin_idx] or \
            #         flag in passage.text[annotation_end_idx : annotation_end_idx + num_char_elbow_room]:
            #         # check if flag overlaps with another annotation:
            #         flag_offset = passage.text[annotation_begin_idx - num_char_elbow_room : annotation_end_idx + num_char_elbow_room]
                    
            #             errors.append(("WARNING", input_filename, passage_id, "nearby_flag is nearby annotation", "Annotation #{} ({}) is nearby the flag word {}".format(annotation.id, annotation.text, flag)))
                    
    print("Found " + str(len(errors)))
    return errors



def process_path(input_paths, standardizer, cell_types_dict):
    paths_queue = collections.deque(input_paths)

    errors = list()
    while paths_queue:
        path = paths_queue.popleft()
        print("Processing directory {}".format(path))
        dir_entries = os.listdir(path)
        for entry in dir_entries:
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path):
                print("Processing file {}".format(full_path))
                errors.extend(process_file(full_path, standardizer, cell_types_dict))
            elif os.path.isdir(full_path):
                paths_queue.append(full_path)
            else:
                errors.append(("WARN", entry, None, "Path is not a directory or normal file", "Path \"{}\" ignored: not a directory or normal file".format(full_path)))

    return errors


if __name__ == "__main__":
    if len(sys.argv) == 5:
        input_path = sys.argv[1]
        names_filename = sys.argv[2]
        docids_filename = sys.argv[3]
        output_path = sys.argv[4]
    else:
        raise Exception("expecting 4 commandline args")
    
    with open(names_filename, "r") as file:
        cell_types_dict = json.load(file)

    docids = BioCXMLUtils.read_docids(docids_filename)
    docids = [(pmid, pmc) for pmid, pmc, ft_avail in docids]
    print(f"Loaded (pmid, pmc) for {len(docids)} docids")
    standardizer = BioCXMLUtils.DocIDStandardizer(docids)
    
    start = datetime.datetime.now()
    errors = process_path([input_path], standardizer, cell_types_dict)
    print("Total processing time = " + str(datetime.datetime.now() - start))

    # Output
    with open(output_path, "w") as output_file:
        for error in errors:
            error_text = list(map(str, error))
            output_file.write("{}\n".format("\t".join(error_text)))
    print("Done.")
