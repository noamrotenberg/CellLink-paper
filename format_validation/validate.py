import datetime
import os
import sys
import json
import collections

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
                errors.append(("ERROR", "Identifier not found in cell types dictionary", "Identifier not found in cell types dictionary: \"{}\"".format(lookup_identifier)))
    
    return errors


def overlaps(ann1, ann2):
    ann1_offset = ann1.locations[0].offset
    ann2_offset = ann2.locations[0].offset
    ann1_end = ann1.locations[0].end
    ann2_end = ann2.locations[0].end
    if ((ann1_offset >= ann2_offset) and (ann1_offset < ann2_end)) or \
       ((ann2_offset >= ann1_offset) and (ann2_offset < ann1_end)):
           return True
    else:
        return False

def encapsulates(ann1, ann2):
    # check whether ann1 completely encapsulates ann2
    ann1_offset = ann1.locations[0].offset
    ann2_offset = ann2.locations[0].offset
    ann1_end = ann1.locations[0].end
    ann2_end = ann2.locations[0].end
    return ((ann1_offset <= ann2_offset) and (ann1_end >= ann2_end))

def get_cell_vague_length(ann1, ann2):
    # given 2 annotations, one of which is cell_vague, get its length
    cell_vague_annotations = [ann for ann in [ann1, ann2] if ann.infons['type'] == 'cell_vague']
    if len(cell_vague_annotations) != 1:
        raise Exception("Only 1 cell_vague annotation was expected.")
    else:
        return len(cell_vague_annotations[0].text)


def process_file(input_filename, standardizer, cell_types_dict):
    if not input_filename.endswith(".xml"):
        return [("WARN", input_filename, None, "File does not end in \".xml\"", "File \"{}\" ignored: does not end in \".xml\"".format(input_filename))]
    errors = list()
    with open(input_filename, "r", encoding='utf-8') as fp:
        collection = biocxml.load(fp)
    for document in collection.documents:
        for passage_index, passage in enumerate(document.passages):
            passage_id = passage.infons["passage_id"] if "passage_id" in passage.infons else "#{}".format(passage_index)
            if not "passage_id" in passage.infons:
                errors.append(("ERROR", input_filename, passage_id, "Passage does not contain \"passage_id\" infon",  "Passage does not contain \"passage_id\" infon"))
                continue
            pmid = passage.infons.get("article-id_pmid")
            if pmid is None:
                errors.append(("ERROR", input_filename, passage_id, "Passage does not contain \"article-id_pmid\" infon",  "Passage does not contain \"article-id_pmid\" infon"))
                continue
            elif "passage_id" in passage.infons and not passage.infons["passage_id"].startswith(pmid + "_"):
                # Check that the passage_id starts with the pmid
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
                if not "type" in annotation.infons:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation does not contain \"type\" infon",  "Annotation does not contain \"type\" infon"))
                elif not annotation.infons["type"] in {"cell_phenotype", "cell_hetero", "cell_vague"}:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation \"type\" infon value is unknown",  "Annotation \"type\" infon value is unknown: {}".format(annotation.infons["type"])))
                if not "identifier" in annotation.infons:
                    errors.append(("ERROR", input_filename, passage_id, "Annotation does not contain \"identifier\" infon",  "Annotation does not contain \"identifier\" infon"))
                else:
                    identifier_errors = validate_identifier_list(annotation, annotation.infons["identifier"], cell_types_dict)
                    for error_type, general_error, specific_error in identifier_errors:
                        errors.append(("ERROR", input_filename, passage_id, general_error, "Annotation #{} ({}): {}".format(annotation.id, annotation.text, specific_error)))
                if "type" in annotation.infons and "identifier" in annotation.infons:
                    # Check that desc does not have an identifier
                    if annotation.infons["type"] == "cell_vague" and annotation.infons["identifier"] != "None":
                        errors.append(("ERROR", input_filename, passage_id, "Annotation with type \"cell_vague\" lists an identifier", "Annotation #{} ({}) with type \"cell_vague\" lists an identifier: {}".format(annotation.id, annotation.text, annotation.infons["identifier"])))
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
                
                # check for overlapping annotations
                for ann2 in passage.annotations[i+1:]:
                    if overlaps(annotation, ann2):
                           
                           # XNOR (exclusive not OR): cell_vague can't overlap with cell_vague; not cell_vague can't overlap with not cell_vague
                           # ("not cell_vague" means cell_phenotype or cell_hetero)
                           if not ((annotation.infons['type'] == 'cell_vague') ^ (ann2.infons['type'] == 'cell_vague')):
                               errors.append(("ERROR", input_filename, passage_id, "Overlapping annotations", "Annotation #{} ({}) overlaps with Annotation #{} ({})".format(annotation.id, annotation.text, ann2.id, ann2.text)))
                           else:
                               # ensure that the cell_vague annotation fully encapsulates the other annotation
                               if not (encapsulates(annotation, ann2) or encapsulates(ann2, annotation)):
                                   errors.append(("ERROR", input_filename, passage_id, "Partially overlapping annotations", "Annotation #{} ({}) overlaps with Annotation #{} ({}); cell_vague must completely encapsulate the other annotation".format(annotation.id, annotation.text, ann2.id, ann2.text)))
                               else:
                                   # since 1 annotation encapsulates another, ensure that the cell_vague annotation is longer
                                   if min(len(annotation.text), len(ann2.text)) >= get_cell_vague_length(annotation, ann2):
                                       errors.append(("ERROR", input_filename, passage_id, "cell_vague is encapsulated", "Annotation #{} ({}) overlaps with Annotation #{} ({}); cell_vague must encapsulate the other annotation".format(annotation.id, annotation.text, ann2.id, ann2.text)))
                                   
                    
                    if ((annotation.infons['type'] == 'cell_vague') ^ (ann2.infons['type'] == 'cell_vague')):
                        # overlapping cell_vague & something else. make sure the span isn't exactly the same
                        if annotation.text == ann2.text:
                            errors.append(("WARNING", input_filename, passage_id, "identical annotations", "cell_vague annotation overlaps exactly with another annotation: ('{}' and '{}')".format(annotation.text, ann2.text)))    
                
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
        names_filename = sys.argv[2] # "../Cell-Ontology_v2025-01-08.json"
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
