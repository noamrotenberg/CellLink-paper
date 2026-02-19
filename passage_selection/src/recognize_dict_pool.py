import collections
import datetime
import sys
import multiprocessing
import json

import bioc

from abbreviations import AbbreviationExpander
from dictionary_ner import DictionaryRecognizer
import string_utils
import pooled_multi_document_processor

def process_document(document):
	# Clear previous annotations
	for passage in document.passages:
		passage.annotations.clear()
	# Get dictionary annotations
	for p in processors:
		p.process(document)
	# Extract dictionary annotations
	dict_annotations = extract_annotations("DICT", document)
	# Apply final annotations
	annotation_count = 0
	for passage in document.passages:
		passage.annotations.clear()
		annotations = list(dict_annotations[passage.text])
		#print("annotations = " + str(annotations))
		warn_ovelapping(annotations)
		for ann in annotations:
			#print("MERGE ADDING Adding annotation \"" + str(ann))
			annb = bioc.BioCAnnotation()
			annb.id = str(annotation_count)
			annb.text = ann[3]
			if not ann[5] is None:
				annb.infons["identifier"] = ann[5]
			annb.infons["type"] = ann[4]
			annb.locations.append(bioc.BioCLocation(ann[1], ann[2]))
			passage.annotations.append(annb)
			annotation_count += 1
	return document

# Extracts annotations from document
# MUST have an identifier
# All annotations will be cleared
def extract_annotations(context, document):
	annotations = collections.defaultdict(set)
	for passage in document.passages:
		for annotation in passage.annotations:
			identifier = annotation.infons.get("identifier")
			if identifier is None:
				identifier = annotation.infons.get("Identifier")
			if identifier == "None":
				identifier = None
			ann = (document.id, annotation.locations[0].offset, annotation.locations[0].length, annotation.text, annotation.infons["type"], identifier)
			#print("EXTRACT " + context + " Extracted annotation \"" + str(ann) + "\"")
			annotations[passage.text].add(ann)
		passage.annotations.clear()
	return annotations

def overlaps(ann1, ann2):
	return ann1[0] == ann2[0] and ann1[1] < ann2[1] + ann2[2] and ann2[1] < ann1[1] + ann1[2]

def warn_ovelapping(annotations):
	for i in range(0, len(annotations)):
		ann1 = annotations[i]
		for j in range(i + 1, len(annotations)):
			ann2 = annotations[j]
			if overlaps(ann1, ann2):
				ann1_text = "\"" + ann1[3] + "\" (" + ann1[4] + ")"
				ann2_text = "\"" + ann2[3] + "\" (" + ann2[4] + ")"
				print("WARN OVERLAPPING annotations in " + ann1[0] + ": " + ann1_text + " & " + ann2_text)

def convert_string(token_text):
	# Map to ASCII, lower case
	p_template = string_utils.map_to_ASCII(token_text).lower().strip()
	## Remove plurals
	p_template = string_utils.s_stem(p_template)
	return p_template

if __name__ == "__main__":
	start = datetime.datetime.now()
	if len(sys.argv) != 7:
		print("arguments = " + str(sys.argv))
		print("Usage: <dict_path> <input_path> <abbr_path> <abbr_freq_filename> <process_count> <output_path>")
		exit()
	dict_path = sys.argv[1]
	input_path = sys.argv[2]
	abbr_path = sys.argv[3]
	abbr_freq_filename = sys.argv[4]
	process_count = int(sys.argv[5])
	output_path = sys.argv[6]

	# Load the abbreviation frequency file
	with open(abbr_freq_filename, "r") as abbr_freq_file:
		abbr_freq_dict = json.load(abbr_freq_file)
	abbr = AbbreviationExpander(abbr_freq_dict)

	# Load the abbreviations
	print("Loading abbreviations")
	abbr.load(abbr_path)

	ner = DictionaryRecognizer(convert_string, string_utils.tokenize)
	ner.load_terms(dict_path)
	
	global processors
	processors = [ner]
	
	pool = multiprocessing.Pool(processes = process_count)
	processor = pooled_multi_document_processor.PooledMultiDocumentProcessor(pool, process_document)
	processor.process_path(input_path, output_path)

	print("Done.")
	
	