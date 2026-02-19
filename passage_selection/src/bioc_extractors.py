import collections
import re
import json

import string_utils

# filters tokens containing 
p1 = re.compile(r"[^\W_]+", re.UNICODE)
p2 = re.compile(r"[0-9]+")

cache = dict()

def annotations(pmid, pmc, passage, params):
	mappings_filename = params[0]
	if mappings_filename in cache:
		mappings = cache[mappings_filename]
	else:
		print("Loading mappings_filename: {}".format(mappings_filename))
		with open(mappings_filename, "r") as mappings_file:
			mappings = json.load(mappings_file)
			cache[mappings_filename] = mappings
		print("Loaded {} mappings".format(len(mappings)))
	annotations = list()
	for annotation in passage.annotations:
		source_name = annotation.infons.get("source", "None")
		loc = annotation.locations[0] if len(annotation.locations) == 1 else None
		start = loc.offset if not loc is None else None
		end = loc.end if not loc is None else None
		annotation_type = annotation.infons.get("type", "None")
		mapped_type = mappings.get(source_name + "/" + annotation_type, "None")
		identifier_key = "identifier" if "identifier" in annotation.infons else "Identifier"
		annotation_identifier = annotation.infons.get(identifier_key, "None")
		annotation_tuple = [source_name, annotation.text, annotation_type, mapped_type, annotation_identifier, start, end]
		annotations.append(annotation_tuple)
	return annotations

def passage_type(pmid, pmc, passage, params):
	section_type = passage.infons.get("section_type", "None")
	type = passage.infons.get("type", "None")
	return "{}/{}".format(section_type, type)

def passage_size(pmid, pmc, passage, params):
	return len(passage.text) + len(passage.infons.get("subtitle", ""))

def get_tokens(text):
	tokens = p1.findall(string_utils.map_to_ASCII(text.lower()))
	tokens = [token for token in tokens if p2.fullmatch(token) is None]
	return tokens

def vocabulary(pmid, pmc, passage, params):
	tokens = get_tokens(passage.text)
	if "subtitle" in passage.infons:
		tokens.extend(passage.infons["subtitle"])
	return collections.Counter(tokens)
