import datetime
import os
import sys
import json
import gzip
import copy
import hashlib
import random

import bioc

import BioCXMLUtils

# required by getattr use below
import bioc_extractors

def read_meta(filename):
	# TODO use the config to identify the extractors, like for bioc_extractors
	print("Loading " + str(filename))
	if filename.endswith(".gz"):
		file = gzip.open(filename, "rt") 
	else:
		file = open(filename, "r") 
	pmid2pub = dict()
	for line in file:
		pub_dict = json.loads(line)
		pmid = pub_dict["pmid"]
		pmid2pub[pmid] = pub_dict
	return pmid2pub
		
def update_or_check(d, k, v):
	if not k in d:
		d[k] = v
		return
	if d[k] != v:
		print("WARN: previous value for key \"{}\" = \"{}\" does not equal new value \"{}\"".format(k, d[k], v))

def process_file(input_filename, docid_standardizer, pmid2pub, extract_document_dimensions, extract_passage_dimensions):
	passages = list()

	if input_filename.endswith(".gz"):
		file = gzip.open(input_filename, "rt")
	else:
		file = open(input_filename, "r")
	collection = bioc.biocxml.load(file)
	file.close()

	for document in collection.documents:
		docid = document.id
		header = document.passages[0]
		pmid = header.infons.get("article-id_pmid")
		pmc = header.infons.get("article-id_pmc")
		pmid, pmc = docid_standardizer.standardize(docid, pmid, pmc)
		if pmid is None:
			raise ValueError("PMID is None for document {} pmid {} pmc {} in filename {}".format(docid, pmid, pmc, input_filename))
		
		# Get the document-level info
		document_instance = pmid2pub.get(pmid, dict())
		update_or_check(document_instance, "pmid", pmid)
		update_or_check(document_instance, "pmc", pmc)
		document_instance["document_title"] = header.text
		if not "data" in document_instance:
			document_instance["data"] = dict()
		for dimension_name, (dimension_function, parameters) in extract_document_dimensions:
			name_fields = dimension_function.split(".")
			value = getattr(sys.modules[name_fields[0]], name_fields[1])(pmid, pmc, header, parameters)
			update_or_check(document_instance["data"], dimension_name, value)

		# Get the passage-level info
		for passage_index, passage in enumerate(document.passages):
			if len(passage.text.strip()) == 0:
				continue
			passage_instance = copy.deepcopy(document_instance)
			passages.append(passage_instance)
			passage_instance["passage_index"] = passage_index
			passage_instance["passage_id"] = "{}_{}".format(pmid, passage_index)
			passage_text = passage.text
			passage_instance["passage_md5"] = hashlib.md5(passage_text.encode("utf-8")).hexdigest()

			for dimension_name, (dimension_function, parameters) in extract_passage_dimensions:
				name_fields = dimension_function.split(".")
				value = getattr(sys.modules[name_fields[0]], name_fields[1])(pmid, pmc, passage, parameters)
				passage_instance["data"][dimension_name] = value

	print("Found {} passages".format(len(passages)))
	return passages

if __name__ == "__main__":
	start = datetime.datetime.now()
	if len(sys.argv) != 5:
		print("Usage: <metadata> <input> <config> <output>")
		exit()
	config_filename = sys.argv[1]
	metadata_filename = sys.argv[2]
	input_path = sys.argv[3]
	output_filename = sys.argv[4]
	
	with open(config_filename) as config_file:
		config = json.load(config_file)
	print("Loaded {} dimensions".format(len(config)))
	extract_document_dimensions = [(dimension_name, dimension_config["bioc_extractor"]) for dimension_name, dimension_config in config.items() if "bioc_extractor" in dimension_config and dimension_config.get("rank") == "document"]
	print("extract_document_dimensions = {}".format(extract_document_dimensions))
	extract_passage_dimensions = [(dimension_name, dimension_config["bioc_extractor"]) for dimension_name, dimension_config in config.items() if "bioc_extractor" in dimension_config and dimension_config.get("rank") == "passage"]
	print("extract_passage_dimensions = {}".format(extract_passage_dimensions))
	
	pmid2pub = read_meta(metadata_filename)
	print("Loaded metadata for {} documents".format(len(pmid2pub)))
	docids = [(pmid, pub.get("pmc")) for pmid, pub in pmid2pub.items()]
	docid_standardizer = BioCXMLUtils.DocIDStandardizer(docids)
	
	filenames = list()
	if os.path.isdir(input_path):
		print("Processing directory " + input_path)
		# Process any xml files found
		dir = os.listdir(input_path)
		random.shuffle(dir)
		for item in dir:
			input_filename = input_path + "/" + item
			if os.path.isfile(input_filename) and (input_filename.endswith(".xml") or input_filename.endswith(".xml.gz")):
				filenames.append(input_filename)
	elif os.path.isfile(input_path):
		filenames.append(input_path)
	else:  
		raise RuntimeError("Path is not a directory or normal file: " + input_path)
	print("Found {} filenames to process".format(len(filenames)))

	if output_filename.endswith(".gz"):
		output_file = gzip.open(output_filename, "wt") 
	else:
		output_file = open(output_filename, "w") 

	start = datetime.datetime.now()
	total = len(filenames)
	for filename_index, input_filename in enumerate(filenames):
		print("Processing file " + input_filename)
		passage_list = process_file(input_filename, docid_standardizer, pmid2pub, extract_document_dimensions, extract_passage_dimensions)
		for passage_instance in passage_list:
			output_file.write(json.dumps(passage_instance) + "\n")
		completed = filename_index + 1
		expected = (total - completed) * ((datetime.datetime.now() - start) / completed) + datetime.datetime.now()
		print("{} / {}; expected {}".format(completed, total, expected.strftime("%Y-%m-%d %H:%M:%S")))
	output_file.close()

	print("Total processing time = " + str(datetime.datetime.now() - start))
	print("Done.")
