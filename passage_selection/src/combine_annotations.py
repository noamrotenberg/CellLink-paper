import sys
import os
import datetime
import gzip
import hashlib
import random

import bioc

import BioCXMLUtils

def get_passage_keys(input_filename, standardizer):
	# Load collection
	if input_filename.endswith(".gz"):
		file = gzip.open(input_filename, "rt")
	else:
		file = open(input_filename, "r")
	collection = bioc.biocxml.load(file)
	file.close()
	
	# Extract passage keys
	passage_keys = set()
	for document in collection.documents:
		header = document.passages[0]
		pmid = header.infons.get("article-id_pmid")
		pmc = header.infons.get("article-id_pmc")
		pmid, pmc = standardizer.standardize(document.id, pmid, pmc)
		for passage_index, passage in enumerate(document.passages):
			passage_md5 = hashlib.md5(passage.text.encode("utf-8")).hexdigest()
			passage_key = (pmid, pmc, passage_index, passage_md5)
			passage_keys.add(passage_key)
	return passage_keys

def extract_annotations(source, source_filename, standardizer, annotations):
	# Load collection
	if source_filename.endswith(".gz"):
		file = gzip.open(source_filename, "rt")
	else:
		file = open(source_filename, "r")
	collection = bioc.biocxml.load(file)
	file.close()
	
	# Extract annotations
	for document in collection.documents:
		header = document.passages[0]
		pmid = header.infons.get("article-id_pmid")
		pmc = header.infons.get("article-id_pmc")
		pmid, pmc = standardizer.standardize(document.id, pmid, pmc)
		for passage_index, passage in enumerate(document.passages):
			passage_md5 = hashlib.md5(passage.text.encode("utf-8")).hexdigest()
			passage_key = (pmid, pmc, passage_index, passage_md5)
			passage_annotations = annotations[passage_key]
			for annotation in passage.annotations:
				locations = [(loc.offset, loc.length) for loc in annotation.locations]
				annotation = (source, tuple(locations), annotation.text, annotation.infons["type"], annotation.infons.get("identifier"))
				passage_annotations.append(annotation)
	return annotations

def process_filename(filename, input_path, output_path, source2path, standardizer):
	input_filename = input_path + filename
	if not os.path.isfile(input_filename):
		return
	if not input_filename.endswith(".xml") and not input_filename.endswith(".xml.gz"):
		return
	output_filename = output_path + filename
	print("Processing file {} to {}".format(input_filename, output_filename))
	# Prepare annotations dictionary with passage keys
	annotations = {passage_key: [] for passage_key in get_passage_keys(input_filename, standardizer)}
	print("\tFound {} passage keys".format(len(annotations)))
	# Get annotations
	for source, source_path in source2path.items():
		source_filename = source_path + filename
		print("\tGetting annotations from source {} filename {}".format(source, source_filename))
		extract_annotations(source, source_filename, standardizer, annotations)

	# Load collection
	print("\tLoading collection from {}".format(input_filename))
	if input_filename.endswith(".gz"):
		file = gzip.open(input_filename, "rt")
	else:
		file = open(input_filename, "r")
	collection = bioc.biocxml.load(file)
	file.close()
	
	# Add annotations
	print("\tAdding annotations")
	for document in collection.documents:
		header = document.passages[0]
		pmid = header.infons.get("article-id_pmid")
		pmc = header.infons.get("article-id_pmc")
		pmid, pmc = standardizer.standardize(document.id, pmid, pmc)
		annotation_count = 0
		for passage_index, passage in enumerate(document.passages):
			passage_md5 = hashlib.md5(passage.text.encode("utf-8")).hexdigest()
			passage_key = (pmid, pmc, passage_index, passage_md5)
			passage_annotations = annotations[passage_key]
			for source, location_tuple, mention_text, entity_type, identifier in passage_annotations:
				annotation = bioc.BioCAnnotation()
				annotation.id = str(annotation_count)
				annotation.text = mention_text
				annotation.infons["source"] = source
				annotation.infons["type"] = entity_type
				if not identifier is None:
					annotation.infons["identifier"] = identifier
				for offset, length in location_tuple:
					annotation.add_location(bioc.BioCLocation(offset, length))
				passage.add_annotation(annotation)
				annotation_count += 1

	print("\tWriting collection to {}".format(output_filename))
	if output_filename.endswith(".gz"):
		file = gzip.open(output_filename, "wt")
	else:
		file = open(output_filename, "w")
	bioc.dump(collection, file)
	file.close()

if __name__ == "__main__":
	docids_filename = sys.argv[1]
	input_path = sys.argv[2]
	output_path = sys.argv[-1]
	source_path_pairs = sys.argv[3:-1]
	source2path = {source: path for source, path in zip(source_path_pairs[::2], source_path_pairs[1::2])}
	print("docids_filename = {}".format(docids_filename))
	print("input_path = {}".format(input_path))
	print("output_path = {}".format(output_path))
	print("source2path = {}".format(source2path))

	docids = BioCXMLUtils.read_docids(docids_filename)
	docids = [(pmid, pmc) for pmid, pmc, ft_avail in docids]
	print("Loaded {} docids".format(len(docids)))
	standardizer = BioCXMLUtils.DocIDStandardizer(docids)

	# Verify directories
	if not input_path.endswith("/"):
		input_path += "/"
	if not os.path.isdir(input_path):
		raise RuntimeError("Input path \"{}\" must be a directory".format(input_path))
	if not output_path.endswith("/"):
		output_path += "/"
	if not os.path.isdir(output_path):
		raise RuntimeError("Output path \"{}\" must be a directory".format(output_path))
	for source, source_path in source2path.items():
		if not source_path.endswith("/"):
			source_path += "/"
			source2path[source] = source_path
		if not os.path.isdir(source_path):
			raise RuntimeError("Source path \"{}\" must be a directory".format(source_path))
	
	start = datetime.datetime.now()
	dir = os.listdir(input_path)
	random.shuffle(dir)
	total = len(dir)
	for filename_index, filename in enumerate(dir):
		process_filename(filename, input_path, output_path, source2path, standardizer)
		completed = filename_index + 1
		expected = total * ((datetime.datetime.now() - start) / completed) + datetime.datetime.now()
		print("{} / {}; expected {}".format(completed, total, expected.strftime("%Y-%m-%d %H:%M:%S")))
	print("Total processing time = " + str(datetime.datetime.now() - start))

	
		
