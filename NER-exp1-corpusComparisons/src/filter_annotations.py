import sys
import os

import bioc

def load_updates(filename):
	filters = set()
	updates = dict()
	with open(filename, "r") as file:
		for line in file:
			line = line.strip()
			if len(line) == 0:
				continue
			fields = line.split("\t")
			# mention text, entity type, identifier --> None | entity type, identifier
			if len(fields) != 3 and len(fields) != 5:
				raise ValueError("Dictionary line has incorrect number of fields: \"{}\"".format(line))
			mention_text = fields[0].strip()
			if len(mention_text) < 0:
				mention_text = None
			mention_text = mention_text if mention_text != "*" else None
			entity_type = fields[1].strip()
			entity_type = entity_type if entity_type != "*" else None
			identifier = fields[2].strip()
			identifier = identifier if identifier != "*" else None
			filter_key = (mention_text, entity_type, identifier)
			if len(fields) == 5:
				entity_type2 = fields[3].strip()
				entity_type2 = entity_type2 if entity_type2 != "*" else None
				identifier2 = fields[4].strip()
				identifier2 = identifier2 if identifier2 != "*" else None
				if not entity_type2 is None or not identifier2 is None:
					updates[filter_key] = (entity_type2, identifier2)
			else:
				filters.add(filter_key)
	return filters, updates

# TODO Rerun update if any changes
def update_annotation(mention_text, entity_type, identifier, updates):
	for (mention_text2, entity_type2a, identifier2a), (entity_type2b, identifier2b) in updates.items():
		if mention_text != mention_text2 and not mention_text2 is None:
			continue
		if entity_type != entity_type2a and not entity_type2a is None:
			continue
		if identifier != identifier2a and not identifier2a is None:
			continue
		entity_type_update = entity_type2b if not entity_type2b is None else entity_type
		identifier_update = identifier2b if not identifier2b is None else identifier
		return entity_type_update, identifier_update
	return entity_type, identifier
	
def filter_annotation(mention_text, entity_type, identifier, filters):
	for mention_text2, entity_type2, identifier2 in filters:
		if mention_text != mention_text2 and not mention_text2 is None:
			continue
		if entity_type != entity_type2 and not entity_type2 is None:
			continue
		if identifier != identifier2 and not identifier2 is None:
			continue
		return True
	return False

def process_file(input_filename, filters, updates, output_filename):
	output_collection = bioc.BioCCollection()
	with open(input_filename, 'r') as fp:
		input_collection = bioc.load(fp)
	for document in input_collection.documents:
		for passage in document.passages:
			annotations = list(passage.annotations)
			passage.annotations.clear()
			for annotation in annotations:
				identifier_key = "identifier" if "identifier" in annotation.infons else "Identifier"
				identifier = annotation.infons.get(identifier_key, "None")
				entity_type = annotation.infons.get("type", "None")
				mention_text = annotation.text if not annotation.text is None else ""
				entity_type2, identifier2 = update_annotation(mention_text, entity_type, identifier, updates)
				if entity_type != entity_type2 or identifier != identifier2:
					#print("UPDATE ({}, {}, {}) to ({}, {})".format(mention_text, entity_type, identifier, entity_type2, identifier2))
					entity_type = entity_type2
					identifier = identifier2
					annotation.infons["type"] = entity_type2
					annotation.infons["identifier"] = identifier2
				if not filter_annotation(mention_text, entity_type, identifier, filters):
					passage.add_annotation(annotation)
				#else: 
				#	print("FILTER ({}, {}, {})".format(mention_text, entity_type, identifier))
		document.relations.clear()
		output_collection.add_document(document)
	with open(output_filename, 'w') as fp:
		bioc.dump(output_collection, fp)

if __name__ == "__main__":
	input_path = sys.argv[1]
	update_filename = sys.argv[2]
	output_path = sys.argv[3]
	
	filters, updates = load_updates(update_filename)
	
	if os.path.isdir(input_path):
		if not os.path.isdir(output_path):
			raise RuntimeError("If input path is a directory then output path must be a directory: " + output_path)
		print("Processing directory " + input_path)
		# Process any xml files found
		dir = os.listdir(input_path)
		for item in dir:
			input_filename = input_path + "/" + item
			output_filename = output_path + "/" + item
			if os.path.isfile(input_filename) and input_filename.endswith(".xml"):
				print("Processing file " + input_filename + " to " + output_filename)
				process_file(input_filename, filters, updates, output_filename)
	elif os.path.isfile(input_path):
		# TODO If output_path exists, it must be a file
		# TODO If output_path does not exist, then its location must be a directory that exists
		if os.path.isdir(output_path):
			raise RuntimeError("If input path is a file then output path may not be a directory: " + output_path)
		print("Processing file " + input_path + " to " + output_path)
		# Process directly
		process_file(input_path, filters, updates, output_path)
	else:  
		raise RuntimeError("Path is not a directory or normal file: " + input_path)
	print("Done.")
	