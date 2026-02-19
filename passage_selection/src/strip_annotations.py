import datetime
import os
import sys

import bioc

# Removes annotations from input file / directory and places a copy in the output file / directory

def process_file(input_filename, output_filename):
	with open(input_filename, "r") as fp:
		collection = bioc.biocxml.load(fp)
	for document in collection.documents:
		for passage in document.passages:
			passage.annotations.clear()
			passage.relations.clear()
		document.relations.clear()
	with open(output_filename, 'w') as fp:
		bioc.dump(collection, fp)
		
if __name__ == "__main__":
	start = datetime.datetime.now()
	if len(sys.argv) != 3:
		print("Usage: <input> <output>")
		exit()
	input_path = sys.argv[1]
	output_path = sys.argv[2]

	start = datetime.datetime.now()
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
				process_file(input_filename, output_filename)
	elif os.path.isfile(input_path):
		# TODO If output_path exists, it must be a file
		# TODO If output_path does not exist, then its location must be a directory that exists
		if os.path.isdir(output_path):
			raise RuntimeError("If input path is a file then output path may not be a directory: " + output_path)
		print("Processing file " + input_path + " to " + output_path)
		# Process directly
		process_file(input_path, output_path)
	else:  
		raise RuntimeError("Path is not a directory or normal file: " + input_path)
	print("Total processing time = " + str(datetime.datetime.now() - start))
