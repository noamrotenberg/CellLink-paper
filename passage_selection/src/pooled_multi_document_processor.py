import datetime
import os
import pathlib
import random
import time
import gzip

import bioc

# TODO Extend to handle BioCJSON files
# TODO Extend to handle files with arbitrary extensions or no extension

lock_extension = ".lock"
lock_identifier = "." + str(int(time.time())) + str(random.randint(0, 1000))

class PooledMultiDocumentProcessor:

	def __init__(self, pool, document_handler):
		self.pool = pool
		self.document_handler = document_handler
		print("lock_identifier = {}".format(lock_identifier))

	def process_path(self, input_path, output_path):
		process_list = list()
		if not os.path.isdir(input_path):
			raise RuntimeError("Input path must be a directory: " + input_path)
			
		if not output_path is None:
			if not os.path.exists(output_path):
				os.makedirs(output_path)
			elif not os.path.isdir(output_path):
				raise ValueError("If input path is a directory then output path must be a directory: " + output_path)
		print("Processing directory " + input_path + " to " + output_path)
		# Process any xml files found
		dir = os.listdir(input_path)
		for item in dir:
			input_filename = input_path + "/" + item
			if os.path.isfile(input_filename) and (input_filename.endswith(".xml") or input_filename.endswith(".xml.gz")):
				#process_list.append((input_filename, output_filename, self.document_handler))
				process_list.append((item, input_path, output_path, self.document_handler))
		random.shuffle(process_list)
		print("len(process_list) = {}".format(len(process_list)))
		start = datetime.datetime.now()
		self.pool.map(process_file, process_list)
		print("Total processing time = " + str(datetime.datetime.now() - start))

def process_file(process_info):
	item, input_path, output_path, document_handler = process_info
	
	input_filename = input_path + "/" + item
	output_filename = output_path + "/" + item

	print("Checking file " + input_filename + " to " + output_filename)
	
	# Check if already processed
	if os.path.exists(output_filename):
		print("File " + input_filename + " has already been processed to " + output_filename)
		return
	
	# Check if already locked
	lock_filepattern = output_filename + lock_extension
	locks = [output_path + "/" + filename for filename in os.listdir(output_path)]
	locks = [filename for filename in locks if filename.startswith(lock_filepattern)]
	if len(locks) > 0:
		print("File " + input_filename + " is already locked (" + output_filename + ")")
		return
	
	# Attempt to lock
	lock_filename = output_filename + lock_extension + lock_identifier
	pathlib.Path(lock_filename).touch()
	locks = [output_path + "/" + filename for filename in os.listdir(output_path)]
	locks = [filename for filename in locks if filename.startswith(lock_filepattern)]
	locks.sort()
	#print("len(locks) = {}".format(len(locks)))
	#print("locks = {}".format(locks))
	if locks[0] != lock_filename:
		print("Tried to lock file " + input_filename + " but already locked (" + str(locks) + ")")
		pathlib.Path(lock_filename).unlink()
		return
	
	# Process the file
	print("Processing file " + input_filename + " to " + output_filename)
	if input_filename.endswith(".gz"):
		input_file = gzip.open(input_filename, "rt") 
	else:
		input_file = open(input_filename, "r")
	collection = bioc.biocxml.load(input_file)
	input_file.close()
	new_collection = bioc.BioCCollection()
	for document in collection.documents:
		print("Processing document {}".format(document.id))
		new_document = document_handler(document)
		new_collection.add_document(new_document)
	with open(output_filename, "w") as fp:
		bioc.biocxml.dump(new_collection, fp)
	pathlib.Path(lock_filename).unlink()
