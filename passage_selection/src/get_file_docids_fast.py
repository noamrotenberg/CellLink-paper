import datetime
import sys
import random
import gzip
from collections import deque
from multiprocessing import Pool

import BioCXMLUtils

patterns = [("<id>", "</id>"), ("<infon key=\"article-id_pmid\">", "</infon>"), ("<infon key=\"article-id_pmc\">", "</infon>")]

def process_file(input_filename):
	docids = set()
	error_status = None
	try:
		#with open(input_filename, "r") as file:
		#	filedata = file.read()

		# Get file data
		if input_filename.endswith(".gz"):
			file = gzip.open(input_filename, "rt") 
		else:
			file = open(input_filename, "r") 
		filedata = file.read()
		file.close()

		# First find the documents
		document_spans = list()
		index = 0
		while index < len(filedata):
			start = filedata.find("<document>", index)
			if start < 0:
				break
			end = filedata.find("</document>", start)
			if end < 0:
				print("ERROR")
				break
			document_spans.append((start, end))
			index = end
		#print(document_spans)
		# Now find document IDs
		for document_start, document_end in document_spans:
			document_text = filedata[document_start:document_end]
			docid_results = list()
			for start_pattern, end_pattern in patterns:
				match_start = document_text.find(start_pattern, 0)
				if match_start < 0:
					docid_results.append(None)
					continue
				match_end = document_text.find(end_pattern, match_start)
				if match_end < 0:
					print("ERROR")
					docid_results.append(None)
					continue
				docid_results.append(document_text[match_start + len(start_pattern): match_end])
			if not docid_results[2] is None:
				docid_results[2] = docid_results[2].upper()
				if not docid_results[2].startswith("PMC"):
					docid_results[2] = "PMC" + docid_results[2]
			docids.add(tuple(docid_results))
	except Exception as error:
		error_status = str(error)
	return (input_filename, error_status, docids)

def process_batch(input_filenames):
	# Process the filenames
	start = datetime.datetime.now()
	p = Pool(pcount)
	with p:
		result = p.map(process_file, input_filenames)	
	print("Total processing time = " + str(datetime.datetime.now() - start))
	combined_results = set()
	for filename, error_status, docids in result:
		if error_status is None:
			combined_results.update({(docid, pmid, pmc, filename) for docid, pmid, pmc in docids})
		else:
			print("ERROR reading filename {}: {}".format(filename, error_status))
	print("Processed {} filenames, with {} total docids".format(len(result), len(combined_results)))
	return combined_results

if __name__ == "__main__":
	start = datetime.datetime.now()
	if len(sys.argv) != 6:
		print("Usage: <docids> <input> <output> <process count> <batch size>")
		exit()
	docids_filename = sys.argv[1]
	filename_list_filename = sys.argv[2]
	output_filename = sys.argv[3]
	pcount = int(sys.argv[4])
	batch_size = int(sys.argv[5]) #1000
	
	docids = BioCXMLUtils.read_docids(docids_filename)
	pmid2ft_avail = {pmid: ft_avail for pmid, pmc, ft_avail in docids if not pmid is None}
	docids = [(pmid, pmc) for pmid, pmc, ft_avail in docids]
	standardizer = BioCXMLUtils.DocIDStandardizer(docids)
	print("Loaded {} docids".format(len(docids)))

	input_filenames = list()
	with open(filename_list_filename, "r") as file:
		for line in file:
			line = line.strip()
			if len(line) > 0:
				input_filenames.append(line)
				#print("Adding filename {}".format(line))
	print("Number of filenames = {}".format(len(input_filenames)))		
	random.shuffle(input_filenames)
	input_filenames = deque(input_filenames)
	
	batch_index = 0
	combined_results = set()
	while len(input_filenames) > 0:
		batch_index += 1
		batch = list()
		while len(batch) < batch_size and len(input_filenames) > 0:
			batch.append(input_filenames.popleft())
		print("Processing batch {}, filename count: {}".format(batch_index, len(batch)))
		combined_results.update(process_batch(batch))
		# Output the results
		print("Writing batch {}, remaining filenames: {}".format(batch_index, len(input_filenames)))
		with open(output_filename, "w") as output_file:
			for docid, pmid, pmc, filename in combined_results:
				pmid2, pmc2 = standardizer.standardize(docid, pmid, pmc)
				ft_avail = pmid2ft_avail[pmid2]
				output_file.write("{}\t{}\t{}\t{}\n".format(pmid2, pmc2, ft_avail, filename))
	print("Done.")	

