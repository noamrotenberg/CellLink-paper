import datetime
import time
import os
import gzip
import urllib.request
import urllib.error
from collections import deque

import bioc

def read_docids(filename):
	docids = set()
	with open(filename, "r") as file:
		for line_index, line in enumerate(file):
			line = line.strip()
			if len(line) == 0:
				continue
			fields = line.split("\t")
			pmid = normalize_identifier(fields[0].strip())
			pmc = normalize_PMC(fields[1].strip())
			ft_avail_text = fields[2].lower()
			if ft_avail_text == "true":
				ft_avail = True
			elif ft_avail_text == "false":
				ft_avail = False
			else:
				raise ValueError("ft_avail value is not boolean: {}".format(ft_avail))
			docids.add((pmid, pmc, ft_avail))
	return docids

def normalize_identifier(identifier):
	if identifier is None:
		return None
	identifier = identifier.strip()
	if len(identifier) == 0:
		return None
	if identifier == "None":
		return None
	return identifier

def normalize_PMC(pmc):
	if pmc is None:
		return None
	pmc = pmc.strip()
	if len(pmc) == 0:
		return None
	if pmc == "None":
		return None
	if pmc.startswith("PMC"):
		return pmc
	return "PMC" + pmc

class DocIDStandardizer:
	
	def __init__(self, pmid_pmc_pairs):
		self.update_pairs(pmid_pmc_pairs)

	def update_pairs(self, pmid_pmc_pairs):
		self.pmid2pmc = dict()
		self.pmc2pmid = dict()
		for pmid, pmc in pmid_pmc_pairs:
			if not pmid is None:
				self.pmid2pmc[pmid] = pmc
			if not pmc is None:
				self.pmc2pmid[pmc] = pmid
	
	def validate(self, pmid, pmc, docid_tuple):
		if not pmid is None:
			if not pmid in self.pmid2pmc:
				raise ValueError("PMID {} is unknown for {}".format(pmid, docid_tuple))
			if pmc != self.pmid2pmc[pmid]:
				print("WARN: PMC {} is incorrect for {}, updating to {}".format(pmc, docid_tuple, self.pmid2pmc[pmid]))
			return (pmid, self.pmid2pmc[pmid])
		if not pmc is None:
			if not pmc in self.pmc2pmid:
				raise ValueError("PMC ID {} is unknown for {}".format(pmc, docid_tuple))
			return (self.pmc2pmid[pmc], pmc)
		raise ValueError("Both PMID and PMC are None for {}".format(docid_tuple))
	
	def standardize(self, docid, pmid, pmc):
		docid = normalize_identifier(docid)
		pmid = normalize_identifier(pmid)
		pmc = normalize_PMC(pmc)
		docid_tuple = (docid, pmid, pmc)
		if pmid is None:
			if pmc is None:
				return self.validate(docid, None, docid_tuple)
			elif docid == pmc[3:]:
				return self.validate(None, None, pmc, docid_tuple)
			else:
				return self.validate(docid, pmc, docid_tuple)
		elif pmc is None:
			if docid == pmid:
				return self.validate(pmid, None, docid_tuple)
			else:
				docid_pmc = normalize_PMC(docid)
				return self.validate(pmid, docid_pmc, docid_tuple)
		else:
			return self.validate(pmid, pmc, docid_tuple)

class BioCXMLReader:
	
	def __init__(self):
		self.collection_filenames = deque()
		self.documents = deque()
	
	def add_filename_list(self, filename_list_filename):
		with open(filename_list_filename, "r") as file:
			for line in file:
				line = line.strip()
				self.collection_filenames.append(line)
				print("Adding filename {}".format(line))
		print("Number of filenames = {}".format(len(self.collection_filenames)))		
	
	def add_filenames(self, input_filenames):
		self.collection_filenames.extend(input_filenames)
	
	def add_path(self, input_path, filename_extension = None):
		if os.path.isfile(input_path):
			#print("Adding filename {}".format(input_path))
			self.collection_filenames.append(input_path)
		elif os.path.isdir(input_path):
			#print("Adding filename {}".format(input_path))
			dir = os.listdir(input_path)
			for item in dir:
				input_filename = input_path + "/" + item
				if not os.path.isfile(input_filename) or (not filename_extension is None and not item.endswith(filename_extension)):
					#print("Skipping filename {}".format(input_filename))
					continue
				#print("Adding filename {}".format(input_filename))
				self.collection_filenames.append(input_filename)
		else:
			raise RuntimeError("Path is not a directory or normal file: " + input_path)
		print("Number of filenames = {}".format(len(self.collection_filenames)))
	
	def load_next_collection(self):
		if len(self.documents) > 0:
			return
		if len(self.collection_filenames) == 0:
			return
		input_filename = self.collection_filenames.popleft()
		print("BioCXMLReader.load_next_collection() input_filename = {}".format(input_filename))
		try:
			if input_filename.endswith(".gz"):
				input_file = gzip.open(input_filename, "rt")
			else:
				input_file = open(input_filename, "r")
			collection = bioc.biocxml.load(input_file)
			input_file.close()
			self.documents.extend(collection.documents)
		except IOError as err:
			print("ERROR IOError loading file {}: {}".format(input_filename, err))
		print("BioCXMLReader.load_next_collection() Number of filenames = {} Number of documents = {}".format(len(self.collection_filenames), len(self.documents)))
	
	def next_document(self):
		while len(self.documents) == 0 and len(self.collection_filenames) > 0:
			self.load_next_collection()
		if len(self.documents) == 0:
			#print("BioCXMLReader.next_document() Returning None")
			return None
		document = self.documents.popleft()
		return document

class BioCXMLRetriever:
	
	def __init__(self, needed_docids, base_url, batch_size, wait_seconds, last_request = None):
		self.needed_docids = deque(set(needed_docids))
		self.documents = deque()
		if last_request == None:
			self.last_request = datetime.datetime.now()
		else:
			self.last_request = last_request
		self.total = len(needed_docids)
		self.count = 1
		self.base_url = str(base_url)
		self.batch_size = int(batch_size)
		self.wait_seconds = float(wait_seconds)
	
	def next_download(self):
		if len(self.documents) > 0:
			return
		batch = list()
		while len(batch) < self.batch_size and len(self.needed_docids) > 0:
			batch.append(self.needed_docids.popleft())
		if len(batch) == 0:
			return
		batch_text = ",".join(batch)
		url = self.base_url.format(batch_text)
		print("Retreiving batch; url = \"{}\"".format(url))
		# Check if we need to slow down
		diff = (datetime.datetime.now() - self.last_request).total_seconds()
		if diff < self.wait_seconds:
			sleep_time = self.wait_seconds - diff
			print("Sleeping for {}".format(sleep_time))
			time.sleep(sleep_time)
		self.last_request = datetime.datetime.now()
		xml_str = ""
		try:
			with urllib.request.urlopen(url) as response:
				xml = response.read()
			xml_str = xml.decode("UTF-8")
			#print("XML = \"{}\"".format(xml_str))
			batch_collection = bioc.loads(xml_str)
			print("Retreived {} documents".format(len(batch_collection.documents)))
			self.documents.extend(batch_collection.documents)
		except Exception as error:
			print("Error {} with xml string \"{}\"".format(error, xml_str))

	def next_document(self):
		while len(self.documents) == 0 and len(self.needed_docids) > 0:
			self.next_download()
		if len(self.documents) == 0:
			return None
		print("Returning document {} of {}".format(self.count, self.total))
		self.count += 1
		document = self.documents.popleft()
		return document

class BioCXMLWriter():
	
	def __init__(self, output_path, maximum_count):
		if not output_path.endswith("/"):
			output_path += "/"
		self.output_path = output_path
		self.maximum_count = maximum_count
		self.collection_index = 0
		self.current_collection = None
		self.current_count = 0
		self.timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%dT%H%M%S%f")
	
	def get_collection(self):
		# Check if current collection is full
		if not self.current_collection is None and self.current_count >= self.maximum_count:
			self.flush()
		# Check if need to create a new collection
		if self.current_collection is None:
			self.current_collection = bioc.BioCCollection()
			self.current_count = 0
		return self.current_collection
	
	def process(self, document):
		collection = self.get_collection()
		self.current_count += 1
		collection.add_document(document)
	
	def flush(self):
		if self.current_collection is None:
			return
		if len(self.current_collection.documents) == 0:
			return
		# TODO Add ability to configure file name
		filename = self.output_path + "collection_{}_{}.xml".format(self.timestamp, self.collection_index)
		print("Writing {} documents to {}".format(len(self.current_collection.documents), filename))
		with open(filename, 'w') as fp:
			bioc.dump(self.current_collection, fp)
		self.collection_index += 1
		self.current_collection = None
		self.current_count = 0
	
	def close(self):
		self.flush()


