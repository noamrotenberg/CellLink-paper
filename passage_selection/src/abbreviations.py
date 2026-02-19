import codecs
import gzip
import os
import re

from bioc import biocxml

class AbbreviationExpander:

	def __init__(self, abbr_freq_dict = dict()):
		self.abbr_freq_dict = abbr_freq_dict
		self.abbr_dict = dict()

	def load(self, path):
		if os.path.isdir(path):
			# Load abbreviations from any files found
			dir = os.listdir(path)
			for item in dir:
				if os.path.isfile(path + "/" + item):
					self.load_file(path + "/" + item)				
		elif os.path.isfile(path):  
			# load directly
			self.load_file(path)
		else:  
			raise RuntimeError("Path is not a directory or normal file: " + path)

	def load_file(self, filename):
		if filename.endswith(".xml"):
			self.load_biocxml(filename)
		elif filename.endswith(".tsv"):
			self.load_tsv(filename)
		else:
			print("Abbreviation file does not end in xml or tsv, ignoring: \"{}\"".format(filename))

	def load_tsv(self, filename):
		print("Loading abbreviations from TSV file " + filename)
		count = 0
		if filename.endswith(".gz"):
			file = gzip.open(filename, 'rt', encoding="utf-8") 
		else:
			file = codecs.open(filename, 'r', encoding="utf-8") 
		for line in file:
			line = line.strip()
			if len(line) == 0:
				continue
			try:
				fields = line.split("\t")
				document_ID = fields[0]
				short = fields[1]
				long = fields[2]
				self.add(document_ID, short, long)
				# Handle plural abbreviations
				if short.endswith("s") and long.endswith("s"):
					self.add(document_ID, short[:-1], long[:-1])
				count += 1
			except:
				print("Abbreviation line malformed: \"{}\"".format(line))
		file.close()
		print("Loaded " + str(count) + " abbreviations")
	
	def load_biocxml(self, filename):
		print("Loading abbreviations from BioC file " + filename)
		count = 0
		with open(filename, 'r') as input_file:
			collection = biocxml.load(input_file)
		for document in collection.documents:
			for passage in document.passages:
				annotation_id2text = dict()
				for annotation in passage.annotations:
					if not "type" in annotation.infons or annotation.infons["type"] != "ABBR":
						continue
					annotation_id2text[annotation.id] = annotation.text
				for relation in passage.relations:
					if not "type" in relation.infons or relation.infons["type"] != "ABBR":
						continue
					long = None
					short = None
					for node in relation.nodes:
						if node.role=="LongForm":
							long=annotation_id2text.get(node.refid)
						elif node.role=="ShortForm":
							short=annotation_id2text.get(node.refid)
					if not long is None and not short is None:
						self.add(document.id, short, long)
						# Handle plural abbreviations
						if short.endswith("s") and long.endswith("s"):
							self.add(document.id, short[:-1], long[:-1])
						count += 1
					else:
						print("WARN Could not identify long form & short form for document " + document.id + " relation " + relation.id)
		print("Loaded " + str(count) + " abbreviations")
	
	def add(self, document_ID, short, long):
		if not document_ID in self.abbr_dict:
			self.abbr_dict[document_ID] = dict()
		doc_dict = self.abbr_dict[document_ID]
		# TODO Figure out why Java version used word boundaries
		regex = re.compile("\\b" + re.escape(short) + "\\b")
		if regex.search(long):
			print("INFO Ignoring abbreviation \"" + short + "\" -> \"" + long + "\" because long form contains short form")
			return
		previous_long = doc_dict.get(short)
		if previous_long is None or long == previous_long:
			# Either not already present or the same
			doc_dict[short] = long
			return
		count = self.get_abbr_freq(short, long)
		previous_count = self.get_abbr_freq(short, previous_long)
		#print("WARN Abbreviation \"" + short + "\" -> \"" + long + "\" (count "+str(count)+") is already defined as \"" + previous_long + "\" (count "+str(previous_count)+")")
		if count > previous_count:
			doc_dict[short] = long			

	def get_abbr_freq(self, short, long):
		if not short in self.abbr_freq_dict:
			return 0
		return self.abbr_freq_dict[short].get(long, 0)

	def expand(self, document_ID, text):
		if not document_ID in self.abbr_dict:
			return text
		doc_dict = self.abbr_dict[document_ID]
		result_history = set()
		short_history = set()
		result = text
		while not result in result_history:
			result_history.add(result)
			#print("Expanding \"" + result + "\"")
			for short, long in doc_dict.items():
				if result.find(short) >= 0 and not short in short_history:
					updated = None
					try:
						if result.find(long) >= 0:
							updated = re.sub("\\s*\\(\\s*" + re.escape(short) + "\\s*\\)\\s*", " ", result);
						else:
							updated = re.sub("\\b" + re.escape(short) + "\\b", long, result)
						if updated != result:
							result = updated
							short_history.add(short)
					except:
						print("WARN Exception applying abbreviation \"{}\" -> \"{}\" for document {} and text {}". format(short, long, document_ID, text))
						# Do not update result
						short_history.add(short)
		return result
	
