import datetime
import json
import logging
import os
import re
import sys
from typing import Generator, Tuple, List
from multiprocessing import Pool
import random

import bioc
import spacy

nlp = spacy.load("en_core_sci_sm", exclude=["ner"])

# TODO make max length configurable
max_length = 96
# TODO: How does this handle disjunct spans?

def main():
	if len(sys.argv) < 3:
		print("Usage: <input_path> <output_file> <process_count>")
		exit()

	input_path = sys.argv[1]
	output_path = sys.argv[2]
	process_count = int(sys.argv[3])
	processing_pairs = get_processing_pairs(input_path, output_path, ".xml", ".json")
	random.shuffle(processing_pairs)
	print("Found " + str(len(processing_pairs)) + " input files")
	
	start = datetime.datetime.now()
	p = Pool(process_count)
	with p: 
		result = p.starmap(convert_bioc_to_json, processing_pairs)
	print("Total processing time = " + str(datetime.datetime.now() - start))
	print(result)
	print("Done.")

def get_processing_filenames(input_path, output_path, item, input_file_extension = None, output_file_extension = None):
	input_filename = input_path + "/" + item
	output_filename = output_path + "/" + item
	if input_file_extension is None:
		if not output_file_extension is None:
			raise ValueError("If input file extension is None then output file extension must also be None")
		# No changes needed
	elif not input_filename.endswith(input_file_extension):
		return None, None
	elif not output_file_extension is None: 
		output_filename = output_filename[:-len(input_file_extension)] + output_file_extension
	return input_filename, output_filename

def get_processing_pairs(input_path, output_path, input_file_extension = None, output_file_extension = None):
	processing_pairs = list()
	if os.path.isdir(input_path):
		if not os.path.isdir(output_path):
			raise RuntimeError("If input path is a directory then output path must be a directory: " + output_path)
		print("Processing directory " + input_path)
		# Process any files found
		dir = os.listdir(input_path)
		for item in dir:
			input_filename, output_filename = get_processing_filenames(input_path, output_path, item, input_file_extension, output_file_extension)
			if input_filename is None or not os.path.isfile(input_filename):
				continue
			processing_pairs.append((input_filename, output_filename))
	elif os.path.isfile(input_path):
		# TODO If output_path does not exist, then its location must be a directory that exists
		if os.path.isdir(output_path):
			raise RuntimeError("If input path is a file then output path may not be a directory: " + output_path)
		if not input_file_extension is None and not input_path.endswith(input_file_extension):
			raise RuntimeError("Filename {} does not end with extension {}".format(input_path, input_file_extension))
		if not output_file_extension is None and not output_path.endswith(output_file_extension):
			raise RuntimeError("Filename {} does not end with extension {}".format(output_path, output_file_extension))
		print("Processing file " + input_path + " to " + output_path)
		# Process directly
		processing_pairs.append((input_path, output_path))
	else:  
		raise RuntimeError("Path is not a directory or normal file: " + input_path)
	return processing_pairs

def split_punct(text: str, start: int) -> Generator[Tuple[str, int, int], None, None]:
	# yield text, start, start + len(text)
	for m in re.finditer(r"""[\w']+|[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", text):
		yield m.group(), m.start() + start, m.end() + start

def tokenize_text(text: str, id, offset: int = 0) -> List[bioc.BioCSentence]:
	sentences = []
	doc = nlp(text)
	for sent in doc.sents:
		sentence = bioc.BioCSentence()
		sentence.infons["document_id"] = id
		sentence.offset = sent.start_char + offset
		sentence.text = text[sent.start_char:sent.end_char]
		sentences.append(sentence)
		i = 0
		for token in sent:
			for t, start, end in split_punct(token.text, token.idx):
				ann = bioc.BioCAnnotation()
				ann.id = f'a{i}'
				ann.text = t
				ann.add_location(bioc.BioCLocation(start + offset, end - start))
				sentence.add_annotation(ann)
				i += 1
	return sentences

def print_ner_debug(sentences: List[bioc.BioCSentence], start: int, end: int):
	anns = []
	for sentence in sentences:
		for ann in sentence.annotations:
			span = ann.total_span
			if start <= span.offset <= end \
					or start <= span.offset + span.length <= end:
				anns.append(ann)
	logging.debug('-' * 80)
	if len(anns) != 0:
		for ann in anns:
			logging.debug(ann)
	logging.debug('-' * 80)
	ss = [s for s in sentences if s.offset <= start <= s.offset + len(s.text)]
	if len(ss) != 0:
		for s in ss:
			logging.debug(s.offset, s.text)
	else:
		for s in sentences:
			logging.debug(s.offset, s.text)

def _find_toks(sentences, start, end):
	toks = []
	for sentence in sentences:
		for ann in sentence.annotations:
			span = ann.total_span
			if start <= span.offset and span.offset + span.length <= end:
				toks.append(ann)
			elif span.offset <= start and end <= span.offset + span.length:
				toks.append(ann)
	return toks

# TODO use actual tokenization
# TODO only split between two "O" labels

def write_bert_ner_file(total_sentences, filename):
	cnt = 0
	elements = []
	for sentence in total_sentences:
		ner_tags = []
		tokens = []
		spans = []
		for i, ann in enumerate(sentence.annotations):
			tokens.append(ann.text)
			ner_tags.append(ann.infons.get('NE_label', "O"))
			spans.append((ann.total_span.offset, ann.total_span.end))
		
		# TODO Do we want to drop sentences with zero length?
		if len(ner_tags) != len(tokens) or len(ner_tags) !=  len(spans):
			raise ValueError("lengths: ner_tags {} tokens {} spans {}".format(len(ner_tags), len(tokens), len(spans)))
		#print("lengths: ner_tags {} tokens {} spans {}".format(len(ner_tags), len(tokens), len(spans)))
		
		while len(ner_tags) > max_length:
			ner_tags2 = ner_tags[:max_length]
			tokens2 = tokens[:max_length]
			spans2 = spans[:max_length]
			element = {"id": len(elements), "document_id": sentence.infons["document_id"], "ner_tags": ner_tags2, "tokens": tokens2, "spans": spans2}
			elements.append(element)
			ner_tags = ner_tags[max_length:]
			tokens = tokens[max_length:]
			spans = spans[max_length:]
		
		element = {"id": len(elements), "document_id": sentence.infons["document_id"], "ner_tags": ner_tags, "tokens": tokens, "spans": spans}
		elements.append(element)
		cnt += 1
	with open(filename, 'w') as file:
		for element in elements:
			file.write(json.dumps(element) + "\n")
	return len(elements)

def convert_bioc_to_json(src, dest, entity_type = None):
	print("Converting {} to {}".format(src, dest))
	total_sentences = []
	with open(src, "r") as fp:
		collection = bioc.biocxml.load(fp)
	for document in collection.documents:
		#print("Processing document " + str(document.id) + ", number of sentences = " + str(len(total_sentences)))
		for passage in document.passages:
			text = passage.text
			sentences = tokenize_text(text, document.id, offset=passage.offset)
			total_sentences.extend(sentences)

			for ann in passage.annotations:
				anns = _find_toks(sentences, ann.total_span.offset, ann.total_span.end)
				if len(anns) == 0:
					logging.debug('%s: Cannot find %s', document.id, ann)
					print_ner_debug(sentences, ann.total_span.offset, ann.total_span.end)
					continue
				entity_type = ann.infons.get('type', "Unknown")
				has_first = False
				for ann in anns:
					if not has_first:
						ann.infons['NE_label'] = "B-" + entity_type
						has_first = True
					else:
						ann.infons['NE_label'] = "I-" + entity_type

	cnt = write_bert_ner_file(total_sentences, dest)
	logging.debug("Number of mentions: %s", cnt)

if __name__ == '__main__':
	main()