import collections
import json
import logging
import os
import re
import sys
from typing import Generator, Tuple, List

import bioc
import spacy

nlp = spacy.load("en_core_sci_sm")

# TODO: How does this handle disjunct spans?

def main():
	if len(sys.argv) < 3:
		print("Usage: <input_path>* <output_file>")
		exit()
	input_paths = sys.argv[1:-1]
	output_file = sys.argv[-1]

	# Get the list of input files
	file_list = list()
	for input_path in input_paths:
		file_list.extend(list_files_recursive(input_path))
	print("Found " + str(len(file_list)) + " input files")

	count = convert_bioc_to_json(file_list, output_file)
	print("Total number of sentences = " + str(count))
	
	print("Done.")

def list_files_recursive(input_path):
	file_list = list()
	dir_list = list()
	if os.path.isfile(input_path):
		file_list.append(input_path)
	elif os.path.isdir(input_path):
		dir_list.append(input_path)
	else:
		raise RuntimeError("Input path must be a file or directory: " + input_path)
	while len(dir_list) > 0:
		dir_name = dir_list.pop()
		print("Processing directory " + dir_name)
		dir = os.listdir(dir_name)
		for item in dir:
			input_filename = dir_name
			if not input_filename.endswith("/"):
				input_filename += "/"
			input_filename += item
			#print("Checking item " + input_filename)
			if os.path.isfile(input_filename):
				file_list.append(input_filename)
			elif os.path.isdir(input_filename):
				dir_list.append(input_filename)
	return file_list

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

def write_bert_ner_file_ORIGINAL(total_sentences, filename):
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
		element = {"id": len(elements), "document_id": sentence.infons["document_id"], "ner_tags": ner_tags, "tokens": tokens, "spans": spans}
		elements.append(element)
		cnt += 1
	with open(filename, 'w') as file:
		for element in elements:
			file.write(json.dumps(element) + "\n")
	return len(elements)

max_length = 96
# TODO use actual tokenization
# TODO make max length configurable
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

def convert_bioc_to_json(srcs, dest, entity_type = None):
	total_sentences = []
	for src in srcs:
		with open(src, "r") as fp:
			collection = bioc.biocxml.load(fp)
		for document in collection.documents:
			print("Processing document " + str(document.id) + ", number of sentences = " + str(len(total_sentences)))
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
	return cnt

def verify_ann(src):
	with open(src, encoding='utf8') as fp:
		collection = bioc.biocxml.load(fp)

	cnt = collections.Counter()
	for doc in collection.documents:
		for passage in tqdm.tqdm(doc.passages):
			for ann in passage.annotations:
				expected_text = ann.text
				start = ann.total_span.offset - passage.offset
				end = ann.total_span.end - passage.offset
				actual_text = passage.text[start: end]
				if expected_text != actual_text:
					logging.debug('%s:%s: %s vs %s', doc.id, passage.offset,
								  expected_text, actual_text)
				if ' ' in expected_text:
					cnt['has space'] += 1
	logging.debug('%s: %s', src, cnt)

if __name__ == '__main__':
	main()