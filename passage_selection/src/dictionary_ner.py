import bioc

from trie import Trie
import string_utils
import PerformanceProfiler

class DictionaryRecognizer:
	
	def __init__(self, convert_string, tokenize):
		self.convert_string = convert_string
		self.tokenize = tokenize
		self.terms = Trie()
	
	def add_term(self, term, type, identifier):
		token_spans = self.tokenize(term)
		unparen_spans = unparen(token_spans)
		tokens = [self.convert_string(token) for token, start, end in token_spans]
		tokens = [token for token in tokens if len(token) > 0]
		self.terms.add(tokens, (type, identifier))
		if not unparen_spans is None:
			unparen_tokens = [self.convert_string(token) for token, start, end in unparen_spans]
			unparen_tokens = [token for token in unparen_tokens if len(token) > 0]
			self.terms.add(unparen_tokens, (type, identifier))
	
	def load_terms(self, filename):
		with open(filename, "r") as file:
			for line in file:
				line = line.strip()
				if len(line) == 0:
					continue
				fields = line.split("\t")
				if len(fields) != 3:
					raise ValueError("Dictionary line has incorrect number of fields: \"{}\"".format(line))
				self.add_term(fields[0], fields[1], fields[2])
	
	def process(self, document):
		PerformanceProfiler.start("dictionary_ner.process()")
		annotation_count = 0
		for passage in document.passages:
			annotations = self.get_annotations(document.id, passage.text)
			while self.combine_annotations(passage.text, annotations):
				pass
			self.filter_annotations(annotations)
			for ann in annotations:
				#print("ADDING Adding annotation \"" + str(ann))
				annb = bioc.BioCAnnotation()
				annb.id = str(annotation_count)
				annb.text = ann[3]
				annb.infons["type"] = ann[4]
				annb.infons["identifier"] = ann[5]
				annb.locations.append(bioc.BioCLocation(passage.offset + ann[1], ann[2]))
				passage.annotations.append(annb)
				annotation_count += 1
		PerformanceProfiler.end("dictionary_ner.process()")
	
	def get_annotations(self, doc_ID, passage_text):
		PerformanceProfiler.start("dictionary_ner.get_annotations()")
		token_spans = self.tokenize(passage_text)
		unparen_spans = unparen(token_spans)
		converted_spans = [(self.convert_string(token), start, end) for token, start, end in token_spans]
		converted_spans = [(token, start, end) for token, start, end in converted_spans if len(token) > 0]
		converted_tokens = [token for token, start, end in converted_spans]
		annotations = set()
		self.get_annotations_for_tokens(doc_ID, passage_text, converted_tokens, converted_spans, annotations)
		if not unparen_spans is None:
			unparen_converted_spans = [(self.convert_string(token), start, end) for token, start, end in unparen_spans]
			unparen_converted_spans = [(token, start, end) for token, start, end in unparen_converted_spans if len(token) > 0]
			unparen_converted_tokens = [token for token, start, end in unparen_converted_spans]
			self.get_annotations_for_tokens(doc_ID, passage_text, unparen_converted_tokens, unparen_converted_spans, annotations)
		annotations = list(annotations)
		annotations.sort()
		PerformanceProfiler.end("dictionary_ner.get_annotations()")
		return annotations

	def get_annotations_for_tokens(self, doc_ID, passage_text, tokens, spans, annotations):
		PerformanceProfiler.start("dictionary_ner.get_annotations_for_tokens()")
		for (type, identifier), start_index, end_index in self.terms.lookup(tokens):
			char_start = spans[start_index][1]
			char_end = spans[end_index-1][2]
			mention_text = passage_text[char_start:char_end]
			if not string_utils.balanced_paren(mention_text):
				continue
			ann = (doc_ID, char_start, char_end - char_start, mention_text, type, identifier)
			lookup = tokens[start_index:end_index]
			#print("FOUND Found annotation \"{}\" from mention \"{}\" and lookup text \"{}\"".format(ann, mention_text, lookup))
			annotations.add(ann)
		PerformanceProfiler.end("dictionary_ner.get_annotations_for_tokens()")

	# Filters annotations by looking for ones that are contained within another annotation
	def filter_annotations(self, annotations):
		PerformanceProfiler.start("dictionary_ner.filter_annotations()")
		keep = set(annotations)
		for i in range(0, len(annotations)):
			ann1 = annotations[i]
			for j in range(0, len(annotations)):
				if i == j:
					continue
				ann2 = annotations[j]
				if ann1[2] > ann2[2] and ann1[1] <= ann2[1] and ann1[1] + ann1[2] >= ann2[1] + ann2[2]:
					#print("FILTER Dropping annotation \"" + str(ann2) + "\" because it is contained in \"" + str(ann1) + "\"")
					keep.discard(ann2)
		annotations.clear()
		annotations.extend(keep)
		annotations.sort()
		PerformanceProfiler.end("dictionary_ner.filter_annotations()")

	def combine_annotations(self, passage_text, annotations):
		PerformanceProfiler.start("dictionary_ner.combine_annotations()")
		changed = False
		keep = set()
		handled = set()
		for i in range(0, len(annotations)):
			ann1 = annotations[i]
			if ann1 in handled:
				continue
			for j in range(0, len(annotations)):
				ann2 = annotations[j]
				if i == j or ann2 in handled:
					continue
				if ann1[1] < ann2[1] + ann2[2] and ann2[1] < ann1[1] + ann1[2] and ann1[4] == ann2[4] and ann1[5] == ann2[5]:
					start = min(ann1[1], ann2[1])
					end = max(ann1[1] + ann1[2], ann2[1] + ann2[2])
					mention_text = passage_text[start:end]
					ann3 = (ann1[0], start, end - start, mention_text, ann1[4], ann1[5])
					#print("COMBINE Combining annotation \"" + str(ann1) + "\" with \"" + str(ann2) + "\" to make \"" + str(ann3) + "\"")
					keep.add(ann3)
					handled.add(ann1)
					handled.add(ann2)
					changed = True
			if not ann1 in handled:
				keep.add(ann1)
				handled.add(ann1)
		annotations.clear()
		annotations.extend(keep)
		annotations.sort()
		PerformanceProfiler.end("dictionary_ner.combine_annotations()")
		return changed

open_paren = ["[","{","("] 
close_paren = ["]","}",")"] 

def unparen(token_spans):
	token_spans2 = list()
	stack = [] 
	changed = False
	for token, start, end in token_spans: 
		if token in open_paren: 
			stack.append(token) 
			changed = True
		elif token in close_paren: 
			pos = close_paren.index(token) 
			if ((len(stack) > 0) and
				(open_paren[pos] == stack[len(stack)-1])): 
				stack.pop() 
			else: 
				return token_spans2
		elif len(stack) == 0:
			token_spans2.append((token, start, end))
	if not changed:
		return None
	return token_spans2
