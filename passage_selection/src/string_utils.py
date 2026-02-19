import re
import sys

import unidecode

# Translation: looks for 4 classes
# Class 1: Sequences of letters: [^\W\d_]+
# Class 2: Sequences of digits: [\d]+
# Class 3: Single symbol or punctuation characters: [^\w\s]
# Class 4: Single underscores or whitespace: [_\s]
# Note: "word" characters are alphabetic, digits and underscore
token_regex = "[^\\W\\d_]+|[\\d]+|[^\\w\\s]|_|,"
p_tokens = re.compile(token_regex, re.UNICODE)

def tokenize(text):
	return [(m.group(), m.start(), m.end()) for m in p_tokens.finditer(text)]

p2 = re.compile("\\s+")
p5 = re.compile("[A-Za-z]")

custom_ASCII_map = {
	# Suppressed (not useful for name matching)
	u"\u2122": u" ", # (TM)
	# Look-alikes: Russian
	u"\u0412": u"B",
	u"\u041D": u"H",
	u"\u0421": u"P",
	u"\u0425": u"X",
	u"\u0432": u"b",
	u"\u043D": u"h",
	u"\u0441": u"p",
	u"\u0445": u"x"
}

custom_ASCII_expansion = {
	# No exact mapping: Greek upper case
	u"\u0391": u"alpha",
	u"\u0392": u"beta",
	u"\u0393": u"gamma",
	u"\u0394": u"delta",
	u"\u0395": u"epsilon",
	u"\u0396": u"zeta",
	u"\u0397": u"eta",
	u"\u0398": u"theta",
	u"\u0399": u"iota",
	u"\u039A": u"kappa",
	u"\u039B": u"lamda",
	u"\u039C": u"mu",
	u"\u039D": u"nu",
	u"\u039E": u"xi",
	u"\u039F": u"omicron",
	u"\u03A0": u"pi",
	u"\u03A1": u"rho",
	u"\u03A3": u"sigma",
	u"\u03A4": u"tau",
	u"\u03A5": u"upsilon",
	u"\u03A6": u"phi",
	u"\u03A7": u"chi",
	u"\u03A8": u"psi",
	u"\u03A9": u"omega",
	# No exact mapping: Greek lower case
	u"\u03B1": u"alpha",
	u"\u03B2": u"beta",
	u"\u03B3": u"gamma",
	u"\u03B4": u"delta",
	u"\u03B5": u"epsilon",
	u"\u03B6": u"zeta",
	u"\u03B7": u"eta",
	u"\u03B8": u"theta",
	u"\u03B9": u"iota",
	u"\u03BA": u"kappa",
	u"\u03BB": u"lamda",
	u"\u03BC": u"mu",
	u"\u03BD": u"nu",
	u"\u03BE": u"xi",
	u"\u03BF": u"omicron",
	u"\u03C0": u"pi",
	u"\u03C1": u"rho",
	u"\u03C2": u"sigma",
	u"\u03C3": u"sigma",
	u"\u03C4": u"tau",
	u"\u03C5": u"upsilon",
	u"\u03C6": u"phi",
	u"\u03C7": u"chi",
	u"\u03C8": u"psi",
	u"\u03C9": u"omega"
}

def map_to_ASCII(text):
	# Apply custom mappings
	new_text = __custom_map_to_ASCII__(text)
	# Map remainder
	new_text = unidecode.unidecode(new_text)
	# Verify result is ASCII
	new_text = __restrict_to_ASCII__(new_text)
	new_text = p2.sub(" ", new_text.strip())
	return new_text

def __custom_map_to_ASCII__(text):
	# Apply custom mappings
	new_text = ""
	for i in range(len(text)):
		c = text[i]
		if c in custom_ASCII_map:
			new_text += custom_ASCII_map[c]
		elif c in custom_ASCII_expansion:
			# Add spaces if needed
			if i > 0 and p5.fullmatch(text[i-1]):
				new_text += " "
			new_text += custom_ASCII_expansion[c]
			if i+1 < len(text) and p5.fullmatch(text[i+1]):
				new_text += " "
		else:
			new_text += c
	return new_text

def __restrict_to_ASCII__(text):
	new_text = ""
	for c in text:
		if ord(c) < 128:
			new_text += c
		else:
			sys.stderr.write("WARN MISSING Unicode: " + hex(ord(c)) + " " + c + "\n")
			new_text += " "
	return new_text

# A conservative stemmer that tries to only remove plurals
# Known errors: "foxes"
def s_stem(string):
	# TODO compile these so it's faster
	# Note assumption str is already lowercase
	if not string.endswith("s"):
		return string
	if (string.endswith("viruses")):
		return re.sub("uses$", "us", string)
	# If word ends in "ies" but not "eies" or "aies" then "ies" --> "y"
	if (string.endswith("ies") and not string.endswith("eies") and not string.endswith("aies")):
		return re.sub("ies$", "y", string)
	# If a word ends in "es" but not "aes" "ees" or "oes" --> "es" --> "e"
	if (string.endswith("es") and not string.endswith("aes") and not string.endswith("ees") and not string.endswith("oes")):
		if string.endswith("sses"):
			return re.sub("es$", "", string)
		else:
			return re.sub("es$", "e", string)			
	# If a word ends in "s" but not "us" or "ss" then "s" --> null
	if (not string.endswith("us") and not string.endswith("ss")):
		return re.sub("s$", "", string)
	# Return as-is
	return string

# balanced parentheses in an expression 
open_paren = ["[","{","("] 
close_paren = ["]","}",")"] 
  
# Function to check parentheses 
def balanced_paren(str):
	stack = [] 
	for i in str: 
		if i in open_paren: 
			stack.append(i) 
		elif i in close_paren: 
			pos = close_paren.index(i) 
			if ((len(stack) > 0) and
				(open_paren[pos] == stack[len(stack)-1])): 
				stack.pop() 
			else: 
				return False
	return len(stack) == 0
