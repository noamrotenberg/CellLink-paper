import re
import json
import collections
import math

import string_utils

p1 = re.compile(r"[^\W_]+", re.UNICODE)
p2 = re.compile(r"[0-9]+")
cache = dict()

def mean_stdev_scaling(dimension_name, mean, stdev, params):
	mean_adj, stdev_adj = params
	return mean * mean_adj, stdev * stdev_adj

def exponential_scaling(dimension_name, per_doc_mean, item_means, params):
	per_doc_mean_adjustment = float(params[0])
	exponent = float(params[1])
	adjusted_means = {item_name: math.pow(original_mean, exponent) for item_name, original_mean in item_means.items()}
	normalizer = 1.0 / sum(adjusted_means.values())
	adjusted_means = {item_name: normalizer * adjusted_mean for item_name, adjusted_mean in adjusted_means.items()}
	return per_doc_mean_adjustment * per_doc_mean, adjusted_means

def get_tokens(text):
	tokens = p1.findall(string_utils.map_to_ASCII(text.lower()))
	tokens = [token for token in tokens if p2.fullmatch(token) is None]
	return tokens

def lexicon_tokens_TSV(dimension_name, per_doc_mean, item_means, params):
	per_doc_mean_adjustment = float(params[0])
	filename = params[1]
	alpha = float(params[2])
	cache_key = ("lexicon_tokens_TSV", filename)
	if not cache_key in cache:
		tokens = list()
		with open(filename) as file:
			for line in file:
				fields = line.split("\t")
				name_tokens = get_tokens(fields[0])
				tokens.extend(name_tokens)
		counts = collections.Counter(tokens)
		normalizer = 1.0 / sum(counts.values())
		cache[cache_key] = {token : normalizer * count for token, count in counts.items()}
	cached_adjustments = cache[cache_key]
	adjusted_means = {item_name: (1.0 - alpha) * original_mean + alpha * cached_adjustments.get(item_name, 0.0) for item_name, original_mean in item_means.items()}
	normalizer = 1.0 / sum(adjusted_means.values())
	adjusted_means = {item_name: normalizer * adjusted_mean for item_name, adjusted_mean in adjusted_means.items()}
	return per_doc_mean_adjustment * per_doc_mean, adjusted_means

def manual_JSON(dimension_name, per_doc_mean, item_means, params):
	filename = params[0]
	cache_key = ("manual_JSON", filename)
	if not cache_key in cache:
		with open(filename) as file:
			cache[cache_key] = json.load(file)
	per_doc_mean_adjustment, item_mean_adjustments = cache[cache_key].get(dimension_name, [1.0, dict()])
	adjusted_means = {item_name: original_mean * item_mean_adjustments.get(item_name, 1.0) for item_name, original_mean in item_means.items()}
	normalizer = 1.0 / sum(adjusted_means.values())
	adjusted_means = {item_name: normalizer * adjusted_mean for item_name, adjusted_mean in adjusted_means.items()}
	return per_doc_mean_adjustment * per_doc_mean, adjusted_means

