import copy
import json
import math
import re

import scipy

import string_utils

# filters tokens containing 
p1 = re.compile(r"[^\W_]+", re.UNICODE)
p2 = re.compile(r"[0-9]+")

cache = dict()

def make_copy(original_data, measurements, dimension_name, config, transformed_data, params):
	original_dimension_values = original_data.get(dimension_name)
	if original_dimension_values is None:
		return
	transformed_dimension_values = copy.deepcopy(original_dimension_values)
	transformed_data[dimension_name] = transformed_dimension_values

def item_mapping(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, mappings_filename = params
	if mappings_filename in cache:
		item_mappings = cache[mappings_filename]
	else:
		print("Loading mappings_filename: {}".format(mappings_filename))
		with open(mappings_filename, "r") as mappings_file:
			item_mappings = json.load(mappings_file)
			cache[mappings_filename] = item_mappings
	data_type = config[source_dimension_name]["data_type"]
	if data_type == "number":
		raise ValueError("Cannot apply item_mapping to data type {}".format(data_type))
	if data_type == "singleton":
		item_name = original_data[source_dimension_name]
		transformed_data[dimension_name] = item_mappings.get(item_name)
	elif data_type == "count_dict":
		#print("item_mapping@1")
		if not dimension_name in transformed_data:
			transformed_data[dimension_name] = dict()
		original_item_values = original_data[source_dimension_name]
		for item_name, item_count in original_item_values.items():
			#print("item_mapping, item_name: {}".format(item_name))
			if not item_name in item_mappings:
				continue
			item_name2 = item_mappings[item_name]
			#print("item_mapping, item_name2: {}".format(item_name2))
			if not item_name2 in transformed_data[dimension_name]:
				transformed_data[dimension_name][item_name2] = 0
			transformed_data[dimension_name][item_name2] += item_count
	else:
		raise ValueError("Unknown data type: {}".format(data_type))

def item_multi_mapping(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, mappings_filename = params
	if mappings_filename in cache:
		item_mappings = cache[mappings_filename]
	else:
		print("Loading mappings_filename: {}".format(mappings_filename))
		with open(mappings_filename, "r") as mappings_file:
			item_mappings = json.load(mappings_file)
			cache[mappings_filename] = item_mappings
	data_type = config[source_dimension_name]["data_type"]
	if not dimension_name in transformed_data:
		transformed_data[dimension_name] = dict()
	original_item_values = original_data[source_dimension_name]
	for item_name, item_count in original_item_values.items():
		#print("item_mapping, item_name: {}".format(item_name))
		if not item_name in item_mappings:
			continue
		for item_name2 in item_mappings[item_name]:
			#print("item_mapping, item_name2: {}".format(item_name2))
			if not item_name2 in transformed_data[dimension_name]:
				transformed_data[dimension_name][item_name2] = 0
			transformed_data[dimension_name][item_name2] += item_count

def passage_zscore(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, ndigits = params
	original_data_type = config[source_dimension_name]["data_type"]
	if original_data_type != "number":
		raise ValueError("Cannot apply passage_zscore to data type {}".format(original_data_type))
	transformed_data_type = config[dimension_name]["data_type"]
	if transformed_data_type != "singleton":
		raise ValueError("Cannot apply passage_zscore to data type {}".format(transformed_data_type))
	mean, std = measurements[source_dimension_name]
	value = original_data[source_dimension_name]
	transformed_data[dimension_name] = str(round(abs(mean - value) / std, ndigits))

def passage_signed_zscore(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, ndigits = params
	original_data_type = config[source_dimension_name]["data_type"]
	if original_data_type != "number":
		raise ValueError("Cannot apply passage_zscore to data type {}".format(original_data_type))
	transformed_data_type = config[dimension_name]["data_type"]
	if transformed_data_type != "singleton":
		raise ValueError("Cannot apply passage_zscore to data type {}".format(transformed_data_type))
	mean, std = measurements[source_dimension_name]
	value = original_data[source_dimension_name]
	zscore = round((value - mean) / std, ndigits)
	if zscore == -0.0:
		zscore = abs(zscore)
	transformed_data[dimension_name] = str(zscore)

def rare_threshold(original_data, measurements, dimension_name, config, transformed_data, params):
	rare_name, sample_size, probability_threshold = params
	probability_threshold = 1.0 - probability_threshold
	data_type = config[dimension_name]["data_type"]
	if data_type == "number":
		raise ValueError("Cannot apply rare_threshold to data type {}".format(data_type))
	per_doc_mean, item_means = measurements[dimension_name]
	cache_key = (dimension_name, "rare_threshold")
	if cache_key in cache:
		rare_items = cache[cache_key]
	else:
		#print("Creating rare mappings for: {}".format(cache_key))
		item_rates = list()
		for item_name, item_mean in item_means.items():
			item_rates.append((per_doc_mean * item_mean, item_name))
		item_rates.sort()
		rare_items = set()
		total_rate = 0.0
		for item_rate, item_name in item_rates:
			item_prob = 1.0 - scipy.stats.poisson.cdf(1, sample_size * (total_rate + item_rate))
			#print("probability for {} with rate {} is {}".format(item_name, item_rate, item_prob))
			if item_prob <= probability_threshold:
				rare_items.add(item_name)
				total_rate += item_rate
			else:
				break
		cache[cache_key] = rare_items
		#print("Found: {}".format(rare_items))
	if data_type == "singleton":
		item_name = original_data[dimension_name]
		item_name2 = item_name if not item_name in rare_items else rare_name
		transformed_data[dimension_name] = item_name2
	elif data_type == "count_dict":
		if not dimension_name in transformed_data:
			transformed_data[dimension_name] = dict()
		original_item_values = original_data[dimension_name]
		for item_name, item_count in original_item_values.items():
			item_name2 = item_name if not item_name in rare_items else rare_name
			if not item_name2 in transformed_data[dimension_name]:
				transformed_data[dimension_name][item_name2] = 0
			transformed_data[dimension_name][item_name2] += item_count
	else:
		raise ValueError("Unknown data type: {}".format(data_type))

def uniqueness(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name = params[0]
	data_type = config[dimension_name]["data_type"]
	if data_type != "count_dict":
		raise ValueError("Cannot apply uniqueness to data type {}".format(data_type))
	per_passage_mean, item_means = measurements[source_dimension_name]
	cache_key = (dimension_name, "uniqueness")
	if cache_key in cache:
		item_scores = cache[cache_key]
	else:
		#print("Creating item counts for: {}".format(cache_key))
		item_scores = dict()
		passage_count = measurements["PASSAGE_COUNT"]
		# TODO Split item name into type and mention
		for item_name, item_mean in item_means.items():
			fields = item_name.split("'")
			item_type = fields[1]
			item_value = fields[3]
			if not item_type in item_scores:
				item_scores[item_type] = dict()
			count = passage_count * per_passage_mean * item_mean
			score = 1.0 / count
			#print("UNIQENESS: item_name = \"{}\" item_type = \"{}\" item_value = \"{}\" count = {} score = {}".format(item_name, item_type, item_value, count, score))
			item_scores[item_type][item_value] = score
		cache[cache_key] = item_scores
		#print("Found: {}".format(rare_items))
	original_item_values = original_data[source_dimension_name]
	unique_counts = dict()
	for item_name, item_count in original_item_values.items():
		fields = item_name.split("'")
		item_type = fields[1]
		item_value = fields[3]
		if not item_type in unique_counts:
			unique_counts[item_type] = [0.0, 0.0]
		unique_counts[item_type][0] += item_scores[item_type][item_value]
		unique_counts[item_type][1] += 1.0
	if not dimension_name in transformed_data:
		transformed_data[dimension_name] = dict()
	for item_type, (unique_count, total) in unique_counts.items():
		transformed_data[dimension_name][item_type] = (unique_count / total) * math.log(unique_count + 1.0)

def raw_list_counts(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, item_key_indexes, count_set_indexes = params
	#print("raw_list_counts: item_key_indexes {} count_set_indexes {}".format(item_key_indexes, count_set_indexes))
	if config[source_dimension_name]["data_type"] != "raw":
		raise ValueError("Cannot extract raw_list_counts from data type {}".format(config[source_dimension_name]["data_type"]))
	if config[dimension_name]["data_type"] != "count_dict":
		raise ValueError("Cannot apply raw_list_counts to data type {}".format(config[dimension_name]["data_type"]))
	count_sets = dict()
	for annotation_tuple in original_data.get(source_dimension_name, []):
		key = list()
		for item_key_index in item_key_indexes:
			key.append(annotation_tuple[item_key_index])
		key = tuple(key)
		if not key in count_sets:
			count_sets[key] = set()
		count_set_value = list()
		for count_set_index in count_set_indexes:
			count_set_value.append(annotation_tuple[count_set_index])
		count_set_value = tuple(count_set_value)
		#print("raw_list_counts: annotation_tuple {} becomes key {} count_set_value {}".format(annotation_tuple, key, count_set_value))
		count_sets[key].add(count_set_value)
	if not dimension_name in transformed_data:
		transformed_data[dimension_name] = dict()
	for key, count_set in count_sets.items():
		key_str = str(key)
		if not key_str in transformed_data[dimension_name]:
			transformed_data[dimension_name][key_str] = 0
		transformed_data[dimension_name][key_str] += len(count_set)

def get_tokens(text):
	tokens = p1.findall(string_utils.map_to_ASCII(text.lower()))
	tokens = [token for token in tokens if p2.fullmatch(token) is None]
	return tokens

def raw_list_token_counts(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, item_key_indexes, key_token_index, count_set_indexes = params
	#print("raw_list_counts: item_key_indexes {} count_set_indexes {}".format(item_key_indexes, count_set_indexes))
	if config[source_dimension_name]["data_type"] != "raw":
		raise ValueError("Cannot extract raw_list_counts from data type {}".format(config[source_dimension_name]["data_type"]))
	if config[dimension_name]["data_type"] != "count_dict":
		raise ValueError("Cannot apply raw_list_counts to data type {}".format(config[dimension_name]["data_type"]))
	if not key_token_index in item_key_indexes:
		raise ValueError()
	count_sets = dict()
	for annotation_tuple in original_data.get(source_dimension_name, []):
		# prepare value first
		count_set_value = list()
		for count_set_index in count_set_indexes:
			count_set_value.append(annotation_tuple[count_set_index])
		count_set_value = tuple(count_set_value)
		# Get tokens
		pre_token_text = annotation_tuple[key_token_index]
		tokens = get_tokens(pre_token_text)
		for token in tokens:
			# prepare keys
			key = list()
			for item_key_index in item_key_indexes:
				if item_key_index == key_token_index:
					key.append(token)
				else:
					key.append(annotation_tuple[item_key_index])
			key = tuple(key)
			print("raw_list_token_counts: annotation_tuple {} becomes key {} count_set_value {}".format(annotation_tuple, key, count_set_value))
			if not key in count_sets:
				count_sets[key] = set()
			count_sets[key].add(count_set_value)
		if not dimension_name in transformed_data:
			transformed_data[dimension_name] = dict()
		for key, count_set in count_sets.items():
			key_str = str(key)
			if not key_str in transformed_data[dimension_name]:
				transformed_data[dimension_name][key_str] = 0
			transformed_data[dimension_name][key_str] += len(count_set)

def raw_list_bitvector_counts(original_data, measurements, dimension_name, config, transformed_data, params):
	source_dimension_name, item_key_indexes, count_set_indexes, bitvector_index, bitvector_items = params
	bitvector_item2index = dict()
	for index, bitvector_item in enumerate(bitvector_items):
		bitvector_item2index[bitvector_item] = index
	#print("raw_list_bitvector_counts: item_key_indexes {} count_set_indexes {}".format(item_key_indexes, count_set_indexes))
	if config[source_dimension_name]["data_type"] != "raw":
		raise ValueError("Cannot extract raw_list_bitvector_counts from data type {}".format(config[source_dimension_name]["data_type"]))
	if config[dimension_name]["data_type"] != "count_dict":
		raise ValueError("Cannot apply raw_list_bitvector_counts to data type {}".format(config[dimension_name]["data_type"]))
	count_sets = dict()
	for annotation_tuple in original_data.get(source_dimension_name, []):
		key = list()
		for item_key_index in item_key_indexes:
			key.append(annotation_tuple[item_key_index])
		key = tuple(key)
		if not key in count_sets:
			count_sets[key] = dict()
		count_set_value = list()
		for count_set_index in count_set_indexes:
			count_set_value.append(annotation_tuple[count_set_index])
		count_set_value = tuple(count_set_value)
		if not count_set_value in count_sets[key]:
			count_sets[key][count_set_value] = ["0"] * len(bitvector_items)
		bitvector_item = annotation_tuple[bitvector_index]
		count_sets[key][count_set_value][bitvector_item2index[bitvector_item]] = "1"
		#print("raw_list_bitvector_counts: annotation_tuple {} becomes key {} count_set_value {}".format(annotation_tuple, key, count_set_value))
	if not dimension_name in transformed_data:
		transformed_data[dimension_name] = dict()
	for key, count_set2bitvector in count_sets.items():
		for count_set_value, bitvector in count_set2bitvector.items():
			key2 = list(key)
			key2.append("".join(bitvector))
			key_str = str(tuple(key2))
			if not key_str in transformed_data[dimension_name]:
				transformed_data[dimension_name][key_str] = 0
			transformed_data[dimension_name][key_str] += 1
