import sys
import gzip 
import json
import collections

# required by getattr use below
import data_filters

if __name__ == '__main__':
	config_filename = sys.argv[1]
	passage_input_filename = sys.argv[2]
	measurements_input_filename = sys.argv[3]
	passage_output_filename = sys.argv[4]

	with open(config_filename) as config_file:
		config = json.load(config_file)
	print("Loaded {} dimensions".format(len(config)))
	filtered_dimensions = {dimension_name for dimension_name, dimension_config in config.items() if "data_filter" in dimension_config}

	print("Loading measurements")
	if measurements_input_filename.endswith(".gz"):
		with gzip.open(measurements_input_filename, "rt") as file:
			measurements = json.load(file)
	else:
		with open(measurements_input_filename, "r") as file:
			measurements = json.load(file)
	print("Number of measurements is {}".format(len(measurements)))

	print("Filtering passages")
	if passage_input_filename.endswith(".gz"):
		passage_input_file = gzip.open(passage_input_filename, "rt") 
	else:
		passage_input_file = open(passage_input_filename, "r") 
	if passage_output_filename.endswith(".gz"):
		passage_output_file = gzip.open(passage_output_filename, "wt") 
	else:
		passage_output_file = open(passage_output_filename, "w") 

	filtered_counts = collections.Counter()
	keep_values = set()
	passage_count = 0
	for line in passage_input_file:
		passage = json.loads(line)
		passage_count += 1
		passage_data = passage["data"]
		filter_passage = False
		for dimension_name, values in passage_data.items():
			if not dimension_name in filtered_dimensions:
				continue
			data_type = config[dimension_name]["data_type"]
			filter_function, params = config[dimension_name]["data_filter"]
			name_fields = filter_function.split(".")
			filter_passage2 = getattr(sys.modules[name_fields[0]], name_fields[1])(passage_data, measurements, dimension_name, data_type, params)
			if filter_passage2:
				filter_passage = True
				filtered_counts[dimension_name] += 1
		if filter_passage:
			continue
		for dimension_name, values in passage_data.items():
			if not dimension_name in config:	
				continue
			data_type = config[dimension_name]["data_type"]
			if data_type == "singleton":
				value_name = values
				keep_values.add((dimension_name, value_name))
			elif data_type == "count_dict":
				for value_name in values.keys():
					keep_values.add((dimension_name, value_name))
			elif data_type == "number":
				keep_values.add((dimension_name, None))
			elif data_type != "raw":
				raise ValueError("Unknown data type: {}".format(data_type))
		passage_output_file.write(json.dumps(passage) + "\n")
	
	passage_input_file.close()
	passage_output_file.close()

	# Log filtering results
	for dimension_name, count in filtered_counts.items():
		print("Dimension {} filtered {} passages ({}%)".format(dimension_name, count, count / passage_count))
	for dimension_name, count in filtered_counts.items():
		print("Dimension {} filtered {} values ({}%)".format(dimension_name, count, count / len(measurements[dimension_name])))
	for dimension_name, value_name in keep_values:	
		print("Keeping dimension {} value {}".format(dimension_name, value_name))
	for dimension_name, values in measurements.items():
		if not dimension_name in config:
			print(f"INFO: dimension {dimension_name} not in configuration, ignoring...")
			continue
		data_type = config[dimension_name]["data_type"]
		if data_type == "singleton" or data_type == "count_dict":
			per_doc_mean, item_values = values
			for item_name, value in item_values.items():
				check = (dimension_name, item_name)
				if not check in keep_values:
					print("Dropping measurement ({}, {}) = {}".format(dimension_name, item_name, value))
					continue
		elif data_type == "number":
			check = (dimension_name, None)
			if not check in keep_values:
				print("Dropping measurement ({},) = {}".format(dimension_name, values))
				continue
		else:
			raise ValueError("Unknown data type: {}".format(data_type))

	print("Done.")
