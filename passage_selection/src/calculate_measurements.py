import sys
import gzip 
import json

# required by getattr use below
import summary_calculators

if __name__ == '__main__':
	config_filename = sys.argv[1]
	summary_calculator = sys.argv[2]
	input_filename = sys.argv[3]
	output_filename = sys.argv[4]

	with open(config_filename) as config_file:
		config = json.load(config_file)
	print("Loaded {} dimensions".format(len(config)))

	print("Loading passages")
	if input_filename.endswith(".gz"):
		input_file = gzip.open(input_filename, "rt") 
	else:
		input_file = open(input_filename, "r") 
	# Get intermediate measurements from data
	intermediate_measurements = dict()
	passage_count = 0
	for line in input_file:
		passage = json.loads(line)
		passage_data = passage["data"]
		passage_count += 1
		for dimension_name, dimension_config in config.items():
			if not summary_calculator in dimension_config:
				continue
			data_type = config[dimension_name]["data_type"]
			dimension_values = passage_data.get(dimension_name)
			if dimension_values is None:
				if data_type == "singleton":
					dimension_values = None
				elif data_type == "count_dict":
					dimension_values = {}
				elif data_type == "number":
					dimension_values = 0
				else:
					raise ValueError("Unknown data type: {}".format(data_type))
			summary_function, params = config[dimension_name][summary_calculator]
			name_fields = summary_function.split(".")
			getattr(sys.modules[name_fields[0]], name_fields[1])(intermediate_measurements, dimension_name, data_type, dimension_values, params)
	input_file.close()
	print("Number of passages is {}".format(passage_count))
		
	# Get summaries
	summaries = dict()
	summaries["PASSAGE_COUNT"] = passage_count
	for dimension_name, intermediate_values in intermediate_measurements.items():
		summaries[dimension_name] = intermediate_values.summarize()

	# Output
	if output_filename.endswith(".gz"):
		with gzip.open(output_filename, "wt") as output_file:
			json.dump(summaries, output_file, indent = 3)
	else:
		with open(output_filename, "w") as output_file:
			json.dump(summaries, output_file, indent = 3)

	print("Done.")
