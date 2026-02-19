import sys
import gzip 
import json

# required by getattr use below
import adjuster_functions

epsilon = 1e-8

if __name__ == '__main__':
	config_filename = sys.argv[1]
	measurements_input_filename = sys.argv[2]
	measurements_output_filename = sys.argv[3]

	with open(config_filename) as config_file:
		config = json.load(config_file)
	print("Loaded {} dimensions".format(len(config)))

	print("Loading measurements")
	if measurements_input_filename.endswith(".gz"):
		with gzip.open(measurements_input_filename, "rt") as file:
			measurements = json.load(file)
	else:
		with open(measurements_input_filename, "r") as file:
			measurements = json.load(file)
	print("Number of measurements is {}".format(len(measurements)))

	# TODO: Verify summary is "summary_calculators.mean"
	# Run adjuster functions to get adjusted means
	adjusted_measurements = dict()
	for dimension, dimension_values in measurements.items():
		print("adjusting dimension {}".format(dimension))
		if dimension == "PASSAGE_COUNT":
			adjusted_measurements[dimension] = dimension_values
			continue
		dimension_config = config[dimension]
		if not "adjusters" in dimension_config:
			adjusted_measurements[dimension] = dimension_values
			continue
		data_type = dimension_config["data_type"]
		print("data_type is {}".format(data_type))
		if data_type == "number":
			print("@1")
			mean, stdev = dimension_values
			if len(config[dimension]["adjusters"]) != 1:
				raise ValueError("Not implemented")
			adjuster_function, params = config[dimension]["adjusters"][0]
			name_fields = adjuster_function.split(".")
			mean_adj, stdev_adj = getattr(sys.modules[name_fields[0]], name_fields[1])(dimension, mean, stdev, params)
			adjusted_measurements[dimension] = [mean_adj, stdev_adj]
		elif data_type == "singleton" or data_type == "count_dict":
			print("@2")
			per_doc_mean, item_values = dimension_values
			item_means = item_values.copy()
			for adjuster_function, params in config[dimension]["adjusters"]:
				name_fields = adjuster_function.split(".")
				per_doc_mean, item_means = getattr(sys.modules[name_fields[0]], name_fields[1])(dimension, per_doc_mean, item_means, params)
			if len(item_means) != len(item_values):
				raise ValueError("Adjusted length does not match length")
			if abs(sum(item_means.values()) - 1.0) > epsilon:
				raise ValueError("Total item mean for dimension {} != 1.0: {}".format(dimension, sum(item_means.values())))
			if data_type == "singleton" and abs(per_doc_mean - 1.0) > epsilon:
				raise ValueError("Per doc mean for dimension {} != 1.0: {}".format(dimension, per_doc_mean))
			item_mean_list = [(item_mean, item_name) for item_name, item_mean in item_means.items()]
			item_mean_list.sort(reverse = True)
			adjusted_measurements[dimension] = [per_doc_mean, {item_name: item_mean for item_mean, item_name in item_mean_list}]
		else:
			raise ValueError("Not implemented")
			
	# Output
	if measurements_output_filename.endswith(".gz"):
		with gzip.open(measurements_output_filename, "wt") as output_file:
			json.dump(adjusted_measurements, output_file, indent = 3)
	else:
		with open(measurements_output_filename, "w") as output_file:
			json.dump(adjusted_measurements, output_file, indent = 3)

	print("Done.")
