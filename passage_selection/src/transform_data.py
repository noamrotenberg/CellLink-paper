import sys
import gzip 
import json

# required by getattr use below
import data_transforms

if __name__ == '__main__':
	config_filename = sys.argv[1]
	data_transformer = sys.argv[2] # "data_transform"
	passage_input_filename = sys.argv[3]
	measurements_input_filename = sys.argv[4]
	passage_output_filename = sys.argv[5]

	with open(config_filename) as config_file:
		config = json.load(config_file)
	print("Loaded {} dimensions".format(len(config)))
	transformed_dimensions = {dimension_name for dimension_name, dimension_config in config.items() if data_transformer in dimension_config}
	for dimension_name in transformed_dimensions:
		transform_function, params = config[dimension_name][data_transformer]
		print("Transforming dimension {} using function {} and parameters {}".format(dimension_name, transform_function, params))

	print("Loading measurements")
	if measurements_input_filename.endswith(".gz"):
		with gzip.open(measurements_input_filename, "rt") as file:
			measurements = json.load(file)
	else:
		with open(measurements_input_filename, "r") as file:
			measurements = json.load(file)
	print("Number of measurements is {}".format(len(measurements)))
	passage_count = measurements["PASSAGE_COUNT"]
	print("Number of passages is {}".format(passage_count))

	print("Transforming passages")
	if passage_input_filename.endswith(".gz"):
		passage_input_file = gzip.open(passage_input_filename, "rt") 
	else:
		passage_input_file = open(passage_input_filename, "r") 
	if passage_output_filename.endswith(".gz"):
		passage_output_file = gzip.open(passage_output_filename, "wt") 
	else:
		passage_output_file = open(passage_output_filename, "w") 

	for line_index, line in enumerate(passage_input_file):
		if (line_index + 1) % 5000 == 0:
			print("Processing {} of {}".format(line_index + 1, passage_count))
		passage = json.loads(line)
		original_passage_data = passage["data"]
		transformed_passage_data = dict()
		for dimension_name in transformed_dimensions:
			transform_function, params = config[dimension_name][data_transformer]
			name_fields = transform_function.split(".")
			transformed_passage_data_before = None
			getattr(sys.modules[name_fields[0]], name_fields[1])(original_passage_data, measurements, dimension_name, config, transformed_passage_data, params)
		passage["data"] = transformed_passage_data
		passage_text = json.dumps(passage)
		#print(passage_text)
		passage_output_file.write(passage_text + "\n")
	
	passage_input_file.close()
	passage_output_file.close()
	
	print("Done.")
