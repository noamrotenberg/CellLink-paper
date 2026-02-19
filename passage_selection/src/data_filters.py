
def passage_names(data, measurements, dimension_name, data_type, params):
	if not dimension_name in data:
		return False
	names = set(params)
	if data_type == "singleton":
		value_name = data[dimension_name]
		if value_name in names:
			return True
	elif data_type == "count_dict":
		for value_name in data[dimension_name].keys():
			if value_name in names:
				return True
	else:
		raise ValueError("Cannot apply filter passage_names to data type {}".format(data_type))
	return False

def passage_notempty(data, measurements, dimension_name, data_type, params):
	if not dimension_name in data:
		return True
	if data_type == "singleton":
		return False
	elif data_type == "count_dict":
		if len(data[dimension_name]) == 0:
			return True
	else:
		raise ValueError("Cannot apply filter passage_names to data type {}".format(data_type))
	return False

def passage_name_required(data, measurements, dimension_name, data_type, params):
	if not dimension_name in data:
		return False
	if data_type != "count_dict":
		raise ValueError("Cannot apply filter mesh_required to data type {}".format(data_type))
	# Removes the passage if it has non-zero items in the dimension
	# but does NOT include at least one of the ones passed as params
	required = set(params)
	values = data[dimension_name]
	for value_name in values.keys():
		if value_name in required:
			return False
	return True

def value_names(data, measurements, dimension_name, data_type, params):
	if not dimension_name in data:
		return False
	if data_type != "count_dict":
		raise ValueError("Cannot apply filter value_names to data type {}".format(data_type))
	names = set(params)
	values = data[dimension_name]
	filtered_values = {name: value for name, value in values.items() if not name in names}
	data[dimension_name] = filtered_values
	return False

def passage_zscore(data, measurements, dimension_name, data_type, params):
	if not dimension_name in data:
		return False
	if data_type != "number":
		raise ValueError("Cannot apply filter passage_names to data type {}".format(data_type))
	mean, std = measurements[dimension_name]
	value = data[dimension_name]
	zscore_threshold = params[0]
	zscore = abs(mean - value) / std
	return zscore > zscore_threshold
