import math

class MeanStdCounter():
	
	def __init__(self):
		self.count = 0.0
		self.sums = dict()
	
	def increment(self, name, value):
		self.count += 1.0
		if not name in self.sums:
			self.sums[name] = [0.0, 0.0]
		self.sums[name][0] += value
		self.sums[name][1] += value * value

	def update(self, value_dict):
		self.count += 1.0
		for name, value in value_dict.items():
			if not name in self.sums:
				self.sums[name] = [0.0, 0.0]
			self.sums[name][0] += value
			self.sums[name][1] += value * value
	
	def summarize(self):
		# Generate list of data & sort
		summary_list = [(sum / self.count, math.sqrt((self.count * sum_sqr - sum * sum) / (self.count * (self.count - 1))), name) for name, (sum, sum_sqr) in self.sums.items()]
		summary_list.sort(reverse = True)
		# If "number" type, return only mean & std
		if len(summary_list) == 1 and summary_list[0][2] is None:
			return [summary_list[0][0], summary_list[0][1]]
		# Return dict of mean & std
		return {name: [mean, std] for mean, std, name in summary_list}

def mean_std(intermediate_measurements, dimension_name, data_type, data_value, params):
	if not dimension_name in intermediate_measurements:
		intermediate_measurements[dimension_name] = MeanStdCounter()
	counter = intermediate_measurements[dimension_name]
	if data_type == "singleton":
		counter.increment(data_value, 1.0)
	elif data_type == "count_dict":
		counter.update(data_value)
	elif data_type == "number":
		counter.increment(None, data_value)
	else:
		raise ValueError("Unknown data type: {}".format(data_type))

class PDistSummary():
	
	def __init__(self):
		self.count = 0.0
		self.total = 0.0
		self.sums = dict()
	
	def increment(self, name, value):
		self.count += 1.0
		if not name in self.sums:
			self.sums[name] = 0.0
		self.sums[name] += value
		self.total += 1.0

	def update(self, value_dict):
		self.count += 1.0
		for name, value in value_dict.items():
			if not name in self.sums:
				self.sums[name] = 0.0
			self.sums[name] += value
			self.total += value
	
	def summarize(self):
		# Generate list of data & sort
		summary_list = [(sum / self.total, name) for name, sum in self.sums.items()]
		summary_list.sort(reverse = True)
		return [self.total / self.count, {name: mean for mean, name in summary_list}]
	
def mean(intermediate_measurements, dimension_name, data_type, data_value, params):
	if not dimension_name in intermediate_measurements:
		intermediate_measurements[dimension_name] = PDistSummary()
	counter = intermediate_measurements[dimension_name]
	if data_type == "singleton":
		counter.increment(data_value, 1.0)
	elif data_type == "count_dict":
		counter.update(data_value)
	elif data_type == "number":
		raise ValueError("Not implemented")
	else:
		raise ValueError("Unknown data type: {}".format(data_type))
