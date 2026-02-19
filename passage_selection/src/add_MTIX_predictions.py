import json
import gzip
import collections
import sys

if __name__ == '__main__':
	mtix_filename = sys.argv[1]
	measurements_input_filename = sys.argv[2]
	passage_input_filename = sys.argv[3]
	passage_output_filename = sys.argv[4]

	print("Loading MTIX predictions")
	with gzip.open(mtix_filename, "rt") as file:
		mtix_predictions = json.load(file)
	print("Number of predictions is {}".format(len(mtix_predictions)))
	mtix_term2type = dict()
	mtix_type_scores = dict()
	mtix_term_scores = dict()
	for article in mtix_predictions:
		for prediction in article["Indexing"]:
			term = prediction["Term"]
			type = prediction["Type"]
			score = prediction["Reasons"][0]["Score"]
			mtix_term2type[term] = type
			if not type in mtix_type_scores:
				mtix_type_scores[type] = list()
			mtix_type_scores[type].append(score)
			if not term in mtix_term_scores:
				mtix_term_scores[term] = list()
			mtix_term_scores[term].append(score)

	print("Loading measurements")
	if measurements_input_filename.endswith(".gz"):
		with gzip.open(measurements_input_filename, "rt") as file:
			measurements = json.load(file)
	else:
		with open(measurements_input_filename, "r") as file:
			measurements = json.load(file)
	print("Number of measurements is {}".format(len(measurements)))

	mesh_type_counts = collections.Counter()
	mesh_term_counts = collections.Counter()
	mesh_count_per_article = measurements["MESH_TERMS"][0]
	mesh_term_rates = measurements["MESH_TERMS"][1]
	for term, type in mtix_term2type.items():
		# TODO Add a smoothing count
		count = len(mtix_predictions) * mesh_count_per_article * mesh_term_rates.get(term, 0)
		mesh_type_counts[type] += count
		mesh_term_counts[term] += count
	
	print("Determining type thresholds")
	types = set()
	types.update(mtix_type_scores.keys())
	types.update(mesh_type_counts.keys())
	type2threshold = dict()
	for type in types:
		#print("type = {}".format(type))
		expected_count = int(round(mesh_type_counts.get(type, 0)))
		#print("\texpected count = {}".format(expected_count))
		scores = mtix_type_scores.get(type, [])
		#print("\tnumber of scores = {}".format(len(scores)))
		if expected_count <= 0:
			threshold = 1.0
		elif len(scores) <= expected_count:
			threshold = 0.0
		else:
			scores.sort(reverse = True)
			score2count = dict()
			for index, score in enumerate(scores):
				if score in score2count:
					continue
				score2count[score] = index
			error_scores = [(abs(count - expected_count), count, score) for score, count in score2count.items()]
			error_scores.sort()
			#print("\tcount@t = {}".format(error_scores[0][1]))
			threshold = error_scores[0][2]
		#print("\tthreshold = {}".format(threshold))
		type2threshold[type] = threshold
	
	print("Determining term thresholds")
	terms = set()
	terms.update(mtix_term_scores.keys())
	terms.update(mesh_term_counts.keys())
	term2threshold = dict()
	for term in terms:
		#print("term = {}".format(term))
		expected_count = int(round(mesh_term_counts.get(term, 0)))
		#print("\texpected count = {}".format(expected_count))
		scores = mtix_term_scores.get(term, [])
		#print("\tnumber of scores = {}".format(len(scores)))
		scores.sort(reverse = True)
		if expected_count <= 0:
			threshold = 1.0
		elif len(scores) <= expected_count:
			threshold = 0.0
		else:
			scores.sort(reverse = True)
			score2count = dict()
			for index, score in enumerate(scores):
				if score in score2count:
					continue
				score2count[score] = index
			error_scores = [(abs(count - expected_count), count, score) for score, count in score2count.items()]
			error_scores.sort()
			#print("\tcount@t = {}".format(error_scores[0][1]))
			threshold = error_scores[0][2]
		#print("\tterm threshold = {}".format(threshold))
		type = mtix_term2type.get(term)
		if not type is None:
			#print("\ttype = {}".format(type))
			type_threshold = type2threshold[type]
			#print("\ttype threshold = {}".format(type_threshold))
			threshold = 0.5 * threshold + 0.5 * type_threshold
			#print("\tcombo threshold = {}".format(threshold))
		term2threshold[term] = threshold

	print("Determining usable terms")
	pmid2mesh_list = dict()
	for article in mtix_predictions:
		pmid = str(article["PMID"])
		mesh_list = list()
		for prediction in article["Indexing"]:
			term = prediction["Term"]
			if prediction["Reasons"][0]["Score"] > term2threshold.get(term, 1.0):
				mesh_list.append(term)
		if len(mesh_list) > 0:
			pmid2mesh_list[pmid] = mesh_list
	print("Adding {} MeSH terms to {} articles".format(sum(len(mesh_list) for mesh_list in pmid2mesh_list.values()), len(pmid2mesh_list)))

	print("Loading " + str(passage_input_filename))
	if passage_input_filename.endswith(".gz"):
		passage_input_file = gzip.open(passage_input_filename, "rt") 
	else:
		passage_input_file = open(passage_input_filename, "r") 
	if passage_output_filename.endswith(".gz"):
		passage_output_file = gzip.open(passage_output_filename, "wt") 
	else:
		passage_output_file = open(passage_output_filename, "w") 
	for line in passage_input_file:
		passage_dict = json.loads(line)
		pmid = passage_dict["pmid"]
		if not pmid in pmid2mesh_list:
			passage_output_file.write(json.dumps(passage_dict) + "\n")
			continue
		#print("pmid = {}".format(pmid))
		mesh_list = pmid2mesh_list[pmid]
		#print("len(mesh_list) = {}".format(len(mesh_list)))
		mesh_counts = passage_dict["data"]["MESH_TERMS"]
		#print("len(mesh_counts) = {}".format(len(mesh_counts)))
		for mesh in mesh_list:
			mesh_counts[mesh] = 1
		#print("len(mesh_counts) = {}".format(len(mesh_counts)))
		passage_output_file.write(json.dumps(passage_dict) + "\n")
	passage_input_file.close()
	passage_output_file.close()

	print("Done.")	
		







		