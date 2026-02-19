import math
import random

import numpy 
import scipy

import PerformanceProfiler

epsilon = 1e-8

class numeric_lognorm_scorer:
	
	def __init__(self, mean, stdev, weight):
		self.mean = mean
		self.stdev = stdev
		self.neg_weight = -weight
		variance = self.stdev * self.stdev
		# This is the method of moments estimate
		lognorm_sigma = math.sqrt(math.log((variance) / (self.mean * self.mean) + 1.0))
		# This sets the mode at the original mean
		lognorm_mu = math.log(self.mean) + lognorm_sigma * lognorm_sigma
		self.dist = scipy.stats.lognorm(lognorm_sigma, scale=math.exp(lognorm_mu))
		
	def score(self, value):
		PerformanceProfiler.start("numeric_scorer.score()")
		result = self.dist.pdf(value)
		result *= self.neg_weight
		PerformanceProfiler.end("numeric_scorer.score()")
		return result

	def __repr__(self):
		return "numeric_lognorm_scorer(mean={}, stdev={}, weight={})".format(self.mean, self.stdev, -self.neg_weight)

class cached_scorer:

	def __init__(self, scorer):
		self.scorer = scorer
		self.cache = dict()
	
	def score_internal(self, value):
		PerformanceProfiler.start("cached_scorer.score_internal()")
		result = self.scorer.score(value)
		PerformanceProfiler.end("cached_scorer.score_internal()")
		return result
	
	def score(self, value):
		PerformanceProfiler.start("cached_scorer.score()")
		result = self.cache.get(value)
		if not result is None:
			PerformanceProfiler.end("cached_scorer.score()")
			return result
		result = self.score_internal(value)
		self.cache[value] = result
		PerformanceProfiler.end("cached_scorer.score()")
		return result

	def __repr__(self):
		return "cached_scorer({})".format(repr(self.scorer))

class annotation_prior_scorer:

	def __init__(self):
		self.annotations = list() # Needed for logging
		self.annotation_index2scorer = list()
		self.annotation2index = dict()
		self.scorer_cache = dict()
	
	def add_annotation(self, annotation, scorer):
		PerformanceProfiler.start("annotation_prior_scorer.add_annotation()")
		index = self.annotation2index.get(annotation)
		if not index is None:
			PerformanceProfiler.end("annotation_prior_scorer.add_annotation()")
			return index
		# Not yet indexed
		index = len(self.annotation_index2scorer)
		self.annotation2index[annotation] = index
		# Cache the scorer
		scorer_repr = repr(scorer)
		cs = self.scorer_cache.get(scorer_repr)
		if cs is None:
			cs = cached_scorer(scorer)
			self.scorer_cache[scorer_repr] = cs
		self.annotation_index2scorer.append(cs)
		self.annotations.append(annotation)
		PerformanceProfiler.end("annotation_prior_scorer.add_annotation()")
	
	def get_indexed_counts(self, annotation_counts):
		PerformanceProfiler.start("annotation_prior_scorer.get_indexed_counts()")
		indexed_counts = dict()
		for annotation, count in annotation_counts.items():
			index = self.annotation2index.get(annotation)
			if index is None:
				continue
			indexed_counts[index] = count
		PerformanceProfiler.end("annotation_prior_scorer.get_indexed_counts()")
		return indexed_counts
	
	def score_indexed_counts(self, indexed_counts):
		PerformanceProfiler.start("annotation_prior_scorer.score_indexed_counts()")
		score = 0.0
		for index, count in indexed_counts.items():
			score += self.annotation_index2scorer[index].score(count)
		PerformanceProfiler.end("annotation_prior_scorer.score_indexed_counts()")
		return score

	def analyze_indexed_counts_score(self, indexed_counts):
		PerformanceProfiler.start("annotation_prior_scorer.analyze_indexed_counts_score()")
		analysis = dict()
		score = 0.0
		for index, count in indexed_counts.items():
			index_score = self.annotation_index2scorer[index].score(count)
			analysis[self.annotations[index]] = index_score
			score += index_score
		PerformanceProfiler.end("annotation_prior_scorer.analyze_indexed_counts_score()")
		return score, analysis

class kldiv_scorer():
	
	def __init__(self, p):
		self.p = p
		self.plogp = p * math.log(p)
		self.cache = dict()

	def plog(self, value):
		result = self.cache.get(value)
		if result is None:
			result = self.p * math.log(value)
			self.cache[value] = result
		return result
	
	def score(self, count, total):
		result = self.plogp
		result -= self.plog(count)
		result += self.plog(total)
		return result

class annotation_indexed_kldiv_dimension_scorer():

	def __init__(self, dimension, per_doc_mean, smoothing):
		self.dimension = dimension
		self.smoothing = per_doc_mean * smoothing
		if self.smoothing <= 0.0:
			raise ValueError(f"Smoothing must be greate than zero: {self.smoothing}")
		self.annotations = list() # Needed for logging
		self.annotation2index = dict()
		self.annotation_index2p = list()
		self.finalized = False
		self.annotation_index2count = None
		self.annotation_index2scorer = None
		self.total_count = None
		self.total_delta_cache = None
		self.accepted_count = smoothing
		
	def add_annotation(self, annotation, p):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.add_annotation()")
		if self.finalized:
			raise ValueError("Cannot add annotations after being finalized")
			
		# Set up dimension
		if self.dimension != annotation[0]:
			raise ValueError("Can only handle one dimension")
		
		# Set up annotation index	
		annotation_index = self.annotation2index.get(annotation)
		if not annotation_index is None:
			PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.add_annotation()")
			return annotation_index
		# NOTE: This shows that there is zero overlap in p between the dimensions
		#print("CHECK\t{}\t{}".format(self.dimension, p))
		# Not yet indexed
		annotation_index = len(self.annotations)
		self.annotation2index[annotation] = annotation_index
		self.annotations.append(annotation)

		# Add parameters
		self.annotation_index2p.append(p)
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.add_annotation()")

	def finalize(self):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.finalize()")
		self.annotation_index2p = numpy.asarray(self.annotation_index2p)
		total_p = numpy.sum(self.annotation_index2p)
		if abs(total_p - 1.0) > epsilon:
			print("WARN: total p for dimension {} != 1.0; normalizing: {}".format(self.dimension, total_p))
			self.annotation_index2p /= total_p
		self.annotation_index2count = self.annotation_index2p * self.smoothing
		self.total_count = self.smoothing
		self.finalized = True
		self.total_delta_cache = dict()
		
		# Prepare scorers
		scorer_cache = dict() # PERFORMANCE: could share scorers across dimensions
		self.annotation_index2scorer = list()
		for annotation_index, p in enumerate(self.annotation_index2p):
			scorer = scorer_cache.get(p)
			if scorer is None:
				scorer = kldiv_scorer(p)
				scorer_cache[p] = scorer
			self.annotation_index2scorer.append(scorer)
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.finalize()")

	def get_indexed_counts(self, annotation_counts):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.get_indexed_counts()")
		indexed_counts = dict()
		for annotation, count in annotation_counts.items():
			index = self.annotation2index.get(annotation)
			if index is None:
				continue
			indexed_counts[index] = count
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.get_indexed_counts()")
		return indexed_counts

	def score_total_delta(self, total_delta):
		if total_delta in self.total_delta_cache:
			return self.total_delta_cache[total_delta]
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.score_total_delta()")
		q = self.annotation_index2count.copy()
		q_denominator_factor = 1.0 / (self.total_count + total_delta)
		q *= q_denominator_factor
		#print(f"TRACE score_total_delta self.annotation_index2count = {self.annotation_index2count} self.total_count = {self.total_count} total_delta = {total_delta} q_denominator_factor = {q_denominator_factor} q = {q}")
		index2score = self.annotation_index2p / q
		index2score = numpy.log(index2score)
		index2score = self.annotation_index2p * index2score
		total_score = numpy.sum(index2score)
		self.total_delta_cache[total_delta] = (total_score, index2score)
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.score_total_delta()")
		return total_score, index2score
	
	def score_delta(self, index, index_delta, total_delta):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.score_delta()")
		p = self.annotation_index2p[index]
		q = (self.annotation_index2count[index] + index_delta) / (self.total_count + total_delta)
		result = p * math.log(p / q)
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.score_delta()")
		return result
	
	def score_indexed_counts(self, indexed_counts):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.score_indexed_counts()")
		doc_total = sum(indexed_counts.values())
		total_score, index2score = self.score_total_delta(doc_total)
		doc_total += self.total_count

		total_score += sum(self.annotation_index2scorer[annotation_index].score(self.annotation_index2count[annotation_index] + doc_count, doc_total) - index2score[annotation_index] for annotation_index, doc_count in indexed_counts.items())

		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.score_indexed_counts()")
		return total_score

	def analyze_indexed_counts_score(self, indexed_counts):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.analyze_indexed_counts_score()")
		analysis = dict()

		doc_total = sum(indexed_counts.values())
		total_score, index2score = self.score_total_delta(doc_total)
		doc_total += self.total_count

		for annotation_index, doc_count in indexed_counts.items():
			annotation_score = self.annotation_index2scorer[annotation_index].score(self.annotation_index2count[annotation_index] + doc_count, doc_total) - index2score[annotation_index]
			#total_score += mse * annotation_score
			total_score += annotation_score
			annotation = self.annotations[annotation_index]
			analysis[(self.dimension, annotation)] = (doc_count, annotation_score)
			
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.analyze_indexed_counts_score()")
		return total_score, analysis

	def add_indexed_counts(self, indexed_counts):
		PerformanceProfiler.start("annotation_indexed_kldiv_dimension_scorer.add_indexed_counts()")
		for annotation_index, doc_count in indexed_counts.items():
			self.annotation_index2count[annotation_index] += doc_count
			self.total_count += doc_count
			self.accepted_count += 1
		self.total_delta_cache = dict()
		PerformanceProfiler.end("annotation_indexed_kldiv_dimension_scorer.add_indexed_counts()")

class passage_scorer:
	
	def __init__(self, prior_annotation_scorer, dimension_scorer_list):
		self.prior_annotation_scorer = prior_annotation_scorer
		self.dimension_weights = list()
		self.dimension_scorers = list()
		for dimension_weight, dimension_scorer in dimension_scorer_list:
			self.dimension_weights.append(dimension_weight)
			self.dimension_scorers.append(dimension_scorer)
		self.passageid2info = dict()
		self.finalized = False

	def add_passage(self, passageid, annotation_counts):
		PerformanceProfiler.start("passage_scorer.add_passage()")
		# NOTE: annotation_counts needs to be in (annotation, value) format
		if self.finalized:
			raise ValueError("Cannot add passages after being finalized")
		if passageid in self.passageid2info:
			raise ValueError("Passage {} has already been added".format(passageid))
		# Handle prior
		prior_indexed_counts = self.prior_annotation_scorer.get_indexed_counts(annotation_counts)
		passage_prior = self.prior_annotation_scorer.score_indexed_counts(prior_indexed_counts)
		#print("Prior score for passage {} is {}".format(passageid, passage_prior))
		# Handle dimensions
		indexed_count_list = list()
		for dimension_scorer in self.dimension_scorers:
			indexed_count_list.append(dimension_scorer.get_indexed_counts(annotation_counts))
		self.passageid2info[passageid] = (passage_prior, indexed_count_list)
		PerformanceProfiler.end("passage_scorer.add_passage()")
	
	def finalize(self):
		if self.finalized:
			raise ValueError("Cannot finalize after being finalized")
		self.finalized = True

	def score_passages(self):
		PerformanceProfiler.start("passage_scorer.score_passage()")
		
		# Returns the score if we were to select this passage
		scores = dict()
		lowest_score = math.inf
		lowest_passageids = []
		
		# Get dimension scores
		for passageid, (passage_prior, indexed_count_list) in self.passageid2info.items():
			passage_score = passage_prior
			for dim_index in range(len(self.dimension_scorers)):
				indexed_counts = indexed_count_list[dim_index]
				dimension_weight = self.dimension_weights[dim_index]
				dimension_scorer = self.dimension_scorers[dim_index]
				passage_score += dimension_weight * dimension_scorer.score_indexed_counts(indexed_counts)
			scores[passageid] = passage_score
			if passage_score < lowest_score:
				lowest_score = passage_score
				lowest_passageids.clear()
				lowest_passageids.append(passageid)
			elif passage_score == lowest_score:
				lowest_passageids.append(passageid)
		
		PerformanceProfiler.end("passage_scorer.score_passage()")
		return scores, lowest_score, lowest_passageids

	def analyze_passage_score(self, passageid):
		passage_prior, indexed_count_list = self.passageid2info[passageid]
		dimension_summary = list()
		dimension_summary.append(("PRIOR", 1.0, passage_prior))
		# TODO make a way to get the prior analysis easily, since its indexed counts are not the same as the selection indexed counts
		item_analysis = dict()
		for dim_index in range(len(self.dimension_scorers)):
			indexed_counts = indexed_count_list[dim_index]
			dimension_weight = self.dimension_weights[dim_index]
			dimension_scorer = self.dimension_scorers[dim_index]
			dimension_score, dimension_analysis = dimension_scorer.analyze_indexed_counts_score(indexed_counts)
			dimension_summary.append((dimension_scorer.dimension, dimension_weight, dimension_score))
			item_analysis.update(dimension_analysis)
		return dimension_summary, item_analysis

	def select_passage(self, passageid):
		PerformanceProfiler.start("passage_scorer.select_passage()")
		if not passageid in self.passageid2info:
			PerformanceProfiler.end("passage_scorer.select_passage()")
			return
		passage_prior, indexed_count_list = self.passageid2info.pop(passageid)
		for indexed_counts, dimension_scorer in zip(indexed_count_list, self.dimension_scorers):
			dimension_scorer.add_indexed_counts(indexed_counts)
		PerformanceProfiler.end("passage_scorer.select_passage()")

	def drop_passage(self, passageid):
		PerformanceProfiler.start("passage_scorer.drop_passage()")
		if not passageid in self.passageid2info:
			PerformanceProfiler.end("passage_scorer.drop_passage()")
			return
		self.passageid2info.pop(passageid)
		PerformanceProfiler.end("passage_scorer.drop_passage()")

	def select_lowest(self):
		# NOTE: This keeps the MINIMUM found
		PerformanceProfiler.start("passage_scorer.select_lowest()")
		scores, lowest_score, lowest_passageids = self.score_passages()
		if len(lowest_passageids) == 0:
			PerformanceProfiler.end("passage_scorer.select_lowest()")
			return math.inf, None, None, None
		random.shuffle(lowest_passageids)
		passageid = lowest_passageids[0]
		dimension_summary, item_analysis = self.analyze_passage_score(passageid)
		self.select_passage(passageid)
		PerformanceProfiler.end("passage_scorer.select_lowest()")
		return lowest_score, passageid, dimension_summary, item_analysis

class normalized_passage_scorer(passage_scorer):

	def __init__(self, prior_annotation_scorer, dimension_scorer_list):
		super().__init__(prior_annotation_scorer, dimension_scorer_list)
		
	def score_passages(self):
		PerformanceProfiler.start("normalized_passage_scorer.score_passage()")
		
		# Initialize
		passageids = list()
		priors = numpy.zeros(len(self.passageid2info), numpy.float64)
		dimension_scores = list()
		for dim_index in range(len(self.dimension_scorers)):
			dimension_scores.append(numpy.zeros(len(self.passageid2info), numpy.float64))
		print(f"TRACE normalized_passage_scorer.score_passages()@init: len(passageids) = {len(passageids)} len(priors) = {len(priors)} len(self.passageid2info) = {len(self.passageid2info)} len(dimension_scores) = {len(dimension_scores)}")

		# Store scores
		for passage_index, (passageid, (passage_prior, indexed_count_list)) in enumerate(self.passageid2info.items()):
			passageids.append(passageid)
			priors[passage_index] = passage_prior
			for dim_index in range(len(self.dimension_scorers)):
				indexed_counts = indexed_count_list[dim_index]
				dimension_scorer = self.dimension_scorers[dim_index]
				dimension_scores[dim_index][passage_index] = dimension_scorer.score_indexed_counts(indexed_counts)
		print(f"TRACE normalized_passage_scorer.score_passages()@score: len(passageids) = {len(passageids)} len(priors) = {len(priors)} len(self.passageid2info) = {len(self.passageid2info)} len(dimension_scores) = {len(dimension_scores)}")
		
		# Get score z-scores
		print(f"priors: min = {numpy.min(priors)} mean = {numpy.mean(priors)} max = {numpy.max(priors)}")
		if numpy.count_nonzero(priors) > 0:
			priors = scipy.stats.zscore(priors)
		print(f"z-priors: min = {numpy.min(priors)} mean = {numpy.mean(priors)} max = {numpy.max(priors)}")
		for dim_index in range(len(self.dimension_scorers)):
			dim_scores = dimension_scores[dim_index]
			print(f"dim_scores{dim_index}: min = {numpy.min(dim_scores)} mean = {numpy.mean(dim_scores)} max = {numpy.max(dim_scores)}")
			dim_scores = scipy.stats.zscore(dim_scores)
			print(f"z-dim_scores{dim_index}: min = {numpy.min(dim_scores)} mean = {numpy.mean(dim_scores)} max = {numpy.max(dim_scores)}")
			dimension_scores[dim_index] = dim_scores

		# Returns the score if we were to select this passage
		scores = dict()
		lowest_score = math.inf
		lowest_passageids = []

		for passage_index, passageid in enumerate(passageids):
			passage_score = priors[passage_index]
			for dim_index in range(len(self.dimension_scorers)):
				dimension_weight = self.dimension_weights[dim_index]
				passage_score += dimension_weight * dimension_scores[dim_index][passage_index]
			scores[passageid] = passage_score
			if passage_score < lowest_score:
				lowest_score = passage_score
				lowest_passageids.clear()
				lowest_passageids.append(passageid)
			elif passage_score == lowest_score:
				lowest_passageids.append(passageid)

		PerformanceProfiler.end("normalized_passage_scorer.score_passage()")
		print(f"TRACE normalized_passage_scorer.score_passages()@return: len(scores) = {len(scores)} lowest_score = {lowest_score} len(lowest_passageids) = {len(lowest_passageids)}")
		return scores, lowest_score, lowest_passageids

	