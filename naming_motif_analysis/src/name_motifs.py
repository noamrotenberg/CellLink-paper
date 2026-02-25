import json
import logging
from collections import Counter
import random
import math

import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from string_normalization import SpacyStringNormalizer
from term_encoder import TermEncoder
from trie import Trie
import PerformanceProfiler

ROOT_LABEL = "root"
STOPWORD_LABEL = "stopword"
eps = 1e-12


def overlaps(span1, span2):
    return span1[0] < span2[1] and span2[0] < span1[1]

class NameMotifParser:

    def __init__(
        self,
        name_motif_map_filename,
        scispacy_model_name,
    ):
        logging.info("Loading scispacy")
        self.nlp = spacy.load(scispacy_model_name)
        self.name_part_updater = SpacyStringNormalizer(self.nlp)

        self._load_name_motifs(name_motif_map_filename)

        # Initialized by add_terms:
        self.term2dict = None
        # Initialized by process:
        self.tokseq2dict = None
        self.seq_label_counts = None
        self.classifier = None
        self.class_names = None
        self._parser_status("__init__")

    def _load_name_motifs(self, name_motif_map_filename: str):
        logging.info(f"Loading name motif map from {name_motif_map_filename}")
        with open(name_motif_map_filename, "r") as file:
            name_motif_map = json.load(file)
        self.name_motif_map = dict()
        self.name_motif_trie = Trie()
        for name_motif, label in name_motif_map.items():
            updated_name_motif = self.name_part_updater.update(name_motif)
            current_label = name_motif_map.get(updated_name_motif)
            if not current_label is None and current_label != label:
                raise ValueError(
                    f"Name motif {name_motif} is mapped to both {current_label} and {label}"
                )
            self.name_motif_map[updated_name_motif] = label
            doc = self.nlp(updated_name_motif)
            name_motif_token_tuple = tuple(token.text for token in doc)
            self.name_motif_trie.add(name_motif_token_tuple, label)
        logging.info(f"Loaded {len(self.name_motif_map)} motif mappings")

    def _parser_status(self, state_string):
        logging.debug(f"NameMotifParser status: {state_string}")
        logging.debug("NameMotifParser.term2dict: {}".format("is None" if self.term2dict is None else f"len={len(self.term2dict)}"))
        logging.debug("NameMotifParser.tokseq2dict: {}".format("is None" if self.tokseq2dict is None else f"len={len(self.tokseq2dict)}"))
        logging.debug("NameMotifParser.seq_label_counts: {}".format("is None" if self.seq_label_counts is None else f"len={len(self.seq_label_counts)}"))
        logging.debug("NameMotifParser.classifier: {}".format("is None" if self.classifier is None else "is not None"))
        logging.debug("NameMotifParser.class_names: {}".format("is None" if self.class_names is None else f"len={len(self.class_names)}"))
        logging.debug("NameMotifParser.name_motif_map: {}".format("is None" if self.name_motif_map is None else f"len={len(self.name_motif_map)}"))
        logging.debug("NameMotifParser.name_motif_trie: {}".format("is None" if self.name_motif_trie is None else f"len={len(self.name_motif_trie)}"))

    @classmethod
    def load(
        cls,
        ontology,
        mention_texts,
        name_motif_map_filename,
        term_cache_filename,
        vector_cache_filename,
        encoder_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        encoder_max_term_length=50,
        scispacy_model_name="en_core_sci_sm",
        maxseqlen=3,
        min_margin=0.0,
    ):
        parser = cls(
            name_motif_map_filename,
            scispacy_model_name,
        )
        logging.info("Adding terms: roots")
        root_texts = []
        for name_motif, label in parser.name_motif_map.items():
            if not label == "root":
                continue
            root_texts.append(name_motif)
        parser.add_terms(root_texts, "name_motif_map:roots")
        logging.info("Adding terms: ontology terms")
        ontology_term_texts = list()
        for term_id, term_data in ontology.items():
            term_texts = set()
            if "name" in term_data:
                term_texts.add(term_data["name"])
            if "synonyms" in term_data:
                term_texts.update(term_data["synonyms"])
            # remove endings like "(sensu Vertebrata)" and "(sensu Nematoda and Protostomia)"
            for term in term_texts:
                i = term.find(" (sensu ")
                if i < 0:
                    ontology_term_texts.append(term)
                else:
                    ontology_term_texts.append(term[:i])
        logging.info(
            f"Found {len(ontology_term_texts)} term names and synonyms; {len(set(ontology_term_texts))} unique"
        )
        parser.add_terms(ontology_term_texts, "ontology")
        logging.info("Adding terms: corpus mentions")
        parser.add_terms(mention_texts, "corpus")
        logging.info("Loading encoder model")
        term_encoder = TermEncoder(
            model_name=encoder_model_name,
            max_term_length=encoder_max_term_length,
        )
        term_encoder.load(term_cache_filename, vector_cache_filename)
        logging.info("Training classifier")
        parser.process(term_encoder, maxseqlen, min_margin)
        logging.info("Saving term and vector caches")
        term_encoder.save(term_cache_filename, vector_cache_filename, sort_terms=True)
        parser._parser_status("load()")
        return parser

    def _create_term_dict(self, updated_term_text, original_term_text, term_source):
        PerformanceProfiler.start("NameMotifParser._create_term_dict()")
        term_dict = dict()
        term_dict["term"] = updated_term_text
        term_dict["original_term_text"] = original_term_text
        term_dict["source_count"] = Counter()
        term_dict["source_count"][term_source] += 1
        doc = self.nlp(updated_term_text)
        term_dict["doc"] = doc
        tokens = [token.text for token in doc]
        term_dict["tokens"] = tokens
        term_dict["labels"] = self.name_motif_trie.lookup(tokens)
        PerformanceProfiler.end("NameMotifParser._create_term_dict()")
        return term_dict

    def add_terms(self, term_texts, term_source):
        PerformanceProfiler.start("NameMotifParser.add_terms()")
        if self.term2dict == None:
            self.term2dict = dict()
        for term_text in term_texts:
            PerformanceProfiler.start("NameMotifParser.add_terms()@loop")
            updated_term_text = self.name_part_updater.update(term_text)
            if updated_term_text in self.term2dict:
                self.term2dict[updated_term_text]["source_count"][term_source] += 1
                PerformanceProfiler.end("NameMotifParser.add_terms()@loop")
                continue
            term_dict = self._create_term_dict(
                updated_term_text, term_text, term_source
            )
            self.term2dict[updated_term_text] = term_dict
            PerformanceProfiler.end("NameMotifParser.add_terms()@loop")
        PerformanceProfiler.end("NameMotifParser.add_terms()")

    def _filter_subsequence(self, tokseq):
        if len(tokseq) == 0:
            return True
        first = self.name_part_updater.update(tokseq[0].text)
        if self.name_motif_map.get(first) == STOPWORD_LABEL:
            return True
        last = self.name_part_updater.update(tokseq[-1].text)
        if self.name_motif_map.get(last) == STOPWORD_LABEL:
            return True
        return False

    def process(self, term_encoder: TermEncoder, maxseqlen: int, min_margin: float):
        PerformanceProfiler.start("NameMotifParser.process()")
        self._prepare_subsequences(maxseqlen)
        self._encode(term_encoder)
        if not self._train():
            return
        self._predict_subsequences()
        self._predict_terms(min_margin=min_margin, maxseqlen=maxseqlen)
        self._parser_status("process()")
        PerformanceProfiler.end("NameMotifParser.process()")

    def _prepare_subsequences(self, maxseqlen: int):
        PerformanceProfiler.start("NameMotifParser._prepare_subsequences()")
        if self.term2dict is None:
            raise RuntimeError("Need to add terms first")
        logging.info("Preparing subsequences")
        # Iterate through subsequences of length up to N
        self.seq_label_counts = Counter()
        filtered = set()
        self.tokseq2dict = dict()
        for term, term_dict in self.term2dict.items():
            doc = term_dict["doc"]
            labels = term_dict["labels"]
            count = term_dict["source_count"].total()
            for n in range(1, maxseqlen + 1):
                for i in range(len(doc) - n + 1):
                    PerformanceProfiler.start(
                        "NameMotifParser._prepare_subsequences()@loop"
                    )
                    PerformanceProfiler.start(
                        "NameMotifParser._prepare_subsequences()@loop:filter1"
                    )
                    tokens = doc[i : i + n]
                    if self._filter_subsequence(tokens):
                        tokens = str(tokens)
                        if not tokens in filtered:
                            logging.debug(f'Filtering token sequence "{tokens}"')
                            filtered.add(tokens)
                        PerformanceProfiler.end(
                            "NameMotifParser._prepare_subsequences()@loop:filter1"
                        )
                        PerformanceProfiler.end(
                            "NameMotifParser._prepare_subsequences()@loop"
                        )
                        continue
                    PerformanceProfiler.end(
                        "NameMotifParser._prepare_subsequences()@loop:filter1"
                    )
                    PerformanceProfiler.start(
                        "NameMotifParser._prepare_subsequences()@loop:filter2"
                    )
                    tokseq = tuple(token.text for token in tokens)
                    label = self.name_motif_trie.get(tokseq)
                    # If no label AND overlaps a span with a label, then continue
                    span = (i, i + n)
                    if label is None and any(
                        overlaps(span, (start, end)) for _label, start, end in labels
                    ):
                        PerformanceProfiler.end(
                            "NameMotifParser._prepare_subsequences()@loop:filter2"
                        )
                        PerformanceProfiler.end(
                            "NameMotifParser._prepare_subsequences()@loop"
                        )
                        continue
                    PerformanceProfiler.end(
                        "NameMotifParser._prepare_subsequences()@loop:filter2"
                    )
                    PerformanceProfiler.start(
                        "NameMotifParser._prepare_subsequences()@loop:add"
                    )
                    if not tokseq in self.tokseq2dict:
                        seq_dict = dict()
                        seq_dict["terms"] = Counter()
                        seq_dict["seq_texts"] = Counter()
                        seq_dict["label"] = label
                        seq_dict["usage_counts"] = Counter()
                        self.seq_label_counts[str(label)] += 1
                        self.tokseq2dict[tokseq] = seq_dict
                    else:
                        seq_dict = self.tokseq2dict[tokseq]

                    seq_dict["terms"][term] += count
                    seq_text = ("".join(token.text_with_ws for token in tokens)).strip()
                    seq_dict["seq_texts"][seq_text] += count
                    PerformanceProfiler.end(
                        "NameMotifParser._prepare_subsequences()@loop:add"
                    )
                    PerformanceProfiler.end(
                        "NameMotifParser._prepare_subsequences()@loop"
                    )
        PerformanceProfiler.end("NameMotifParser._prepare_subsequences()")

    def _encode(self, term_encoder: TermEncoder):
        PerformanceProfiler.start("NameMotifParser._encode()")
        if self.tokseq2dict is None:
            raise RuntimeError("Must run _prepare_subsequences() first")
        logging.info("Encoding texts")
        texts = set()
        for seq_dict in self.tokseq2dict.values():
            texts.update(seq_dict["seq_texts"].keys())
        texts = list(texts)
        # Shuffling gives better time estimates for the encoding
        random.shuffle(texts)
        embeds = term_encoder.encode_terms(texts)
        text2embed = {text: embed for text, embed in zip(texts, embeds)}

        logging.info("Calculating mean embeddings")
        for seq_dict in self.tokseq2dict.values():
            embeds = []
            weights = []
            for seq_text, seq_count in seq_dict["seq_texts"].items():
                embeds.append(text2embed[seq_text])
                weights.append(seq_count)
            embeds = np.vstack(embeds)
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            mean_embed = np.average(embeds, axis=0, weights=weights)
            mean_embed = mean_embed / (np.linalg.norm(mean_embed) + eps)
            seq_dict["embed"] = mean_embed
        PerformanceProfiler.end("NameMotifParser._encode()")

    def _train(self):
        PerformanceProfiler.start("NameMotifParser._train()")
        if self.tokseq2dict is None:
            raise RuntimeError("Must run _prepare_subsequences() first")
        logging.info("Preparing training points")
        X = []
        y_names = []
        for seq_dict in self.tokseq2dict.values():
            label = seq_dict["label"]
            if label is None:
                continue
            X.append(seq_dict["embed"])
            y_names.append(label)

        if len(X) < 2 or len(set(y_names)) < 2:
            logging.error("Not enough data to train a classifier")
            PerformanceProfiler.end("NameMotifParser._train()")
            return False

        logging.info("Training classifier")
        X = np.vstack(X)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_names)

        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.classifier.fit(X, y)
        class_indices = np.asarray(self.classifier.classes_)
        self.class_names = list(
            map(str, label_encoder.inverse_transform(class_indices))
        )
        PerformanceProfiler.end("NameMotifParser._train()")
        return True

    def _predict_subsequences(self):
        PerformanceProfiler.start("NameMotifParser._predict()")
        if self.tokseq2dict is None or self.classifier is None or self.class_names is None:
            raise RuntimeError("Must run _train() first")
        logging.info("Processing predictions")
        for seq_dict in self.tokseq2dict.values():
            label = seq_dict["label"]
            if label is None:
                X_vec = np.asarray(seq_dict["embed"], dtype=np.float32).reshape(1, -1)
                probs = self.classifier.predict_proba(X_vec)[0]
                pairs = list((c, float(p)) for c, p in zip(self.class_names, probs))
            else:
                pairs = list((c, 1.0 if c == label else 0.0) for c in self.class_names)
            pairs.sort(key=lambda x: x[1], reverse=True)
            seq_dict["prediction"] = {c: p for c, p in pairs}
        PerformanceProfiler.end("NameMotifParser._predict()")

    def _predict_terms(self, min_margin: float, maxseqlen: int):
        if self.term2dict is None or self.tokseq2dict is None:
            raise RuntimeError("Must run _predict_subsequences() first")
        for term in self.term2dict.keys():
            term_dict = self.term2dict[term]
            doc = term_dict["doc"]
            source_counts = term_dict["source_count"]
            potential_predictions = list()  # margin, span, tokseq, prediction
            for n in range(1, maxseqlen + 1):
                for i in range(len(doc) - n + 1):
                    tokens = doc[i : i + n]
                    tokseq = tuple(token.text for token in tokens)
                    if not tokseq in self.tokseq2dict:
                        continue
                    prediction = self.tokseq2dict[tokseq]["prediction"]
                    pairs = list(prediction.items())
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    margin = pairs[0][1] - pairs[1][1]
                    if margin < min_margin:
                        continue
                    potential_prediction_tuple = (
                        margin,
                        n,
                        (i, i + n),
                        pairs[0][0],
                        tokseq,
                    )
                    potential_predictions.append(potential_prediction_tuple)
            potential_predictions.sort(key=lambda x: (x[0], x[1]), reverse=True)
            predictions = list()
            covered = [0] * len(term_dict["tokens"])
            for margin, span_len, span, label, tokseq in potential_predictions:
                covered_count = sum(covered[i] for i in range(span[0], span[1]))
                if covered_count > 0:
                    continue
                predictions.append((label, span[0], span[1]))
                self.tokseq2dict[tokseq]["usage_counts"].update(source_counts)
                for i in range(span[0], span[1]):
                    covered[i] = 1
            predictions.sort(key=lambda x: x[1])
            term_dict["predictions"] = predictions
            logging.debug(f"Predicting term \"{term}\": {predictions}")

    def find_motifs(self, term_text):
        if self.term2dict is None:
            raise RuntimeError("Must run process() first")
        updated_term_text = self.name_part_updater.update(term_text)
        term_dict = self.term2dict.get(updated_term_text)
        if term_dict is None:
            logging.warning(f'MotifFinder could not process term "{term_text}"')
            return [], {"unmapped": 1}
        motif_types_found = Counter()
        tokens = term_dict["tokens"]
        covered = [0] * len(tokens)
        for motif, start, end in term_dict["predictions"]:
            motif_types_found[motif] += 1
            for i in range(start, end):
                covered[i] = 1
        unused_tokens = [
            token for token, token_covered in zip(tokens, covered) if token_covered == 0
        ]
        if len(motif_types_found) == 0:
            motif_types_found["unmapped"] += 1
        if len(unused_tokens) > 0:
            motif_types_found["unknown token"] += len(unused_tokens)
        return unused_tokens, motif_types_found

    def cross_validate(self, n_folds: int, sources: list[str]):
        if self.tokseq2dict is None:
            raise RuntimeError("Must run process() first")
        logging.info("Stratifying folds by label")
        label2points = dict()
        for tokseq, seq_dict in self.tokseq2dict.items():
            label = seq_dict["label"]
            usage_counts = seq_dict["usage_counts"]
            usage_count = sum(
                count for source, count in usage_counts.items() if source in sources
            )
            if label is None or usage_count == 0:
                continue
            if not label in label2points:
                label2points[label] = list()
            label2points[label].append((tokseq, seq_dict["embed"], label, usage_count))
            if not "cv_predicted" in seq_dict:
                seq_dict["cv_predicted"] = [0, 0]
                seq_dict["cv_prediction"] = Counter()

        logging.info("Creating folds")
        folds = [list() for _ in range(n_folds)]
        for label, points in label2points.items():
            random.shuffle(points)
            fold_size = math.ceil(len(points) / n_folds)
            logging.info(f"fold_size for label {label} is {fold_size}")
            label_folds = [
                points[i : i + fold_size] for i in range(0, len(points), fold_size)
            ]
            random.shuffle(label_folds)
            while len(label_folds) < n_folds:
                label_folds.append(list())
            for fold_index in range(n_folds):
                folds[fold_index].extend(label_folds[fold_index])

        results = list()
        for eval_fold_index in range(n_folds):
            logging.info(f"Creating data for fold {eval_fold_index}")
            X = list()
            y_names = list()
            for fold_index in range(n_folds):
                if fold_index == eval_fold_index:
                    continue
                for point in folds[fold_index]:
                    _tokseq, embed, label, usage_count = point
                    X.append(embed)
                    y_names.append(label)

            if len(X) < 2 or len(set(y_names)) < 2:
                logging.error("Not enough data to train a classifier")
                continue

            logging.info(f"Training classifier for fold {eval_fold_index}")
            X = np.vstack(X)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_names)

            cv_classifier = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            )
            cv_classifier.fit(X, y)
            class_indices = np.asarray(cv_classifier.classes_)
            cv_class_names = list(
                map(str, label_encoder.inverse_transform(class_indices))
            )
            logging.info(f"Evaluating predictions for fold {eval_fold_index}")
            tp_weighted = 0
            tp_unweighted = 0
            total_weighted = 0
            total_unweighted = 0
            for tokseq, embed, label, usage_count in folds[eval_fold_index]:
                seq_dict = self.tokseq2dict[tokseq]
                X_vec = np.asarray(embed, dtype=np.float32).reshape(1, -1)
                probs = cv_classifier.predict_proba(X_vec)[0]
                pairs = list((c, float(p)) for c, p in zip(cv_class_names, probs))
                pairs.sort(key=lambda x: x[1], reverse=True)
                # prediction = {c: p for c, p in pairs}
                prediction = pairs[0][0]
                correct = label == prediction
                if correct:
                    tp_weighted += usage_count
                    tp_unweighted += 1
                    seq_dict["cv_predicted"][0] += 1
                else:
                    seq_dict["cv_predicted"][1] += 1
                seq_dict["cv_prediction"][prediction] += 1
                total_weighted += usage_count
                total_unweighted += 1
            fold_results = {
                "fold": eval_fold_index,
                "weighted": {
                    "tp": tp_weighted,
                    "total": total_weighted,
                    "accuracy": tp_weighted / total_weighted,
                },
                "unweighted": {
                    "tp": tp_unweighted,
                    "total": total_unweighted,
                    "accuracy": tp_unweighted / total_unweighted,
                }
            }
            results.append(fold_results)
            logging.info(
                f"Weighted accuracy for fold {eval_fold_index}: {tp_weighted} / {total_weighted} = {tp_weighted/ total_weighted}"
            )
            logging.info(
                f"Unweighted accuracy for fold {eval_fold_index}: {tp_unweighted} / {total_unweighted} = {tp_unweighted/ total_unweighted}"
            )
        return results
