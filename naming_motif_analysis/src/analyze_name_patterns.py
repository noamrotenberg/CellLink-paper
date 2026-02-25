import gzip
import json
import pathlib
from collections import Counter
from collections import defaultdict
import math
import csv
import logging
import argparse

from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.multitest import multipletests

import cell_analysis_utils as cau
from annotation_filter import SimpleAnnotationFilter
from abbreviations import AbbreviationExpander
import name_motifs
import file_utils
from multi_counter import MultiCounter

# Configure basic logging from console
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

LINEAGES = [
            "Epithelial",
            "Endothelial",
            "Mesenchymal/stromal",
            "Muscle",
            "Hematopoietic",
            "Neuronal",
            "Glial",
            "Stem/progenitor",
            "Germ line",
            "Trophoblast/placental",
            "Other",
        ]
MOTIFS = [
            "root",
            "anatomical",
            "lineage",
            "developmental",
            "appearance",
            "role",
            "variant",
            "molecular signaling",
            "molecular signature",
            "eponym",
            "species",
            "state",
            "disease",
            "stimulus",
            "unknown token",
        ]

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.examples_filename is not None and args.example_count <= 0:
        parser.error("--example-count must be greater than 0 when --examples is provided")
    if args.cv_results_filename is not None and args.cv_iter_count <= 0:
        parser.error("--cv-iter-count must be greater than 0 when --cv-results is provided")
    if args.cv_results_filename is not None and args.cv_fold_count <= 0:
        parser.error("--cv-fold-count must be greater than 0 when --cv-results is provided")

    # Load the ontology dictionary
    with open(args.cl_filename, "r") as f:
        ontology = json.load(f)

    file_map = file_utils.map_path(args.input_path, ["*.xml", "*.xml.gz"])
    logging.info("Found {} files to process".format(len(file_map)))
    if len(file_map) == 0:
        exit(0)

    # Load abbreviations
    logging.info("Loading abbreviations")
    abbr_freq_dict = dict()
    if not args.abbr_freq_filename is None:
        # Load the abbreviation frequency file
        abbr_freq_path = pathlib.Path(args.abbr_freq_filename)
        open_func = gzip.open if abbr_freq_path.suffix == ".gz" else open
        with open_func(abbr_freq_path, "rt") as abbr_freq_file:
            abbr_freq_dict = json.load(abbr_freq_file)
    abbr = AbbreviationExpander(abbr_freq_dict)
    abbr.load(args.abbr_path)

    annotations = list()
    for input_filename in file_map:
        annotations.extend(cau.process_collection(input_filename, abbr))
    logging.info(f"Found {len(annotations)} annotation pairs")

    filters = dict()
    filters["cell_phenotype(exact)"] = SimpleAnnotationFilter(
        allowed_mention_types={"cell_phenotype"},
        allowed_qualifiers={"(skos:exact)"},
        allowed_coordination_lengths={1},
    )
    filters["cell_phenotype(related)"] = SimpleAnnotationFilter(
        allowed_mention_types={"cell_phenotype"},
        allowed_qualifiers={"(skos:related)"},
        allowed_coordination_lengths={1},
    )
    filters["cell_hetero"] = SimpleAnnotationFilter(
        allowed_mention_types={"cell_hetero"},
        allowed_qualifiers={"(skos:exact)", "(skos:related)"},
        allowed_coordination_lengths={1},
    )
    mention_key2info = get_mention_info(ontology, annotations, filters)
    parser = name_motifs.NameMotifParser.load(
        ontology,
        [mention_text for mention_text, _mention_type, _identifier_list in mention_key2info.keys()],
        args.name_motif_map_filename,
        args.term_cache_filename,
        args.vector_cache_filename,
    )
    if parser.tokseq2dict is None or parser.seq_label_counts is None:
        raise RuntimeError("parser not initialized")
    tokseq_usage = Counter(
        [
            (seq_dict.get("usage_count", 0) > 0, seq_dict.get("label") is None)
            for seq_dict in parser.tokseq2dict.values()
        ]
    )
    logging.info(f"tokseq_usage = {tokseq_usage}")
    remaining_tokens = Counter()
    for (mention_text, _mention_type, _identifier_list), mention_dict in mention_key2info.items():
        unused_tokens, motif_counts = parser.find_motifs(mention_text)
        remaining_tokens.update(unused_tokens)
        mention_dict["motif_counts"] = dict(motif_counts)
    logging.info(f"Found {len(mention_key2info)} unique mentions matching the filter")
    logging.info(f"Remaining tokens: {len(remaining_tokens)}:")
    for token, count in remaining_tokens.most_common():
        logging.info(f"\t{token}\t{count}")
        # logging.info(f"{token}")

    motifs = [motif for motif, _count in parser.seq_label_counts.most_common() if not motif is None]
    motifs.append("unknown token")

    count_and_print(filters, mention_key2info, motifs)
    load_and_analyze(filters, mention_key2info, args.lineage_analysis_filename, args.motif_analysis_filename)
    if not args.examples_filename is None:
        prepare_and_output_examples(mention_key2info, parser, args.examples_filename, args.example_count)
    if args.cv_results_filename is not None:
        cross_validate(parser, args.cv_results_filename, args.cv_iter_count, args.cv_fold_count)
    logging.info("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run lineage/motif analysis with cached resources and optional outputs.")

    # Inputs (required)
    p.add_argument("--input", required=True, dest="input_path", help="Main input path.")
    p.add_argument("--abbr", required=True, dest="abbr_path", help="Abbreviation source file/path.")
    p.add_argument("--abbr-freq", dest="abbr_freq_filename", help="Abbreviation frequency filename.")
    p.add_argument("--cl", required=True, dest="cl_filename", help="Cell ontology / CL filename.")
    p.add_argument("--name-motif-map", required=True, dest="name_motif_map_filename", help="Name motif map filename.")
    p.add_argument("--term-cache", required=True, dest="term_cache_filename", help="Term cache filename.")
    p.add_argument("--vector-cache", required=True, dest="vector_cache_filename", help="Vector cache filename.")

    # Outputs (optional)
    p.add_argument(
        "--lineage-analysis",
        dest="lineage_analysis_filename",
        default=None,
        help="Output filename for lineage analysis.",
    )
    p.add_argument(
        "--motif-analysis", dest="motif_analysis_filename", default=None, help="Output filename for motif analysis."
    )
    p.add_argument("--examples", dest="examples_filename", default=None, help="Output filename for examples.")
    p.add_argument(
        "--example-count",
        dest="example_count",
        type=int,
        default=100,
        help="Number of examples to output (default=100).",
    )
    p.add_argument(
        "--cv-results", dest="cv_results_filename", default=None, help="Output filename for cross-validation results."
    )
    p.add_argument(
        "--cv-iter-count",
        dest="cv_iter_count",
        type=int,
        default=1,
        help="Number of CV iterations (default=1).",
    )
    p.add_argument(
        "--cv-fold-count",
        dest="cv_fold_count",
        type=int,
        default=10,
        help="Number of CV folds (default=10).",
    )

    return p

def get_mention_key(expanded_text, mention_type, identifier_list):
    return (expanded_text, mention_type, str(identifier_list))


def get_mention_info(ontology, annotations, filters):
    mentions_kept = list()
    for pmid, mention_text, expanded_text, mention_type, identifier_list in annotations:
        keep = False
        filter_dict = dict()
        for filter_name, annotation_filter in filters.items():
            if annotation_filter.filter(mention_text, expanded_text, mention_type, identifier_list):
                filter_dict[filter_name] = False
            else:
                keep = True
                filter_dict[filter_name] = True
        logging.debug(
            f'Annotation filtering: mention "{expanded_text}" type: {mention_type} identifier_list: {identifier_list} keep: {keep} filters: {filter_dict}'
        )
        if keep:
            mention_key = get_mention_key(expanded_text, mention_type, identifier_list)
            identifier_counts = cau.count_fractional_identifiers(identifier_list)
            lineage_counts = cau.count_fractional_lineage(identifier_counts, ontology)
            mentions_kept.append((mention_key, filter_dict, lineage_counts))
    logging.info(f"Kept {len(mentions_kept)} mentions out of {len(annotations)} total annotations")
    mention_key2info = dict()
    for mention_key, filter_dict, lineage_counts in mentions_kept:
        if mention_key in mention_key2info:
            mention_dict = mention_key2info[mention_key]
        else:
            mention_dict = dict()
            mention_dict["filters"] = filter_dict
            mention_dict["count"] = 0
            mention_dict["lineage_counts"] = Counter()
            mention_key2info[mention_key] = mention_dict
        # Update
        mention_dict["count"] += 1
        mention_dict["lineage_counts"].update(lineage_counts)
    for (expanded_text, mention_type, identifier_list), mention_dict in mention_key2info.items():
        count = mention_dict["count"]
        lineage_counts = list(mention_dict["lineage_counts"].items())
        lineage_counts.sort()
        logging.info(
            f"Mention info: mention: {expanded_text} type: {mention_type} identifiers {identifier_list} => count: {count} lineage_counts: {lineage_counts}"
        )
    return mention_key2info


def cross_validate(parser, cv_results_filename, cv_iter_count, cv_fold_count):
    with open(cv_results_filename, "w") as file:
        file.write("Iteration\tFold\tWeighted accuracy\tUnweighted accuracy\n")
        for iteration in range(cv_iter_count):
            cv_result = parser.cross_validate(cv_fold_count, ["corpus"])
            for fold_result in cv_result:
                file.write(
                    f"{iteration}\t{fold_result['fold']}\t{fold_result['weighted']['accuracy']}\t{fold_result['unweighted']['accuracy']}\n"
                )

        file.write("\n")
        file.write("tokseq\tlabel\tcv_label\tcv_count\tcorpus_count\tcv_predicted[0]\tcv_predicted[1]\n")
        for tokseq, seq_dict in parser.tokseq2dict.items():
            label = seq_dict["label"]
            usage_counts = seq_dict["usage_counts"]
            corpus_count = usage_counts.get("corpus", 0)
            cv_predicted = seq_dict.get("cv_predicted", [0, 0])
            cv_prediction = seq_dict.get("cv_prediction", Counter())
            cv_label, cv_count = cv_prediction.most_common(1)[0] if len(cv_prediction) > 0 else (None, 0)
            file.write(
                f"{tokseq}\t{label}\t{cv_label}\t{cv_count}\t{corpus_count}\t{cv_predicted[0]}\t{cv_predicted[1]}\n"
            )

def load_and_analyze(filters, mention_key2info, lineage_analysis_filename=None, motif_analysis_filename=None):
    """
    Main orchestration function.

    :param filters: dict, keys are filter names
    :param mention_key2info: dict, mention data
    :param motifs: list, all possible motif names
    :param lineage_analysis_filename: output filename for lineage analysis
    :param motif_analysis_filename: output filename for motif analysis
    """

    # Aggregate all counts
    counts = MultiCounter(dimensions=["filter", "lineage", "motif"])
    counts.add_keys("lineage", LINEAGES)
    counts.add_keys("motif", MOTIFS)
    _calculate_counts(filters, mention_key2info, counts)

    # Perform lineage-level p-value analysis
    lineage_analysis = dict()
    _perform_lineage_analysis(lineage_analysis, counts, "cell_phenotype(exact)", None)
    _perform_lineage_analysis(lineage_analysis, counts, "cell_phenotype(related)", "cell_phenotype(exact)")
    _perform_lineage_analysis(lineage_analysis, counts, "cell_hetero", "cell_phenotype(exact)")

    # Perform motif-level p-value analysis
    motif_analysis = dict()
    _perform_motif_analysis(motif_analysis, counts, "cell_phenotype(related)", "cell_phenotype(exact)")
    _perform_motif_analysis(motif_analysis, counts, "cell_hetero", "cell_phenotype(exact)")

    # Apply FDR correction
    # This modifies lineage_analysis and motif_analysis in-place
    _apply_fdr_correction(lineage_analysis, motif_analysis)

    # Write new output files
    if not lineage_analysis_filename is None:
        _write_lineage_output(lineage_analysis, counts, lineage_analysis_filename)
    if not motif_analysis_filename is None:
        _write_motif_output(motif_analysis, counts, motif_analysis_filename)

    logging.info(f"Analysis complete.")


def _calculate_counts(filters, mention_key2info, counts):
    """
    Aggregates fractional counts for each filter based on mention data.
    """
    logging.info("Calculating counts for each filter...")
    for filter in filters.keys():
        logging.info(f"  Processing filter = {filter}")
        for mention_dict in mention_key2info.values():
            if not mention_dict["filters"][filter]:
                continue

            lineage_counts = mention_dict["lineage_counts"]
            motif_type_counts = mention_dict["motif_counts"]
            # Factor to convert mention counts to fractional counts
            mention_factor = 1.0 / mention_dict["count"]

            for lineage, lineage_count in lineage_counts.items():
                fractional_lineage = mention_factor * lineage_count
                for motif_type, motif_type_count in motif_type_counts.items():
                    # If motif_type_count > 0, this lineage has this motif
                    # The fractional count is the same as the fractional_lineage
                    fractional_motif = fractional_lineage if motif_type_count > 0 else 0
                    if fractional_motif > 0:
                        # Increment specific (lineage, motif) count
                        counts.add(
                            fractional_motif,
                            filter=filter,
                            lineage=lineage,
                            motif=motif_type,
                        )


def _perform_lineage_analysis(analysis, counts, filter, comparison_filter):
    """
    Performs the original p-value calculations for each (filter, lineage, motif).

    pvalue1: (lineage, motif) rate vs. (MTOTAL, motif) rate [within same filter]
    pvalue2: (lineage, motif) rate vs. (lineage, motif) rate [in 'exact' filter]
    """
    lineages = counts.get_keys("lineage")
    motifs = counts.get_keys("motif")
    for lineage in lineages:
        for motif in motifs:
            data_dict = {}

            # Get counts for the current (filter, lineage, motif)
            k = counts.get(filter=filter, lineage=lineage, motif=motif)
            n = counts.get(filter=filter, lineage=lineage)  # sum over all motifs

            # Store raw counts
            data_dict["k_count"] = k
            data_dict["n_total"] = n
            data_dict["rate"] = (k / n) if n > 0 else 0.0

            if comparison_filter is None:
                k2 = counts.get(filter=filter, motif=motif)  # sum over all lineages
                n2 = counts.get(filter=filter)  # sum for filter
            else:
                k2 = counts.get(filter=comparison_filter, lineage=lineage, motif=motif)
                n2 = counts.get(filter=comparison_filter, lineage=lineage)

            data_dict["comp_k_count"] = k2
            data_dict["comp_n_total"] = n2
            data_dict["comp_rate"] = (k2 / n2) if n2 > 0 else 0.0

            pvalue = math.nan
            if n > 0 and n2 > 0:
                try:
                    # Test if rate k/n is different from the 'exact' rate k2/n2
                    result = binomtest(
                        round(k),
                        round(n),
                        data_dict["comp_rate"],
                        alternative="two-sided",
                    )
                    pvalue = result.pvalue
                except ValueError as e:
                    # (e.g., k > n after rounding, or n=0)
                    # logging.info(f"  WARN: binomtest 2 failed for {filter}/{lineage}/{motif}: {e}")
                    pass
            data_dict["pvalue"] = pvalue
            data_dict["qvalue"] = math.nan
            analysis[(filter, lineage, motif)] = data_dict

    return analysis


def _perform_motif_analysis(analysis, counts, filter, comparison_filter):
    """
    Performs motif-level comparison between each filter and the 'exact' filter.

    pvalue3: (MTOTAL, motif) rate vs. (MTOTAL, motif) rate [in 'exact' filter]
    Uses Fisher's Exact Test on the 2x2 contingency table.
    """
    logging.info("Performing motif-level analysis...")
    o_filter = counts.get(filter=filter)
    o_comparison = counts.get(filter=comparison_filter)

    logging.info(f"  Analyzing filter: {filter} vs. cell_phenotype(exact)")
    for c_motif in counts.get_keys("motif"):
        data_dict = {}

        # Counts for current filter
        m = counts.get(filter=filter, motif=c_motif)  # sum over all lineages

        # Counts for 'exact' filter
        m_comparison = counts.get(filter=comparison_filter, motif=c_motif)
        # o_exact defined above

        data_dict["motif_count"] = m
        data_dict["filter_total"] = o_filter
        data_dict["rate"] = (m / o_filter) if o_filter > 0 else 0.0

        data_dict["comp_motif_count"] = m_comparison
        data_dict["comp_filter_total"] = o_comparison
        data_dict["comp_rate"] = (m_comparison / o_comparison) if o_comparison > 0 else 0.0

        pvalue3 = math.nan
        if o_filter > 0 or o_comparison > 0:
            # Create 2x2 table:
            #         [motif_count, other_count]
            # [filter]
            # [exact]
            table = [
                [round(m), round(o_filter - m)],
                [round(m_comparison), round(o_comparison - m_comparison)],
            ]

            try:
                # oddsr, p-value
                _, pvalue3 = fisher_exact(table)
            except ValueError as e:
                # (e.g., negative counts after rounding)
                # logging.info(f"  WARN: fisher_exact failed for {filter}/{motif}: {e}")
                pass

        data_dict["pvalue"] = pvalue3
        data_dict["qvalue"] = math.nan  # Placeholder

        analysis[(filter, c_motif)] = data_dict


def _apply_fdr_correction(lineage_analysis, motif_analysis):
    """
    Applies Benjamini-Hochberg FDR correction to the p-values.
    Modifies the analysis dicts in-place.
    """
    logging.info("Applying Benjamini-Hochberg FDR correction...")

    # --- Correct Lineage-Level p-values (p1 and p2) ---
    # We correct all p1 and p2 values (from 'related' and 'hetero' filters)
    # as one large family of tests.

    p_map_lineage = {key: data["pvalue"] for key, data in lineage_analysis.items() if not math.isnan(data["pvalue"])}

    if p_map_lineage:
        # Unpack keys and p-values for multipletests
        original_keys = list(p_map_lineage.keys())
        pvals = [p_map_lineage[k] for k in original_keys]

        # Run BH correction
        rejected, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        # Map q-values back to the main analysis dictionary
        for key_tuple, qval in zip(original_keys, qvals):
            lineage_analysis[key_tuple]["qvalue"] = qval
        logging.info("  Lineage-level FDR correction applied.")
    else:
        logging.info("  No lineage-level p-values found to correct.")

    # --- Correct Motif-Level p-values (p3) ---
    p_map_motif = {}  # Map for pvalue3
    p_map_motif = {key: data["pvalue"] for key, data in motif_analysis.items() if not math.isnan(data["pvalue"])}

    if p_map_motif:
        original_keys = list(p_map_motif.keys())
        pvals = [p_map_motif[k] for k in original_keys]

        rejected, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        for key, qval in zip(original_keys, qvals):
            motif_analysis[key]["qvalue"] = qval
        logging.info("  Motif-level FDR correction applied.")
    else:
        logging.info("  No motif-level p-values found to correct.")

    logging.info("Complete.")
    # No return value, modifies dicts in-place


def _write_lineage_output(lineage_analysis, counts, lineage_analysis_filename):
    """
    Writes the detailed lineage-by-motif analysis to a TSV file.
    """
    logging.info("Writing lineage analysis output file...")

    filters = list(counts.get_keys("filter"))
    values = ["k_count", "rate", "comp_rate", "pvalue", "qvalue"]
    lineages = list(counts.get_keys("lineage"))
    motifs = list(counts.get_keys("motif"))
    header = [
        "filter",
        "value",
        "motif",
    ]
    header.extend(lineages)

    logging.info(f"  Writing {lineage_analysis_filename}")
    with open(lineage_analysis_filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for filter in filters:
            for value in values:
                writer.writerow(header)
                for motif in motifs:
                    row = [filter, value, motif]
                    for lineage in lineages:
                        key = (filter, lineage, motif)
                        data = lineage_analysis[key]
                        row.append(data[value])
                    writer.writerow(row)
    logging.info("Complete.")


def _write_motif_output(motif_analysis, counts, motif_analysis_filename):
    """
    Writes the new motif-level comparison analysis to a single TSV file.
    """
    logging.info("Writing motif comparison output file...")

    filters = list(counts.get_keys("filter"))
    motifs = list(counts.get_keys("motif"))
    filters_analyzed = set()
    for filter in filters:
        for motif in motifs:
            key = (filter, motif)
            if key in motif_analysis:
                filters_analyzed.add(filter)

    values = [
        "rate",
        "comp_rate",
        "pvalue",
        "qvalue",
        "motif_count",
        "filter_total",
        "comp_motif_count",
        "comp_filter_total",
    ]

    header = ["Filter", "Motif"]
    header.extend(values)

    with open(motif_analysis_filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for filter in filters:
            if not filter in filters_analyzed:
                continue
            writer.writerow(header)
            for motif in motifs:
                key = (filter, motif)
                data = motif_analysis.get(key, dict())
                row = [filter, motif]
                for value in values:
                    row.append(data.get(value, "N/A"))
                writer.writerow(row)

    logging.info(f"  Writing {motif_analysis_filename}")
    logging.info("Complete.")


def count_and_print(filters, mention_key2info, motif_types):
    for filter in filters.keys():
        logging.info(f"filter = {filter}")
        lineage_totals = Counter()
        motif_type_set = set()
        motif_type_lineage_counts = Counter()
        for mention_dict in mention_key2info.values():
            if not mention_dict["filters"][filter]:
                continue
            mention_count = mention_dict["count"]
            lineage_counts = mention_dict["lineage_counts"]
            motif_type_counts = mention_dict["motif_counts"]

            mention_factor = 1.0 / mention_count
            for lineage, lineage_count in lineage_counts.items():
                fractional_lineage = mention_factor * lineage_count
                lineage_totals[lineage] += fractional_lineage
                for motif_type, motif_type_count in motif_type_counts.items():
                    motif_type_lineage_counts[(lineage, motif_type)] += fractional_lineage * (
                        1 if motif_type_count > 0 else 0
                    )
            motif_type_set.update(motif_type_counts.keys())

        error_count = 0
        for lineage in lineage_totals.keys():
            if not lineage in LINEAGES:
                logging.info(f'ERROR: lineage "{lineage}" not found')
                error_count += 1
        for motif_type in motif_type_set:
            if not motif_type in motif_types:
                logging.info(f'ERROR: motif_type "{motif_type}" not found')
                error_count += 1
        if error_count > 0:
            exit()

        header = ["Motif type"]
        for lineage in LINEAGES:
            header.append(lineage)
        logging.info("\t".join(header))
        for motif_type in motif_types:
            line = [motif_type]
            for lineage in LINEAGES:
                line.append(str(motif_type_lineage_counts[(lineage, motif_type)]))
            logging.info("\t".join(line))
        trailer = ["TOTAL"]
        for lineage in LINEAGES:
            trailer.append(str(lineage_totals[lineage]))
        logging.info("\t".join(trailer))


def prepare_and_output_examples(mention_key2info, parser, examples_filename, n_examples=10):
    # Get examples for each filter
    filter_counts = defaultdict(Counter)
    lineage_counts = defaultdict(Counter)
    motif_counts = defaultdict(Counter)
    for (mention_text, _mention_type, _identifier_list), mention_dict in mention_key2info.items():
        count = mention_dict["count"]
        # Filter
        for filter_name, keep in mention_dict["filters"].items():
            if not keep:
                continue
            filter_counts[filter_name][mention_text] += count
        # Lineage
        for lineage, fractional_count in mention_dict["lineage_counts"].items():
            lineage_counts[lineage][mention_text] += fractional_count * count
        # Motif
        updated_term_text = parser.name_part_updater.update(mention_text)
        term_dict = parser.term2dict.get(updated_term_text)
        if term_dict is None:
            continue
        doc = term_dict["doc"]
        for motif, start, end in term_dict["predictions"]:
            tokens = doc[start:end]
            seq_text = ("".join(token.text_with_ws for token in tokens)).strip()
            motif_counts[motif][seq_text] += count
    with open(examples_filename, "w") as file:
        file.write("EXAMPLES\n")
        file.write("\nFILTERS: Most common mentions annotated for each filter\n=====\n")
        for filter_name, mention_counts in filter_counts.items():
            file.write(f"{filter_name}:\n")
            for mention_text, count in mention_counts.most_common(n_examples):
                file.write(f"\t{mention_text}: {count}\n")
        file.write("\nLINEAGE: Most common mentions annotated for each lineage\n=====\n")
        for lineage, mention_counts in lineage_counts.items():
            file.write(f"{lineage}:\n")
            for mention_text, count in mention_counts.most_common(n_examples):
                file.write(f"\t{mention_text}: {count}\n")
        file.write("\nMOTIFS: Most common motif texts found for each motif type\n=====\n")
        for motif, motif_counts in motif_counts.items():
            file.write(f"{motif}:\n")
            for motif_text, count in motif_counts.most_common(n_examples):
                file.write(f"\t{motif_text}: {count}\n")


if __name__ == "__main__":
    main()
