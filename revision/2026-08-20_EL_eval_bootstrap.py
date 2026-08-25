#!/usr/bin/env python3
"""Evaluate entity-linking predictions with passage-level bootstrap CIs.

The point estimates match ``2026-08-20_EL_eval.py``.  Confidence intervals
are percentile intervals from bootstrap samples of passage IDs, with each
sample containing the same number of passage IDs as the input collection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, Sequence

DEFAULT_MODELS = ("SapBERT", "MedCPT-Query", "OpenAI-txt-emb-3-L")
ENTITY_TYPES = ("cell_phenotype", "cell_hetero")
TOP_K = (1, 5, 10)


def default_path(script_dir: Path, relative_path: str) -> Path:
    """Resolve a default path relative to this script, independent of cwd."""
    return (script_dir / relative_path).resolve()


def parse_args(script_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="BioC XML file containing reference annotations and predictions."
    )
    parser.add_argument(
        "--cell-types", type=Path, default=default_path(script_dir, "../Cell-Ontology_v2025-01-08.json")
    )
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="Model prefixes used in the BioC infon fields."
    )
    parser.add_argument(
        "--entity-types", nargs="+", choices=ENTITY_TYPES, default=list(ENTITY_TYPES), help="Entity types to evaluate."
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of bootstrap samples (default: 10000).")
    parser.add_argument("--seed", type=int, default=None, help="Optional NumPy random seed for reproducibility.")
    parser.add_argument(
        "--chunk-size", type=int, default=256, help="Bootstrap replicates processed at once (default: 256)."
    )
    parser.add_argument("--hide-bootstrap", action="store_true", help="Flag to hide bootstrap values, making output exactly match previous versions")
    args = parser.parse_args()
    if args.n_samples < 1 or args.chunk_size < 1:
        parser.error("--n-samples and --chunk-size must be positive.")
    return args


def load_inputs(input_path: Path, cell_types_path: Path):
    import bioc

    if not input_path.is_file():
        raise FileNotFoundError(f"Input BioC file not found: {input_path}")
    if not cell_types_path.is_file():
        raise FileNotFoundError(f"Cell-types JSON file not found: {cell_types_path}")
    with cell_types_path.open(encoding="utf-8") as file:
        cell_types = json.load(file)
    with input_path.open(encoding="utf-8") as file:
        collection = bioc.load(file)
    return collection, cell_types


def validate_ids(ids: Sequence[str], cell_types: dict) -> None:
    invalid = [identifier for identifier in ids if identifier not in cell_types]
    if invalid:
        raise ValueError(f"Could not find {invalid} in the supplied cell-types JSON.")


def exact_ids_iterator(collection, cell_types: dict) -> Iterator[tuple]:
    """Yield annotations having one non-coordination, exact identifier."""
    for document in collection.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                identifier = annotation.infons["identifier"]
                if (
                    identifier
                    and "none" not in identifier.lower()
                    and ";" not in identifier
                    and "," not in identifier
                    and "related" not in identifier
                ):
                    identifier = identifier.replace("(skos:exact)", "")
                    validate_ids((identifier,), cell_types)
                    yield passage, annotation, (identifier,)


def all_labels_iterator(collection, cell_types: dict) -> Iterator[tuple]:
    """Yield all identifiers for cell phenotype and cell hetero annotations."""
    for document in collection.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                if annotation.infons["type"] not in ENTITY_TYPES:
                    continue
                identifier = annotation.infons["identifier"]
                identifier = identifier.replace("(skos:related)", "")
                identifier = identifier.replace("(skos:exact)", "")
                identifier = identifier.replace("None", "")
                identifiers = [item for item in re.split(",|;", identifier) if item not in ("-", "")]
                validate_ids(identifiers, cell_types)
                yield passage, annotation, identifiers


def prediction_tuples(passage, annotation, model_name: str, limit: int = 10) -> list[tuple]:
    predictions = []
    for index in range(limit):
        identifier = annotation.infons.get(f"{model_name}_id_{index}", "")
        if identifier is None:
            identifier = ""
        if str(identifier).strip().lower() in ("", "-", "none"):
            continue
        predictions.append((passage.infons["passage_id"], annotation.infons["type"], str(identifier)))
    return predictions


def passage_ids(collection) -> list[str]:
    """Return the ordered, unique passage IDs used as bootstrap units."""
    ids = dict.fromkeys(
        passage.infons["passage_id"] for document in collection.documents for passage in document.passages
    )
    if not ids:
        raise ValueError("The collection contains no passages.")
    return list(ids)


def passage_counts(
    collection,
    cell_types: dict,
    iterator,
    current_types: Sequence[str],
    models: Sequence[str],
    passage_id_list: Sequence[str],
):
    """Build per-passage TP, predicted, and reference counts once.

    Counts are based on sets, matching the original evaluator.  Keeping only
    counts after this pass makes each bootstrap replicate an array operation.
    """
    records = {(model, k): {pid: [set(), set()] for pid in passage_id_list} for model in models for k in TOP_K}
    for passage, annotation, reference_ids in iterator(collection, cell_types):
        if annotation.infons["type"] not in current_types:
            continue
        # Keep predictions for annotations whose reference identifier is
        # "None". allLabels_iterator yields these with an empty reference
        # label list, while the established evaluator counts their
        # predictions as false positives.
        pid = passage.infons["passage_id"]
        for model in models:
            top10 = prediction_tuples(passage, annotation, model)
            for k in TOP_K:
                reference, predicted = records[(model, k)][pid]
                reference.update((pid, annotation.infons["type"], identifier) for identifier in reference_ids)
                predicted.update(top10[:k])

    import numpy as np

    arrays = {}
    for key, record in records.items():
        tp = []
        predicted = []
        reference = []
        for pid in passage_id_list:
            ref, pred = record[pid]
            tp.append(len(ref.intersection(pred)))
            predicted.append(len(pred))
            reference.append(len(ref))
        arrays[key] = (
            np.asarray(tp, dtype=np.int64),
            np.asarray(predicted, dtype=np.int64),
            np.asarray(reference, dtype=np.int64),
        )
    return arrays


def bootstrap_metrics(counts, n_samples: int, seed: int | None, chunk_size: int):
    import numpy as np

    true_positive, predicted, reference = counts
    n_passages = len(true_positive)
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, 3), dtype=float)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        choices = rng.integers(0, n_passages, size=(stop - start, n_passages))
        tp = true_positive[choices].sum(axis=1)
        pp = predicted[choices].sum(axis=1)
        rr = reference[choices].sum(axis=1)
        precision = np.divide(tp, pp, out=np.zeros_like(tp, dtype=float), where=pp != 0)
        recall = np.divide(tp, rr, out=np.zeros_like(tp, dtype=float), where=rr != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp, dtype=float),
            where=(precision + recall) != 0,
        )
        result[start:stop] = np.column_stack((precision, recall, f1))
    return result


def report(model: str, k: int, counts, bootstrap: object, show_bootstrap: bool) -> None:
    import numpy as np

    tp, predicted, reference = (int(values.sum()) for values in counts)
    point = np.array([tp / predicted if predicted else 0.0, tp / reference if reference else 0.0, 0.0])
    point[2] = 2 * point[0] * point[1] / (point[0] + point[1]) if point[0] + point[1] else 0.0
    low, high = np.percentile(bootstrap, (2.5, 97.5), axis=0)
    if show_bootstrap:
        print(
            f"{model} top-{k} results: precision {point[0]:.3f} "
            f"(95% CI {low[0]:.3f}-{high[0]:.3f}; {tp}/{predicted}), "
            f"recall {point[1]:.3f} (95% CI {low[1]:.3f}-{high[1]:.3f}; {tp}/{reference}), "
            f"F1 {point[2]:.3f} (95% CI {low[2]:.3f}-{high[2]:.3f})"
        )
    else:
        print(
            f"{model} top-{k} results: precision {point[0]:.3f} "
            f"({tp}/{predicted}), "
            f"recall {point[1]:.3f} ({tp}/{reference}), "
            f"F1 {point[2]:.3f}"
        )


def evaluate_linking(
    collection,
    cell_types: dict,
    models: Sequence[str],
    entity_types: Sequence[str],
    n_samples: int,
    seed: int | None,
    chunk_size: int,
    show_bootstrap: bool,
) -> None:
    all_passage_ids = passage_ids(collection)
    iterator_specs = (("exactIDsOnly_iterator", exact_ids_iterator), ("allLabels_iterator", all_labels_iterator))
    for selected_types in (list(entity_types),):
        # Evaluate each requested type separately, then the combined set.
        type_sets = [[entity_type] for entity_type in selected_types]
        if len(selected_types) > 1:
            type_sets.append(list(selected_types))
        for current_types in type_sets:
            print(current_types)
            for iterator_name, iterator in iterator_specs:
                print(iterator_name)
                arrays = passage_counts(collection, cell_types, iterator, current_types, models, all_passage_ids)
                for model_index, model in enumerate(models):
                    for k_index, k in enumerate(TOP_K):
                        bootstrap = bootstrap_metrics(
                            arrays[(model, k)],
                            n_samples,
                            None if seed is None else seed + model_index * 3 + k_index,
                            chunk_size,
                        )
                        report(model, k, arrays[(model, k)], bootstrap, show_bootstrap)
            print()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    args = parse_args(script_dir)
    collection, cell_types = load_inputs(args.input, args.cell_types)
    evaluate_linking(collection, cell_types, args.models, args.entity_types, args.n_samples, args.seed, args.chunk_size, not args.hide_bootstrap)


if __name__ == "__main__":
    main()
