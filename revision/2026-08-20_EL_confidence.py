#!/usr/bin/env python3
"""Evaluate entity-linking predictions in a BioC XML collection.

The defaults reproduce the inputs, models, and analyses used by the original
2025 evaluator, while allowing them to be changed from the command line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_MODELS = ("SapBERT", "MedCPT-Query", "OpenAI-txt-emb-3-L", "GPT-5.2_Agent")
ENTITY_TYPES = ("cell_phenotype", "cell_hetero")
LINKAGE_TYPES = ("exact", "related", "none")
TOP_K = (1, 5, 10)


def default_path(script_dir: Path, relative_path: str) -> Path:
    """Resolve a default path relative to this script, independent of cwd."""
    return (script_dir / relative_path).resolve()


def parse_args(script_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="BioC XML file containing the reference annotations and predictions.",
    )
    parser.add_argument(
        "--cell-types",
        type=Path,
        default=default_path(script_dir, "../Cell-Ontology_v2025-01-08.json"),
        help="JSON file containing valid cell ontology identifiers.",
    )
    parser.add_argument(
        "--confidence-model",
        default="SapBERT",
        help="Model whose confidence fields are used for the confidence analysis (default: SapBERT).",
    )
    parser.add_argument(
        "--entity-types",
        nargs="+",
        choices=ENTITY_TYPES,
        default=list(ENTITY_TYPES),
        help="Entity types to evaluate (default: both).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Save generated plots in this directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively.",
    )
    return parser.parse_args()


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


def confidence_dataframe(collection, cell_types: dict, model: str) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for document in collection.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                if annotation.infons["type"] not in ENTITY_TYPES:
                    continue
                identifier = annotation.infons["identifier"]
                if ";" in identifier:
                    continue
                if "(skos:related)" in identifier:
                    linkage_type, identifier = "related", identifier.replace("(skos:related)", "")
                elif "(skos:exact)" in identifier:
                    linkage_type, identifier = "exact", identifier.replace("(skos:exact)", "")
                elif identifier == "None":
                    linkage_type, identifier = "none", ""
                else:
                    raise ValueError(f"Unsupported identifier format: {identifier}")
                identifiers = [item for item in re.split(",", identifier) if item not in ("-", "")]
                validate_ids(identifiers, cell_types)
                prediction = annotation.infons[f"{model}_id_0"]
                rows.append(
                    {
                        "entity_type": annotation.infons["type"],
                        "identifiers": identifiers,
                        "linkage_type": linkage_type,
                        "correct": prediction in identifiers,
                        "confidence": float(annotation.infons[f"{model}_identifier_score_0"]),
                        "identifier": annotation.infons["identifier"],
                        "annotation_text": annotation.text,
                    }
                )
    return pd.DataFrame(rows)


def save_or_show(figure, name: str, args: argparse.Namespace) -> None:
    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot_dir / name, bbox_inches="tight")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


def confidence_analysis(collection, cell_types: dict, model: str, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn.metrics

    data = confidence_dataframe(collection, cell_types, model)
    if data.empty:
        print("No confidence-analysis annotations found.")
        return

    plt.figure()
    for linkage_type in LINKAGE_TYPES:
        plt.hist(data.loc[data.linkage_type == linkage_type, "confidence"], bins=np.linspace(0, 1, 21), edgecolor="black", alpha=0.5, label=linkage_type)
    plt.legend()
    plt.title(f"{model} confidence by linkage type")
    plt.xlabel("cosine similarity")
    plt.ylabel("number of mentions")
    save_or_show(plt.gcf(), "confidence_by_linkage_type.png", args)

    data["successful_match"] = data["linkage_type"].eq("exact") & data["correct"]
    fpr, tpr, _ = sklearn.metrics.roc_curve(data["successful_match"], data["confidence"])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.xlabel("False Positive Rate (1-specificity)")
    plt.title(f"ROC curve of {model} successful matching based on confidence")
    save_or_show(plt.gcf(), "confidence_success_roc.png", args)
    print(f"Overall successful-match AUROC: {sklearn.metrics.auc(fpr, tpr):.3f}")

    negatives = data[~data.successful_match].sort_values("confidence", ascending=False)
    positives = data[data.successful_match].sort_values("confidence")
    print("\nHighest-confidence unsuccessful terms:\n", negatives.head(10).annotation_text, sep="")
    print("\nLowest-confidence successful terms:\n", positives.head(10).annotation_text, sep="")

    type_sets = [[entity_type] for entity_type in args.entity_types]
    if len(args.entity_types) > 1:
        type_sets.append(list(args.entity_types))
    for selected_types in type_sets:
        subset = data[data.entity_type.isin(selected_types)]
        for positive_label, positive_mask, description in (
            ("exact", subset.linkage_type.eq("exact"), "exact vs related & no ID"),
            ("non-none", subset.linkage_type.ne("none"), "exact & related vs no ID"),
        ):
            if positive_mask.nunique() < 2:
                print(selected_types, description, "AUROC: unavailable (only one class)")
                continue
            fpr, tpr, _ = sklearn.metrics.roc_curve(positive_mask, subset.confidence)
            print(selected_types, description, "AUROC:", sklearn.metrics.auc(fpr, tpr))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    args = parse_args(script_dir)
    collection, cell_types = load_inputs(args.input, args.cell_types)
    confidence_analysis(collection, cell_types, args.confidence_model, args)


if __name__ == "__main__":
    main()
