#!/usr/bin/env python3
"""Copy selected annotation infons from one BioC XML file into another.

The BioC Python API stores annotation locations in ``annotation.locations``;
the first location's ``(offset, length)`` is used as the positional part of
an annotation's identity.
"""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path

import bioc


log = logging.getLogger(__name__)
AnnotationKey = tuple[object, object, object, object, object, object]
PassageKey = tuple[object, object]


def load_collection(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"BioC XML file not found: {path}")
    with path.open(encoding="utf-8") as input_file:
        return bioc.load(input_file)


def annotation_key(document, passage, annotation) -> AnnotationKey:
    if not annotation.locations:
        log.warning(
            "Annotation %r in document %r, passage %r has no location; "
            "using None for both positional fields.",
            annotation.id, document.id, passage.infons.get("passage_id"),
        )
        offset, length = None, None
    else:
        location = annotation.locations[0]
        offset, length = location.offset, location.length
    return (
        document.id,
        passage.infons.get("passage_id"),
        offset,
        length,
        annotation.infons.get("type"),
        annotation.text,
    )


def passage_key(document, passage) -> PassageKey:
    # missing passage_id is an error
    return document.id, passage.infons["passage_id"]


def collection_structure(collection):
    """Return document, passage, and annotation occurrence counters."""
    documents = collections.Counter(document.id for document in collection.documents)
    passages = collections.Counter(
        passage_key(document, passage)
        for document in collection.documents
        for passage in document.passages
    )
    annotations = collections.Counter(
        annotation_key(document, passage, annotation)
        for document in collection.documents
        for passage in document.passages
        for annotation in passage.annotations
    )
    return documents, passages, annotations


def warn_structure_difference(name: str, first: collections.Counter,
                              second: collections.Counter) -> None:
    if first == second:
        return
    only_first = first - second
    only_second = second - first
    if only_first:
        log.warning("%s present in input1 but not input2: %s", name, list(only_first.elements()))
    if only_second:
        log.warning("%s present in input2 but not input1: %s", name, list(only_second.elements()))


def collect_infons(collection, infon_key_prefix: str):
    stored: dict[AnnotationKey, dict] = {}
    for document in collection.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                key = annotation_key(document, passage, annotation)
                matching_infons = {
                    infon_key: value
                    for infon_key, value in annotation.infons.items()
                    if infon_key.startswith(infon_key_prefix)
                }
                if key in stored:
                    for infon_key, value in matching_infons.items():
                        if infon_key in stored[key] and stored[key][infon_key] != value:
                            log.warning(
                                "Conflicting input2 value for annotation %r, infon %r: "
                                "%r overwritten by %r.",
                                key, infon_key, stored[key][infon_key], value,
                            )
                        stored[key][infon_key] = value
                else:
                    stored[key] = matching_infons
    return stored


def copy_infons(collection, stored: dict[AnnotationKey, dict]) -> None:
    for document in collection.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                key = annotation_key(document, passage, annotation)
                if key not in stored:
                    log.warning("Annotation in input1 has no corresponding input2 annotation: %r", key)
                    continue
                for infon_key, value in stored[key].items():
                    if infon_key in annotation.infons and annotation.infons[infon_key] != value:
                        log.warning(
                            "Conflicting value for annotation %r, infon %r: %r overwritten by %r.",
                            key, infon_key, annotation.infons[infon_key], value,
                        )
                    annotation.infons[infon_key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input1", type=Path, help="BioC XML file to update.")
    parser.add_argument("input2", type=Path, help="BioC XML file supplying infons.")
    parser.add_argument("infon_key_prefix", help="Prefix selecting annotation infon keys.")
    parser.add_argument("output", type=Path, help="Output BioC XML filename.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="Logging level for warnings and diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    collection1 = load_collection(args.input1)
    collection2 = load_collection(args.input2)

    structure1 = collection_structure(collection1)
    structure2 = collection_structure(collection2)
    for name, first, second in zip(
        ("Documents", "Passages", "Annotations"), structure1, structure2
    ):
        warn_structure_difference(name, first, second)

    stored = collect_infons(collection2, args.infon_key_prefix)
    copy_infons(collection1, stored)
    with args.output.open("w", encoding="utf-8") as output_file:
        bioc.dump(collection1, output_file)


if __name__ == "__main__":
    main()
