"""Add article-level fields from an enriched article TSV to document metadata JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_HEADERS = ["PMID", "PMC", "DOI", "Full text", "License"]


def load_articles(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_HEADERS:
            raise ValueError(
                f"expected TSV headers {EXPECTED_HEADERS!r}; found {reader.fieldnames!r}"
            )
        articles: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            pmid = (row["PMID"] or "").strip()
            if not pmid:
                raise ValueError(f"missing PMID on TSV row {row_number}")
            if pmid in articles:
                raise ValueError(f"duplicate PMID {pmid!r} on TSV row {row_number}")
            full_text = (row["Full text"] or "").strip()
            if full_text not in {"True", "False"}:
                raise ValueError(
                    f"Full text must be True or False on TSV row {row_number}; "
                    f"found {full_text!r}"
                )
            articles[pmid] = {
                "doi": (row["DOI"] or "").strip() or None,
                "full_text": full_text == "True",
                "license_URL": (row["License"] or "").strip() or None,
            }
    return articles


def enrich(input_path: Path, articles_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as handle:
        metadata: dict[str, dict[str, Any]] = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("document metadata JSON must contain a PMID-keyed object")

    articles = load_articles(articles_path)
    metadata_pmids = set(metadata)
    article_pmids = set(articles)
    if metadata_pmids != article_pmids:
        missing_from_tsv = sorted(metadata_pmids - article_pmids)
        missing_from_json = sorted(article_pmids - metadata_pmids)
        details = []
        if missing_from_tsv:
            details.append(f"missing from articles TSV: {missing_from_tsv[:10]}")
        if missing_from_json:
            details.append(f"missing from metadata JSON: {missing_from_json[:10]}")
        raise ValueError("PMID sets do not match (" + "; ".join(details) + ")")

    for pmid, values in articles.items():
        metadata[pmid].update(values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=3, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", type=Path, help="Enriched articles.tsv file")
    parser.add_argument("metadata", type=Path, help="Input document_metadata.json file")
    parser.add_argument("output", type=Path, help="Output enriched document_metadata.json file")
    args = parser.parse_args()
    enrich(args.metadata, args.articles, args.output)


if __name__ == "__main__":
    main()
