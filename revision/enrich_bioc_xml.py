"""Enrich CellLink BioC XML files with article metadata from articles.tsv."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import bioc


EXPECTED_HEADERS = ["PMID", "PMC", "DOI", "Full text", "License"]
EXPECTED_PASSAGE_HEADERS = ["PMID", "passage_id"]
log = logging.getLogger(__name__)


def load_articles(path: Path) -> dict[str, dict[str, str | None]]:
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
                "pmc": (row["PMC"] or "").strip() or None,
                "doi": (row["DOI"] or "").strip() or None,
                "full_text": full_text == "True",
                "license_URL": (row["License"] or "").strip() or None,
            }
    return articles

def load_ann_or_decoy_passages(path: Path, articles: dict[str, dict[str, Any]]) -> None:
    print(f"Loading annotated or decoy passages from {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_PASSAGE_HEADERS:
            raise ValueError(
                f"expected TSV headers {EXPECTED_PASSAGE_HEADERS!r}; found {reader.fieldnames!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            #print(f"#{row_number}: row = {row}")
            pmid = (row["PMID"] or "").strip()
            if not pmid:
                raise ValueError(f"missing PMID on TSV row {row_number}")
            if not pmid in articles:
                raise ValueError(f"unknown PMID {pmid!r} on TSV row {row_number}")
            passage_id = (row["passage_id"] or "").strip()
            if not passage_id:
                raise ValueError(f"missing passage_id on TSV row {row_number}")
            if "ann_or_decoy_passage_ids" not in articles[pmid]:
                articles[pmid]["ann_or_decoy_passage_ids"] = set()
            articles[pmid]["ann_or_decoy_passage_ids"].add(passage_id)


def first_infon(passage: Any, key: str) -> str | None:
    value = passage.infons.get(key)
    return None if value is None else str(value)


def set_infon(passage: Any, key: str, value: str) -> None:
    passage.infons[key] = value


def normalized_pmc(value: str) -> str:
    value = value.strip()
    return value if value.upper().startswith("PMC") else "PMC" + value


def check_PMID(document: Any, articles: dict[str, dict[str, str | None]], source: Path) -> str:
    if not document.passages:
        raise ValueError(f"document {document.id!r} in {source} has no passages")
    passage = document.passages[0]
    # Full-document IDs are PMIDs; split-set document IDs may be passage IDs
    # such as ``38173036_67``.
    document_pmid = str(document.id).strip().split("_", 1)[0]
    existing_pmid = first_infon(passage, "article-id_pmid")
    if document_pmid and existing_pmid and document_pmid != existing_pmid:
        raise ValueError(
            f"document {document.id!r} in {source} has conflicting document and passage PMID"
        )
    pmid = existing_pmid or document_pmid
    if not pmid or pmid not in articles:
        raise ValueError(f"document {document.id!r} in {source} has unknown PMID {pmid!r}")
    if not existing_pmid:
        set_infon(passage, "article-id_pmid", pmid)
    return pmid

def enrich_document(pmid: str, document: Any, articles: dict[str, dict[str, str | None]], source: Path) -> None:
    if not document.passages:
        raise ValueError(f"document {document.id!r} in {source} has no passages")
    passage = document.passages[0]
    # Full-document IDs are PMIDs; split-set document IDs may be passage IDs
    # such as ``38173036_67``.
    document_pmid = str(document.id).strip().split("_", 1)[0]
    existing_pmid = first_infon(passage, "article-id_pmid")
    if document_pmid and existing_pmid and document_pmid != existing_pmid:
        raise ValueError(
            f"document {document.id!r} in {source} has conflicting document and passage PMID"
        )
    pmid = existing_pmid or document_pmid
    if not pmid or pmid not in articles:
        raise ValueError(f"document {document.id!r} in {source} has unknown PMID {pmid!r}")
    if not existing_pmid:
        set_infon(passage, "article-id_pmid", pmid)

    expected = articles[pmid]
    existing_pmc = first_infon(passage, "article-id_pmc")
    if expected["pmc"] is None:
        if existing_pmc is not None:
            raise ValueError(f"PMID {pmid} in {source} has PMC {existing_pmc}, but TSV has none")
    elif existing_pmc is not None and normalized_pmc(existing_pmc).casefold() != normalized_pmc(expected["pmc"]).casefold():
        raise ValueError(
            f"PMID {pmid} in {source} has PMC {existing_pmc}, expected {expected['pmc']}"
        )

    existing_doi = first_infon(passage, "article-id_doi")
    if expected["doi"] is None:
        if existing_doi is not None:
            raise ValueError(f"PMID {pmid} in {source} has DOI {existing_doi}, but TSV has none")
    elif existing_doi is None:
        set_infon(passage, "article-id_doi", expected["doi"])
    elif existing_doi.casefold() != expected["doi"].casefold():
        raise ValueError(
            f"PMID {pmid} in {source} has DOI {existing_doi}, expected {expected['doi']}"
        )

    if expected["full_text"] is not None:
        set_infon(passage, "full_text", expected["full_text"])
    if expected["license_URL"] is not None and first_infon(passage, "license_URL") is None:
        set_infon(passage, "license_URL", expected["license_URL"])

def fix_passages(pmid: str, document: Any, article_dict: dict[str, Any]) -> None:
    full_text = article_dict.get("full_text", False)
    if full_text:
        return
    ann_or_decoy_passage_ids = article_dict.get("ann_or_decoy_passage_ids", set())
    kept_passages = []
    for passage in document.passages:
        passage_id = passage.infons.get("passage_id", None)
        section_type = passage.infons.get("section_type", None)
        passage_type = passage.infons.get("type", None)
        if section_type == "TITLE" or  passage_type == "title":
            kept_passages.append(passage)
            continue
        if section_type == "ABSTRACT" or  passage_type == "abstract":
            kept_passages.append(passage)
            continue
        ann_or_decoy = passage_id in ann_or_decoy_passage_ids
        if ann_or_decoy:
            kept_passages.append(passage)
            continue
        log.info(
            "Removing passage %r from non-full-text article %s; "
            "section/passage type: %s / %s",
            passage_id,
            pmid,
            section_type,
            passage_type,
        )
    document.passages = kept_passages

def enrich(
    input_root: Path,
    output_root: Path,
    articles: dict[str, dict[str, str | None]],
) -> None:
    for input_path in sorted(input_root.rglob("*.xml")):
        print(f"Processing: {input_path}")
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        with input_path.open(encoding="utf-8") as handle:
            collection = bioc.biocxml.load(handle)
        for document in collection.documents:
            pmid = check_PMID(document, articles, input_path)
            enrich_document(pmid, document, articles, input_path)
            fix_passages(pmid, document, articles[pmid])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            bioc.biocxml.dump(collection, handle, pretty_print=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path)
    parser.add_argument(
        "--ann-or-decoy-passages",
        "--passages",
        dest="ann_or_decoy_passages",
        type=Path,
        default=None,
        help="Optional TSV file listing annotated or decoy passage IDs.",
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    articles = load_articles(args.articles)
    if args.ann_or_decoy_passages is not None:
        load_ann_or_decoy_passages(args.ann_or_decoy_passages, articles)
    enrich(args.input_root, args.output_root, articles)


if __name__ == "__main__":
    main()
