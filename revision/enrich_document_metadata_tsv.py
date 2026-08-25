"""Add DOI and PMC license metadata to a CellLink passage-key TSV file.

The script uses PubMed EFetch to obtain DOI identifiers and PMC EFetch to
    obtain the license from the full-text XML. Results are written as TSV,
    preserving every input row and inserting DOI and license columns.
"""

from __future__ import annotations

import argparse
import csv
from http.client import IncompleteRead
import logging
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
LOG = logging.getLogger(__name__)


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_xml(
    db: str,
    ids: list[str],
    *,
    email: str | None,
    tool: str,
    api_key: str | None,
    retries: int,
    timeout: float,
) -> ET.Element:
    params: dict[str, str] = {"db": db, "id": ",".join(ids), "retmode": "xml", "tool": tool}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    request = Request(f"{EUTILS_URL}?{urlencode(params)}", headers={"User-Agent": tool})

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return ET.fromstring(response.read())
        except (HTTPError, URLError, TimeoutError, IncompleteRead, ET.ParseError) as exc:
            if attempt == retries:
                raise RuntimeError(f"eUtils request failed for {db} IDs {ids[:3]}...") from exc
            wait = 2**attempt
            LOG.warning("eUtils request failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def get_dois(
    pmids: list[str], *, batch_size: int, delay: float, email: str | None, tool: str,
    api_key: str | None, retries: int, timeout: float,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for batch_number, batch in enumerate(chunks(pmids, batch_size)):
        root = fetch_xml("pubmed", batch, email=email, tool=tool, api_key=api_key,
                         retries=retries, timeout=timeout)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID")
            if not pmid:
                continue
            for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if article_id.attrib.get("IdType", "").lower() == "doi" and article_id.text:
                    result[pmid] = article_id.text.strip()
                    break
        LOG.info("DOI lookup: batch %d (%d/%d PMIDs)", batch_number + 1,
                 min((batch_number + 1) * batch_size, len(pmids)), len(pmids))
        if delay and batch_number + 1 < (len(pmids) + batch_size - 1) // batch_size:
            time.sleep(delay)
    return result


def license_value(article: ET.Element) -> str | None:
    """Return the first value found by the requested license XPath sequence."""
    namespaces = {
        "ali": "http://www.niso.org/schemas/ali/1.0/",
        "xlink": "http://www.w3.org/1999/xlink",
    }
    paths = (
        (".//front/article-meta/permissions/license/{" + namespaces["ali"] + "}license_ref", "text"),
        (".//front/article-meta/permissions/license/@xlink:href", "attribute"),
    )
    # Additional path, aparrently not needed:
    #    (".//front/article-meta/permissions/license/license-p/ext-link/@xlink:href", "attribute"),
    for path, value_type in paths:
        if value_type == "text":
            element = article.find(path, namespaces)
            value = None if element is None else element.text
        else:
            element_path, attribute = path.rsplit("/@", 1)
            element = article.find(element_path, namespaces)
            attribute_namespace, attribute_name = attribute.split(":", 1)
            value = None if element is None else element.attrib.get(
                "{" + namespaces[attribute_namespace] + "}" + attribute_name
            )
        if value and value.strip():
            return value.strip()
    return None


def get_licenses(
    pmcids: list[str], *, batch_size: int, delay: float, email: str | None, tool: str,
    api_key: str | None, retries: int, timeout: float,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for batch_number, batch in enumerate(chunks(pmcids, batch_size)):
        root = fetch_xml("pmc", batch, email=email, tool=tool, api_key=api_key,
                         retries=retries, timeout=timeout)
        # PMC returns one article per requested PMCID; use the article's own
        # article-id rather than relying on response order.
        for article in root.iter():
            if article.tag.rsplit("}", 1)[-1] != "article":
                continue
            pmcid = next((node.text.strip() for node in article.iter()
                          if node.tag.rsplit("}", 1)[-1] == "article-id"
                          and node.attrib.get("pub-id-type") in {"pmc", "pmcid"}
                          and node.text), None)
            value = license_value(article)
            if pmcid and value:
                result[pmcid] = value
        LOG.info("license lookup: batch %d (%d/%d PMCIDs)", batch_number + 1,
                 min((batch_number + 1) * batch_size, len(pmcids)), len(pmcids))
        if delay and batch_number + 1 < (len(pmcids) + batch_size - 1) // batch_size:
            time.sleep(delay)
    return result


def enrich(input_path: Path, output_path: Path, *, batch_size: int, delay: float,
           email: str | None, tool: str, api_key: str | None, retries: int,
           timeout: float, limit: int | None = None) -> None:
    with input_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if any(len(row) != 5 for row in rows):
        bad_row = next(index + 1 for index, row in enumerate(rows) if len(row) != 5)
        raise ValueError(f"expected 5 tab-separated columns; invalid row {bad_row}")
    if limit is not None:
        rows = rows[:limit]
    pmids = list(dict.fromkeys(row[0] for row in rows if row[0]))
    pmcids = list(dict.fromkeys(row[1] for row in rows if row[1] and row[2] == "True"))
    dois = get_dois(pmids, batch_size=batch_size, delay=delay, email=email, tool=tool,
                    api_key=api_key, retries=retries, timeout=timeout)
    licenses = get_licenses(pmcids, batch_size=batch_size, delay=delay, email=email,
                            tool=tool, api_key=api_key, retries=retries, timeout=timeout)
    pmids_output = set()
    enriched_rows = [["PMID", "PMC", "DOI", "Full text", "License"]]
    for row in rows:
        pmid, pmcid, full_text, _split, _passage_id = row
        if pmid in pmids_output:
            continue
        pmids_output.add(pmid)
        enriched_rows.append([
            pmid,
            pmcid,
            dois.get(pmid, ""),
            full_text,
            licenses.get(pmcid, "") if full_text == "True" and pmcid else "",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(enriched_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input docid_passage_key.tsv")
    parser.add_argument("output", type=Path, help="Output enriched TSV file")
    parser.add_argument("--email", help="Email address for NCBI requests")
    parser.add_argument("--tool", default="celllink-metadata-enricher")
    parser.add_argument("--api-key", help="NCBI API key (raises the request rate limit)")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.34,
                        help="Seconds between batches (default: 0.34)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--limit", type=int, help="Process only the first N records (for testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.retries < 0 or args.delay < 0:
        parser.error("batch-size must be positive; retries and delay cannot be negative")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    enrich(args.input, args.output, batch_size=args.batch_size, delay=args.delay,
           email=args.email, tool=args.tool, api_key=args.api_key, retries=args.retries,
           timeout=args.timeout, limit=args.limit)


if __name__ == "__main__":
    main()
