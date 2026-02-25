import gzip
from collections import Counter

import bioc


def process_collection(input_filename, abbr):
    mentions = list()
    print("Processing file " + input_filename)
    open_func = gzip.open if input_filename.endswith(".gz") else open
    with open_func(input_filename, "rt") as fp:
        collection = bioc.load(fp)
    for document in collection.documents:
        header = document.passages[0]
        docid = document.id
        pmid = header.infons.get("article-id_pmid")
        pmc = header.infons.get("article-id_pmc")
        if pmid is None:
            raise ValueError("PMID is None for document ({}, {}, {})".format(docid, pmid, pmc))
        for passage in document.passages:
            for annotation in passage.annotations:
                identifier_list = parse_identifier_list(annotation.infons.get("identifier"))
                mention_type = annotation.infons.get("type")
                mention_text = annotation.text
                expanded_text = abbr.expand(pmid, mention_text)
                mentions.append((pmid, mention_text, expanded_text, mention_type, identifier_list))
    return mentions


def parse_identifier_list(identifier_list_text):
    if identifier_list_text is None:
        return []
    return [parse_ellipsis_element(element) for element in identifier_list_text.split(";")]


def parse_ellipsis_element(element):
    if element is None:
        return (None, [])
    element = element.strip()
    if element == "None" or element == "-":
        return (None, [])
    end_paren_index = element.find(")")
    accession_index = end_paren_index + 1 if end_paren_index >= 0 else -1
    qualifier_text = element[:accession_index] if accession_index >= 0 else None
    element_identifier_text = element[accession_index:] if accession_index >= 0 else element
    element_identifiers = element_identifier_text.split(",")
    return (qualifier_text, element_identifiers)


def count_fractional_identifiers(identifier_list):
    # print(f"count_fractional_identifiers: identifier_list = {identifier_list}")
    identifier_counts = Counter()
    coordination_factor = 1.0 / len(identifier_list)
    # print(f"count_fractional_identifiers: coordination_factor = {coordination_factor}")
    for qualifier, identifier_elements in identifier_list:
        if len(identifier_elements) == 0:
            continue
        # print(f"count_fractional_identifiers: identifier_elements = {identifier_elements}")
        multi_match_factor = 1.0 / len(identifier_elements)
        # print(f"count_fractional_identifiers: multi_match_factor = {multi_match_factor}")
        for identifier in identifier_elements:
            identifier_counts[identifier] += coordination_factor * multi_match_factor
    # print(f"count_fractional_identifiers: sum(identifier_counts.values()) = {sum(identifier_counts.values())}")
    # print(f"count_fractional_identifiers: identifier_counts = {identifier_counts}")
    return identifier_counts


def count_fractional_lineage(identifier_counts, ontology):
    # print(f"count_fractional_lineage: identifier_counts = {identifier_counts}")
    lineage_counts = Counter()
    for identifier, identifier_count in identifier_counts.items():
        lineages = calculate_slim(identifier, ontology)
        # print(f"count_fractional_lineage: identifier = {identifier} lineages = {lineages}")
        lineage_count = identifier_count / len(lineages)
        # print(f"count_fractional_lineage: lineage_count = {lineage_count}")
        for lineage in lineages:
            lineage_counts[lineage] += lineage_count
    # print(f"count_fractional_lineage: sum(lineage_counts.values()) = {sum(lineage_counts.values())}")
    # print(f"count_fractional_lineage: lineage_counts = {lineage_counts}")
    return lineage_counts


def calculate_slim(term_id, ontology):
    if term_id in slim_categories:
        return slim_categories[term_id]
    if not term_id in ontology:
        return ("Unknown",)
    term_data = ontology[term_id]
    if "slim" in term_data:
        return term_data["slim"]
    slims_dict = dict()
    for parent in term_data.get("parents", []):
        for slim in calculate_slim(parent, ontology):
            if not slim in slims_dict:
                slims_dict[slim] = set()
            slims_dict[slim].add(parent)
    name = term_data["name"]
    slims = break_ties(set(slims_dict.keys()))
    if len(slims) > 1 or "Unclassified" in slims:
        for slim, parents in slims_dict.items():
            if not slim in slims:
                continue
            parents = list(parents)
            parents.sort()
            parents_text = ", ".join(parents)
            print(f"SLIM    {term_id}    {name}    {slim}    {parents_text}")
    slims = list(slims)
    slims.sort()
    slims = tuple(slims)
    term_data["slim"] = slims
    return slims


def break_ties(slims):
    if len(slims) == 1:
        return slims
    if "Unclassified" in slims:
        slims.remove("Unclassified")
    if len(slims) == 1:
        return slims
    if "Other" in slims:
        slims.remove("Other")
    if len(slims) == 1:
        return slims
    if "Stem/progenitor" in slims:
        slims.remove("Stem/progenitor")
    return slims


slim_categories = {
    "CL:0000000": ("Other",),  # Cell
    "CL:0000066": ("Epithelial",),
    "CL:0000311": ("Epithelial",),  # keratin accumulating cell
    "CL:0000677": ("Epithelial",),  # gut absorptive cell
    "CL:0000115": ("Endothelial",),
    "CL:0000423": ("Endothelial",),  # tip cell
    "CL:0008019": ("Mesenchymal/stromal",),  # Mesenchymal (fibroblasts, smooth muscle, etc.)
    "CL:0000499": ("Mesenchymal/stromal",),  # Stromal
    "CL:0000669": ("Mesenchymal/stromal",),  # pericyte
    "CL:0000186": ("Mesenchymal/stromal",),  # myofibroblast cell
    "CL:0008034": ("Mesenchymal/stromal",),  # mural cell
    "CL:0000136": ("Mesenchymal/stromal",),  # adipocyte
    "CL:0000148": ("Mesenchymal/stromal",),  # melanocyte
    "CL:0000137": ("Mesenchymal/stromal",),  # osteocyte
    "CL:0000178": ("Mesenchymal/stromal",),  # Leydig cell
    "CL:0007001": ("Mesenchymal/stromal",),  # skeletogenic cell
    "CL:0017004": ("Mesenchymal/stromal",),  # telocyte
    "CL:0007005": ("Mesenchymal/stromal",),  # notochordal cell
    "CL:0002564": ("Mesenchymal/stromal",),  # nucleus pulposus cell of intervertebral disc
    "CL:0000140": ("Mesenchymal/stromal",),  # odontocyte
    "CL:0000183": ("Mesenchymal/stromal",),  # contractile cell
    "CL:0002522": ("Mesenchymal/stromal",),  # renal filtration cell
    "CL:0000988": ("Hematopoietic",),
    "CL:0000442": ("Hematopoietic",),  # follicular dendritic cell (better than "Other")
    "CL:0000353": ("Stem/progenitor",),  # Blastoderm cell
    "CL:0000034": ("Stem/progenitor",),  # Stem cell
    "CL:0011026": ("Stem/progenitor",),  # Progenitor cell
    "CL:0011020": ("Stem/progenitor",),  # Neural progenitor cell
    "CL:0011115": ("Stem/progenitor",),  # Precursor cell
    "CL:0000222": ("Stem/progenitor",),  # Mesodermal cell
    "CL:0002321": ("Stem/progenitor",),  # embryonic cell (metazoa)
    "CL:0010017": ("Stem/progenitor",),  # zygote
    "CL:0009010": ("Stem/progenitor",),  # transit amplifying cell (better than "Other")
    "CL:0000039": ("Germ line",),
    "CL:0000187": ("Muscle",),
    "CL:0000188": ("Muscle",),  # cell of skeletal muscle
    "CL:0000540": ("Neuronal",),
    "CL:0000047": ("Neuronal",),  # neural stem cell
    "CL:0002319": ("Neuronal",),  # neural cell (better than "Other")
    "CL:0000197": ("Neuronal",),  # sensory receptor cell
    "CL:0000125": ("Glial",),
    "CL:0000349": ("Trophoblast/placental",),
    "CL:0002092": ("Other",),
    "CL:0009004": ("Other",),
    "CL:0000064": ("Other",),
    "CL:0000630": ("Other",),
    "CL:0000080": ("Other",),
    "CL:1000497": ("Other",),
    "CL:2000004": ("Other",),
    "CL:0002494": ("Other",),
    "CL:0001035": ("Other",),
    "CL:0009002": ("Other",),
    "CL:0002559": ("Other",),
    "CL:0009005": ("Other",),
    "CL:1000600": ("Other",),
    "CL:2000021": ("Other",),
    "CL:2000030": ("Other",),
    "CL:1001319": ("Other",),
    "CL:0000219": ("Other",),
    "CL:0000163": ("Other",),
    "CL:0000293": ("Other",),
    "CL:0000151": ("Other",),
    "CL:4033054": ("Other",),
    "CL:4023072": ("Other",),
    "CL:0000147": ("Other",),
    "CL:4030031": ("Other",),
    "CL:0000212": ("Other",),
    "CL:0000445": ("Other",),
    "CL:0000001": ("Other",),
    "CL:0000010": ("Other",),
    "CL:0001034": ("Other",),
    "CL:0000255": ("Other",),
    "CL:0000415": ("Other",),
    "CL:0000413": ("Other",),
    "CL:0000371": ("Other",),
    "CL:0000228": ("Other",),
    "CL:0002242": ("Other",),
    "CL:0000226": ("Other",),
    "CL:0002369": ("Other",),  # fungal spore
    "CL:0000225": ("Other",),  # anucleate cell
    "CL:0001064": ("Other",),  # malignant cell
    "CL:0001063": ("Other",),  # neoplastic cell
    "CL:0001061": ("Other",),  # abnormal cell
    "CL:0000596": ("Other",),  # sexual spore
    "CL:2000020": ("Other",),  # inner cell mass cell
    "CL:4052002": ("Other",),  # syncytial cell
    "CL:0000607": ("Other",),  # ascospore
    "CL:0002520": ("Other",),  # nephrocyte
    "CL:0000326": ("Other",),  # glycogen accumulating cell
    "CL:0000306": ("Other",),  # crystallin accumulating cell
    "CL:0000524": ("Other",),  # spheroplast
}
