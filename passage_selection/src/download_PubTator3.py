import sys
import random

import BioCXMLUtils

base_urls = {False: "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids={}", True: "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids={}&full=true"}
batch_size = 100
collection_size = 100
wait_seconds = 2.0 / 3.0

def retrieve(docids, ft_avail, output_directory, last_request = None):
	retriever = BioCXMLUtils.BioCXMLRetriever([pmid for pmid, pmc in docids], base_urls[ft_avail], batch_size, wait_seconds, last_request)
	standardizer = BioCXMLUtils.DocIDStandardizer(docids)
	writer = BioCXMLUtils.BioCXMLWriter(output_directory, collection_size)
	document = retriever.next_document()
	downloaded = set()
	while not document is None:
		docid = document.id
		header = document.passages[0]
		pmid = header.infons.get("article-id_pmid")
		pmc = header.infons.get("article-id_pmc")
		pmid2, pmc2 = standardizer.standardize(docid, pmid, pmc)
		if not (pmid2, pmc2) in docids:
			raise ValueError("Retrieved unrequested document: ({}, {}, {}) -> ({}, {})".format(docid, pmid, pmc, pmid2, pmc2))
		#print("Adding document ({}, {}, {}) -> ({}, {})".format(docid, pmid, pmc, pmid2, pmc2))
		pmid = pmid2
		pmc = pmc2
		downloaded.add((pmid, pmc))
		# Update the IDs:
		if ft_avail and "article-id_pmid" in header.infons:
			document.id = pmc[3:]
			header.infons["article-id_pmid"] = pmid
			header.infons["article-id_pmc"] = pmc
		else:
			document.id = pmid
			if pmc is None:
				header.infons.pop("article-id_pmc", None)
			else:
				header.infons["article-id_pmc"] = pmc
		writer.process(document)
		document = retriever.next_document()
	writer.flush()
	print("Number of downloaded documents is {}".format(len(downloaded)))
	print("Number of records not returned is {}".format(len(docids) - len(downloaded)))	
	return retriever.last_request, downloaded

if __name__ == "__main__":
	docids_filename = sys.argv[1]
	output_directory = sys.argv[2]
	
	docids = BioCXMLUtils.read_docids(docids_filename)
	print("Loaded {} docids requested".format(len(docids)))
	
	# Separate full text / TIAB
	needed_docids_TA = [(pmid, pmc) for pmid, pmc, ft_avail in docids if ft_avail == False]
	needed_docids_FT = [(pmid, pmc) for pmid, pmc, ft_avail in docids if ft_avail == True]
	print("Number of needed TIAB documents is {}".format(len(needed_docids_TA)))
	print("Number of needed full text documents is {}".format(len(needed_docids_FT)))
	random.shuffle(needed_docids_TA)
	random.shuffle(needed_docids_FT)
	
	last_request = None
	if len(needed_docids_TA) > 0:
		last_request, downloaded = retrieve(needed_docids_TA, False, output_directory)
	if len(needed_docids_FT) > 0:
		retrieve(needed_docids_FT, True, output_directory, last_request)
	print("Done.")
