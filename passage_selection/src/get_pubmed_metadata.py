import datetime
import gzip
import json
import random
import sys
import time

import urllib.request
import xml.etree.ElementTree as ET

import file_utils

chunk_size = 250
wait_seconds = 0.4 # EUtils allows 3 requests per second; we wait slightly longer than 1/3 second
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="

def process_document_element(document_element):
	pub_dict = dict()
	# PMID
	result = document_element.find("./MedlineCitation/PMID")
	pmid = result.text if not result is None else None
	pub_dict["pmid"] = pmid
	# PMC
	result = document_element.find("./PubmedData/ArticleIdList/ArticleId/[@IdType='pmc']")
	pmc = None
	if not result is None:
		pmc = result.text
		if not pmc.startswith("PMC"):
			pmc = "PMC" + pmc
	pub_dict["pmc"] = pmc
	pub_dict["data"] = dict()
	# Journal abbreviation
	result = document_element.find("./MedlineCitation/Article/Journal/ISOAbbreviation")
	if result is None:
		result = document_element.find("./MedlineCitation/Article/Journal/Title")
	pub_dict["data"]["JOURNAL_NAME"] = result.text if not result is None else None
	# Publication year
	result = document_element.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
	if result is None:
		result = document_element.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
	pub_dict["data"]["PUBLICATION_YEAR"] = result.text[:4] if not result is None else None
	# MeSH headings
	pub_dict["data"]["MESH_TERMS"] = dict()
	result = document_element.findall("./MedlineCitation/MeshHeadingList/MeshHeading")
	for term in result:
		# Each heading has one descriptor and zero or more qualifiers
		descriptor = term.find("DescriptorName")
		pub_dict["data"]["MESH_TERMS"][descriptor.text] = 1
	result = document_element.findall("./MedlineCitation/MeshHeadingList/SupplMeshList")
	for term in result:
		pub_dict["data"]["MESH_TERMS"][term.text] = 1
	return pub_dict

if __name__ == "__main__":
	pmids = file_utils.read_pmids(sys.argv[1])
	print("Requested " + str(len(pmids)) + " pmids")
	pmid_list = list(pmids)
	output_filename = sys.argv[2]

	# This version updates all pmids
	random.shuffle(pmid_list)
	pmid_chunks = [pmid_list[i:i + chunk_size] for i in range(0, len(pmid_list), chunk_size)] 
	
	if output_filename.endswith(".gz"):
		output_file = gzip.open(output_filename, "wt") 
	else:
		output_file = open(output_filename, "w") 
	
	last_request = datetime.datetime.now() - datetime.timedelta(seconds = wait_seconds)
	for index, chunk in enumerate(pmid_chunks):
		pmid_list_str = ",".join(chunk)
		url = base_url + pmid_list_str
		diff = (datetime.datetime.now() - last_request).total_seconds()
		if diff < wait_seconds:
			sleep_time = wait_seconds - diff
			print("Sleeping for {}".format(sleep_time))
			time.sleep(sleep_time)
		with urllib.request.urlopen(url) as response:
			xml = response.read()
		last_request = datetime.datetime.now()
		root = ET.fromstring(xml)
		for document_element in root.findall("./PubmedArticle"):
			pub_dict = process_document_element(document_element)
			output_file.write(json.dumps(pub_dict) + "\n")
		print("Request {} / {}".format(index, len(pmid_chunks)))

	output_file.close()

	print("Done.")
