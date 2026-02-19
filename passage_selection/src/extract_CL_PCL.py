import sys
import codecs
import datetime

entities_dict = dict()

id2parents = dict()
id2parents["CL:4042013"] = {"CL:4023011"}
id2keep = dict()
id2keep["CL:0000000"] = True # Cell (ie root)
id2keep["BFO:0000002"] = True
id2keep["CL:0000163"] = True # Breaks loop in is_a hierarchy
id2keep["UBERON:0004121"] = True
id2keep["UBERON:0000122"] = True
id2keep["GO:0005737"] = False # seems to be cytoplasm
id2keep["GO:0005634"] = False # Seems to be nucleus
id2keep["GO:0000792"] = False # Seems to be nucleus
id2keep["SO:0000704"] = False # Gene
id2keep["SO:0001260"] = False # Sequence collection; seems to define a cell type, but the names do not seem to be useful

sensu_dict = dict()
sensu_dict["(sensu Animalia)"] = True
sensu_dict["(sensu Arthopoda)"] = False
sensu_dict["(sensu Arthropoda)"] = False
sensu_dict["(sensu Endopterygota)"] = False
sensu_dict["(sensu Fungi)"] = False
sensu_dict["(sensu Mus)"] = True
sensu_dict["(sensu Nematoda and Protostomia)"] = False
sensu_dict["(sensu Nematoda)"] = False
sensu_dict["(sensu Teleostei)"] = False
sensu_dict["(sensu Vertebrata)"] = True

messages = set()

def log_message(msg):
	if msg in messages:
		return
	print(msg)
	messages.add(msg)

def main():
	cl_ontology_filename = sys.argv[1]
	pcl_ontology_filename = sys.argv[2]
	output_filename = sys.argv[3]
	load_obo(cl_ontology_filename)
	print("Loaded " + str(len(entities_dict)) + " entities from CL ontology")
	load_obo(pcl_ontology_filename)
	print("Loaded " + str(len(entities_dict)) + " entities from PCL ontology")
	
	print("DEBUG CL:0000451 = {}".format(entities_dict.get("CL:0000451")))
	
	name_type_counts = dict()
	entities_dict2 = dict()
	for id, entity in entities_dict.items():
		k = keep(id, set())
		names2 = set()
		for name, type in entity.get("names", set()):
			if not type in name_type_counts:
				name_type_counts[type] = 0
			name_type_counts[type] += 1
			if type != "BROAD" and type != "NARROW" and type != "RELATED":
				names2.add(name)
			if type == "}":
				log_message("WARN: name {} type {}".format(name, type))
		entity["names"] = names2
		if k is None:
			log_message("WARN No keep value for ID " + id)
		elif k:
			entities_dict2[id] = entity
	print("Kept " + str(len(entities_dict2)) + " entities")
	for type, count in name_type_counts.items():
		print("name type {}, count {}".format(type, count))

	print("Writing out dictionary, time = " + str(datetime.datetime.now()))
	#entities.write(entities_dict2, output_filename)
	print("Writing out dictionary, time = " + str(datetime.datetime.now()))
	with open(output_filename, "w") as entites_file:
		for id, entity in entities_dict2.items():
			if not "names" in entity:
				continue
			for name in entity["names"]:
				entites_file.write("{}\t{}\n".format(name, id))
	print("Done, time = " + str(datetime.datetime.now()))

def create(id, types, names, parents, xrefs):
	entity = dict()
	entity["id"] = id
	entity["types"] = types
	entity["names"] = names
	entity["parents"] = parents
	entity["xrefs"] = xrefs
	return entity


def keep(id, history):
	if id in history:
		log_message("WARN ID " + id + " has a cycle: " + str(history))
		return None
	if id in id2keep:
		return id2keep[id]
	if not id in id2parents:
		log_message("WARN ID " + id + " does not have parents listed")
		return None
	parents = id2parents[id]
	keep_values = set()
	for parent in parents:
		history2 = set()
		history2.add(id)
		history2.update(history)
		keep2 = keep(parent, history2)
		keep_values.add(keep2)
	if len(keep_values) == 0:
		log_message("WARN ID " + id + " returned no keep values: " + str(keep_values))
		return None
	if len(keep_values) > 1:
		log_message("WARN ID " + id + " returned multiple keep values: " + str(keep_values))
		return None
	keep_values = list(keep_values)
	return keep_values[0]

def load_obo(filename):
	print("Loading file " + filename)
	with codecs.open(filename, 'r', encoding="utf-8") as f:
		state = 0 #Ignore
		id = ""
		names = set()
		parents = set()
		xrefs = set()
		for line in f:
			line = line.strip()
			# print(line)
			if line == "[Term]":
				# Output record
				if len(id) > 0:
					entities_dict[id] = create(id, set(), names, parents, xrefs)
				# print("New record")
				state = 1 #Term
				id = ""
				names = set()
				parents = set()
				xrefs = set()
			elif line == "[Typedef]":
				state = 0 #Ignore
				id = ""
				names = set()
				parents = set()
				xrefs = set()
			elif state == 1:
				if line.startswith("id: "):
					id = line[4:]
					# print("Found id \"" + id + "\"")
				elif line.startswith("name: "):
					name = line[6:]
					# print("Found name \"" + name + "\"")
					#sensu_index = name.find("(sensu")
					#if sensu_index > 0:
					#	name = name[:sensu_index].strip()
					names.add((name, "NAME"))
				elif line.startswith("alt_id: "):
					alt_id = line[8:]
					# print("Found alternate id \"" + alt_id + "\"")
					xrefs.add(alt_id)
				elif line.startswith("synonym: "):
					index1 = line.find("\"")
					index2 = line.find("\"", index1 + 1)
					name = line[index1 + 1:index2]
					#sensu_index = name.find("(sensu")
					#if sensu_index > 0:
					#	name = name[:sensu_index].strip()
					remainder_fields = line[index2 + 1:].strip().split(" ")
					names.add((name, remainder_fields[0]))
				elif line.startswith("xref: "):
					fields = line.split(" ");
					#print("Found xref \"" + fields[1] + "\"")
					xref = fields[1]
					resource = xref.split(":")[0]					
					accession = xref[len(resource) + 1:]
					if resource in resource_map:
						resource = resource_map[resource]
					else:
						log_message("WARN resource \"" + resource + "\" is unknown") 
						resource = None
					if not resource is None:
						xrefs.add(resource + ":" + accession)
				elif line.startswith("is_a: "):
					fields = line.split(" ");
					#print("Found is_a \"" + fields[1] + "\"")
					parent = fields[1]
					# Ignore is-a relationships outside of the cell ontology
					# e.g. "PR:000050567 ! protein-containing material entity"
					if parent.startswith("CL:"):
						parents.add(parent)
						if not id in id2parents:
							id2parents[id] = set()
						id2parents[id].add(parent)
				elif line == "is_obsolete: true":
					state = 0 #Ignore
					id = ""
					names = set()
					parents = set()
					xrefs = set()
		# Output record
		if len(id) > 0:
			entities_dict[id] = create(id, set(), names, parents, xrefs)

resource_map = dict()
resource_map["FMA"] = "FMA" 
resource_map["ZFA"] = "ZFA" 
resource_map["BTO"] = "BTO" 
resource_map["CALOHA"] = "CALOHA" 
resource_map["KUPO"] = "KUPO" 
resource_map["MESH"] = "MESH" 
resource_map["FBbt"] = "FBbt" 
resource_map["ILX"] = "ILX" 
resource_map["WBbt"] = "WBbt" 
resource_map["FAO"] = "FAO" 
resource_map["VHOG"] = "VHOG" 
resource_map["BAMS"] = "BAMS" 

if __name__ == '__main__':
    main()
