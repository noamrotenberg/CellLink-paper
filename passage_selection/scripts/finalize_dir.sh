set -e

date

INPUT_XML=$(realpath ${1})
INPUT_JSON=$(realpath ${2})
OUTPUT_JSON=$(realpath ${3})
OUTPUT_XML=$(realpath ${4})

mkdir -p ${OUTPUT_XML}

for input_json_filename in "${INPUT_JSON}"/*.json; do
	echo "input_json_filename=${input_json_filename}"
	json_name=$(basename ${input_json_filename})
	xml_name=$(echo ${json_name} | sed "s/\.json/.xml/g")
	input_xml_filename="${INPUT_XML}/${xml_name}"
	echo "input_xml_filename=${input_xml_filename}"
	output_json_filename="${OUTPUT_JSON}/${json_name}"
	echo "output_json_filename=${output_json_filename}"
	output_xml_filename="${OUTPUT_XML}/${xml_name}"
	echo "output_xml_filename=${output_xml_filename}"
	python -u src/convert_hfjson_to_bioc.py ${input_xml_filename} ${input_json_filename} ${output_json_filename} ${output_xml_filename}
done

date
