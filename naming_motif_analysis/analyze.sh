set -e 

CORPUS_DIR="***ADD THIS***"

mkdir -p output

python -u src/analyze_name_patterns.py \
	--input ${CORPUS_DIR} \
	--abbr "data/CellLink_abbreviations.tsv" \
	--cl "../Cell-Ontology_v2025-01-08.json" \
	--name-motif-map "data/name_motif_map.json" \
	--term-cache "data/term_cache.json" \
	--vector-cache "data/vector_cache.npy" \
	--lineage-analysis "output/result_lineage_analysis.tsv" \
	--motif-analysis "output/result_motif_analysis.tsv" \
	--cv-results "output/cv_results.tsv" \
	--cv-iter-count 5 \
	--cv-fold-count 10 \
2>&1 | tee analysis.log
