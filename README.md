This GitHub repository contains code created to analyze the National Library of Medicine CellLink corpus, as reported in the [manuscript cited below](https://doi.org/10.64898/2026.02.11.705457). The analysis includes computations of corpus statistics, inter-annotator agreement (IAA), comparisons with other corpora, named entity recognition (NER) of cell populations, and entity linking (EL) of cell populations.


## Repository Structure
This repository includes the following Python scripts, bash files, and other useful files. The cited figures and tables correspond to the manuscript cited below.
- paper_basic_stats.py – Computes basic statistics about the corpus (Tables 2, 4, 5, and 6; Figure 3)
- comparing_other_corpora_stats.py – Computes statistics comparing CellLink with other corpora that contain cell types (Table 7)
- NER-exp1-corpusComparisons/run_corpusComparisons.sh – Performs and analyzes NER experiments comparing models trained on CellLink or on other corpora that contain cell types (Table 9)
- NER-exp2-multipleModels/run_NER-exp2_BERT.sh and NER-exp2-multipleModels/run_NER-exp2_GPT.sh – Performs and analyzes NER experiments comparing different types of models trained on CellLink (Table 8)
- EL/run_EL.sh – Performs entity linking experiments (Tables 10 and 11, Figure 8)
- EL/EL_eval.py - Analyzes entity linking experiments (Tables 10 and 11, Figure 8)
- inter-annotator_agreement_evaluation.py – Computes inter-annotator agreement (Table 6)
- general_scripts/ – Shared utilities and helper scripts
- format_validation/ – Performs CellLink data format validation
- environments/ – Python environment specifications for BERT and GPT experiments
- Cell-Ontology_v2025-01-08.json – Cell Ontology version used for CellLink
- revision/ - Python code for the paper revision: added DOI and license to BioC XML, added bootstrap sampling to benchmark experiments.

## Data Availability
The exact versions of the training and validation sets of the NLM CellLink corpus can be found here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18090009.svg)](https://doi.org/10.5281/zenodo.18090009). The test set is currently not released in full; further information can be found in the latest version of the same [Zenodo data repository](https://doi.org/10.5281/zenodo.18090008).


## Citation
If you use this code or the CellLink corpus, please cite the following paper:
Rotenberg N, Leaman R, Islamaj R, Kuivaniemi H, Tromp G, Fluharty B, Richardson S, Eastwood C, Diller M, Xu B, Pankajam A, Osumi-Sutherland D, Lu Z, & Scheuermann, R. H. Cell phenotypes in the biomedical literature: a systematic analysis and text mining corpus. bioRxiv. [doi:10.64898/2026.02.11.705457](https://doi.org/10.64898/2026.02.11.705457).

References for other corpora containing cell types:
- AnatEM (Anatomical Entity Mention) corpus: Pyysalo S, Ananiadou S. Anatomical entity mention recognition at literature scale. Bioinformatics 30, 868-875 (2014).
- BioID (BioCreative VI BioID) corpus: Arighi C, et al. Bio-ID track overview. Proc. BioCreative Workshop 482, 376 (2017).
- CRAFT (Colorado Richly Annotated Full Text) corpus: Verspoor K, et al. A corpus of full-text journal articles is a robust evaluation tool for revealing differences in performance of biomedical natural language processing tools. BMC Bioinformatics 13, 207 (2012).
- JNLPBA (2004 Joint Workshop on Natural Language Processing in Biomedicine and its Applications) corpus: Collier N, Ohta T, Tsuruoka Y, Tateisi Y, Kim J-D. Introduction to the Bio-entity Recognition Task at JNLPBA. International Joint Workshop on Natural Language Processing in Biomedicine and its Applications (NLPBA/BioNLP), 73-78 (2004).

