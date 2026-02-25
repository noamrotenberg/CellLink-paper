# CellLink Naming Motif Analysis

This directory contains code for analyzing CellLink annotations to characterize how authors name cell populations in the biomedical literature. The goal is to identify recurring naming motifs - lexical components such as anatomical context, molecular signature, or developmental stage - and to quantify how these motifs are combined across biological lineages.

### Scientific Rationale

To characterize how authors refer to cell populations, we analyzed the CellLink annotations to identify frequently used naming motifs - such as anatomical context (e.g., peripheral), molecular signature (e.g., Foxp3+), developmental stage (e.g., immature), or functional role (e.g., suppressor) - using a combination of manual curation and automated labeling (see Methods: Naming motif analysis in the manuscript).

Because cell population names are hierarchical and compositional, this analysis focuses on how authors combine motifs to distinguish phenotypes. Quantifying motif prevalence and co-occurrence provides insight into naming conventions, lexical heterogeneity, and lineage-specific descriptive strategies. These patterns inform ontology extension, entity recognition, and downstream biomedical knowledge extraction.

## Overview of the Approach
The analysis proceeds in two stages:
1. Lineage assignment
2. Motif identification and labeling

Annotations linked to zero or multiple Cell Ontology identifiers are excluded.

## Lineage Assignment
We define 11 high-level lineages:
- epithelial
- endothelial
- mesenchymal/stromal
- muscle
- hematopoietic
- neuronal
- glial
- stem/progenitor
- germ line
- trophoblast/placental
- other

Lineages are inferred from the Cell Ontology hierarchy.
- High-level cell types are manually mapped to a lineage (see ```src/cell_analysis_utils.py```).
- Unmapped cell types inherit lineage from parent types.
- If multiple parent lineages are identified, lineage assignment is fractional with equal weighting across parents.
   - Example: if two parent lineages are identified, each receives weight 0.5.

Fractional lineage weights are used when computing aggregate motif statistics.

## Naming Motif Taxonomy

A motif is defined as a contiguous token substring (length 1-3 tokens) of a cell population name that captures a recurring semantic component.

We define 14 motif categories:

| Motif | Definition | Examples |
|-------|------------|----------|
| Root | Fundamental identity nouns corresponding to canonical cell classes, serving as the primary identity term in the name | neuron, macrophage, fibroblast |
| Anatomical context | Terms that localize the cell within the body, such as tissues, organs, regions or directional descriptors. | vascular, thymic, anterior, retinal |
| Lineage | Terms characterizing developmental origin | epithelial, mesenchymal, hematopoietic |
| Molecular signature | Terms indicating identifying gene/transcript markers or protein expression patterns | CD8+, SOX2+, double negative |
| Appearance | Descriptors of visually observable traits such as morphology, structural features, or characteristic staining. | pyramidal, ciliated, acidophilic |
| Functional role | Terms describing biological function. | natural killer, suppressor, excitatory, secretory |
| Developmental | Terms indicating the position along a differentiation trajectory, such as maturation stage or lineage commitment. | embryonic, immature, multipotent, terminally differentiated |
| State | Descriptors of dynamic, reversible, or transient physiological conditions. | activated, circulating, exhausted |
| Variant | Labels denoting subtypes or alternative forms within a broader category according to an established classification scheme. | type 1, conventional, non-classical |
| Molecular signaling | Terms describing chemicals used for intercellular communication, including neurotransmitters, hormones, or cytokines. | GABAergic, adrenergic, androgen secreting, calcitonin secreting, histaminergic, interferon-producing |
| Disease | Terms derived from pathological conditions. | tumor-associated, leukemia, rheumatoid, neoplastic |
| Eponym | Names derived from an individual historically associated with the cell type. | Schwann, Purkinje |
| Stimulus | Terms characterizing responsiveness to external physical or chemical stimuli. | photosensitive, NO-sensitive, cold-sensing |
| Species/Sex | Terms indicating origin by organism or biological sex. These terms were not common. | human, mouse, female, male |

## Motif Labeling Methodology

### Preprocessing
- Abbreviations are expanded using ```data/CellLink_abbreviations.tsv``` (extracted via Ab3P).
- Names are tokenized and lemmatized using the SciSpaCy tokenizer (```en_core_sci_sm```).
   - Hyphenated tokens are not split.
   - All tokens except stopwords are eligible for motif labeling.

### Candidate Phrase Extraction
For each name, all contiguous token sequences of length 1-3 are extracted, excluding spans that begin or end with a stopword.

## Manual + Automated Hybrid Labeling
### Step 1 - Manual Motif Identification
Manually labeled motif spans from ```data/name_motif_map.json``` are applied first using longest non-overlapping match.

Constraints:
- Motifs are assigned as non-overlapping spans.
- Each token may receive at most one motif label.
- Motif labels are mutually exclusive and non-hierarchical by design.
   - For example, ```CD8+``` is labeled as a molecular signature motif only.

### Step 2 - Automated Classification
Remaining candidate phrases are labeled using a multinomial logistic regression classifier trained on the manually labeled examples.
- Embeddings are generated using SapBERT (```SapBERT-from-PubMedBERT-fulltext```).
- Each candidate phrase is embedded independently.
- The classifier is implemented in scikit-learn 1.8.0:
   - Solver: ```lbfgs```
   - Penalty: L2
   - Multinomial loss

#### Margin Computation
Candidates are added in order of decreasing margin: the difference between the highest probability predicted for any label and the second-highest probability. The default min_margin is 0.0, so margin is used only for ordering, not thresholding.

## Cross-Validation
Cross-validation is optional and controlled by CLI parameters.

When enabled:
- Entire labeled motif dataset is partitioned into k folds (default k=10).
- k-1 folds are used for training, 1 for testing.
- Accuracy is computed over unweighted motif instances in the test fold.
- Reported accuracy reflects phrase-level classification accuracy.

In the manuscript, we report:
0.85 ± 0.03 accuracy (5 iterations of 10-fold cross-validation)

Note:
- No explicit random seed is fixed, though this only affects the CV fold assignment.
- Minor fold variation may occur across runs but does not materially affect aggregate statistics.
- Our analysis included the test set, which is not provided.

## Hardware Requirements
- GPU is not required but is used to compute embeddings if available. Embeddings are computed once and cached to disk.
- Runs on 16 GB RAM machines.
- Typical runtime without cross-validation under 10 minutes. With cross-validation, should be under 20 minutes.

## Input Data Requirements
This analysis requires:
- CellLink corpus, in BioC XML format.
   - The training and validation sets are distributed separately.
   - NOTE: We do *not* distribute the test set.
- Several files provided in the repository:
   - Name motif mapping file
   - Cell Ontology json file
   - Abbreviation mapping file

## Usage

Commands and scripts were tested with Python 3.11.3 and are designed to be run from directory ```CellLink-paper/naming_motif_analysis```.

1. Install requirements:
```
pip install -r requirements.txt
```
2. Run the analysis:
```
python main.py \
  --input <input_path> \
  --abbr <abbreviation_file> \
  --cl ../Cell-Ontology_v2025-01-08.json \
  --name-motif-map data/name_motif_map.json \
  --term-cache data/term_cache.json \
  --vector-cache data/vector_cache.npy \
  --lineage-analysis output/lineage.tsv \
  --motif-analysis output/motif.tsv
```

A pre-prepared script is provided: ```./analysis.sh```.

Optional arguments: 
| Argument | Description |
|----------|-------------|
| ```--examples``` | Output filename for example motifs |
| ```--example-count``` | Number of examples (default=100) |
| ```--cv-results``` | Output filename for cross-validation results. If omitted, cross-validation is skipped. |
| ```--cv-iter-count``` | Number of CV iterations (default=1) |
| ```--cv-fold-count``` | Number of CV folds (default=10) |

### Output
Output files are written to the specified filenames. Example outputs are provided in the ```output/``` directory.

## Citation
If you use this code or the CellLink corpus, please cite the following paper:
> Rotenberg N, Leaman R, Islamaj R, Kuivaniemi H, Tromp G, Fluharty B, Richardson S, Eastwood C, Diller M, Xu B, Pankajam A, Osumi-Sutherland D, Lu Z, & Scheuermann R. Cell phenotypes in the biomedical literature: a systematic analysis and text mining corpus. bioRxiv. doi:10.64898/2026.02.11.705457.

