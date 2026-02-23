# CellLink Passage Selection

This directory contains code for selecting a representative set of passages from a corpus of PubMed and PMC documents while ensuring topical and structural variety.
The goal is to construct a subset that:
* Reflects the statistical distribution of key corpus features.
* Preserves long-tail phenomena.
* Is substantially more useful than random sampling for training named entity recognition (NER) and entity linking (EL) systems.

Selection is formulated as an optimization problem over feature distributions.

## Algorithm Overview
Each passage is represented as a set of features. A feature consists of values and associated counts, which define a probability distribution over the corpus.

Examples:
* Journal (categorical; one-hot)
* MeSH terms (multi-label)
* Token distribution (bag-of-words)
* Passage length (numeric)
* Model-derived annotations

The full corpus defines a target distribution for each feature. The algorithm greedily selects passages to minimize the KL-divergence between the feature distributions of the selected subset and those of the full corpus.

### Distribution Adjustment
Before selection, feature distributions can be modified:
* Individual values can be multiplicatively reweighted.
* Entire distributions can be shifted toward long-tail values.

These transformations allow controlled over- or under-sampling of specific phenomena.

## Algorithm Phases
1. Download articles (BioC XML and metadata).
2. Annotate articles using external NER models (not provided).
3. Download metadata from eUtils.
4. Extract passages and primary features.
5. Derive secondary features.
6. Filter passages.
7. Adjust feature distributions.
8. Perform greedy KL-based selection.

Because KL-divergence must be recomputed after each selection, runtime is approximately quadratic in the number of passages per batch due to recomputation of KL-divergence after each selection pass. To manage this, the algorithm operates in batches:
   * The dataset is randomly partitioned into batches.
   * Selection is performed independently within each batch.
   * Selected passages are merged.
   * The process repeats until the final selection size is reached.

## Limitations
1. No explicit modeling of annotation cost or difficulty.
2. No explicit near-duplicate suppression.
   * Embedding-based filtering (e.g., MedCPT) would improve this.
   * Embedding-based clustering could further improve variety.
3. KL-based greedy optimization is not monotonic, limiting caching optimizations.

## Repeatability
We evaluated repeatability using 5,000 randomly sampled documents (103,574 passages).

Selection parameters:
* Output size: 3,000
* Batch size: 18,000

Input and output of the experiment are found in directory ```CellLink-paper/passage_selection/repeatability```.

Across three runs:
* Pairwise overlap: 2513 ± 1 passages
* Triple overlap: 2320 passages

Under random sampling, the probability of ≥2513 overlap between two samples is < 10^-3983 (hypergeometric test). This demonstrates strong non-random structure in the selection. However, 2320 < 3000, confirming that the algorithm is not fully deterministic.

## Performance
In the repeatability experiment, selection completed in 8 hours 2 minutes ± 35 minutes on a single CPU core, using under 20 GB of memory. 
At that rate, processing the entire set in one step would take about 5 days. Realistically, you may want to separate the data into subsets and process them separately.

## Usage

Commands and scripts were tested with Python 3.11.3 and are designed to be run from directory ```CellLink-paper/passage_selection```.

1. Install requirements:
```
pip install -r requirements.txt
```

The provided requirements.txt includes exact packages versions for repeatability. This code should also run with the following minimal installation:
```
pip install bioc spacy unidecode scipy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```
2. Download corpus:
```
./scripts/update_download.sh
```
Input document IDs are found in ```data/docids.tsv```.

3. Run external NER models (not included).

Place outputs in:
   - ```data/BioCXML_AnatEM```
   - ```data/BioCXML_BioID```
   - ```data/BioCXML_CRAFT```
   - ```data/BioCXML_JNLPBA```

A sample script is provided at ```./scripts/run_NER.sh```

4. Run feature extraction:
```
./scripts/update_pipeline.sh
```

The feature configuration is found at ```data/config.json```

5. Run selection:
```
./scripts/run_selection.sh
```
Adjust ```SELECTION_COUNT``` and ```BATCH_SIZE``` as needed. Output will be a list of the selected passage IDs, one per line, in a single file (```CellLink-paper/passage_selection/selected_passageids.txt```).

## Configuration
Features are configured via the file data/config.json. The file consists of a dictionary where keys represent the feature names and the values are dictionaries containing configuration values.

The configuration items for each feature are as follows:
* *data_type*: required for all features. Supported values are: {number, count_dict, singleton, raw}.
   * "number" data types are numeric
   * "count_dict" data types are dictionaries with strings as keys and integer counts as values.
   * "singleton" data types are a single string, equivalent to a "count_dict" with a single key with a count of 1.
   * "raw" data types allow data to be added to the passage and extracted during transformation via custom code; they are otherwise ignored (e.g., during selection).
* *summary_calculator*: optional, specifies the process used to summarize the distribution of this feature.
* *rank*: whether this feature is associated with documents or passages. Required if bioc_extractor is given.
* *bioc_extractor*: optional, specifies the code to extract this feature from the BioC XML. If provided, "rank" must also be provided.
* *data_transform1*: required, defines a list of transformations to apply to this feature. If no transformation is desired, use "data_transforms.make_copy".
* *data_transform2*: required, defines a list of transformations to apply to this feature after data_transform1 is applied. If no transformation is desired, use "data_transforms.make_copy".
* *adjusters*: optional, specifies what process to use to adjust the distribution for this feature.
* *scorer*: optional, specifies what process to use for scoring this feature during passage selection.

## Citation
If you use this code or the CellLink corpus, please cite the following paper:
> Rotenberg N, Leaman R, Islamaj R, Kuivaniemi H, Tromp G, Fluharty B, Richardson S, Eastwood C, Diller M, Xu B, Pankajam A, Osumi-Sutherland D, Lu Z, & Scheuermann R. Cell phenotypes in the biomedical literature: a systematic analysis and text mining corpus. bioRxiv. doi:10.64898/2026.02.11.705457.
