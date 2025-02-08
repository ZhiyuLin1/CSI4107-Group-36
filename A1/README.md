# CSI4107 Assignment 1 

## Group number: 36

### Members:  Hengjing Zhang    300288003 	

### 		      Tom  Cui      300345709	

### 		     Zhiyu Lin      300255509

______

## work distribution

Henjing Zhang	-Preprocessing

Tom Cui	-Indexing

Zhiyu Lin	-Ranking



## Assignment 1 Overview

This Assignment is designed to perform text similarity checking using a multi-step process involving preprocessing, indexing, and ranking. The system uses natural language processing (NLP) techniques to tokenize text, remove stop-words, create an inverted index, and rank documents based on their similarity to user queries using Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity algorithms.



## Folder Layout

```
A1/
├── dataset/                  # Dataset folder
│   ├── corpus.jsonl         # Corpus file
│   ├── queries.jsonl        # Queries file
│   └── qrels/               # Folder with relevance judgments
│       └── test.tsv         # Test file with relevance judgments
├── stopwords.txt            # Text file containing the list of stopwords (one per line)
├── preprocessing.py         # Module for preprocessing (tokenization, filtering, etc.)
├── indexing.py              # Module for building the inverted index
├── ranking.py               # Module for retrieval and ranking (e.g., using cosine similarity or BM25)
└── main.py                  # Main driver script to run the IR system
```



## Program Functionality

1. **Preprocessing (preprocessing.py)**

   - **Tokenization:** Converts input text into lowercase tokens using NLTK's `word_tokenize`, which splits text into meaningful units.
   - **Removing Non-Alphabetic Tokens:** Filters out tokens that contain numbers or special characters to maintain clean, analyzable text.
   - **Stop-word Removal:** Loads stop-words from `stopwords.txt` and removes them from the token list to reduce noise.
   - **Stemming:** Applies the **Porter Stemming Algorithm**, which reduces words to their root forms (e.g., "running" to "run"), helping to normalize the vocabulary.

2. **Indexing (indexing.py)**

   - **Inverted Index Construction:** Maps each token to the list of document IDs where it appears, along with its frequency count in each document. This data structure enables fast retrieval of documents that contain specific terms.
   - **Saving and Loading Index:** The inverted index is saved as a JSON file (`inverted_index.json`) for reuse and quick loading in future runs.

3. **Ranking (ranking.py)**

   - **IDF Calculation:** Computes the Inverse Document Frequency (IDF) using the smoothed formula:
     $$
     IDF(t) = \log\left(\frac{N + 1}{df + 1}\right) + 1
     $$
     where NN is the total number of documents, and dfdf is the document frequency of the term tt.

   - **TF-IDF Vectorization:** For each document and query, the Term Frequency (TF) is multiplied by the IDF to generate TF-IDF vectors, representing the importance of each term.

   - **Cosine Similarity:** Measures the cosine of the angle between the query and document vectors to determine similarity. The formula is: 
     $$
     {Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}
     $$
      where AA and BB are the TF-IDF vectors.

4. **Main Operations (main.py)**

   - **Corpus and Query Loading:** Loads the corpus and query datasets from JSON files.
   - **Text Preprocessing:** Applies all preprocessing steps to the corpus text.
   - **Inverted Index Building:** Constructs and saves the inverted index for fast query processing.
   - **Document Ranking:** For each query, ranks documents based on cosine similarity scores and outputs the results in `Results.txt`.

## Running the Program

1. **Dependencies:**

   - Install NLTK: `pip install nltk`
   - Ensure `stopwords.txt`, `corpus.jsonl`, and `queries.jsonl` are in the `dataset/` folder.

2. **Execution:** Run the main script:

   ```bash
   python main.py
   ```

   This will process the corpus and queries, generate the inverted index, and output ranking results to `Results.txt`.

## Algorithms, Data Structures, and Optimizations

1. **Algorithms:**
   - **Tokenization:** Uses NLTK's `word_tokenize` for splitting text into tokens.
   - **Stop-word Removal:** Removes common, non-informative words to reduce noise.
   - **Porter Stemming Algorithm:** Reduces words to their base forms to consolidate similar terms.
   - **Inverted Index:** Maps tokens to document IDs and frequencies for efficient retrieval.
   - **TF-IDF Calculation:** Quantifies term importance in documents relative to the corpus.
   - **Cosine Similarity:** Measures the similarity between document and query vectors.
2. **Data Structures:**
   - **Dictionaries:** Used extensively for the inverted index, IDF storage, and TF-IDF vectors.
   - **Lists:** To store tokens, documents, and queries.
   - **Sets:** Optimizes membership checks during query processing.
3. **Optimizations:**
   - **Efficient Preprocessing:** Reduces computational overhead by filtering out unnecessary tokens early.
   - **Sparse Representations:** Only stores non-zero TF-IDF values to save memory.
   - **Batch Processing:** Processes documents and queries in batches to improve I/O performance.

## Mean Average Precision (MAP) Score

The MAP score is computed using `trec_eval` based on the results from the `Results.txt` file. Please make sure to install `trec_eval` and use `make` command to build the executable for MAP.

- **Test Setup:**

  - **Run 1:** Using only titles from the queries.
  - **Run 2:** Using both titles and full text from the queries.

- After running for the MAP (Mean Average Precision), these were our results:

  ```
  runid                   all     my_run
  map                     all     0.4258
  ```

- **Discussion:**

  Typically, using both titles and full text provides richer context, leading to better ranking accuracy and higher MAP scores. However, this can vary depending on the dataset and query nature. The additional context helps in better term matching and relevance estimation. MAP represents an overall performance of our searching, and we managed to achieve a 42.58% Mean Average Precision. Considering we have a limited data set, we think this is pretty good.

## Sample tokens from the vocabulary

```
aa
aaa
aab
aabenhus
aacr
aad
aag
aai
aams
aarhus
aaronquinlan
aas
aasv
aatf
aauaaa
aav
ab
abad
abandoned
abandoning
abasic
abbe
abberant
abbott
abbreviated
abc
abciximab
abd
abdb
abdomen
abdominal
abduction
aberrant
aberrantly
aberration
aberrations
abeta
abi
abilities
ability
abiotic
abiraterone
abl
ablate
ablated
ablating
ablation
able
abmd
abms
abnormal
abnormalities
abnormality
abnormally
abolish
abolished
abolishes
abolishing
abort
aborted
aborting
abortion
abortions
abortive
aborts
abound
abounds
abp
abpi
abrb
abroad
abrogate
abrogated
abrogates
abrogating
abrogation
abrupt
abruption
abruptly
abs
abscess
abscesses
abscises
abscisic
abscission
absence
absent
absolute
absolutely
absorbable
absorbance
absorbed
absorbing
absorbs
absorptiometry
absorption
absorptive
abstain
abstained
abstainers
```

## Sample queries

**Query 0 (10 result)**

```
0 10906636 The carboxyl terminus of human cytomegalovirus-encoded 7 transmembrane receptor US28 camouflages agonism by mediating constitutive endocytosis.
0 26731863 Distinct and essential roles of transcription factors IRF-3 and IRF-7 in response to viruses for IFN-alpha/beta gene induction.
0 26071782 Latent membrane protein 1 of Epstein–Barr virus coordinately regulates proliferation with control of apoptosis
0 994800 TCR ligand density and affinity determine peripheral induction of Foxp3 in vivo
0 37949139 The in vitro effect of dandelions antioxidants on microsomal lipid peroxidation.
0 21439640 Macrophages induce COX-2 expression in breast cancer cells: role of IL-1β autoamplification.
0 7581911 Human embryonic stem cells with biological and epigenetic characteristics similar to those of mouse ESCs.
0 35008773 Neural induction and early patterning in vertebrates.
0 2566674 Ribose 2′-O-methylation provides a molecular signature for the distinction of self and non-self mRNA dependent on the RNA sensor Mda5
0 13231899 In situ regulation of DC subsets and T cells mediates tumor regression in mice.
```

**Query 1(10 result)**

```
1 10906636 The carboxyl terminus of human cytomegalovirus-encoded 7 transmembrane receptor US28 camouflages agonism by mediating constitutive endocytosis.
1 994800 TCR ligand density and affinity determine peripheral induction of Foxp3 in vivo
1 37949139 The in vitro effect of dandelions antioxidants on microsomal lipid peroxidation.
1 21439640 Macrophages induce COX-2 expression in breast cancer cells: role of IL-1β autoamplification.
1 26071782 Latent membrane protein 1 of Epstein–Barr virus coordinately regulates proliferation with control of apoptosis
1 7581911 Human embryonic stem cells with biological and epigenetic characteristics similar to those of mouse ESCs.
1 26731863 Distinct and essential roles of transcription factors IRF-3 and IRF-7 in response to viruses for IFN-alpha/beta gene induction.
1 35008773 Neural induction and early patterning in vertebrates.
1 13231899 In situ regulation of DC subsets and T cells mediates tumor regression in mice.
1 10786948 An efficient nonviral method to generate integration-free human-induced pluripotent stem cells from cord blood and peripheral blood cells.
1 6227220 Autophagy deficiency leads to protection from obesity and insulin resistance by inducing Fgf21 as a mitokine
```



## Conclusion

This similarity checking project efficiently processes large text corpora, leveraging NLP techniques and Information Retrieval (IR) models to deliver accurate document rankings for user queries. The modular structure allows easy adjustments and optimizations for future enhancements. The inclusion of algorithms like Porter Stemming, TF-IDF weighting, and cosine similarity ensures robust and scalable performance for real-world text similarity applications.