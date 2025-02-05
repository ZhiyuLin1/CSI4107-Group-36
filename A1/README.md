# CSI4107 Assignment 1 

## Group number: 36

### Members:  Hengjing Zhang  300288003	Tom Cui 300345709	ZhiyuLin300255509

______

## Assignment 1 Overview

This Assignment is designed to perform text similarity checking using a multi-step process involving preprocessing, indexing, and ranking. The system uses natural language processing (NLP) techniques to tokenize text, remove stop-words, create an inverted index, and rank documents based on their similarity to user queries using Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity algorithms.



## Folder Layout:

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

## Vocabulary Details

- **Vocabulary Size:** The vocabulary consists of all unique tokens after preprocessing, including tokenization, stop-word removal, and stemming.
- **Sample of 100 Tokens:** (This sample will be generated from the dataset during actual execution.)

## Query Results

- First 10 Answers for the First 2 Queries:

  Example from 

  ```
  Results.txt
  ```

  :

  ```
  0 Q0 10906636 1 0.1253 my_run
  0 Q0 26731863 2 0.1180 my_run
  0 Q0 26071782 3 0.1099 my_run
  0 Q0 994800 4 0.0968 my_run
  0 Q0 37949139 5 0.0892 my_run
  0 Q0 21439640 6 0.0879 my_run
  0 Q0 7581911 7 0.0834 my_run
  0 Q0 35008773 8 0.0825 my_run
  ```

## Mean Average Precision (MAP) Score

The MAP score is computed using `trec_eval` based on the results from the `Results.txt` file.

- **Test Setup:**

  - **Run 1:** Using only titles from the queries.
  - **Run 2:** Using both titles and full text from the queries.

- **Discussion:**

  Typically, using both titles and full text provides richer context, leading to better ranking accuracy and higher MAP scores. However, this can vary depending on the dataset and query nature. The additional context helps in better term matching and relevance estimation.

## Conclusion

This similarity checking project efficiently processes large text corpora, leveraging NLP techniques and Information Retrieval (IR) models to deliver accurate document rankings for user queries. The modular structure allows easy adjustments and optimizations for future enhancements. The inclusion of algorithms like Porter Stemming, TF-IDF weighting, and cosine similarity ensures robust and scalable performance for real-world text similarity applications.