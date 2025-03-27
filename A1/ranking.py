# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import math
import json
from preprocessing import preprocess_text
from indexing import load_inverted_index


def compute_idf(inverted_index, total_docs):
    # Compute the inverse document frequency (IDF) for each term.
    # Uses the smoothed formula: log((N + 1) / (df + 1)) + 1.
    idf = {}
    for token, doc_dict in inverted_index.items():
        df = len(doc_dict)
        idf[token] = math.log((total_docs + 1) / (df + 1)) + 1
    return idf


def compute_vector(tokens, idf):
    # Compute a TF-IDF vector for a list of tokens.
    # The vector is a dictionary mapping tokens to TF-IDF weights.
    vector = {}
    # Compute term frequency (TF)
    for token in tokens:
        vector[token] = vector.get(token, 0) + 1
    # Multiply each TF by the token's IDF
    for token in vector:
        vector[token] *= idf.get(token, 0)
    return vector


def cosine_similarity(vec1, vec2):
    # Compute cosine similarity between two TF-IDF vectors (dictionaries).
    dot_product = 0.0
    for token, weight in vec1.items():
        if token in vec2:
            dot_product += weight * vec2[token]
    norm1 = math.sqrt(sum(weight ** 2 for weight in vec1.values()))
    norm2 = math.sqrt(sum(weight ** 2 for weight in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def rank_documents_for_query(query, corpus, inverted_index, idf):
    # Preprocess the query.
    query_tokens = preprocess_text(query)
    query_vector = compute_vector(query_tokens, idf)

    # Determine candidate document IDs (documents containing any query token)
    candidate_doc_ids = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token].keys())

    # Build a mapping from doc_id to document for fast lookup.
    doc_map = {doc['_id']: doc for doc in corpus}

    results = []
    for doc_id in candidate_doc_ids:
        # If the document lacks 'preprocessed_text', process it now.
        if 'preprocessed_text' not in doc_map[doc_id] and 'text' in doc_map[doc_id]:
            doc_map[doc_id]['preprocessed_text'] = preprocess_text(doc_map[doc_id]['text'])
        # Get the preprocessed tokens; if missing, default to an empty list.
        doc_tokens = doc_map[doc_id].get('preprocessed_text', [])
        doc_vector = compute_vector(doc_tokens, idf)
        score = cosine_similarity(query_vector, doc_vector)
        results.append((doc_id, score))

    # Sort results by similarity score (highest first).
    results.sort(key=lambda x: x[1], reverse=True)
    return results


#####################
# BM25 Implementation
#####################

def rank_documents_for_query_bm25(query, corpus, inverted_index, avg_doc_len, k1=1.5, b=0.75):
    """
    Ranks documents using the BM25 scoring function.

    Parameters:
      query: a string representing the query.
      corpus: list of document dictionaries.
      inverted_index: the inverted index built from the corpus.
      avg_doc_len: average document length (number of tokens) in the corpus.
      k1, b: BM25 parameters (default values: k1=1.5, b=0.75).

    Returns a list of tuples (doc_id, score), sorted by descending BM25 score.
    """
    query_tokens = preprocess_text(query)
    candidate_doc_ids = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token].keys())

    doc_map = {doc['_id']: doc for doc in corpus}
    N = len(corpus)
    results = []

    for doc_id in candidate_doc_ids:
        doc = doc_map[doc_id]
        # Ensure preprocessed text is available.
        if 'preprocessed_text' not in doc and 'text' in doc:
            doc['preprocessed_text'] = preprocess_text(doc['text'])
        doc_tokens = doc.get('preprocessed_text', [])
        doc_len = len(doc_tokens)
        score = 0.0
        for token in query_tokens:
            # Compute document frequency (df) for the token.
            if token in inverted_index:
                df = len(inverted_index[token])
            else:
                df = 0
            # Compute term frequency (tf) in this document.
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            # BM25 idf component.
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            # BM25 term score.
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))
        results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
