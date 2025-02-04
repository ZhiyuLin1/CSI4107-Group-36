# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import math
import json
from preprocessing import preprocess_text
from indexing import load_inverted_index


def compute_idf(inverted_index, total_docs):
    """
    Compute the inverse document frequency (IDF) for each term.
    Uses the smoothed formula: log((N + 1) / (df + 1)) + 1.
    """
    idf = {}
    for token, doc_dict in inverted_index.items():
        df = len(doc_dict)
        idf[token] = math.log((total_docs + 1) / (df + 1)) + 1
    return idf


def compute_vector(tokens, idf):
    """
    Compute a TF-IDF vector for a list of tokens.
    The vector is a dictionary mapping tokens to TF-IDF weights.
    """
    vector = {}
    # Compute term frequency (TF)
    for token in tokens:
        vector[token] = vector.get(token, 0) + 1
    # Multiply each TF by the token's IDF
    for token in vector:
        vector[token] *= idf.get(token, 0)
    return vector


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two TF-IDF vectors (dictionaries).
    """
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
    """
    Given a query (as a string), the corpus (a list of documents), the inverted index, and the IDF dictionary:
      1. Preprocess the query.
      2. Build the TF-IDF vector for the query.
      3. Identify candidate documents (those that contain at least one query token).
      4. Compute cosine similarity between the query vector and each candidate document's vector.
      5. Return a sorted list of (doc_id, score) tuples.
    """
    # Preprocess the query
    query_tokens = preprocess_text(query)
    query_vector = compute_vector(query_tokens, idf)

    # Determine candidate document IDs (documents containing any query token)
    candidate_doc_ids = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token].keys())

    # Build a mapping from doc_id to document for fast lookup
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

    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

