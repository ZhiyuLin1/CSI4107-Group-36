# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import numpy as np
from models.bert_model import BertEmbedder
from models.use_model import USEEmbedder


def vectorized_cosine_similarity(query_embedding, doc_embeddings):
    """
    Computes cosine similarity in a vectorized fashion.

    :param query_embedding: numpy array of shape (D,)
    :param doc_embeddings: numpy array of shape (N, D)
    :return: numpy array of shape (N,) with cosine similarity scores.
    """
    dot_products = np.dot(doc_embeddings, query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    # Add a small epsilon to avoid division by zero.
    similarities = dot_products / (doc_norms * query_norm + 1e-10)
    return similarities


def neural_rerank_bert(query_text, docs):
    """
    Re-ranks a list of candidate documents using BERT-based embeddings in a vectorized manner.

    :param query_text: A string containing the query.
    :param docs: A list of document dictionaries; each must have a 'text' field.
    :return: A sorted list of tuples (document, similarity score) in descending order.
    """
    bert = BertEmbedder()
    # Encode the query to obtain a (D,) vector.
    query_embedding = bert.encode(query_text)
    # Encode all candidate documents to obtain a (N, D) array.
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert.encode(doc_texts)

    # Compute cosine similarities in one go.
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def neural_rerank_use(query_text, docs):
    """
    Re-ranks a list of candidate documents using USE-based embeddings in a vectorized manner.

    :param query_text: A string containing the query.
    :param docs: A list of document dictionaries; each must have a 'text' field.
    :return: A sorted list of tuples (document, similarity score) in descending order.
    """
    use = USEEmbedder()
    # USE returns a (1, D) array for a single text; extract the vector.
    query_embedding_full = use.encode(query_text)
    query_embedding = query_embedding_full[0]

    # Encode all candidate documents.
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = use.encode(doc_texts)

    # Compute cosine similarities vectorized.
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results



