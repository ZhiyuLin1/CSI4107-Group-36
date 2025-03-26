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
    similarities = dot_products / (doc_norms * query_norm + 1e-10)
    return similarities


def normalize_scores(scores):
    """
    Normalizes an array of scores to the range [0, 1].
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.full_like(scores, 0.5)
    return (scores - min_score) / (max_score - min_score)


def neural_rerank_bert(query_text, docs, bert_embedder=None):
    """
    Re-ranks a list of candidate documents using BERT-based embeddings in a vectorized manner.

    :param query_text: Query string.
    :param docs: List of document dictionaries (each with a 'text' field).
    :param bert_embedder: (Optional) Pre-loaded BertEmbedder instance.
    :return: Sorted list of tuples (document, similarity score) in descending order.
    """
    if bert_embedder is None:
        bert_embedder = BertEmbedder()
    query_embedding = bert_embedder.encode(query_text)
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert_embedder.encode(doc_texts)
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def neural_rerank_use(query_text, docs, use_embedder=None):
    """
    Re-ranks a list of candidate documents using USE-based embeddings in a vectorized manner.

    :param query_text: Query string.
    :param docs: List of document dictionaries (each with a 'text' field).
    :param use_embedder: (Optional) Pre-loaded USEEmbedder instance.
    :return: Sorted list of tuples (document, similarity score) in descending order.
    """
    if use_embedder is None:
        use_embedder = USEEmbedder()
    query_embedding = use_embedder.encode(query_text)[0]
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = use_embedder.encode(doc_texts)
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def neural_rerank_bert_hybrid(query_text, docs, baseline_scores, alpha=0.5, bert_embedder=None):
    """
    Hybrid re-ranking using BERT-based embeddings and baseline scores.

    :param query_text: Query string.
    :param docs: List of document dictionaries (each with a 'text' field).
    :param baseline_scores: List or numpy array of baseline scores corresponding to docs.
    :param alpha: Weight for neural score (0 <= alpha <= 1).
    :param bert_embedder: (Optional) Pre-loaded BertEmbedder instance.
    :return: Sorted list of tuples (document, hybrid score) in descending order.
    """
    if bert_embedder is None:
        bert_embedder = BertEmbedder()
    query_embedding = bert_embedder.encode(query_text)
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert_embedder.encode(doc_texts)
    neural_scores = vectorized_cosine_similarity(query_embedding, doc_embeddings)

    # Normalize both sets of scores.
    neural_norm = normalize_scores(neural_scores)
    baseline_norm = normalize_scores(np.array(baseline_scores))

    # Compute the hybrid score.
    hybrid_scores = alpha * neural_norm + (1 - alpha) * baseline_norm
    results = list(zip(docs, hybrid_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def neural_rerank_use_hybrid(query_text, docs, baseline_scores, alpha=0.5, use_embedder=None):
    """
    Hybrid re-ranking using USE-based embeddings and baseline scores.

    :param query_text: Query string.
    :param docs: List of document dictionaries (each with a 'text' field).
    :param baseline_scores: List or numpy array of baseline scores corresponding to docs.
    :param alpha: Weight for neural score (0 <= alpha <= 1).
    :param use_embedder: (Optional) Pre-loaded USEEmbedder instance.
    :return: Sorted list of tuples (document, hybrid score) in descending order.
    """
    if use_embedder is None:
        use_embedder = USEEmbedder()
    query_embedding = use_embedder.encode(query_text)[0]
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = use_embedder.encode(doc_texts)
    neural_scores = vectorized_cosine_similarity(query_embedding, doc_embeddings)

    neural_norm = normalize_scores(neural_scores)
    baseline_norm = normalize_scores(np.array(baseline_scores))
    hybrid_scores = alpha * neural_norm + (1 - alpha) * baseline_norm
    results = list(zip(docs, hybrid_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    return results





