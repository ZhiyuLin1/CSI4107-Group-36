# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import numpy as np
from models.bert_model import BertEmbedder
from models.use_model import USEEmbedder

# Computes cosine similarity in a vectorized manner.
def vectorized_cosine_similarity(query_embedding, doc_embeddings):
    dot_products = np.dot(doc_embeddings, query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = dot_products / (doc_norms * query_norm + 1e-10)
    return similarities

# Normalizes an array of scores to the range [0, 1].
def normalize_scores(scores):
    scores = np.array(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.full_like(scores, 0.5)
    return (scores - min_score) / (max_score - min_score)

# Re-ranks candidate documents using BERT-based embeddings in a vectorized manner.
def neural_rerank_bert(query_text, docs, bert_embedder=None):
    if bert_embedder is None:
        bert_embedder = BertEmbedder()
    query_embedding = bert_embedder.encode(query_text)
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert_embedder.encode(doc_texts)
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Re-ranks candidate documents using USE-based embeddings in a vectorized manner.
def neural_rerank_use(query_text, docs, use_embedder=None):
    if use_embedder is None:
        use_embedder = USEEmbedder()
    query_embedding = use_embedder.encode(query_text)[0]
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = use_embedder.encode(doc_texts)
    similarities = vectorized_cosine_similarity(query_embedding, doc_embeddings)
    results = list(zip(docs, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Hybrid re-ranking using BERT-based embeddings and baseline scores.
def neural_rerank_bert_hybrid(query_text, docs, baseline_scores, alpha=0.35, bert_embedder=None):
    if bert_embedder is None:
        bert_embedder = BertEmbedder()
    query_embedding = bert_embedder.encode(query_text)
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert_embedder.encode(doc_texts)
    neural_scores = vectorized_cosine_similarity(query_embedding, doc_embeddings)

    # Normalize both neural and baseline scores.
    neural_norm = normalize_scores(neural_scores)
    baseline_norm = normalize_scores(np.array(baseline_scores))

    hybrid_scores = alpha * neural_norm + (1 - alpha) * baseline_norm
    results = list(zip(docs, hybrid_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Hybrid re-ranking using USE-based embeddings and baseline scores.
def neural_rerank_use_hybrid(query_text, docs, baseline_scores, alpha=0.35, use_embedder=None):
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






