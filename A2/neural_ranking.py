# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import numpy as np
from models.bert_model import BertEmbedder
from models.use_model import USEEmbedder


def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two numpy vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def neural_rerank_bert(query_text, docs):
    """
    Re-ranks a list of candidate documents using BERT-based embeddings.

    :param query_text: A string containing the query.
    :param docs: A list of document dictionaries; each must have a 'text' field.
    :return: A sorted list of tuples (document, similarity score) in descending order of similarity.
    """
    # Initialize the BERT embedder (this will load the model).
    bert = BertEmbedder()

    # Compute the embedding for the query.
    query_embedding = bert.encode(query_text)

    # Compute embeddings for all candidate documents.
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = bert.encode(doc_texts)

    # Compute cosine similarity between the query and each document.
    results = []
    for doc, emb in zip(docs, doc_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        results.append((doc, sim))

    # Sort the results based on similarity score (highest first).
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def neural_rerank_use(query_text, docs):
    """
    Re-ranks a list of candidate documents using USE-based embeddings.

    :param query_text: A string containing the query.
    :param docs: A list of document dictionaries; each must have a 'text' field.
    :return: A sorted list of tuples (document, similarity score) in descending order of similarity.
    """
    # Initialize the USE embedder.
    use = USEEmbedder()

    # Compute the embedding for the query.
    # Extract the first element to convert the (1,512) array to a (512,) vector.
    query_embedding = use.encode(query_text)[0]

    # Compute embeddings for all candidate documents.
    doc_texts = [doc['text'] for doc in docs]
    doc_embeddings = use.encode(doc_texts)

    # Compute cosine similarity between the query and each document.
    results = []
    for doc, emb in zip(docs, doc_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        results.append((doc, sim))

    # Sort the results based on similarity score (highest first).
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# Example usage (for testing purposes):
if __name__ == "__main__":
    # Sample query and candidate documents
    query = "machine learning in healthcare"
    docs = [
        {"_id": "1", "text": "Healthcare is leveraging machine learning techniques to improve diagnostics."},
        {"_id": "2", "text": "Climate change remains a significant global challenge."},
        {"_id": "3", "text": "Machine learning can analyze large datasets in the healthcare industry effectively."}
    ]

    print("BERT-based Re-ranking:")
    bert_results = neural_rerank_bert(query, docs)
    for doc, score in bert_results:
        print(f"Doc ID: {doc['_id']}, Score: {score:.4f}")

    print("\nUSE-based Re-ranking:")
    use_results = neural_rerank_use(query, docs)
    for doc, score in use_results:
        print(f"Doc ID: {doc['_id']}, Score: {score:.4f}")
