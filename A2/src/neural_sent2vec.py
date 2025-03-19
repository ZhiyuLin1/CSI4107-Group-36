import numpy as np
import sent2vec



def load_sent2vec_model(model_path):
    """
    Load a pretrained sent2vec model from the specified model path.

    Parameters:
    - model_path: Path to the sent2vec model file (e.g., "models/sent2vec/model.bin")

    Returns:
    - A sent2vec model object.
    """
    model = sent2vec.Sent2vecModel()
    model.load_model(model_path)
    return model


def embed_text(model, text):
    """
    Generate a sentence embedding for the input text using the provided sent2vec model.

    Parameters:
    - model: Pretrained sent2vec model.
    - text: A string to embed.

    Returns:
    - A numpy array representing the sentence embedding.
    """
    # sent2vec expects a list of sentences and returns an array of shape (n_sentences, embedding_dim)
    embedding = model.embed_sentences([text])[0]
    return embedding


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec1, vec2: Numpy arrays.

    Returns:
    - Cosine similarity as a float.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def rerank_candidates_sent2vec(query, candidate_docs, model):
    """
    Re-rank a list of candidate documents using sent2vec embeddings.

    Parameters:
    - query: A string representing the query.
    - candidate_docs: A list of candidate document dictionaries. Each document should contain at least '_id' and 'text'.
    - model: The pretrained sent2vec model.

    Returns:
    - A list of tuples (doc_id, score) sorted in descending order of similarity score.
    """
    query_embedding = embed_text(model, query)
    results = []
    for doc in candidate_docs:
        # Use the 'text' field to compute the document embedding.
        doc_text = doc.get('text', '')
        if not doc_text.strip():
            score = 0.0
        else:
            doc_embedding = embed_text(model, doc_text)
            score = cosine_similarity(query_embedding, doc_embedding)
        results.append((doc['_id'], score))

    results.sort(key = lambda x: x[1], reverse = True)
    return results

# Testing
if __name__ == "__main__":
    # Update this path to point to your sent2vec model file.
    model_path = "models/sent2vec/model.bin"
    model = load_sent2vec_model(model_path)

    # Example query.
    query = "What is the impact of climate change on polar bears?"

    # Example candidate documents.
    candidate_docs = [
        {"_id": "doc1", "text": "Polar bears are greatly affected by the melting of sea ice due to climate change."},
        {"_id": "doc2", "text": "Climate change has significant effects on marine life."},
        {"_id": "doc3", "text": "Polar bears are a species living in the Arctic."}
    ]

    ranked_results = rerank_candidates_sent2vec(query, candidate_docs, model)
    print("Reranked candidates using sent2vec:")
    for doc_id, score in ranked_results:
        print(f"DocID: {doc_id}, Score: {score:.4f}")