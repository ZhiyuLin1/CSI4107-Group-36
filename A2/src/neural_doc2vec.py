import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_doc2vec(corpus, vector_size=100, window=5, min_count=2, epochs=40):
    """
    Train a Doc2Vec model on the given corpus.
    Each document in the corpus should be a dictionary with keys '_id' and 'preprocessed_text'.
    Returns the trained Doc2Vec model.
    """
    tagged_docs = []
    for doc in corpus:
        tokens = doc.get('preprocessed_text', [])
        # Use the document's _id (converted to string) as its tag
        tagged_docs.append(TaggedDocument(words=tokens, tags=[str(doc['_id'])]))

    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def save_doc2vec_model(model, file_path):
    """
    Save the trained Doc2Vec model to the given file path.
    """
    model.save(file_path)


def load_doc2vec_model(file_path):
    """
    Load a Doc2Vec model from the given file path.
    """
    return Doc2Vec.load(file_path)


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def re_rank_docs_doc2vec(query_text, corpus, model, candidate_doc_ids=None):
    """
    Given a query (as text), a corpus (list of documents), and a trained Doc2Vec model,
    compute the doc2vec embedding for the query and for each document (or candidate document).
    If candidate_doc_ids is provided (as a set of document IDs), only those documents are re-ranked.
    Returns a list of tuples (doc_id, score) sorted in descending order of similarity.
    """
    # Import the preprocessing function from your preprocessing module
    from preprocessing import preprocess_text
    query_tokens = preprocess_text(query_text)
    query_vector = model.infer_vector(query_tokens)

    results = []
    for doc in corpus:
        doc_id = str(doc['_id'])
        if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
            continue
        tokens = doc.get('preprocessed_text', [])
        if tokens:
            doc_vector = model.infer_vector(tokens)
            score = cosine_similarity(query_vector, doc_vector)
            results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# For testing purposes
if __name__ == "__main__":
    # Create a small dummy corpus for testing
    corpus = [
        {"_id": "1", "preprocessed_text": ["this", "is", "a", "test", "document"]},
        {"_id": "2", "preprocessed_text": ["another", "test", "document", "for", "doc2vec"]},
        {"_id": "3", "preprocessed_text": ["more", "text", "data", "to", "train", "doc2vec"]}
    ]

    # Train a Doc2Vec model on the dummy corpus
    model = train_doc2vec(corpus, vector_size=50, window=2, min_count=1, epochs=20)

    # Test re-ranking with a sample query
    query = "test document"
    ranked_results = re_rank_docs_doc2vec(query, corpus, model)

    print("Re-ranked documents using doc2vec:")
    for doc_id, score in ranked_results:
        print(f"DocID: {doc_id}, Score: {score:.4f}")