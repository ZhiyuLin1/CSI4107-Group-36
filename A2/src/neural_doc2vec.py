# test_pretrained_doc2vec.py
from gensim.models.doc2vec import Doc2Vec
from preprocessing import preprocess_text


def main():
    # Load the pretrained doc2vec model
    model_path = "../models/doc2vec.bin"
    model = Doc2Vec.load(model_path)

    # Sample text to test the model
    sample_text = "The quick brown fox jumps over the lazy dog."
    # Preprocess the sample text (tokenize, remove non-alphabetic tokens, stopwords, and apply stemming)
    tokens = preprocess_text(sample_text)

    # Infer a vector for the sample text
    inferred_vector = model.infer_vector(tokens)
    print("Inferred vector:")
    print(inferred_vector)

    # Retrieve and print the top-5 most similar documents using the inferred vector
    similar_docs = model.docvecs.most_similar([inferred_vector], topn=5)
    print("\nMost similar documents:")
    for doc_id, score in similar_docs:
        print(f"DocID: {doc_id}, Similarity Score: {score:.4f}")


if __name__ == "__main__":
    main()
