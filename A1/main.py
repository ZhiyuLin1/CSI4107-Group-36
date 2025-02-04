import json
from preprocessing import preprocess_text
from indexing import build_inverted_index, save_inverted_index

# Loads the corpus from a JSON Lines file.
def load_corpus(filepath="dataset/corpus.jsonl"):

    documents = []
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents

# Loads the queries from a JSON Lines file.
def load_queries(filepath="dataset/queries.jsonl"):
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


if __name__ == "__main__":
    # Load the corpus and queries
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")

    # Preprocess the text of each document.
    # This adds a new key 'preprocessed_text' to each document.
    for doc in corpus:
        if 'text' in doc:
            doc['preprocessed_text'] = preprocess_text(doc['text'])
        else:
            doc['preprocessed_text'] = []

    # Build the inverted index using the preprocessed text tokens.
    inverted_index = build_inverted_index(corpus)
    save_inverted_index(inverted_index)
    print("Inverted index built and saved successfully.")
