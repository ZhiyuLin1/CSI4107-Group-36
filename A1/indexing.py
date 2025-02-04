# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import json

# Build an inverted index from the preprocessed documents.
def build_inverted_index(documents):

    inverted_index = {}
    for doc in documents:
        doc_id = doc['_id']
        # Use the preprocessed text tokens; these should be stored in the document.
        tokens = doc.get('preprocessed_text', [])
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0
            inverted_index[token][doc_id] += 1
    return inverted_index

# Saves the inverted index as a JSON file.
def save_inverted_index(inverted_index, filepath="inverted_index.json"):
    with open(filepath, 'w', encoding = 'utf-8') as f:
        json.dump(inverted_index, f, indent = 4)

# Loads the inverted index from a JSON file.
def load_inverted_index(filepath="inverted_index.json"):
    with open(filepath, 'r', encoding = 'utf-8') as f:
        inverted_index = json.load(f)
    return inverted_index

# for testing
'''
if __name__ == "__main__":
    # Example: build a dummy index with a single document.
    docs = [
        {"_id": "doc1", "preprocessed_text": ["this", "is", "a", "test"]},
        {"_id": "doc2", "preprocessed_text": ["this", "test", "is", "another", "example"]}
    ]
    index = build_inverted_index(docs)
    print("Inverted index:", index)
    save_inverted_index(index)
'''
