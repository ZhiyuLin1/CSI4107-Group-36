import json

def load_corpus(filepath="dataset/corpus.jsonl"):
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents

def load_queries(filepath="dataset/queries.jsonl"):
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

if __name__ == "__main__":
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")
