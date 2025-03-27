# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import json
from preprocessing import preprocess_text
from indexing import build_inverted_index, save_inverted_index
from ranking import compute_idf, rank_documents_for_query, rank_documents_for_query_bm25


# Loads the corpus from a JSON Lines file.
def load_corpus(filepath="dataset/corpus.jsonl"):
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
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
    # Load corpus and queries.
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")

    # Preprocess each document's text and store tokens under 'preprocessed_text'.
    for doc in corpus:
        if 'text' in doc:
            doc['preprocessed_text'] = preprocess_text(doc['text'])
        else:
            doc['preprocessed_text'] = []

    # Build and save the inverted index.
    inverted_index = build_inverted_index(corpus)
    save_inverted_index(inverted_index)
    print("Inverted index built and saved successfully.")

    # Compute IDF values for all tokens in the index.
    total_docs = len(corpus)
    idf = compute_idf(inverted_index, total_docs)

    # Compute average document length (for BM25).
    total_length = sum(len(doc.get('preprocessed_text', [])) for doc in corpus)
    avg_doc_len = total_length / total_docs if total_docs > 0 else 0

    # Sort the queries in ascending order by their query ID.
    queries = sorted(queries, key=lambda q: int(q['_id']))

    # Run tag for TREC output.
    run_tag = "my_run"

    # Open the output files to write ranking results.
    output_tf_idf = "Results_TFIDF.txt"
    output_bm25 = "Results_BM25.txt"

    with open(output_tf_idf, 'w', encoding='utf-8') as out_tf_idf, \
            open(output_bm25, 'w', encoding='utf-8') as out_bm25:

        # Process each query.
        for query in queries:
            query_id = query['_id']
            query_text = query.get('text', '')

            # Rank documents using TF-IDF (cosine similarity).
            ranked_tf_idf = rank_documents_for_query(query_text, corpus, inverted_index, idf)
            top_tf_idf = ranked_tf_idf[:100]
            for rank, (doc_id, score) in enumerate(top_tf_idf, start=1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n"
                out_tf_idf.write(line)

            # Rank documents using BM25.
            ranked_bm25 = rank_documents_for_query_bm25(query_text, corpus, inverted_index, avg_doc_len)
            top_bm25 = ranked_bm25[:100]
            for rank, (doc_id, score) in enumerate(top_bm25, start=1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n"
                out_bm25.write(line)

    print(f"TF-IDF ranking results written to '{output_tf_idf}'.")
    print(f"BM25 ranking results written to '{output_bm25}'.")
