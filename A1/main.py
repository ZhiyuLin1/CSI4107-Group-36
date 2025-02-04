# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import json
from preprocessing import preprocess_text
from indexing import build_inverted_index, save_inverted_index
from ranking import compute_idf, rank_documents_for_query


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

    # (Optional) Filter queries if needed (e.g., only test queries).
    # For example, if only queries with odd IDs are test queries:
    # queries = [q for q in queries if int(q['_id']) % 2 == 1]

    # Sort the queries in ascending order by their query ID.
    queries = sorted(queries, key=lambda q: int(q['_id']))

    # run_tag
    run_tag = "my_run"

    # Open the output file "Results.txt" to write ranking results.
    output_filename = "Results.txt"
    with open(output_filename, 'w', encoding='utf-8') as out:
        # Process each query.
        for query in queries:
            query_id = query['_id']
            # Assume the query text is stored under the field "text".
            query_text = query.get('text', '')

            # Rank documents for this query using cosine similarity with TF-IDF.
            ranked_docs = rank_documents_for_query(query_text, corpus, inverted_index, idf)

            # Limit to the top 100 results.
            top_results = ranked_docs[:100]
            for rank, (doc_id, score) in enumerate(top_results, start=1):
                # Write a line in the format: query_id Q0 doc_id rank score tag
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n"
                out.write(line)
    print(f"Ranking results written to file '{output_filename}'.")
