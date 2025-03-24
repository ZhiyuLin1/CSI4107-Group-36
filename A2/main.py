# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import json
from preprocessing import preprocess_text
from indexing import build_inverted_index, save_inverted_index
from ranking import compute_idf, rank_documents_for_query
from neural_ranking import neural_rerank_bert, neural_rerank_use


def load_corpus(filepath="dataset/corpus.jsonl"):
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def load_queries(filepath="dataset/queries.jsonl"):
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


if __name__ == "__main__":
    # Load corpus and queries.
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")

    # Preprocess each document's text.
    for doc in corpus:
        if "text" in doc:
            doc["preprocessed_text"] = preprocess_text(doc["text"])
        else:
            doc["preprocessed_text"] = []

    # Build inverted index and compute IDF.
    inverted_index = build_inverted_index(corpus)
    save_inverted_index(inverted_index)
    total_docs = len(corpus)
    idf = compute_idf(inverted_index, total_docs)

    # Sort queries in ascending order by query id.
    queries.sort(key=lambda q: int(q["_id"]))

    # Define output file names and run tags.
    output_tfidf = "Results_TFIDF.txt"
    output_bert = "Results_BERT.txt"
    output_use = "Results_USE.txt"

    run_tag_tfidf = "run_tfidf"
    run_tag_bert = "run_bert"
    run_tag_use = "run_use"

    # Build a document mapping for quick lookup.
    doc_map = {doc["_id"]: doc for doc in corpus}

    # Open output files for writing the ranking results.
    with open(output_tfidf, "w", encoding="utf-8") as out_tfidf, \
            open(output_bert, "w", encoding="utf-8") as out_bert, \
            open(output_use, "w", encoding="utf-8") as out_use:

        for query in queries:
            query_id = query["_id"]
            query_text = query.get("text", "")

            # ----- Baseline TF-IDF Ranking (Assignment 1) -----
            # Get candidate documents using the original ranking.
            tfidf_ranked = rank_documents_for_query(query_text, corpus, inverted_index, idf)
            tfidf_top = tfidf_ranked[:100]  # top-100 results

            # Write TF-IDF ranking results.
            rank = 1
            for doc_id, score in tfidf_top:
                out_tfidf.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag_tfidf}\n")
                rank += 1

            # ----- Prepare Candidate Docs for Neural Re-ranking -----
            # Get the candidate document objects based on the top TF-IDF ranking.
            candidate_docs = []
            for doc_id, _ in tfidf_top:
                if doc_id in doc_map:
                    candidate_docs.append(doc_map[doc_id])

            # ----- BERT-based Re-ranking -----
            bert_ranked = neural_rerank_bert(query_text, candidate_docs)[:100]
            rank = 1
            for doc, score in bert_ranked:
                out_bert.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_bert}\n")
                rank += 1

            # ----- USE-based Re-ranking -----
            use_ranked = neural_rerank_use(query_text, candidate_docs)[:100]
            rank = 1
            for doc, score in use_ranked:
                out_use.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_use}\n")
                rank += 1

    print("Ranking results written to files:")
    print("Baseline TF-IDF: ", output_tfidf)
    print("BERT-based:       ", output_bert)
    print("USE-based:        ", output_use)
