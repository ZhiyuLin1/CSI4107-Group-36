# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import json
import time
from preprocessing import preprocess_text
from indexing import build_inverted_index, save_inverted_index
from ranking import compute_idf, rank_documents_for_query
from neural_ranking import neural_rerank_bert, neural_rerank_use
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def process_query(query, corpus, inverted_index, idf, doc_map):
    """
    Process a single query:
      - Compute the baseline TF‑IDF ranking.
      - Select the top 100 candidate documents.
      - Perform neural re‑ranking with USE and BERT.
    Returns a tuple with:
         (query_id, tfidf_top, use_ranked, bert_ranked)
    where:
         tfidf_top is a list of (doc_id, score) tuples,
         use_ranked is a list of (document, score) tuples (from USE),
         bert_ranked is a list of (document, score) tuples (from BERT).
    """
    query_id = query["_id"]
    query_text = query.get("text", "")

    # Baseline TF‑IDF Ranking
    tfidf_ranked = rank_documents_for_query(query_text, corpus, inverted_index, idf)
    tfidf_top = tfidf_ranked[:100]

    # Prepare candidate documents (based on the TF‑IDF ranking)
    candidate_docs = [doc_map[doc_id] for doc_id, _ in tfidf_top if doc_id in doc_map]

    # Neural re‑ranking (first USE, then BERT)
    use_ranked  = neural_rerank_use(query_text, candidate_docs)[:100]
    bert_ranked = neural_rerank_bert(query_text, candidate_docs)[:100]

    return (query_id, tfidf_top, use_ranked, bert_ranked)

if __name__ == "__main__":
    start_time = time.time()  # Start timer

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

    # Build a document mapping for quick lookup.
    doc_map = {doc["_id"]: doc for doc in corpus}

    # Process queries concurrently using all available CPU cores.
    results = []
    num_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_query, query, corpus, inverted_index, idf, doc_map)
                   for query in queries]
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results by query id (as integer) to preserve original order.
    results.sort(key=lambda x: int(x[0]))

    # Define output file names and run tags.
    output_tfidf = "Results_TFIDF.txt"
    output_use   = "Results_USE.txt"
    output_bert  = "Results_BERT.txt"

    run_tag_tfidf = "run_tfidf"
    run_tag_use   = "run_use"
    run_tag_bert  = "run_bert"

    # Write TF‑IDF results (first file).
    with open(output_tfidf, "w", encoding="utf-8") as out_tfidf:
        for query_id, tfidf_top, use_ranked, bert_ranked in results:
            rank = 1
            for doc_id, score in tfidf_top:
                out_tfidf.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag_tfidf}\n")
                rank += 1
    print("TF-IDF results written to", output_tfidf)

    # Write USE results (second file).
    with open(output_use, "w", encoding="utf-8") as out_use:
        for query_id, tfidf_top, use_ranked, bert_ranked in results:
            rank = 1
            for doc, score in use_ranked:
                out_use.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_use}\n")
                rank += 1
    print("USE results written to", output_use)

    # Write BERT results (third file).
    with open(output_bert, "w", encoding="utf-8") as out_bert:
        for query_id, tfidf_top, use_ranked, bert_ranked in results:
            rank = 1
            for doc, score in bert_ranked:
                out_bert.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_bert}\n")
                rank += 1
    print("BERT results written to", output_bert)

    # End timer and print total execution time.
    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time: {:.2f} seconds".format(total_time))
