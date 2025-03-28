# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import sys
import time
import json
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import tqdm for progress bars
from tqdm import tqdm

from models.bert_model import BertEmbedder
from models.use_model import USEEmbedder
from neural_ranking import vectorized_cosine_similarity, normalize_scores

# Load corpus documents from a JSON Lines file.
def load_corpus(filepath="dataset/corpus.jsonl"):
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents

# Load queries from a JSON Lines file.
def load_queries(filepath="dataset/queries.jsonl"):
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

# Load baseline results (BM25 from Assignment 1) from a file.
def load_baseline_results(filepath="Results_A1_BM25.txt"):
    results = defaultdict(list)  # {query_id: [(doc_id, rank, score, tag), ...]}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, tag = parts[:6]
                try:
                    rank = int(rank)
                    score = float(score)
                except ValueError:
                    continue
                results[query_id].append((doc_id, rank, score, tag))
    # Sort each query's results by ascending rank.
    for qid in results:
        results[qid].sort(key=lambda x: x[1])
    return results


# Global variables
doc_map = {}
query_map = {}
baseline_results = {}
bert_embedder = None
use_embedder = None
bert_embedding_dict = {}
use_embedding_dict = {}
alpha = 0.35
run_tag_bert = "run_bert_hybrid"
run_tag_use = "run_use_hybrid"

# Process a single query: compute hybrid re-ranking using precomputed candidate embeddings
def process_query(query_id):
    query_text = query_map.get(query_id, "")
    if not query_text:
        return (query_id, [], [])

    # Retrieve the top-100 baseline candidates for this query.
    candidate_info = baseline_results[query_id][:100]
    candidate_docs = []
    candidate_ids = []
    baseline_scores = []
    for doc_id, rank, score, tag in candidate_info:
        if doc_id in doc_map:
            candidate_docs.append(doc_map[doc_id])
            candidate_ids.append(doc_id)
            baseline_scores.append(score)
    if not candidate_docs:
        return (query_id, [], [])

    # Compute query embedding using BERT.
    query_embedding_bert = bert_embedder.encode(query_text)
    candidate_embeddings_bert = np.array([bert_embedding_dict[doc_id] for doc_id in candidate_ids])
    neural_scores_bert = vectorized_cosine_similarity(query_embedding_bert, candidate_embeddings_bert)
    neural_norm_bert = normalize_scores(neural_scores_bert)
    baseline_norm = normalize_scores(np.array(baseline_scores))
    hybrid_scores_bert = alpha * neural_norm_bert + (1 - alpha) * baseline_norm
    bert_results = list(zip(candidate_docs, hybrid_scores_bert))
    bert_results.sort(key=lambda x: x[1], reverse=True)
    bert_lines = []
    rank_counter = 1
    for doc, score in bert_results[:100]:
        bert_lines.append(f"{query_id} Q0 {doc['_id']} {rank_counter} {score:.4f} {run_tag_bert}\n")
        rank_counter += 1

    # Compute query embedding using USE (extracting 1D vector from the 2D output)
    query_embedding_use = use_embedder.encode(query_text)[0]
    candidate_embeddings_use = np.array([use_embedding_dict[doc_id] for doc_id in candidate_ids])
    neural_scores_use = vectorized_cosine_similarity(query_embedding_use, candidate_embeddings_use)
    neural_norm_use = normalize_scores(neural_scores_use)
    hybrid_scores_use = alpha * neural_norm_use + (1 - alpha) * baseline_norm
    use_results = list(zip(candidate_docs, hybrid_scores_use))
    use_results.sort(key=lambda x: x[1], reverse=True)
    use_lines = []
    rank_counter = 1
    for doc, score in use_results[:100]:
        use_lines.append(f"{query_id} Q0 {doc['_id']} {rank_counter} {score:.4f} {run_tag_use}\n")
        rank_counter += 1

    return (query_id, bert_lines, use_lines)


if __name__ == "__main__":
    start_time = time.time()

    # Load corpus, queries, and baseline BM25 results from Assignment 1.
    corpus = load_corpus()
    queries = load_queries()
    baseline_results = load_baseline_results("Results_A1_BM25.txt")
    print(
        f"Loaded {len(corpus)} documents, {len(queries)} queries, and baseline results for {len(baseline_results)} queries.")

    # Build fast lookup mappings.
    doc_map = {doc["_id"]: doc for doc in corpus}
    query_map = {query["_id"]: query.get("text", "") for query in queries}

    # Pre-load neural models.
    bert_embedder = BertEmbedder()
    use_embedder = USEEmbedder()

    # Identify all unique candidate document IDs from baseline results.
    candidate_doc_ids = set()
    for results in baseline_results.values():
        for doc_id, _, _, _ in results:
            candidate_doc_ids.add(doc_id)
    candidate_doc_ids = list(candidate_doc_ids)

    print(f"Precomputing embeddings for {len(candidate_doc_ids)} candidate documents.")
    # Ensure this line prints before we start the tqdm loop:
    sys.stdout.flush()

    # Filter out any doc_ids not in doc_map
    valid_candidate_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in doc_map]

    # Precompute BERT embeddings
    bert_candidate_texts = [doc_map[doc_id]['text'] for doc_id in valid_candidate_ids]
    bert_candidate_embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(bert_candidate_texts), batch_size), desc="Precomputing BERT embeddings"):
        batch_texts = bert_candidate_texts[i: i + batch_size]
        batch_embeddings = bert_embedder.encode(batch_texts)
        bert_candidate_embeddings.extend(batch_embeddings)
    bert_embedding_dict = dict(zip(valid_candidate_ids, bert_candidate_embeddings))

    # Precompute USE embeddings
    use_candidate_texts = [doc_map[doc_id]['text'] for doc_id in valid_candidate_ids]
    use_candidate_embeddings_list = []
    for i in tqdm(range(0, len(use_candidate_texts), batch_size), desc="Precomputing USE embeddings"):
        batch_texts = use_candidate_texts[i: i + batch_size]
        batch_embeddings = use_embedder.encode(batch_texts)
        use_candidate_embeddings_list.extend(batch_embeddings)
    use_embedding_dict = dict(zip(valid_candidate_ids, use_candidate_embeddings_list))

    # Define output file names.
    output_bert = "Results_BERT.txt"
    output_use = "Results_USE.txt"

    # Process queries concurrently using a ThreadPoolExecutor.
    query_ids = sorted(baseline_results.keys(), key=lambda q: int(q))
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(tqdm(
            executor.map(process_query, query_ids),
            total=len(query_ids),
            desc="Processing queries"
        ))

    # Sort results by query id (as integer).
    results_list.sort(key=lambda x: int(x[0]))

    # Write results to output files.
    with open(output_bert, "w", encoding="utf-8") as out_bert, \
            open(output_use, "w", encoding="utf-8") as out_use:
        for query_id, bert_lines, use_lines in tqdm(results_list, desc="Writing output files", total=len(results_list)):
            for line in bert_lines:
                out_bert.write(line)
            for line in use_lines:
                out_use.write(line)

    print("Hybrid re-ranking results have been written to:")
    print("BERT-based:", output_bert)
    print("USE-based:", output_use)
    end_time = time.time()
    print("Total time: {:.2f} seconds".format(end_time - start_time))