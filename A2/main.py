# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import time
import json
from collections import defaultdict
import numpy as np

from models.bert_model import BertEmbedder
from models.use_model import USEEmbedder
from neural_ranking import vectorized_cosine_similarity


def load_corpus(filepath="dataset/corpus.jsonl"):
    """Load corpus documents from a JSON Lines file."""
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def load_queries(filepath="dataset/queries.jsonl"):
    """Load queries from a JSON Lines file."""
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_baseline_results(filepath="Results_A1.txt"):
    """
    Load baseline results from Assignment 1.
    Expected format per line:
      query_id Q0 doc_id rank score tag
    Returns a dictionary mapping query_id to a list of tuples.
    """
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
    # Sort each query's results by rank (ascending)
    for qid in results:
        results[qid].sort(key=lambda x: x[1])
    return results


def normalize_scores(scores):
    """Normalize a list of scores to the range [0, 1]."""
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        # If all scores are the same, return a list of constant values (0.5)
        return [0.5 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]


if __name__ == "__main__":
    start_time = time.time()

    # Load corpus, queries, and baseline results.
    corpus = load_corpus()
    queries = load_queries()
    baseline_results = load_baseline_results("Results_A1.txt")
    print(
        f"Loaded {len(corpus)} documents, {len(queries)} queries, and baseline results for {len(baseline_results)} queries.")

    # Build fast lookup mappings.
    doc_map = {doc["_id"]: doc for doc in corpus}
    query_map = {query["_id"]: query.get("text", "") for query in queries}

    # Pre-load neural models once.
    bert_embedder = BertEmbedder()
    use_embedder = USEEmbedder()

    # Identify all unique candidate document IDs from baseline results.
    candidate_doc_ids = set()
    for results in baseline_results.values():
        for doc_id, _, _, _ in results:
            candidate_doc_ids.add(doc_id)
    candidate_doc_ids = list(candidate_doc_ids)
    print(f"Precomputing embeddings for {len(candidate_doc_ids)} candidate documents.")

    # Precompute BERT embeddings for candidate documents.
    bert_candidate_texts = [doc_map[doc_id]['text'] for doc_id in candidate_doc_ids if doc_id in doc_map]
    bert_candidate_embeddings = bert_embedder.encode(bert_candidate_texts)
    bert_embedding_dict = dict(zip(candidate_doc_ids, bert_candidate_embeddings))

    # Precompute USE embeddings for candidate documents.
    use_candidate_texts = [doc_map[doc_id]['text'] for doc_id in candidate_doc_ids if doc_id in doc_map]
    use_candidate_embeddings = use_embedder.encode(use_candidate_texts)
    use_embedding_dict = dict(zip(candidate_doc_ids, use_candidate_embeddings))

    # Define output file names and run tags.
    output_bert = "Results_BERT_hybrid.txt"
    output_use = "Results_USE_hybrid.txt"
    run_tag_bert = "run_bert_hybrid"
    run_tag_use = "run_use_hybrid"

    # Weight for hybrid interpolation (tune this as needed)
    alpha = 0.75

    # Open output files for writing neural re-ranking results.
    with open(output_bert, "w", encoding="utf-8") as out_bert, \
            open(output_use, "w", encoding="utf-8") as out_use:

        # Process queries in ascending order.
        for query_id in sorted(baseline_results, key=lambda q: int(q)):
            query_text = query_map.get(query_id, "")
            if not query_text:
                continue  # Skip if query text is missing.

            # Get baseline candidate document IDs and scores for this query (top-100).
            candidate_info = baseline_results[query_id][:100]
            candidate_docs = []
            candidate_ids = []
            baseline_scores = []
            for doc_id, _, score, _ in candidate_info:
                if doc_id in doc_map:
                    candidate_docs.append(doc_map[doc_id])
                    candidate_ids.append(doc_id)
                    baseline_scores.append(score)
            if not candidate_docs:
                continue

            # Normalize the baseline scores.
            baseline_norm = normalize_scores(baseline_scores)

            # ----- Hybrid BERT-based Re-ranking -----
            # Compute query embedding.
            query_embedding_bert = bert_embedder.encode(query_text)
            # Get candidate embeddings from precomputed cache.
            candidate_embeddings_bert = np.array([bert_embedding_dict[doc_id] for doc_id in candidate_ids])
            # Compute neural (BERT) cosine similarities.
            neural_scores_bert = vectorized_cosine_similarity(query_embedding_bert, candidate_embeddings_bert)
            neural_norm_bert = normalize_scores(neural_scores_bert.tolist())
            # Combine the scores.
            hybrid_scores_bert = [alpha * n + (1 - alpha) * b for n, b in zip(neural_norm_bert, baseline_norm)]
            bert_results = list(zip(candidate_docs, hybrid_scores_bert))
            bert_results.sort(key=lambda x: x[1], reverse=True)
            rank = 1
            for doc, score in bert_results[:100]:
                out_bert.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_bert}\n")
                rank += 1

            # ----- Hybrid USE-based Re-ranking -----
            # Compute query embedding (extract 1D vector from USE output).
            query_embedding_use = use_embedder.encode(query_text)[0]
            candidate_embeddings_use = np.array([use_embedding_dict[doc_id] for doc_id in candidate_ids])
            neural_scores_use = vectorized_cosine_similarity(query_embedding_use, candidate_embeddings_use)
            neural_norm_use = normalize_scores(neural_scores_use.tolist())
            hybrid_scores_use = [alpha * n + (1 - alpha) * b for n, b in zip(neural_norm_use, baseline_norm)]
            use_results = list(zip(candidate_docs, hybrid_scores_use))
            use_results.sort(key=lambda x: x[1], reverse=True)
            rank = 1
            for doc, score in use_results[:100]:
                out_use.write(f"{query_id} Q0 {doc['_id']} {rank} {score:.4f} {run_tag_use}\n")
                rank += 1

    print("Hybrid re-ranking results have been written to:")
    print("BERT-based:", output_bert)
    print("USE-based:", output_use)
    end_time = time.time()
    print("Total time: {:.2f} seconds".format(end_time - start_time))



