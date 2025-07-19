import ast
import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer, calculate_f2
import utils
import random
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from huggingface_hub import login
from dotenv import load_dotenv
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from reranker import create_reranker, reranking
import itertools
import csv
import llm_final_output
import faiss
load_dotenv()
combine_types = ["weighted_sum"]

def encode_legal_data(data_path, models, wseg):
    corpus = json.load(open(os.path.join(data_path, "corpus.json")))
    list_emb_models = []
    for idx, model in enumerate(models):
        doc_list = []
        for k, doc in tqdm(corpus.items()):
            if wseg[idx]:
                doc = utils.word_segmentation(doc)
            doc_list.append(model.encode(doc, show_progress_bar=False))
        emb2_arr = np.array(doc_list)
        list_emb_models.append(emb2_arr)
    with open("encoded_legal_data.pkl", "wb") as f:
        pickle.dump(list_emb_models, f)
    return list_emb_models

def encode_question(question_data, models, wseg):
    print("Start encoding questions.")
    question_embs = []
    for idx, model in enumerate(models):
        emb_quest_dict = {}
        for qid, q in tqdm(question_data.items()):
            question_id = qid
            question = q
            if wseg[idx]:
                question = utils.word_segmentation(question)
            if idx == 2:  # Qwen model
                emb_quest_dict[question_id] = model.encode(question, show_progress_bar=False, prompt_name="query")
            emb_quest_dict[question_id] = model.encode(question, show_progress_bar=False)
        
        question_embs.append(emb_quest_dict)
    with open("encoded_question_data.pkl", "wb") as f:
        pickle.dump(question_embs, f)
    return question_embs

def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    with open(legal_data_path, "rb") as f1:
        emb_legal_data = pickle.load(f1)
    return emb_legal_data

def load_encoded_question_data(question_data_path):
    print("Start loading question data.")
    with open(question_data_path, "rb") as f2:
        question_embs = pickle.load(f2)
    return question_embs

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_models(root, model_names):
    models = []
    wseg = []
    for model_path in tqdm(model_names):
        models.append(SentenceTransformer(model_path))
        if "wseg" in model_path:
            wseg.append(True)
        else:
            wseg.append(False)
    return models

def load_question_json(data_path):
    question_data = json.load(open(os.path.join(data_path, "queries.json")))
    return question_data

def combine_scores(dense_scores, bm25_scores, combine_type = "default", alpha=0.5, k = 60):
    if combine_type == "default":
        return bm25_scores * dense_scores
    elif combine_type == "weighted_sum":
        from sklearn.preprocessing import normalize
        bm25_scores = normalize(bm25_scores.reshape(1, -1), norm='l2').flatten()
        return alpha * dense_scores + (1 - alpha) * bm25_scores
    elif combine_type == "rrf":
        dense_ranks = dense_scores.argsort().argsort() + 1
        bm25_ranks = bm25_scores.argsort().argsort() + 1
        rrf_scores = 1 / (dense_ranks + k) + 1 / (bm25_ranks + k)
        return rrf_scores

def build_faiss_index(emb_legal_data, index_type="IVF", nlist=100):
    faiss_indices = []
    for model_idx, embeddings in enumerate(emb_legal_data):
        print(f"Building FAISS index for model {model_idx + 1}")
        d = embeddings.shape[1]
        if index_type == "Flat":
            index = faiss.IndexFlatIP(d)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(embeddings.astype('float32'))
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError("Unsupported index type")
        faiss.normalize_L2(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        faiss_indices.append(index)
    return faiss_indices

def save_faiss_index(faiss_indices, save_path="faiss_indices"):
    os.makedirs(save_path, exist_ok=True)
    for i, index in enumerate(faiss_indices):
        faiss.write_index(index, os.path.join(save_path, f"index_model_{i}.faiss"))
    print(f"FAISS indices saved to {save_path}")

def load_faiss_index(load_path="faiss_indices", num_models=2):
    faiss_indices = []
    for i in range(num_models):
        index_path = os.path.join(load_path, f"index_model_{i}.faiss")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            faiss_indices.append(index)
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
    print(f"Loaded {len(faiss_indices)} FAISS indices from {load_path}")
    return faiss_indices

def build_law_id_mapping(doc_refers):
    """
    Build mapping between document indices and law IDs.
    
    Args:
        doc_refers: List of tuples (law_id, article_id, text)
    
    Returns:
        dict: Mapping from law_id to list of document indices
    """
    law_id_to_indices = {}
    for idx, (law_id, article_id, text) in enumerate(doc_refers):
        if law_id not in law_id_to_indices:
            law_id_to_indices[law_id] = []
        law_id_to_indices[law_id].append(idx)
    
    print(f"Built law ID mapping for {len(law_id_to_indices)} laws")
    return law_id_to_indices

def save_law_id_mapping(law_id_mapping, save_path="law_id_mapping.pkl"):
    """Save law ID mapping to disk."""
    with open(save_path, "wb") as f:
        pickle.dump(law_id_mapping, f)
    print(f"Law ID mapping saved to {save_path}")

def load_law_id_mapping(load_path="law_id_mapping.pkl"):
    """Load law ID mapping from disk."""
    with open(load_path, "rb") as f:
        law_id_mapping = pickle.load(f)
    print(f"Loaded law ID mapping with {len(law_id_mapping)} laws")
    return law_id_mapping

def faiss_search_with_law_filter(faiss_indices, question_embs, question_id, law_id_mapping, 
                                law_ids_filter=None, top_k=2000, weights=None):
    """
    Search using FAISS indices with law ID filtering.
    
    Args:
        faiss_indices (list): List of FAISS indices
        question_embs (list): Question embeddings for each model
        question_id (str): Question ID
        law_id_mapping (dict): Mapping from law_id to document indices
        law_ids_filter (list): List of law IDs to search within (None for all)
        top_k (int): Number of top results to retrieve
        weights (list): Weights for combining scores from different models
    
    Returns:
        tuple: (scores, indices) filtered by law IDs
    """
    if weights is None:
        weights = [1.0] * len(faiss_indices)
    
    # Determine search space
    if law_ids_filter is not None:
        # Get indices for filtered law IDs
        valid_indices = []
        for law_id in law_ids_filter:
            if law_id in law_id_mapping:
                valid_indices.extend(law_id_mapping[law_id])
        valid_indices = sorted(list(set(valid_indices)))
        print(f"Filtering search to {len(valid_indices)} documents from {len(law_ids_filter)} laws")
    else:
        # Use all documents
        valid_indices = None
    
    all_scores = []
    all_indices = []
    
    for model_idx, (index, weight) in enumerate(zip(faiss_indices, weights)):
        query_emb = question_embs[model_idx][question_id].reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        if valid_indices is not None:
            # Create a subset index for filtering
            subset_embeddings = np.array([index.reconstruct(i) for i in valid_indices])
            
            # Create temporary index for subset
            d = subset_embeddings.shape[1]
            temp_index = faiss.IndexFlatIP(d)
            temp_index.add(subset_embeddings.astype('float32'))
            
            # Search in subset
            scores, subset_indices = temp_index.search(query_emb, min(top_k, len(valid_indices)))
            
            # Map back to original indices
            original_indices = np.array([valid_indices[i] for i in subset_indices[0]])
            all_scores.append(weight * scores[0])
            all_indices.append(original_indices)
        else:
            # Search in full index
            scores, indices = index.search(query_emb, top_k)
            all_scores.append(weight * scores[0])
            all_indices.append(indices[0])
    
    # Combine scores from all models
    min_length = min(len(scores) for scores in all_scores)
    combined_scores = np.zeros(min_length)
    for scores in all_scores:
        combined_scores += scores[:min_length]
    
    # Use indices from first model (truncated to min_length)
    return combined_scores, all_indices[0][:min_length]

def get_related_law_ids(question, law_id_mapping, doc_refers, top_laws=5):
    """
    Get potentially relevant law IDs based on keyword matching or other heuristics.
    This is a simple implementation - can be enhanced with more sophisticated methods.
    
    Args:
        question (str): The question text
        law_id_mapping (dict): Mapping from law_id to document indices
        doc_refers (list): Document references
        top_laws (int): Number of top law IDs to return
    
    Returns:
        list: List of potentially relevant law IDs
    """
    # Simple keyword-based approach
    question_lower = question.lower()
    law_scores = {}
    
    for law_id in law_id_mapping:
        # Count keyword matches in this law's documents
        score = 0
        for doc_idx in law_id_mapping[law_id]:
            doc_text = doc_refers[doc_idx][2].lower()
            # Simple scoring based on common words
            common_words = set(question_lower.split()) & set(doc_text.split())
            score += len(common_words)
        
        law_scores[law_id] = score
    
    # Return top scoring law IDs
    sorted_laws = sorted(law_scores.items(), key=lambda x: x[1], reverse=True)
    return [law_id for law_id, score in sorted_laws[:top_laws] if score > 0]

def faiss_search(faiss_indices, question_embs, question_id, top_k=2000, weights=None):
    if weights is None:
        weights = [1.0] * len(faiss_indices)
    all_scores = []
    all_indices = []
    for model_idx, (index, weight) in enumerate(zip(faiss_indices, weights)):
        query_emb = question_embs[model_idx][question_id].reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)
        all_scores.append(weight * scores[0])
        all_indices.append(indices[0])
    combined_scores = np.zeros(len(all_indices[0]))
    for scores in all_scores:
        combined_scores += scores
    return combined_scores, all_indices[0]

def evaluation(args, data, models, emb_legal_data, bm25, doc_refers, question_embs, range_score, fixed_scores = 10, reranker = None, tokenizer = None, others = None, save_output=False):
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    with open(os.path.join(args.raw_data, "queries.json"), "r") as f:
        queries = json.load(f)
    qid_list = list(queries.keys())
    random.seed(42)
    random.shuffle(qid_list)
    num_eval = int(len(qid_list) * args.eval_size)
    eval_qid = qid_list[:num_eval]
    k = num_eval
    results = []
    
    for idx, item in tqdm(enumerate(data), total=len(data)):
        question_id = item["question_id"]
        if question_id not in eval_qid:
            continue
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        weighted = [args.model_1_weight, args.model_2_weight, args.model_3_weight] 
        cos_sim = []

        for idx_2, _ in enumerate(models):
            emb1 = question_embs[idx_2][question_id]
            emb2 = emb_legal_data[idx_2]
            scores = util.cos_sim(emb1, emb2)
            cos_sim.append(weighted[idx_2] * scores)
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        if args.hybrid:
            tokenized_query = bm25_tokenizer(question)
            doc_scores = bm25.get_scores(tokenized_query)
            new_scores = combine_scores(cos_sim, doc_scores, combine_type=args.combine_type, alpha=args.alpha)
        else:
            new_scores = cos_sim
        
        max_score = np.max(new_scores)
        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        new_scores = new_scores[predictions]
        
        new_predictions = np.where(new_scores >= (max_score - (range_score if reranker is None else fixed_scores)))[0]
        map_ids = predictions[new_predictions]
        new_scores = new_scores[new_scores >= (max_score - (range_score if reranker is None else fixed_scores))]
        if reranker is not None and len(map_ids) > 1:
            rerank_scores = []
            if len(map_ids) > 100:
                num_chunks = len(map_ids) // 100 + 1
                rerank_scores = []
                for i in range(num_chunks):
                    chunk_ids = map_ids[i * 100: (i + 1) * 100]
                    if len(chunk_ids) == 0:
                        continue
                    rerank_scores.extend(reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in chunk_ids], others))
            else:
                rerank_scores = reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in map_ids], others)
            max_rerank_score = np.max(rerank_scores)
            new_predictions = np.where(rerank_scores >= (max_rerank_score - range_score))[0]
            map_ids = map_ids[new_predictions]
            new_scores = new_scores[rerank_scores >= (max_rerank_score - range_score)]
        true_positive = 0
        false_positive = 0
        saved = {"question_id": question_id, "question": question, "predictions": [], "scores": []}
        
        if len(map_ids) > 1:
            predictions = [{"law_id": doc_refers[i][0], "article_id": doc_refers[i][1], "text": doc_refers[i][2]} for i in map_ids]
            prompt = f"Question: {question}\n{predictions}\nThis is outputs from a legal document retrieval system. Remove any irrelevant information and return a list of true predictions in the format: [{{'law_id': '...', 'article_id': '...'}}]. Only include predictions that are relevant to the question and help to answer the question."
            clean_text = llm_final_output.llm_generate(prompt)
            output = ast.literal_eval(clean_text)
            saved["predictions"] = [{"law_id": pred["law_id"], "article_id": pred["article_id"], "text": doc_refers[doc_refers.index((pred["law_id"], pred["article_id"]))][2]} for pred in output]
            
            for pred in output:
                is_match = False
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                        is_match = True
                        break
                
                if not is_match:
                    false_positive += 1
        else:
            for idx, idx_pred in enumerate(map_ids):
                pred = doc_refers[idx_pred]
                saved["predictions"].append({"law_id": pred[0], "article_id": pred[1], "text": pred[2]})
                is_match = False
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                        is_match = True
                        break
                
                if not is_match:
                    false_positive += 1
        saved["ground_truth"] = relevant_articles  
        results.append(saved)
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    if save_output:
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("Results saved to eval_results.json")
    avg_f2 = total_f2 / k
    avg_precision = total_precision / k
    avg_recall = total_recall / k
    return avg_f2, avg_precision, avg_recall

def evaluation_with_faiss(args, data, models, faiss_indices, bm25, doc_refers, question_embs, range_score, fixed_scores=10, reranker=None, tokenizer=None, others=None, save_output=False):
    """Evaluation function using FAISS for efficient similarity search."""
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    # Load or build law ID mapping
    law_id_mapping_path = os.path.join(args.faiss_path, "law_id_mapping.pkl")
    if os.path.exists(law_id_mapping_path):
        law_id_mapping = load_law_id_mapping(law_id_mapping_path)
    else:
        law_id_mapping = build_law_id_mapping(doc_refers)
        save_law_id_mapping(law_id_mapping, law_id_mapping_path)
    
    with open(os.path.join(args.raw_data, "queries.json"), "r") as f:
        queries = json.load(f)
    qid_list = list(queries.keys())
    random.seed(42)
    random.shuffle(qid_list)
    num_eval = int(len(qid_list) * args.eval_size)
    eval_qid = qid_list[:num_eval]
    k = num_eval
    results = []
    
    for idx, item in tqdm(enumerate(data), total=len(data)):
        question_id = item["question_id"]
        if question_id not in eval_qid:
            continue
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        weighted = [args.model_1_weight, args.model_2_weight, args.model_3_weight]
        
        # Get law ID filter if enabled
        law_ids_filter = None
        if hasattr(args, 'use_law_filter') and args.use_law_filter:
            if hasattr(args, 'target_law_ids') and args.target_law_ids:
                # Use specified law IDs
                law_ids_filter = args.target_law_ids
            else:
                # Auto-detect relevant law IDs
                law_ids_filter = get_related_law_ids(question, law_id_mapping, doc_refers, args.top_laws)
            print(f"Filtering search to laws: {law_ids_filter}")
        
        # Use FAISS search with optional law filtering
        if law_ids_filter is not None:
            cos_sim, predictions = faiss_search_with_law_filter(
                faiss_indices, question_embs, question_id, law_id_mapping,
                law_ids_filter, top_k=2000, weights=weighted[:len(faiss_indices)]
            )
        else:
            cos_sim, predictions = faiss_search(
                faiss_indices, question_embs, question_id, 
                top_k=2000, weights=weighted[:len(faiss_indices)]
            )
        
        if args.hybrid:
            tokenized_query = bm25_tokenizer(question)
            doc_scores = bm25.get_scores(tokenized_query)
            bm25_scores_subset = doc_scores[predictions]
            new_scores = combine_scores(cos_sim, bm25_scores_subset, combine_type=args.combine_type, alpha=args.alpha)
        else:
            new_scores = cos_sim
        
        max_score = np.max(new_scores)
        new_predictions = np.where(new_scores >= (max_score - (range_score if reranker is None else fixed_scores)))[0]
        map_ids = predictions[new_predictions]
        new_scores = new_scores[new_scores >= (max_score - (range_score if reranker is None else fixed_scores))]
        
        # ...existing reranking and evaluation logic...
        if reranker is not None and len(map_ids) > 1:
            rerank_scores = []
            if len(map_ids) > 100:
                num_chunks = len(map_ids) // 100 + 1
                rerank_scores = []
                for i in range(num_chunks):
                    chunk_ids = map_ids[i * 100: (i + 1) * 100]
                    if len(chunk_ids) == 0:
                        continue
                    rerank_scores.extend(reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in chunk_ids], others))
            else:
                rerank_scores = reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in map_ids], others)
            max_rerank_score = np.max(rerank_scores)
            new_predictions = np.where(rerank_scores >= (max_rerank_score - range_score))[0]
            map_ids = map_ids[new_predictions]
            new_scores = new_scores[rerank_scores >= (max_rerank_score - range_score)]
        
        true_positive = 0
        false_positive = 0
        saved = {"question_id": question_id, "question": question, "predictions": [], "scores": []}
        
        if len(map_ids) > 1:
            predictions = [{"law_id": doc_refers[i][0], "article_id": doc_refers[i][1], "text": doc_refers[i][2]} for i in map_ids]
            prompt = f"Question: {question}\n{predictions}\nThis is outputs from a legal document retrieval system. Remove any irrelevant information and return a list of true predictions in the format: [{{'law_id': '...', 'article_id': '...'}}]. Only include predictions that are relevant to the question and help to answer the question."
            clean_text = llm_final_output.llm_generate(prompt)
            output = ast.literal_eval(clean_text)
            saved["predictions"] = [{"law_id": pred["law_id"], "article_id": pred["article_id"], "text": doc_refers[doc_refers.index((pred["law_id"], pred["article_id"]))][2]} for pred in output]
            
            for pred in output:
                is_match = False
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                        is_match = True
                        break
                
                if not is_match:
                    false_positive += 1
        else:
            for idx, idx_pred in enumerate(map_ids):
                pred = doc_refers[idx_pred]
                saved["predictions"].append({"law_id": pred[0], "article_id": pred[1], "text": pred[2]})
                is_match = False
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                        is_match = True
                        break
                
                if not is_match:
                    false_positive += 1
        
        saved["ground_truth"] = relevant_articles
        saved["filtered_laws"] = law_ids_filter  
        results.append(saved)
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    
    if save_output:
        with open("eval_results_faiss.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("Results saved to eval_results_faiss.json")
    
    avg_f2 = total_f2 / k
    avg_precision = total_precision / k
    avg_recall = total_recall / k
    return avg_f2, avg_precision, avg_recall

def grid_search(args, data, models, emb_legal_data, bm25, doc_refers, question_embs):
    results = []
    range_scores_list = [0.0, 1.0, 2.0]
    fixed_scores_list = {
        "default": [10, 15],
        "weighted_sum": [0.05, 0.08, 0.1],
        "rrf": [0.001, 0.005, 0.01]
    }
    alphas = [0.3, 0.5, 0.7]
    for combine_type in tqdm(combine_types):
        for range_score in tqdm(range_scores_list, desc=f"Processing range_score"):
            try:
                args.model_1_weight = 0.5
                args.model_2_weight = 0.5
                args.range_score = range_score
                args.combine_type = combine_type
                args.alpha = 0
                if combine_type == "weighted_sum":
                    for fixed_score in fixed_scores_list[combine_type]:
                        print(f"Evaluating with range_score={range_score}, fixed_score={fixed_score}, combine_type={combine_type}")
                        for alpha in alphas:
                            args.alpha = alpha
                            avg_f2, avg_precision, avg_recall = evaluation(
                                args,
                                data,
                                models,
                                emb_legal_data,
                                bm25,
                                doc_refers,
                                question_embs,
                                range_score,
                                fixed_score,
                                reranker,
                                tokenizer,
                                others
                            )
                            result_row = {
                                "range_score": range_score,
                                "fixed_score": fixed_score,
                                "combine_type": combine_type,
                                "alpha": alpha,
                                "avg_f2": avg_f2,
                                "avg_precision": avg_precision,
                                "avg_recall": avg_recall
                            }
                            results.append(result_row)
                else:
                    for fixed_score in fixed_scores_list[combine_type]:
                        print(f"Evaluating with range_score={range_score}, fixed_score={fixed_score}, combine_type={combine_type}")
                        
                        avg_f2, avg_precision, avg_recall = evaluation(
                            args,
                            data,
                            models,
                            emb_legal_data,
                            bm25,
                            doc_refers,
                            question_embs,
                            range_score,
                            fixed_score,
                            reranker,
                            tokenizer,
                            others
                        )

                        result_row = {
                            "range_score": range_score,
                            "fixed_score": fixed_score,
                            "combine_type": combine_type,
                            "alpha": 0,
                            "avg_f2": avg_f2,
                            "avg_precision": avg_precision,
                            "avg_recall": avg_recall
                        }
                        results.append(result_row)
            except Exception as e:
                print(f"Error in combination: {e}")
                print(f"Skipping combination: range_score={range_score}, fixed_score={fixed_score}, combine_type={combine_type}")

    with open("grid_search_results.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Grid search complete. Results saved to grid_search_results.csv.")
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="ALQAC_2025_data", type=str)
    parser.add_argument("--saved_model", default="saved_model", type=str)
    parser.add_argument("--reranker", default="", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str, help="path to legal corpus for reference")
    parser.add_argument("--range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--fixed-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    parser.add_argument("--combine-type", default="default", type=str, help="number of eval data")
    parser.add_argument("--model_1_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_2_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_3_weight", default=0.0, type=float, help="number of eval data")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    parser.add_argument("--encode_question_data", action="store_true", help="for question data encoding")
    parser.add_argument("--hybrid", action="store_true", help="for legal data encoding")
    parser.add_argument("--find-best-score", action="store_true", help="for legal data encoding")
    parser.add_argument("--step", default=0.1, type=float, help="number of eval data")
    parser.add_argument("--alpha", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--use_faiss", action="store_true", help="use FAISS for efficient similarity search")
    parser.add_argument("--build_faiss", action="store_true", help="build and save FAISS index")
    parser.add_argument("--faiss_index_type", default="IVF", type=str, help="FAISS index type (Flat, IVF, HNSW)")
    parser.add_argument("--faiss_path", default="faiss_indices", type=str, help="path to save/load FAISS indices")
    parser.add_argument("--use_law_filter", action="store_true", help="filter search space by law IDs")
    parser.add_argument("--target_law_ids", nargs="+", help="specific law IDs to search within")
    parser.add_argument("--top_laws", default=5, type=int, help="number of top laws to consider when auto-filtering")
    
    args = parser.parse_args()

    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    model_names = ["phonghoccode/ALQAC_2025_Embedding_top50_round1", "phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg"]

    reranker_name = args.reranker if hasattr(args, 'reranker') else None
    if reranker_name:
        reranker, tokenizer, others = create_reranker(reranker_name)
        print(f"Using reranker: {reranker_name}")
    else:
        reranker = None
        tokenizer = None
        others = None
        print("No reranker specified, using default settings.")

    question_items = load_question_json(args.raw_data)
    train_path = os.path.join(args.raw_data, "alqac25_train.json")
    data = json.load(open(train_path))
    print("Number of questions: ", len(question_items))
    models = []
    bm25 = load_bm25(args.bm25_path)
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    if args.encode_legal_data:
        print("Start loading model.")
        models = [SentenceTransformer(name) for name in model_names]
        wseg = [("wseg" in name) for name in model_names]
        print("Number of pretrained models: ", len(models))
        emb_legal_data = encode_legal_data(args.raw_data, models, wseg)
    else:
        emb_legal_data = load_encoded_legal_corpus('encoded_legal_data.pkl')
    if args.encode_question_data:
        if len(models) == 0:
            print("Start loading model.")
            models = [SentenceTransformer(name) for name in model_names]
            wseg = [("wseg" in name) for name in model_names]
            print("Number of pretrained models: ", len(models))
        question_embs = encode_question(question_items, models, wseg)
    else:
        question_embs = load_encoded_question_data("encoded_question_data.pkl")
    
    faiss_indices = None
    if args.use_faiss or args.build_faiss:
        if args.build_faiss:
            print("Building FAISS indices...")
            faiss_indices = build_faiss_index(emb_legal_data, index_type=args.faiss_index_type)
            save_faiss_index(faiss_indices, args.faiss_path)
        else:
            print("Loading FAISS indices...")
            faiss_indices = load_faiss_index(args.faiss_path, len(model_names))
    
    top_n = 2000
    range_score = args.range_score
    fixed_score = args.fixed_score
    pred_list = []
    if args.find_best_score:
        print("Start finding best score.")
        results = grid_search(args, data, model_names, emb_legal_data, bm25, doc_refers, question_embs)
    else:
        if args.use_faiss and faiss_indices is not None:
            print("Using FAISS for evaluation...")
            avg_f2, avg_precision, avg_recall = evaluation_with_faiss(args, data, model_names, faiss_indices, bm25, doc_refers, question_embs, range_score, fixed_score, reranker, tokenizer, others, True)
        else:
            print("Using traditional cosine similarity for evaluation...")
            avg_f2, avg_precision, avg_recall = evaluation(args, data, model_names, emb_legal_data, bm25, doc_refers, question_embs, range_score, fixed_score, reranker, tokenizer, others, True)
    
        print(f"Average F2: \t\t\t\t{avg_f2}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}\n")