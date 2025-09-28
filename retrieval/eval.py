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
# import llm_system
load_dotenv()
combine_types = ["weighted_sum", "rrf", "default"]

def encode_legal_data(data_path, models, wseg):
    # print(legal_dict_json)
    corpus = json.load(open(os.path.join(data_path, "corpus.json")))
    # print(len(doc_data))
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
    """
    Combine dense scores and BM25 scores using a weighted sum.
    
    Args:
        dense_scores (np.ndarray): Dense model scores.
        bm25_scores (np.ndarray): BM25 model scores.
        alpha (float): Weight for the dense scores.
    
    Returns:
        np.ndarray: Combined scores.
    """
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


def get_law_by_llm(questions, laws):
    prompt = f"Given the following laws: {laws} and a question from user: {question}, please return a list of laws that are relevant to these questions. If there are any Luật (law) mention in the question use it. The laws should be in the format: [\"law_1\', \"law_2\"].  Only return if you sure 100% if not return []."
    response = llm_system.llm_generate(prompt)
    output = ast.literal_eval(response)
    return output

def evaluation(args, data, models, emb_legal_data, bm25, doc_refers, question_embs, rerank_range_score, retrieve_range_scores = 10, reranker = None, tokenizer = None, others = None, save_output=False):
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    laws = []
    law_mapping = {}
    if os.path.exists(os.path.join(args.raw_data, "pre_laws.json")):
        print("Loading pre_laws.json")
        with open(os.path.join(args.raw_data, "pre_laws.json"), "r") as f:
            law_mapping = json.load(f)
    with open(os.path.join(args.raw_data, "alqac25_law.json"), "r") as f:
        corpus = json.load(f)
    for item in corpus:
        laws.append(item["id"])
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
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        weighted = [args.model_1_weight, args.model_2_weight, args.model_3_weight] 
        relevant_laws = law_mapping[question_id]
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
        
        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        # Fix: Filter predictions by relevant laws first
        filtered_predictions = []
        for pred_idx in predictions:
            pred = doc_refers[pred_idx]
            if pred[0] in relevant_laws:
                filtered_predictions.append(pred_idx)
        
        # Check if we have any valid predictions
        if len(filtered_predictions) == 0:
            continue
            
        filtered_predictions = np.array(filtered_predictions)
        filtered_scores = new_scores[filtered_predictions]
        
        # Check for empty scores array
        if len(filtered_scores) == 0:
            continue
            
        max_score = np.max(filtered_scores)
        
        # Fix: Use proper indexing
        score_mask = filtered_scores >= (max_score - (rerank_range_score if reranker is None else retrieve_range_scores))
        final_predictions = filtered_predictions[score_mask]
        final_scores = filtered_scores[score_mask]
        
        if reranker is not None and len(final_predictions) > 1:
            # ...existing reranker code...
            rerank_scores = []
            if len(final_predictions) > 100:
                num_chunks = len(final_predictions) // 100 + 1
                rerank_scores = []
                for i in range(num_chunks):
                    chunk_ids = final_predictions[i * 100: (i + 1) * 100]
                    if len(chunk_ids) == 0:
                        continue
                    rerank_scores.extend(reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in chunk_ids], others))
            else:
                rerank_scores = reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in final_predictions], others)
            
            if len(rerank_scores) > 0:
                max_rerank_score = np.max(rerank_scores)
                rerank_mask = np.array(rerank_scores) >= (max_rerank_score - rerank_range_score)
                final_predictions = final_predictions[rerank_mask]
                final_scores = final_scores[rerank_mask]
        
        true_positive = 0
        false_positive = 0
        saved = {"question_id": question_id, "question": question, "predictions": [], "scores": []}
        if len(final_predictions) > 3:
            mask = np.argpartition(final_scores, len(final_scores) - 3)[-3:]
            final_predictions = final_predictions[mask]
        
        for idx_pred in final_predictions:
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
        print(true_positive, false_positive, actual_positive)
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

def grid_search(args, data, models, emb_legal_data, bm25, doc_refers, question_embs):
    results = []
    # Prepare result logging
    rerank_range_scores_list = [0.0, 1.0, 2.0]
    retrieve_range_scores_list = {
        "default": [10, 15],
        "weighted_sum": [0.05, 0.08, 0.1],
        "rrf": [0.001, 0.005, 0.01]
    }
    alphas = [0.3, 0.5, 0.7]
    for combine_type in tqdm(combine_types):
        for rerank_range_score in tqdm(rerank_range_scores_list, desc=f"Processing rerank_range_score"):
            # try:
            args.model_1_weight = 0.5
            args.model_2_weight = 0.5
            args.rerank_range_score = rerank_range_score
            args.combine_type = combine_type
            args.alpha = 0
            if combine_type == "weighted_sum":
                for retrieve_range_score in retrieve_range_scores_list[combine_type]:
                    # print(f"Evaluating with rerank_range_score={rerank_range_score}, retrieve_range_score={retrieve_range_score}, combine_type={combine_type}")
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
                            rerank_range_score,
                            retrieve_range_score,
                            reranker,
                            tokenizer,
                            others
                        )
                        result_row = {
                            "rerank_range_score": rerank_range_score,
                            "retrieve_range_score": retrieve_range_score,
                            "combine_type": combine_type,
                            "alpha": alpha,
                            "avg_f2": avg_f2,
                            "avg_precision": avg_precision,
                            "avg_recall": avg_recall
                        }
                        results.append(result_row)
            else:
                for retrieve_range_score in retrieve_range_scores_list[combine_type]:
                    # print(f"Evaluating with rerank_range_score={rerank_range_score}, retrieve_range_score={retrieve_range_score}, combine_type={combine_type}")
                    
                    avg_f2, avg_precision, avg_recall = evaluation(
                        args,
                        data,
                        models,
                        emb_legal_data,
                        bm25,
                        doc_refers,
                        question_embs,
                        rerank_range_score,
                        retrieve_range_score,
                        reranker,
                        tokenizer,
                        others
                    )

                    result_row = {
                        "rerank_range_score": rerank_range_score,
                        "retrieve_range_score": retrieve_range_score,
                        "combine_type": combine_type,
                        "alpha": 0,
                        "avg_f2": avg_f2,
                        "avg_precision": avg_precision,
                        "avg_recall": avg_recall
                    }
                    results.append(result_row)
            # except Exception as e:
            #     print(f"Error in combination: {e}")
            #     print(f"Skipping combination: rerank_range_score={rerank_range_score}, retrieve_range_score={retrieve_range_score}, combine_type={combine_type}")

        # Save intermediate results after each run
    with open("grid_search_results.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()  # <- Write header here
        writer.writerows(results)


    print("Grid search complete. Results saved to grid_search_results.csv.")
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./ALQAC_2025_data", type=str)
    parser.add_argument("--saved_model", default="saved_model", type=str)
    parser.add_argument("--reranker", default="", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str, help="path to legal corpus for reference")
    parser.add_argument("--rerank-range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--retrieve-range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    parser.add_argument("--combine-type", default="default", type=str, help="number of eval data")
    parser.add_argument("--model_1_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_2_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_3_weight", default=0.0, type=float, help="number of eval data")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    parser.add_argument("--encode_question_data", action="store_true", help="for question data encoding")
    parser.add_argument("--create_law_mapping", action="store_true", help="for legal data encoding")
    parser.add_argument("--hybrid", action="store_true", help="for legal data encoding")
    parser.add_argument("--find-best-score", action="store_true", help="for legal data encoding")
    parser.add_argument("--step", default=0.1, type=float, help="number of eval data")
    parser.add_argument("--alpha", default=0.5, type=float, help="number of eval data")
    
    args = parser.parse_args()

    # define path to model
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    # model_names = ["phonghoccode/ALQAC_2025_Embedding_top50_round1", "phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg", "phonghoccode/ALQAC_2025_Qwen3_Embedding_top50"]
    model_names = ["phonghoccode/ALQAC_2025_Embedding_final", "phonghoccode/ALQAC_2025_Embedding_final_wseg"]

    
    
    reranker_name = args.reranker if hasattr(args, 'reranker') else None
    if reranker_name:
        reranker, tokenizer, others = create_reranker(reranker_name)
        print(f"Using reranker: {reranker_name}")
    else:
        reranker = None
        tokenizer = None
        others = None
        print("No reranker specified, using default settings.")

    # load question from json file
    question_items = load_question_json(args.raw_data)
    train_path = os.path.join(args.raw_data, "alqac25_private_test_task2.json")
    data = json.load(open(train_path))
    print("Number of questions: ", len(question_items))
    models = []
    # load bm25 model 
    bm25 = load_bm25(args.bm25_path)
    # load corpus to search
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    if args.create_law_mapping:
        
        laws = []
        with open(os.path.join(args.raw_data, "alqac25_law.json"), "r") as f:
            corpus = json.load(f)
        for item in corpus:
            laws.append(item["id"])
        law_mapping = {}
        for idx, item in tqdm(enumerate(data), total=len(data)):
            question_id = item["question_id"]
            
            question = item["text"]
            relevant_laws = get_law_by_llm(question, laws)
            if len(relevant_laws) == 0:
                relevant_laws = laws
            law_mapping[question_id] = relevant_laws
        with open("pre_laws.json", "w") as f:
            json.dump(law_mapping, f, indent=4, ensure_ascii=False)
    # load pre encoded for legal corpus
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
    # define top n for compare and range of score
    top_n = 2000
    rerank_range_score = args.rerank_range_score
    retrieve_range_score = args.retrieve_range_score
    pred_list = []
    if args.find_best_score:
        print("Start finding best score.")
        results = grid_search(args, data, model_names, emb_legal_data, bm25, doc_refers, question_embs)
    else:
        avg_f2, avg_precision, avg_recall = evaluation(args, data, model_names, emb_legal_data, bm25, doc_refers, question_embs, rerank_range_score, retrieve_range_score, reranker, tokenizer, others, True)
    
        print(f"Average F2: \t\t\t\t{avg_f2}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}\n")