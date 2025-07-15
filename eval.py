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


load_dotenv()

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
    return question_embs

def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    with open(legal_data_path, "rb") as f1:
        emb_legal_data = pickle.load(f1)
    return emb_legal_data

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

def evaluation(args, data, models, emb_legal_data, bm25, doc_refers, question_embs, range_score, reranker, tokenizer, others):
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
    for idx, item in tqdm(enumerate(data)):
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
            new_scores = doc_scores * cos_sim
        else:
            new_scores = cos_sim
        
        max_score = np.max(new_scores)
        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        new_scores = new_scores[predictions]
        
        new_predictions = np.where(new_scores >= (max_score - range_score if reranker is None else 2.6))[0]
        map_ids = predictions[new_predictions]
        new_scores = new_scores[new_scores >= (max_score - range_score if reranker is None else 2.6)]
        print(len(map_ids))
        if reranker is not None:
            rerank_scores = reranking(reranker, tokenizer, question, [doc_refers[i][2] for i in map_ids], others)
            max_rerank_score = np.max(rerank_scores)
            new_predictions = np.where(rerank_scores >= (max_rerank_score - range_score))[0]
            map_ids = map_ids[new_predictions]
            new_scores = new_scores[rerank_scores >= (max_rerank_score - range_score)]
        # if new_scores.shape[0] > 5:
        #     predictions_2 = np.argpartition(new_scores, len(new_scores) - 5)[-5:]
        #     map_ids = map_ids[predictions_2]
        true_positive = 0
        false_positive = 0
        
        # post processing character error
        for idx, idx_pred in enumerate(map_ids):
            pred = doc_refers[idx_pred]
            
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    true_positive += 1
                else:
                    false_positive += 1
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    avg_f2 = total_f2 / k
    avg_precision = total_precision / k
    avg_recall = total_recall / k
    return avg_f2, avg_precision, avg_recall

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="ALQAC_2025_data", type=str)
    parser.add_argument("--saved_model", default="saved_model", type=str)
    parser.add_argument("--reranker", default="", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str, help="path to legal corpus for reference")
    parser.add_argument("--range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    parser.add_argument("--model_1_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_2_weight", default=0.5, type=float, help="number of eval data")
    parser.add_argument("--model_3_weight", default=0.0, type=float, help="number of eval data")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    parser.add_argument("--hybrid", action="store_true", help="for legal data encoding")
    parser.add_argument("--find-best-score", action="store_true", help="for legal data encoding")
    parser.add_argument("--step", default=0.1, type=float, help="number of eval data")
    
    args = parser.parse_args()

    # define path to model
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    model_names = ["phonghoccode/ALQAC_2025_Embedding_top50_round1", "phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg", "phonghoccode/ALQAC_2025_Qwen3_Embedding_top50"]
    # model_names = ["phonghoccode/ALQAC_2025_Embedding_top50_round1", "phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg"]

    print("Start loading model.")
    models = [SentenceTransformer(name) for name in model_names]
    wseg = [("wseg" in name) for name in model_names]
    print("Number of pretrained models: ", len(models))
    
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
    train_path = os.path.join(args.raw_data, "alqac25_train.json")
    data = json.load(open(train_path))
    print("Number of questions: ", len(question_items))
    
    # load bm25 model 
    bm25 = load_bm25(args.bm25_path)
    # load corpus to search
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    # load pre encoded for legal corpus
    if args.encode_legal_data:
        emb_legal_data = encode_legal_data(args.raw_data, models, wseg)
    else:
        emb_legal_data = load_encoded_legal_corpus('encoded_legal_data.pkl')

    # encode question for query
    question_embs = encode_question(question_items, models, wseg)

    # define top n for compare and range of score
    top_n = 2000
    range_score = args.range_score

    pred_list = []
    if args.find_best_score:
        print("Start finding best score.")
        if args.hybrid:
            min_score = 1.0
            max_score = 6.0
        else:
            min_score = 0.0
            max_score = 1.0
        best_score_f2 = 0.0
        best_score_precision = 0.0
        best_score_recall = 0.0
        best_f2 = 0.0
        best_f2_precision = 0.0
        best_f2_recall = 0.0
        best_precision = 0.0
        best_recall = 0.0
        for i in np.arange(min_score, max_score, args.step):
            range_score = i
            avg_f2, avg_precision, avg_recall = evaluation(args, data, models, emb_legal_data, bm25, doc_refers, question_embs, range_score, reranker, tokenizer, others)
            # print(f"Score: {i}, F2: {avg_f2}, Precision: {avg_precision}, Recall: {avg_recall}")
            if best_f2 < avg_f2:
                best_f2 = avg_f2
                best_f2_precision = avg_precision
                best_f2_recall = avg_recall
                best_score_f2 = i
            if best_precision < avg_precision:
                best_precision = avg_precision
                best_score_precision = i
            if best_recall < avg_recall:
                best_recall = avg_recall
                best_score_recall = i
        avg_f2 = best_f2
        avg_precision = best_f2_precision
        avg_recall = best_f2_recall
        print(f"Best F2 score: {best_score_f2}, Best F2: {best_f2}")
        print(f"Best Precision score: {best_score_precision}, Best Precision: {best_precision}")
        print(f"Best Recall score: {best_score_recall}, Best Recall: {best_recall}")
    else:
        avg_f2, avg_precision, avg_recall = evaluation(args, data, models, emb_legal_data, bm25, doc_refers, question_embs, reranker, tokenizer, others)
    
    print(f"Average F2: \t\t\t\t{avg_f2}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}\n")