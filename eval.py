import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer

from sentence_transformers import SentenceTransformer, util

def encode_legal_data(legal_dict_json, model):
    # print(legal_dict_json)
    doc_data = json.load(open(legal_dict_json))
    # print(len(doc_data))
    list_emb_model = []
    for model in model:
        emb2_list = []
        for k, doc in tqdm(doc_data.items()):
            emb2 = model.encode(doc_data[k]["title"] + " " + doc_data[k]["text"])
            emb2_list.append(emb2)
        emb2_arr = np.array(emb2_list)
        list_emb_model.append(emb2_arr)
    return list_emb_model

def encode_question(question_data, model):
    print("Start encoding questions.")
    question_embs = []
    for model in model:
        emb_quest_dict = {}
        for _, item in tqdm(enumerate(question_data)):
            question_id = item["question_id"]
            question = item["question"]
            emb_quest_dict[question_id] = model.encode(question)
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

def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def load_question_json(question_path):
    question_path = glob.glob(f"{question_path}/*.json")[0]
    question_data = json.load(open(question_path))
    return question_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", type=str, help="for loading question")
    parser.add_argument("--raw_data", default="ALQAC_2025_data", type=str)
    parser.add_argument("--saved_model", default="saved_model", type=str)
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str, help="path to legal corpus for reference")
    parser.add_argument("--range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    args = parser.parse_args()
    print("Start loading model.")
    model = load_model(args.model_name)

    # load question from json file
    question_items = load_question_json(args.data)["items"]
    
    print("Number of questions: ", len(question_items))
    
    # load bm25 model 
    bm25 = load_bm25(args.bm25_path)
    # load corpus to search
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    # load pre encoded for legal corpus
    if args.encode_legal_data:
        emb_legal_data = encode_legal_data(args.legal_dict_json, model)
    else:
        emb_legal_data = load_encoded_legal_corpus('encoded_legal_data.pkl')

    # encode question for query
    question_embs = encode_question(question_items, model)

    # define top n for compare and range of score
    top_n = 61425
    range_score = args.range_score

    pred_list = []

    print("Start calculating results.")
    for idx, item in tqdm(enumerate(question_items)):
        question_id = item["question_id"]
        question = item["question"]

        weighted = [0.1, 0.3, 0.4, 0.2] 
        cos_sim = []
        emb1 = question_embs[question_id]
        emb2 = emb_legal_data
        cos_sim = util.cos_sim(emb1, emb2)
        max_score = np.max(cos_sim)

        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        new_scores = new_scores[predictions]
        
        new_predictions = np.where(new_scores >= (max_score - range_score))[0]
        map_ids = predictions[new_predictions]
        new_scores = new_scores[new_scores >= (max_score - range_score)]

        if new_scores.shape[0] > 5:
            predictions_2 = np.argpartition(new_scores, len(new_scores) - 5)[-5:]
            map_ids = map_ids[predictions_2]
            
        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["relevant_articles"] = []
        
        # post processing character error
        dup_ans = []
        for idx, idx_pred in enumerate(map_ids):
            pred = doc_refers[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id.endswith("nd-cp"):
                law_id = law_id.replace("nd-cp", "nđ-cp")
            if law_id.endswith("nđ-"):
                law_id = law_id.replace("nđ-", "nđ-cp")
            if law_id.endswith("nð-cp"):
                law_id = law_id.replace("nð-cp", "nđ-cp")
            if law_id == "09/2014/ttlt-btp-tandtc-vksndtc":
                law_id = "09/2014/ttlt-btp-tandtc-vksndtc-btc"
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append({"law_id": law_id, "article_id": article_id})
        pred_list.append(pred_dict)
    
    # extract result
    with open('/result/submission.json', 'w') as outfile:
        json.dump(pred_list, outfile)