import os
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, load_json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=20, type=int)
    parser.add_argument("--model_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--zalo", action="store_true", help="use Zalo data")
    parser.add_argument("--rerank", action="store_true", help="use data for rerank")
    parser.add_argument("--data_path", default="../ALQAC_2025_data", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="pair_data/", type=str, help="path to save pair sentence directory")
    parser.add_argument("--saved_model_path", default="saved_model", type=str, help="path to saved model data")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "alqac25_train.json" if not args.zalo else "zalo_question.json")
    print(train_path)
    data = json.load(open(train_path))

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open(os.path.join(args.saved_model_path, "doc_refers_saved"), "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    corpus = json.load(open(os.path.join(args.data_path, "corpus.json")))
    save_pairs = []
    top_n = args.top_pair
    if args.rerank:
        with open(os.path.join(args.data_path, "queries.json"), "r") as f:
            queries = json.load(f)
        qid_list = list(queries.keys())
        random.seed(42)
        random.shuffle(qid_list)
        num_eval = int(len(qid_list) * args.eval_size)
        eval_qid = qid_list[:num_eval]
        for idx, item in tqdm(enumerate(data)):
            question_id = item["question_id"] if not args.zalo else item["id"]
            if question_id in eval_qid:
                continue
            question = item["text"]
            relevant_articles = item["relevant_articles"]
            actual_positive = len(relevant_articles)
            
            tokenized_query = bm25_tokenizer(question)
            doc_scores = bm25.get_scores(tokenized_query)

            predictions = np.argpartition(doc_scores, len(doc_scores) - (top_n+10))[-(top_n+10):]

            # Save positive pairs
            save_dict = {}
            save_dict["query"] = question
            save_dict["pos"] = []
            save_dict["neg"] = []
            for article in relevant_articles:

                concat_id = article["law_id"] + " " + article["article_id"]
                save_dict["pos"].append(corpus[concat_id])

            # Save negative pairs
            for idx, idx_pred in enumerate(predictions):
                if idx >= top_n:
                    break
                pred = doc_refers[idx_pred]

                check = 0
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        check += 1

                if check == 0:
                    concat_id = pred[0] + " " + pred[1]
                    save_dict["neg"].append(corpus[concat_id])
            save_pairs.append(save_dict)
        save_path = args.save_pair_path
        os.makedirs(save_path, exist_ok=True)
        print(f"Number of pairs: {len(save_pairs)}")
        with open(os.path.join(save_path, f"rerank_data_top{top_n}.jsonl"), "w") as f:
            for item in save_pairs:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")
    else:
        for idx, item in tqdm(enumerate(data)):
            question_id = item["question_id"] if not args.zalo else item["id"]
            question = item["text"]
            relevant_articles = item["relevant_articles"]
            actual_positive = len(relevant_articles)
            
            tokenized_query = bm25_tokenizer(question)
            doc_scores = bm25.get_scores(tokenized_query)

            predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]

            # Save positive pairs
            for article in relevant_articles:
                save_dict = {}
                save_dict["qid"] = question_id
                save_dict["question"] = question
                concat_id = article["law_id"] + " " + article["article_id"]
                save_dict["document"] = corpus[concat_id]
                save_dict["relevant"] = 1
                save_pairs.append(save_dict)

            # Save negative pairs
            for idx, idx_pred in enumerate(predictions):
                pred = doc_refers[idx_pred]

                check = 0
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        check += 1
    
                if check == 0:
                    save_dict = {}
                    save_dict["qid"] = question_id
                    save_dict["question"] = question
                    concat_id = pred[0] + " " + pred[1]
                    save_dict["document"] = corpus[concat_id]
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)
                    
        save_path = args.save_pair_path
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"bm_25_pairs_top{top_n}"), "wb") as pair_file:
            pickle.dump(save_pairs, pair_file)
        print(len(save_pairs))