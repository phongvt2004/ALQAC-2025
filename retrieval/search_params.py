
import json
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
from reranker import create_reranker
import itertools
import csv
from eval import evaluation

load_dotenv()

range_scores_list = [0.0, 1.0, 2.0, 3.0, 4.0]
fixed_scores_list = [5, 10, 15]
model_1_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
model_2_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
combine_types = ["default", "weighted_sum", "rrf"]
alphas = [0.3, 0.5, 0.7]


def grid_search(args, data, models, emb_legal_data, bm25, doc_refers, question_embs):
    # Prepare result logging
    results = []
    search_space = list(itertools.product(
        range_scores_list,
        fixed_scores_list,
        model_1_weights,
        model_2_weights,
        combine_types,
        alphas
    ))

    print(f"Total combinations: {len(search_space)}")
    for idx, (range_score, fixed_score, w1, w2, combine_type, alpha) in tqdm(enumerate(search_space), total=len(search_space)):
        if w1 + w2 != 1.0:
            continue  # Skip combinations where weights do not sum to 1.0
        print(f"Combination {idx + 1}/{len(search_space)}:")
        args.model_1_weight = w1
        args.model_2_weight = w2
        args.range_score = range_score
        args.fixed_score = fixed_score
        args.combine_type = combine_type
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
            "model_1_weight": w1,
            "model_2_weight": w2,
            "combine_type": combine_type,
            "alpha": alpha,
            "avg_f2": avg_f2,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall
        }
        results.append(result_row)

        # Optional: Print result for each iteration
        print(result_row)

        # Save intermediate results after each run
        with open("grid_search_results.csv", "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result_row.keys())
            if idx == 0:
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