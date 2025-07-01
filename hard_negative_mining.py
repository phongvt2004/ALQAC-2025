import pickle
import os
import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import warnings
import utils
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_bert_path", default="", type=str, help="path to round 1 sentence bert model")
    parser.add_argument("--data_path", default="ALQAC_2025_data", type=str, help="path to input data")
    parser.add_argument("--save_path", default="pair_data", type=str)
    parser.add_argument("--top_k", default=20, type=str, help="top k hard negative mining")
    parser.add_argument("--encode_legal_data", action="store_true", help="for legal data encoding")
    parser.add_argument("--path_doc_refer", default="generated_data/doc_refers_saved", type=str, help="path to doc refers")
    parser.add_argument("--saved_model_path", default="saved_model", type=str, help="path to saved model data")
    
    parser.add_argument("--path_legal", default="generated_data/legal_dict.json", type=str, help="path to legal dict")
    args = parser.parse_args()

    # load training data from json
    data = json.load(open(os.path.join(args.data_path, "alqac25_train.json")))

    training_data = data
    print(len(training_data))

    with open(os.path.join(args.saved_model_path, "doc_refers_saved"), "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = {}
    
    # load hard negative model
    model = SentenceTransformer(args.sentence_bert_path)

    # add embedding for data
    # if you already have data with encoded sentence uncoment line 47 - 54
    corpus = json.load(open(os.path.join(args.data_path, "corpus.json")))
    
    if args.encode_legal_data:
        
        import pickle
        embed_list = []
        for k, v in tqdm(corpus.items()):
            if "wseg" in args.sentence_bert_path:
                doc = utils.word_segmentation(v)
            else:
                doc = v
            embed = model.encode(doc, show_progress_bar=False)
            doc_data[k] = embed
        # print(len(doc_data))
        list_emb_models = []

        with open('legal_corpus_embedding.pkl', 'wb') as pkl:
            pickle.dump(doc_data, pkl)
    with open('legal_corpus_embedding.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    pred_list = []
    top_k = args.top_k
    save_pairs = []

    for idx, item in tqdm(enumerate(training_data)):
        question_id = item["question_id"]
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + " " + article["article_id"]
            save_dict["document"] = corpus[concat_id]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
        if "wseg" in args.sentence_bert_path:
            question = utils.word_segmentation(question)
        encoded_question  = model.encode(question)
        list_embs = []

        for k, v in data.items():
            emb_2 = torch.tensor(v).unsqueeze(0)
            list_embs.append(emb_2)

        matrix_emb = torch.cat(list_embs, dim=0)
        all_cosine = util.cos_sim(encoded_question, matrix_emb).numpy().squeeze(0)
        predictions = np.argpartition(all_cosine, len(all_cosine) - top_k)[-top_k:]
        
        
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
                
            check = 0
            for article in relevant_articles:
                check += 1 if pred[0] == article["law_id"] and pred[1] == article["article_id"] else 0

            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + " " + pred[1]
                save_dict["document"] = corpus[concat_id]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"save_pairs_sbert_top{top_k}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)