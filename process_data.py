import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="ALQAC_2025_data", type=str, help="path to input data")
    args = parser.parse_args()
    with open(os.path.join(args.data_path, "alqac25_law.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(os.path.join(args.data_path, "alqac25_train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    corpus_ids = []
    corpus_text = []
    for item in corpus:
        for article in item["articles"]:
            corpus_ids.append(item["id"] + " " + article["id"])
            corpus_text.append(article["text"])
    corpus = dict(zip(corpus_ids, corpus_text))
    query_ids = []
    query_text = []
    relevant_docs = {}
    for item in train_data:
        qid = item["question_id"]
        query_ids.append(qid)
        query_text.append(item["text"])
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        for article in item["relevant_articles"]:
            corpus_id = article["law_id"] + " " + article["article_id"]
            if corpus_id not in corpus_ids:
                print(corpus_id)
            relevant_docs[qid].add(corpus_id)
    queries = dict(zip(query_ids, query_text))

    with open(os.path.join(args.data_path, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(os.path.join(args.data_path, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False)
    with open(os.path.join(args.data_path, "relevant_docs.json"), "wb") as f:
        import pickle
        pickle.dump(relevant_docs, f)