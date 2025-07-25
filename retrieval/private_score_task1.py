import os
import json
from utils import calculate_f2
submission_path = "ALQAC_2025_submission"
ground_truth_path = "../ALQAC_2025_data/alqac25_private_test_task2.json"

with open(ground_truth_path, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

for file_name in os.listdir(submission_path):
    file_path = os.path.join(submission_path, file_name)
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    with open(file_path, "r", encoding="utf-8") as f:
        submission_data = json.load(f)
    for item in submission_data:
        question_id = item["question_id"]
        ground_truth = None
        for ground_truth_item in ground_truth_data:
            if ground_truth_item["question_id"] == question_id:
                ground_truth = ground_truth_item
                break
        relevant_articles = ground_truth["relevant_articles"]
        predictions = item["relevant_articles"]
        
        actual_positive = len(relevant_articles)
        true_positive = 0
        false_positive = 0
        for pred in predictions:
            is_match = False
            for article in relevant_articles:
                if pred["law_id"] == article["law_id"] and pred["article_id"] == article["article_id"]:
                    true_positive += 1
                    is_match = True
                    break
            if not is_match:
                false_positive += 1
        precision = true_positive / (true_positive + false_positive + 1e-20)
        recall = true_positive / actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    k = len(submission_data)
    print(f"File: {file_name}")
    print(f"Average F2: \t\t\t\t{total_f2/k}")
    print(f"Average Precision: \t\t\t{total_precision/k}")
    print(f"Average Recall: \t\t\t{total_recall/k}")
    print("-" * 50)
    print()
    