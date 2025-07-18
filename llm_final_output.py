import ast
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re
load_dotenv()

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HUGGINGFACE_TOKEN"],
)

with open("eval_results.json", "r") as f:
    data = json.load(f)

for item in data:
    predictions = item["predictions"]
    question = item["question"]
    completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {
            "role": "user",
            "content": f"Question: {question}\n{predictions}\nThis is outputs from a legal document retrieval system. Remove any irrelevant information and return a list of true predictions in the format: [{{'law_id': '...', 'article_id': '...'}}]. Only include predictions that are relevant to the question and help to answer the question."
        }
    ],
    )
    
    text = completion.choices[0].message.content
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    output = ast.literal_eval(clean_text)
    item["relevant_articles"] = output
total_f2 = 0
total_precision = 0
total_recall = 0
for item in data:
    true_positive = 0
    false_positive = 0
    ground_truth = item["ground_truth"]
    relevant_articles = item["relevant_articles"]
    for article in relevant_articles:
        for answer in ground_truth:
            if article["law_id"] == answer["law_id"] and article["article_id"] == answer["article_id"]:
                true_positive += 1
            
