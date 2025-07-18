import ast
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re
from tqdm import tqdm
load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa"
)


def llm_generate(prompt: str):
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

with open("eval_results.json", "r") as f:
    data = json.load(f)

for item in tqdm(data):
    predictions = item["predictions"]
    question = item["question"]
    prompt = f"Question: {question}\n{predictions}\nThis is outputs from a legal document retrieval system. Remove any irrelevant information and return a list of true predictions in the format: [{{'law_id': '...', 'article_id': '...'}}]. Only include predictions that are relevant to the question and help to answer the question."
    clean_text = llm_generate(prompt)
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
        is_match = False
        for answer in ground_truth:
            if article["law_id"] == answer["law_id"] and article["article_id"] == answer["article_id"]:
                true_positive += 1
                is_match = True
                break
        if not is_match:
            false_positive += 1
    precision = true_positive / (true_positive + false_positive + 1e-20)
    recall = true_positive / (len(ground_truth) + 1e-20)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-20)
    total_f2 += f2
    total_precision += precision
    total_recall += recall
avg_f2 = total_f2 / len(data)
avg_precision = total_precision / len(data)
avg_recall = total_recall / len(data)
print(f"Average F2: {avg_f2}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
