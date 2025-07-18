import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
load_dotenv()

with open("eval_results.json", "r") as f:
    data = json.load(f)



client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)