# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM

model = LLM(model="Qwen/Qwen3-Embedding-0.6B", task="embed")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def qwen_embedding(query: str, type: str = "query") -> torch.Tensor:
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    if type == "query":
        instruction = get_detailed_instruct(task, query)
    else:
        # For retrieval documents, we can just use the query as is
        instruction = query
    
    model = LLM(model="Qwen/Qwen3-Embedding-0.6B", task="embed")
    outputs = model.embed([instruction])
    
    return outputs[0].outputs.embedding
