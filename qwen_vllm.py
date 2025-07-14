# Requires vllm>=0.8.5
import logging
from typing import Dict, Optional, List

import json
import logging

import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc
import math
from vllm.inputs.data import TokensPrompt

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')
model = LLM(model='Qwen/Qwen3-Reranker-0.6B', max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=0.8)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
max_length=8192
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
task = 'Given a legal query, retrieve relevant laws that answer the query'
sampling_params = SamplingParams(temperature=0, 
    max_tokens=1,
    logprobs=20, 
    allowed_token_ids=[true_token, false_token],
)

def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages =  tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        token_count = len(outputs[i].outputs[0].token_ids)
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores

def reranking(
    query: str,
    documents: List[str]
) -> List[float]:
    """
    Rerank documents based on a query using a pre-trained model.
    
    Args:
        query (str): The search query.
        documents (List[str]): List of documents to be reranked.
    Returns:
        List[float]: A lisr of scores.
    """
    
    
    pairs = list(zip([query] * len(documents), documents))
    inputs = process_inputs(pairs, task, max_length-len(suffix_tokens), suffix_tokens)
    scores = compute_logits(model, inputs, sampling_params, true_token, false_token)
    return scores

def release_model():
    destroy_model_parallel()