import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from qwen_vllm import format_instruction, process_inputs, compute_logits

def create_reranker(model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
    """
    Create a reranker model and tokenizer.
    
    Returns:
        model: The pre-trained reranker model.
        tokenizer: The tokenizer for the reranker model.
    """
    others = None
    if "Qwen" in model_name:
        # For Qwen models, use the specific setup
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto").eval()
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        task = 'Given a legal query, retrieve relevant laws that answer the query'
        
        others = {
            'task': task,
            'max_length': max_length,
            'prefix_tokens': prefix_tokens,
            'suffix_tokens': suffix_tokens,
            'token_true_id': token_true_id,
            'token_false_id': token_false_id,
            'model_name': model_name
        }
        return model, tokenizer, others
    else:
        # For other models, use a generic setup
        MAX_LENGTH = 2304
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        others = {
            'max_length': MAX_LENGTH,
            'model_name': model_name
        }
    return model, tokenizer, others

@torch.no_grad()
def reranking(
    model,
    tokenizer,
    query: str,
    documents: List[str],
    others:Dict[str, any]
) -> List[float]:
    """
    Rerank documents based on a query using a pre-trained model.
    
    Args:
        query (str): The search query.
        documents (List[str]): List of documents to be reranked.
    Returns:
        List[float]: A list of scores.
    """
    scores = []
    if "Qwen" not in others["model_name"]:
        pairs = [[query, doc] for doc in documents]
        max_length = others['max_length']
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
    else:
        task = others['task']
        max_length = others['max_length']
        prefix_tokens = others['prefix_tokens']
        suffix_tokens = others['suffix_tokens']
        token_true_id = others['token_true_id']
        token_false_id = others['token_false_id']
        pairs = [format_instruction(task, query, doc) for doc in documents]
        inputs = process_inputs(pairs, model, tokenizer, max_length, prefix_tokens, suffix_tokens)
        scores = compute_logits(model, inputs, token_true_id, token_false_id)
    return scores