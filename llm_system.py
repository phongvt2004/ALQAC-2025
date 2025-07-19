import ast
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re
from tqdm import tqdm
load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
model_name = "Qwen/Qwen3-0.6B-FP8"

# load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="sdpa"  # Use flash attention for better performance
# )
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1000)

# Initialize the vLLM engine
llm = LLM(model=model_name)

def llm_generate(prompt: str):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Set to False to strictly disable thinking
    )
    

    # Generate outputs
    outputs = llm.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()
    # # prepare the model input
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # # conduct text completion
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=32768
    # )
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # # parsing thinking content
    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    #     index = 0
    
    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content
