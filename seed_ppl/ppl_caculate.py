# import json
# import torch
# import argparse
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Argument parser
# parser = argparse.ArgumentParser(description='Calculate PPL for JSON data')
# parser.add_argument("--model", required=True, type=str, help="Path to the model")
# parser.add_argument("--input_file", required=True, type=str, help="Path to the input JSON file")
# parser.add_argument("--output_file", required=True, type=str, help="Path to the output JSON file")
# parser.add_argument("--batch_size", default=8, type=int, help="Batch size for processing")

# args = parser.parse_args()

# # Load model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(args.model)
# model.eval()

# def single_ppl_batch(prompts):
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(model.device)
#     target_ids = inputs.clone().to(model.device)
#     with torch.no_grad():
#         outputs = model(inputs, labels=target_ids)
#     ppl = outputs.loss.clone().detach().to("cpu")
#     ppl = torch.exp(ppl)
#     if ppl.dim() == 0:  # Check if the array is 0-dimensional
#         ppl = ppl.unsqueeze(0)
#     ppl_r = ppl.numpy()
#     return ppl_r

# def calculate_ppl(input_file, output_file, batch_size):
#     # 读取 JSON 文件
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         data = json.load(infile)
    
#     # 处理数据
#     for i in tqdm(range(0, len(data), batch_size)):
#         batch_data = data[i:i+batch_size]
#         batch_prompts = [item['instruction'] + item['input'] for item in batch_data]
#         batch_pairs = [item['instruction'] + item['input'] + item['output'] for item in batch_data]
        
#         prompt_ppl_batch = single_ppl_batch(batch_prompts)
#         pair_ppl_batch = single_ppl_batch(batch_pairs)
        
#         for j, item in enumerate(batch_data):
#             item['prompt_ppl'] = prompt_ppl_batch[j]
#             item['pair_ppl'] = pair_ppl_batch[j]

#     # 保存更新后的数据到 output_file
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         json.dump(data, outfile, ensure_ascii=False, indent=2)

# if __name__ == '__main__':
#     calculate_ppl(args.input_file, args.output_file, args.batch_size)




import json
import numpy as np
import json
import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
# from utils import single_ppl
import argparse
import tqdm as tqdm

from transformers import AutoTokenizer, LlamaForCausalLM
from filelock import FileLock


from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" # the device to load the model onto

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--input_file", default='./seed_tasks.jsonl', type=str)
parser.add_argument("--output_file", default='./generated.jsonl', type=str)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

model.eval()

def single_ppl(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_ids = inputs.clone().to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=target_ids)
    ppl = outputs.loss.clone().detach().to("cpu")
    del inputs, target_ids, outputs
    ppl = torch.exp(ppl)
    ppl_r = ppl.numpy().item()
    return ppl_r

def calculate_ppl(input_file, output_file):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # 处理每条数据
    for item in data:
        output_str = item['output']
        pair_str = item['instruction'] + item['input'] + item['output']
        output_ppl = single_ppl(output_str)
        pair_ppl = single_ppl(pair_str)
        item['output_ppl'] = output_ppl
        item['pair_ppl'] = pair_ppl
    
    # 保存更新后的数据到 output_file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    calculate_ppl(args.input_file, args.output_file)
