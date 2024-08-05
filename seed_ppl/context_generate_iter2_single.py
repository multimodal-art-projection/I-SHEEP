import json
import os
from functools import partial
from multiprocessing import Pool
from utils import vllm_generator
import tqdm
import numpy as np
from rouge_score import rouge_scorer
from generate_instruction_iter2 import generate_instructions
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--sample_methods", default='', type=str)
parser.add_argument("--form", default=['density', 'ppl', 'all'], type=str)
parser.add_argument("--seed_task_path", default='./seed_tasks.jsonl', type=str)
parser.add_argument("--meta_prompt_path", default='./prompt.txt', type=str)
parser.add_argument("--output", default='./generated.jsonl', type=str)
parser.add_argument("--gpus", default=8, type=int)
parser.add_argument("--num_generation", default=10000, type=int)
# parser.add_argument("--use_vllm", action='store_true', default=False)
# parser.add_argument("--form", default='alpaca', type=str)
# parser.add_argument("--options", default='sft', type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--origin_samples", default=0, type=int)

args = parser.parse_args()
stops = ['17']
generator = vllm_generator(args.model, args.gpus, stops)
# lock_path = args.output + '.lock'
# from filelock import FileLock

# lock = FileLock(lock_path, timeout=10000)  # 10秒超时时间


def generate_context(all_data, bz, origin_samples):
    with open(args.seed_task_path, "r") as f:
        seed_tasks = [json.loads(l) for l in f]
    seed_instruction_data = seed_tasks
    # seed_instruction_data = [
    #     {
    #         "instruction": t["instruction"],
    #         "input": t["input"],
    #         "output": t["output"],
    #         "ppl": t['ppl'] if args.form=='ppl' or args.form=='all' else -1,
    #         "cluster_id": t['cluster_id'] if args.form=='density' or args.form=='all'  else -1
    #     } for t in seed_tasks
    # ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    result = generate_instructions(generator,
                                   seed_instruction_data,
                                   args.meta_prompt_path,
                                   origin_samples,
                                   args.sample_methods,
                                   args.form,
                                   all_have_ppl=all_data
                                   )
    return result


# def add_new_context(filename):
#     with lock:
#         with open(filename, 'r') as f:
#             lines = f.readlines()
#         all_data = []
#         for line in lines:
#             try:
#                 if json.loads(line)['ppl'] is not None:
#                     temp_data =json.loads(line.strip())
#                     all_data.append(temp_data)
#             except Exception as e:
#                 print("error line: ",line)
#         all_have_ppl = all(d['ppl'] is not None for d in all_data)
#         if len(all_data) == 0:
#             all_have_ppl = False
#         if args.form == 'density':
#             all_have_ppl = True

#         # print(len(lines))
#     if all_have_ppl or len(lines) == 0:
#         # with open(filename, 'a') as f:
#         print("filename", filename)
#         print(len(all_data))
#         new_context = generate_context(all_data, args.batch_size, args.origin_samples)
#         with lock:
#             with open(args.output, 'a', encoding='utf-8') as f:
#                 print("new_context:", len(new_context))
#                 # print(new_context)
#                 for r in new_context:
#                     print(r)
#                     # f.write(json.dumps({"context": new_context, "ppl": None}) + '\n')
#                     f.write(json.dumps(r, ensure_ascii=False) + '\n')
#                     f.flush()

def add_new_context(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    all_data = []
    for line in lines:
        try:
            if json.loads(line)['ppl'] is not None:
                temp_data =json.loads(line.strip())
                all_data.append(temp_data)
        except Exception as e:
            print("error line: ",line)
    all_have_ppl = all(d['ppl'] is not None for d in all_data)
    if len(all_data) == 0:
        all_have_ppl = False
    if args.form == 'density':
        all_have_ppl = True

        # print(len(lines))
    all_have_ppl = True
    if all_have_ppl or len(lines) == 0:
        # with open(filename, 'a') as f:
        print("filename", filename)
        print(len(all_data))
        new_context = generate_context(all_data, args.batch_size, args.origin_samples)
        with open(args.output, 'a', encoding='utf-8') as f:
            print("new_context:", len(new_context))
            # print(new_context)
            for r in new_context:
                # print(r)
                # f.write(json.dumps({"context": new_context, "ppl": None}) + '\n')
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()

while True:
    add_new_context(args.output)
    if len(open(args.output).readlines()) > args.num_generation:
        print("generate successfully")
        break

# len_file = 0
# with lock:
#     len_file = len(open(args.output).readlines())  
# while True:
#     add_new_context(args.output)
#     with lock:
#         if len(open(args.output).readlines()) > args.num_generation:
#             break
# add_new_context('./test.jsonl')

