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
parser.add_argument("--seed_task_path", default='./seed_tasks.jsonl', type=str)
parser.add_argument("--output", default='./generated.jsonl', type=str)
parser.add_argument("--num_generation", default=10000, type=int)

args = parser.parse_args()
# model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
# # model.to_bettertransformer()
# tokenizer = AutoTokenizer.from_pretrained(args.model)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

model.eval()
lock_path = args.output + '.lock'
lock = FileLock(lock_path, timeout=1000)  # 10秒超时时间


# _recoverstandard("logs/1k_epoch0.jsonl", model, tokenizer)   #这一步是所有task的起点, fixed during one epoch, 在这一步完成前generator不动
# generator更新， ppl值设为-100，判断是否有-100的，仅更新-100的
def single_ppl(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_ids = inputs.clone().to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=target_ids)
    ppl = outputs.loss.clone().detach().to("cpu")
    del inputs, target_ids, outputs
    ppl = torch.exp(ppl)
    ppl_r = ppl.numpy().item()
    return ppl_r


def update_ppl(filename):
    print(' updating ')
    with lock:
        print(' locked ')
        with open(filename, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()

            for line in tqdm.tqdm(lines):
                try:
                    data = json.loads(line.strip())
                    if data['ppl'] is None:
                        query = data['instruction'] + '\n' + data['input']
                        data['ppl'] = single_ppl(query, model, tokenizer)
                        # data['used'] = data['used'] if 'used' in data else False
                        # data['ppl'] = single_ppl(data['instruction'], model, tokenizer)
                        # if data['ppl'] > 40:
                        #     f2.write(json.dumps(data, ensure_ascii=False) + '\n')
                        #     continue
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    elif data['ppl'] == 0:
                        query = data['instruction'] + '\n' + data['input']
                        data['ppl'] = single_ppl(query, model, tokenizer)
                        # data['used'] = data['used'] if 'used' in data else False
                        # data['ppl'] = single_ppl(data['instruction'], model, tokenizer)
                        # if data['ppl'] > 40:
                        #     f2.write(json.dumps(data, ensure_ascii=False) + '\n')
                        #     continue
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    else:
                        # f.write(line, ensure_ascii=False)
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                except Exception as e:
                    print("An exception occurred:", e)
                    print("except:", line)
                
# update_ppl(args.seed_task_path)

len_file = 0
with lock:
    len_file = len(open(args.output).readlines())
while True:
    import time

    print("Starting sleep...")
    time.sleep(5)
    print("Finished sleeping!")
    #     update_ppl('file.jsonl')
    with lock:
        # print('locking ???')
        if len_file != len(open(args.output).readlines()):
            start_time=time.time()
            update_ppl(args.output)
            print("update_ppl cost time: ", time.time()-start_time)
            len_file = len(open(args.output).readlines())
            print("len_file: ", len_file)
        if len(open(args.output).readlines()) > args.num_generation:
            break

