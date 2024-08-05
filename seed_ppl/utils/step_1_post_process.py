import json
import argparse
import re
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
ppl_model_path = input("Please input the model path: ")
ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_path, torch_dtype="auto", device_map="auto")
ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_path)

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# def add_system_prompt(data):
#     # add system prompt key-value
#     for entry in data:
#         entry["system"] = "You are a helpful assistant."
#     print("add system prompt sucessfully")
#     return data

def delete_special_str(data):
    # 定义正则表达式，仅匹配开头的"答案"
    pattern = r'^(答案[:：]\s?)+'
    print("pre_process data:\n")
    for entry in data:
        new_output = re.sub(pattern, '', entry['output'])
        if new_output != entry['output']:
            print(f"{entry}\n")
            entry['output'] = new_output
    return data

def filter_output_empty(data):
    """Filters the data with empty output."""
    filtered_data = []
    for entry in data:
        if entry['output'] != '':
            filtered_data.append(entry)
        else:
            print(f"empty output item: {entry}")
    print(f"filterd {len(data)-len(filtered_data)} empty output, {len(filtered_data)} left")
    return filtered_data

def caculate_instruction_ppl(data):
    print("caculating instruction ppl...\n")
    # model_path = ppl_model_path
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ppl_model
    tokenizer = ppl_tokenizer
    model.eval()
    for entry in tqdm(data):
        query = entry['instruction']
        inputs = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
        target_ids = inputs.clone().to(model.device)
        with torch.no_grad():
            outputs = model(inputs, labels=target_ids)
        ppl = outputs.loss.clone().detach().to("cpu")
        del inputs, target_ids, outputs
        ppl = torch.exp(ppl)
        ppl_r = ppl.numpy().item()
        entry['instruction_ppl'] = ppl_r
    
    return data

def caculate_instruction_inout_ppl(data):
    print("caculating instruction+input ppl...\n")
    # model_path = ppl_model_path
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ppl_model
    tokenizer = ppl_tokenizer
    model.eval()
    for entry in tqdm(data):
        query = entry['instruction'] + '\n' + entry['input']
        inputs = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
        target_ids = inputs.clone().to(model.device)
        with torch.no_grad():
            outputs = model(inputs, labels=target_ids)
        ppl = outputs.loss.clone().detach().to("cpu")
        del inputs, target_ids, outputs
        ppl = torch.exp(ppl)
        ppl_r = ppl.numpy().item()
        entry['instruction_input_ppl'] = ppl_r
    
    return data

def caculate_ppl_random(data):
    print("caculating instruction+random str ppl...\n")
    # model_path = ppl_model_path
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ppl_model
    tokenizer = ppl_tokenizer
    model.eval()
    random_str_list = ['&', '()', 'N/A']
    for entry in tqdm(data):
        tmp_ppl=0
        for radom_str in random_str_list:
            query = entry['instruction'] + '\n' + radom_str
            inputs = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
            target_ids = inputs.clone().to(model.device)
            with torch.no_grad():
                outputs = model(inputs, labels=target_ids)
            ppl = outputs.loss.clone().detach().to("cpu")
            del inputs, target_ids, outputs
            ppl = torch.exp(ppl)
            ppl_r = ppl.numpy().item()
            tmp_ppl += ppl_r    
        entry['radom_ppl'] = tmp_ppl/len(random_str_list)
    
    return data

def caculate_ppl_output_and_total(data):
    print("caculating output_and_total ppl...\n")
    # model_path = ppl_model_path
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ppl_model
    tokenizer = ppl_tokenizer
    model.eval()
    for entry in tqdm(data):
        query = entry['output']
        inputs = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
        target_ids = inputs.clone().to(model.device)
        with torch.no_grad():
            outputs = model(inputs, labels=target_ids)
        ppl = outputs.loss.clone().detach().to("cpu")
        del inputs, target_ids, outputs
        ppl = torch.exp(ppl)
        ppl_r = ppl.numpy().item()
        entry['out_ppl'] = ppl_r
        
        query1 = entry['instruction'] + '\n' + entry['input'] + '\n' + entry['output']
        inputs1 = tokenizer(query1, return_tensors="pt")["input_ids"].to(model.device)
        target_ids1 = inputs1.clone().to(model.device)
        with torch.no_grad():
            outputs1 = model(inputs1, labels=target_ids1)
        ppl1 = outputs1.loss.clone().detach().to("cpu")
        del inputs1, target_ids1, outputs1
        ppl1 = torch.exp(ppl1)
        ppl_r1 = ppl1.numpy().item()
        entry['total_ppl'] = ppl_r1
    return data

def caculate_cluster_id(data):
    print("caculating cluster id...\n")
    model_name = '/ML-A100/team/mm/eamon/self_instruction/liang_ppl/models/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    batch_size = 128
    n_clusters = 1000 if len(data) > 1000 else len(data)//2
    print(f"n_clusters: {n_clusters}")
    # def update_cluster(filename, n_clusters,batch_size=100,output_file='./'):
    # 读取数据
    df = pd.DataFrame(data)

    # 初始化一个空的numpy数组来保存所有的embeddings
    all_embeddings = np.empty((0, model.config.hidden_size))

    # 将数据分为多个批次，然后一次处理一个批次
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]
        inputs = tokenizer(list(batch['instruction'] + ' ' + batch['input']), 
                           padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = inputs.to(model.device)  # 将输入数据移动到指定设备上

        # 获取文本的BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 将计算结果移动回 CPU

        # 将这个批次的embeddings添加到all_embeddings中
        all_embeddings = np.vstack((all_embeddings, embeddings))

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_embeddings)

    # 添加聚类结果到数据中
    df['cluster_id'] = kmeans.labels_

    # 按cluster_id排序
    df = df.sort_values(by=['cluster_id'])
    sorted_list = df.to_dict('records')
    return sorted_list

def caculate_reward_score(data):
    print("caculating reward score...\n")
    reward_name = "/ML-A100/team/mm/eamon/self_instruction/models/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).cuda(), AutoTokenizer.from_pretrained(reward_name)
    for element in tqdm(data):
        instruction = element['instruction']
        _input = ''
        if 'input' in element.keys():
            _input = element['input']
        _output = element['output']
        question = ''
        if _input == '':
            question = instruction
        else:
            question = instruction + '\n' +_input
        
        answer = _output
        
        try:
            inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
            score = rank_model(**inputs).logits[0].detach()
        except Exception as error:
            print("An error occurred:", error)
            print("\nQuestion:\n", question)
            print("Output:\n", _output)
            continue
        # final_result = {'instruction':instruction,'input':_input,'output':_output,'reward_score':float(score)}
        element['reward_score'] = float(score)
        
    sorted_list = sorted(data, key=lambda x: x['reward_score'], reverse=True)
    return sorted_list 

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in data:
            entry.pop('system', None)
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Filter the top 50% entries with the smallest 'ppl' values from a JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read the data from the input file
    data = read_jsonl(args.input_file)
    ## process the data
    # data = add_system_prompt(data)
    # data = delete_special_str(data)
    filtered_data = filter_output_empty(data)
    ## caculate some metrics
    # data_with_ppl = caculate_instruction_inout_ppl(filtered_data)
    # save_jsonl(data_with_ppl, args.output_file)
    # data_with_random_ppl = caculate_ppl_random(filtered_data)
    # save_jsonl(data_with_random_ppl, args.output_file)
    # data_with_output_and_total_ppl = caculate_ppl_output_and_total(filtered_data)
    # save_jsonl(data_with_output_and_total_ppl, args.output_file)
    data_with_instruction_ppl=caculate_instruction_ppl(filtered_data)
    save_jsonl(data_with_instruction_ppl, args.output_file)
    # data_with_culster_id = caculate_cluster_id(data_with_ppl)
    # data_with_reward_score = caculate_reward_score(data_with_culster_id)
    
    # Save the processed data to the output file
    # save_jsonl(data_with_reward_score, args.output_file)
    # save_jsonl(data_with_output_and_total_ppl, args.output_file)

if __name__ == '__main__':
    main()
