"""Self-instruction generation. Adapted from [https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py]"""
import time
import json
import os
import sys
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import fire
import tqdm
import numpy as np
from rouge_score import rouge_scorer
from utils import vllm_generator
import math
import pandas as pd
from scipy.stats import rv_discrete

import math
import random

def weighted_sample_without_replacement(population, weights, k):
    """Sample k unique elements from population based on weights."""
    assert len(population) == len(weights), "Population and weights must have the same size"
    assert k <= len(population), "Can't sample more elements than exist in population"
    
    samples = []

    for _ in range(k):
        sampled = random.choices(population, weights, k=1)[0]
        samples.append(sampled)
        
        # Mark that the selected element is used
        index = population.index(sampled)
        population[index]['used'] = True
        weights[index] = 0

        # # Remove the sampled element from population and weights
        # index = population.index(sampled)
        # del population[index]
        # del weights[index]

        # Normalize the weights to ensure they sum to 1 (optional, but might be a good practice)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    return samples

def sample_by_ppl(data, num, sampling_method='uniform'):
    if sampling_method == 'uniform':
        return random.sample(data, num)
    else:
        weights = []
        for d in data:
            ans = d['ppl']
            if sampling_method == 'inverse':
                # Using inverse of PPL for weights
                weight = 1.0 / ans if ans > 1e-5 else 1.0  # avoid division by zero
                weight = 0 if d['used'] else weight # If used, it doesn't sample anymore; that is, it sets weight to 0
            else:
                weight = ans if ans < 200 else 200
            weights.append(weight)

        # Normalize the weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        samples = weighted_sample_without_replacement(data, weights, num)
        return samples



def encode_prompt(prompt_instructions, meta_prompt):
    """Encode multiple prompt instructions into a single string."""
    # print(prompt_instructions)

    # print(' ==== ' * 20)

    # print(meta_prompt)
    prompt = meta_prompt + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        instruction, input = task_dict["instruction"], task_dict["input"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def filter_instruction(instruction):
    # filter out instruction which input field contains <noinput> as we consider them as instruction with missing information
    if "noinput" in instruction["input"].lower() or "no input" in instruction["input"].lower() or "noinput" in instruction["instruction"]:
        return False
    # filter out instruction which instruction and input field contains unmatched pattern such as regular expression".\s+(Instruction|Input):"
    if re.search(r".\s+(Instruction|Input):", instruction["instruction"]) or re.search(r".\s+(Instruction|Input|Output):", instruction["input"]):
        return False
    # filter out instruction and input field which contains website such as regular expression "(http|https):", because llm may not be able to access external links.
    if re.search(r"https?:", instruction["instruction"]) or re.search(r"https?:", instruction["input"]):
        return False
    return True

def post_process_gpt3_response(num_prompt_instructions, response):
    # print(raw_instructions)
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response
    raw_instructions = re.split("###", raw_instructions)
    # if the decoding stops due to length, the last example is likely truncated so we discard it
    raw_instructions = raw_instructions[:-1]
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input):", inst)
        if len(splitted_data) != 5:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = ""
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        if any(find_word_in_string(word, input) for word in blacklist):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        generated_instruction = {"instruction": inst, "input": input, "output": output}
        if filter_instruction(generated_instruction):
            instructions.append(generated_instruction)
        else:
            print("fitlered_instruction: ",generated_instruction)
            continue
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

def sample_by_density(data, num):
    df = pd.DataFrame(data)
    
    # df = df[df['cluster_id'] != -1]
    samples = []
    clusters = df['cluster_id'].value_counts().sort_values(ascending=False).index.tolist()

    num_per_cluster = int(np.ceil(num / len(clusters)))

    for cluster in clusters:
        if len(samples) >= num:
            break
            
        cluster_data = df[df['cluster_id'] == cluster]

        if len(cluster_data) <= num_per_cluster:
            samples.append(cluster_data)
        else:
            samples.append(cluster_data.sample(n=num_per_cluster))

    sample_df = pd.concat(samples)
    return sample_df.to_dict(orient='records')


def weighted_sample_without_replacement_ppl(data, weights, num_samples):
    """
    Sample without replacement using weighted probabilities.
    """
    distribution = rv_discrete(values=(np.arange(len(data)), weights))
    indices = distribution.rvs(size=num_samples)
    return [data[i] for i in indices]

def sample_by_density_ppl(data, num, sampling_method='uniform'):
    df = pd.DataFrame(data)

    # Filter out rows where cluster_id is -1
    # df = df[df['cluster_id'] != -1]
    
    samples = []
    clusters = df['cluster_id'].value_counts().sort_values(ascending=False).index.tolist()

    num_per_cluster = int(np.ceil(num / len(clusters)))

    for cluster in clusters:
        if len(samples) >= num:
            break
            
        cluster_data = df[df['cluster_id'] == cluster]

        if sampling_method == 'uniform':
            sample = cluster_data.sample(n=min(num_per_cluster, len(cluster_data)))
        else:
            # Calculate weights based on ppl
            weights = []
            for _, row in cluster_data.iterrows():
                ans = row['ppl']
                if sampling_method == 'inverse':
                    weight = 1.0 / ans if ans > 1e-5 else 1.0
                else:  # Assuming 'density' as the other option
                    weight = ans if ans < 200 else 200
                weights.append(weight)

            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            sample_indices = weighted_sample_without_replacement_ppl(cluster_data.index.tolist(), weights, min(num_per_cluster, len(cluster_data)))
            sample = cluster_data.loc[sample_indices]

        samples.append(sample)

    sample_df = pd.concat(samples)
    return sample_df.to_dict(orient='records')


# def sample_by_ppl(sentences, size, prob):
#     return np.random.choice(a=sentences, size=size, replace=False, p=prob).tolist()

def generate_instructions(
    generator,
    seed_instruction_data,
    meta_prompt_file: str,
    origin_samples: int,
    sample_methods: str,
    form: str,
    all_have_ppl=[],
    num_instructions_to_generate=100,
    num_prompt_instructions=5,
    request_batch_size=32,
    num_cpus=16,
    num_gpus=4
):


    with open(meta_prompt_file, "r") as f:
        meta_prompt = f.read()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in all_have_ppl 
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    on_epoch_seed_instruction_data = seed_instruction_data + all_have_ppl 
    origin_len = len(on_epoch_seed_instruction_data)
    print(" on epoch ", origin_len)
    last_epoch_instructions = []

    #time.sleep(1000)
    pre_process_start = time.time()
    results = []
    while len(results) < num_instructions_to_generate: 
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            # @jiajun ppl based instruction add 
            # prompt_instructions = random.sample(on_epoch_seed_instruction_data, num_prompt_instructions)
            if form == 'ppl':
                if len(all_have_ppl) > num_prompt_instructions:
                    prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
                    # origin_prompt_instructions = sample_by_ppl(seed_instruction_data, origin_samples)
                    # generated_prompt_instructions = sample_by_ppl(all_have_ppl, num_prompt_instructions - origin_samples, sample_methods)
                    # prompt_instructions = origin_prompt_instructions + generated_prompt_instructions
                    # prompt_instructions = random.sample(prompt_instructions, len(prompt_instructions))
                else:
                    prompt_instructions = sample_by_ppl(seed_instruction_data, num_prompt_instructions)
            elif form == 'density':   
                if len(all_have_ppl) > num_prompt_instructions:
                    origin_prompt_instructions = sample_by_density(seed_instruction_data, origin_samples)
                    generated_prompt_instructions = sample_by_density(all_have_ppl, num_prompt_instructions - origin_samples)
                    prompt_instructions = origin_prompt_instructions + generated_prompt_instructions
                    prompt_instructions = random.sample(prompt_instructions, len(prompt_instructions))
                else:
                    prompt_instructions = sample_by_density(seed_instruction_data, num_prompt_instructions)
            elif form == 'all':
                if len(all_have_ppl) > num_prompt_instructions:
                    origin_prompt_instructions = sample_by_density_ppl(seed_instruction_data, origin_samples)
                    generated_prompt_instructions = sample_by_density_ppl(all_have_ppl, num_prompt_instructions - origin_samples, sample_methods)
                    prompt_instructions = origin_prompt_instructions + generated_prompt_instructions
                    prompt_instructions = random.sample(prompt_instructions, len(prompt_instructions))
                else:
                    prompt_instructions = sample_by_density_ppl(seed_instruction_data, num_prompt_instructions)
            else:
                print(' we do not support this methods ')
            prompt = encode_prompt(prompt_instructions, meta_prompt=meta_prompt)
            # print(prompt)
            # exit()
            # print(prompt[0])
            batch_inputs.append(prompt)
        print(f"batch_inputs[0]:\n{batch_inputs[0]}")
        ress = generator.generate(batch_inputs)
        # print(ress[0])
        instruction_data = []
        for result in ress:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            # print(new_instructions)
            instruction_data += new_instructions
        total = len(instruction_data)
        # exit()
        # print(instruction)
        print(" === " * 20)
        print(" generate new data {} ".format(total))
        print(" === " * 20)
        keep = 0
        # results = []
        with Pool(num_cpus) as p:
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if max(rouge_scores) > 0.7:
                    continue
                else:
                    keep += 1
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                instruction_data_entry["ppl"] = None 
                instruction_data_entry["cluster_id"] = -1 
                results.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
            print("results: ", len(results))
    return results



# if __name__ == "__main__":
#     fire.Fire(main)