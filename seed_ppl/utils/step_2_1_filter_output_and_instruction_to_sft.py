import json
import argparse
import re

# def filter_output_empty(data):
#     """Filters the data with empty output."""
#     filtered_data = []
#     for entry in data:
#         if entry['output'] != '':
#             filtered_data.append(entry)
#         else:
#             print(f"empty output item: {entry}")
#     print(f"filterd {len(data)-len(filtered_data)} empty output, {len(filtered_data)} left")
#     return filtered_data

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

def read_jsonl(file_path):
    """Reads a JSON file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # data = json.load(file)
        for line in file:
            data.append(json.loads(line))
    return data

def contains_specified_patterns_output(text):
    # \nassistant\n, \nassistan:,  \nSystem: , \nSystem\n, \nHuman:,  \nHuman\n
    pattern1 = r"(?i)\n(assistant|system|human|question|output|user)[:\n]"
    # Answer Choices in output usually indicate that the model does not directly answer the question.
    # Some special patterns are found manually, which represent non-standard Single Round Question Answering patterns
    pattern2 = r'(?i)Answer\sChoices:|noinput|no input|\n\n?can you|do you .{1,50}?|\n\n.{1,100}\?'
    # pattern3 = r'A\.[\s\S]*B\.[\s\S]*'
    pattern3 = r'(?:\b[A-Ea-e]\.|\([A-Ea-e]\))'
    pattern4 = r'[\u4e00-\u9fff]+'
    # pattern4 = r"noinput"    
    # pattern6 = r'\n\ncan you'
    flag = re.search(pattern1, text) or re.search(pattern2, text) or re.search(pattern3, text) or re.search(pattern4, text)
    return flag is not None 

def contains_specified_patterns_input(text):
    pattern1 = r"(?i)noinput|no input|<.{1,20}>|none"
    flag = re.search(pattern1, text)
    return flag is not None

def pre_process(data):
    # 定义正则表达式，仅匹配开头的"答案"
    pattern = r'^(答案[:：]\s?)+'
    print("pre_process data:\n")
    for entry in data:
        new_output = re.sub(pattern, '', entry['output'])
        if new_output != entry['output']:
            print(f"{entry}\n")
            entry['output'] = new_output
    return data

def filter_ill_formed(data):
    # Filters out ill-formed outputs.
    filtered_data = []
    print(f"Filtered out ill-formed output:\n")
    for entry in data:
        if contains_specified_patterns_output(entry['output']):
            print(f"{entry}")
        elif contains_specified_patterns_input(entry['input']) or contains_specified_patterns_input(entry['instruction']):
            # print("noinput in input or instruction\n")
            print(f"{entry}")
        else:
            filtered_data.append(entry)
    print(f"filter {len(data)-len(filtered_data)} ill-formed data")
    return filtered_data

def filter_ill_pair(data):
    # input is already present in Instruction
    print(f"Filtered out ill-pair output:\n")
    filtered_data = []
    for entry in data:
        if entry['input'] in entry['instruction'] and entry['input'].strip() != "":
            print(f"{entry}")
        else:
            filtered_data.append(entry)
    print(f"filter {len(data)-len(filtered_data)} input in instruction data")
    return filtered_data
            

def add_qwen_prompts(data):
    # Add system prompts for every entry, system prompt is "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    # Reformatting the prompt, which is {instruction\ninput}
    qwen_data=[]
    for entry in data:
        entry['system_prompt'] = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        entry['instruction'] = "<|im_start|>user\n" + entry['instruction']
        entry['input'] = entry["input"] + "<|im_end|>\n" + "<|im_start|>assistant\n"
        entry['output'] = entry["output"] + "<|im_end|>"
        qwen_data.append(entry)
    return qwen_data

def filter_by_reward_score(data, data_len):
    """maintain the data with 'reward_score'>1 """
    print("filtering by reward_score...")
    maintain_data = [entry for entry in data if entry['reward_score']>1]
    # half_size = (len(sorted_data)+1) // 2
    print(f"the filted data len is: {len(maintain_data)}")
    print(f"the final reward_score is: {maintain_data[-1]['reward_score']}")
    return maintain_data

def filter_by_multi_ppl(data, data_len):
    """maintain the data with multi ppl """
    print("filtering by multi ppl...")
    print(f"origin data len is: {len(data)}")
    maintain_data = [entry for entry in data if entry['out_ppl']==float("nan") or (entry['out_ppl']<70 and entry['out_ppl']>1.5)]
    maintain_data = [entry for entry in maintain_data if entry['instruction_input_ppl']> 2.5 and entry['instruction_input_ppl'] < 30]
    # half_size = (len(sorted_data)+1) // 2
    print(f"the maintain data len is: {len(maintain_data)}")
    return maintain_data
    
def filter_half_by_ppl(data, data_len):
    """Filters the half of the data with the smallest 'ppl' values."""
    half_size = data_len // 2
    if len(data) < half_size:
        print(f"the data size is smaller than {half_size}")
        return data
    else:
        sorted_data = sorted(data, key=lambda x: x['ppl'])
        # half_size = (len(sorted_data)+1) // 2
        print(f"origin len is {len(data)}, the filted data len is: {half_size}")
        print(f"the final ppl is: {sorted_data[half_size-1]['ppl']}")
        return sorted_data[:half_size]

def save_json(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        # for entry in data:
        #     json.dump(entry, file)
        #     file.write('\n')

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Filter the top 50% entries with the smallest 'ppl' values and filter out ill-formed ouputs from a JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read the data from the input file
    data = read_jsonl(args.input_file)
    data_len  = len(data)
    # process the data
    filtered_data = pre_process(data)
    # filtered_data = filter_output_empty(data)
    filtered_data = filter_ill_formed(filtered_data)
    filtered_data = filter_ill_pair(filtered_data)
    filtered_data = filter_output_empty(filtered_data)
    # filtered_data = add_qwen_prompts(filtered_data)
    # filtered_data = filter_half_by_ppl(filtered_data, data_len)
    # filtered_data = filter_by_reward_score(filtered_data, data_len)
    # filtered_data = filter_by_multi_ppl(filtered_data, data_len)
    # Save the filtered data to the output file
    save_json(filtered_data, args.output_file)

if __name__ == '__main__':
    main()
