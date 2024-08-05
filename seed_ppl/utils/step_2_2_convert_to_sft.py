import json
import argparse
import re

# Filter out ill-formed outputs and filter out the top 50% of the data according to the ppl of Instrution.


def read_jsonl(file_path):
    """Reads a JSON file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # data = json.load(file)
        for line in file:
            data.append(json.loads(line))
    return data

def contains_specified_patterns(text):
    # \nassistant\n, \nassistan:,  \nSystem: , \nSystem\n, \nHuman:,  \nHuman\n
    pattern1 = r"(?i)\n(assistant|system|human|question|output)[:\n]"
    # Answer Choices in output usually indicate that the model does not directly answer the question.
    pattern2 = r"Answer Choices:"
    pattern3 = r'A\.[\s\S]*B\.[\s\S]*C\.'
    flag = re.search(pattern1, text) or re.search(pattern2, text) or re.search(pattern3, text)
    return flag is not None 



def filter_ill_formed(data):
    """Filters out ill-formed outputs."""
    filtered_data = []
    print(f"Filtered out ill-formed output:\n")
    for entry in data:
        if contains_specified_patterns(entry['output']):
            print(f"{entry}")
        else:
            filtered_data.append(entry)
    print(f"filter {len(data)-len(filtered_data)} ill-formed data")
    return filtered_data

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
    
def add_qwen_prompts(data):
    # Add system prompts for every entry, system prompt is "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    # Reformatting the prompt, which is {instruction\ninput}
    qwen_data=[]
    for entry in data:
        entry['system_prompt'] = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        entry['instruction'] = "<|im_start|>user\n" + entry['instruction'].strip()
        entry['input'] = entry["input"].strip() + "<|im_end|>\n" + "<|im_start|>assistant\n"
        entry['output'] = entry["output"].strip() + "<|im_end|>"
        qwen_data.append(entry)
    return qwen_data

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
    filtered_data = filter_ill_formed(data)
    filtered_data = add_qwen_prompts(filtered_data)
    # filtered_data = filter_half_by_ppl(filtered_data, data_len)
    # Save the filtered data to the output file
    save_json(filtered_data, args.output_file)

if __name__ == '__main__':
    main()
