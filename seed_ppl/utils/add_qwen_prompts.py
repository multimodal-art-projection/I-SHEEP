import json
import argparse

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data



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

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        # for entry in data:
        #     json.dump(entry, file, ensure_ascii=False)
        #     file.write('\n')

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
    # Filter the data
    filtered_data = add_qwen_prompts(data)
    # Save the filtered data to the output file
    save_jsonl(filtered_data, args.output_file)

if __name__ == '__main__':
    main()
