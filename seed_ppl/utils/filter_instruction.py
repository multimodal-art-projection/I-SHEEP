import json
import argparse

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def filter_half_by_ppl(data):
    """Filters the half of the data with the smallest 'ppl' values."""
    sorted_data = sorted(data, key=lambda x: x['ppl'])
    half_size = (len(sorted_data)+1) // 2
    print(f"the final ppl is: {sorted_data[half_size-1]['ppl']}")
    return sorted_data[:half_size]

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
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
    # Filter the data
    filtered_data = filter_half_by_ppl(data)
    # Save the filtered data to the output file
    save_jsonl(filtered_data, args.output_file)

if __name__ == '__main__':
    main()
