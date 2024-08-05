import json
import argparse


def jsonl_to_json(jsonl_path, json_path):
    # 读取 JSONL 文件，每一行是一个 JSON 对象
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 将数据写入 JSON 文件，使用 UTF-8 编码，保留中文字符，美化输出
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON")
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    jsonl_to_json(args.input, args.output)

if __name__ == '__main__':
    main()
